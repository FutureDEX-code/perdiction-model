#!/usr/bin/env python3
"""Email the latest futures tape. Gmail to the same Gmail. Never the company mailbox."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import hour_model
import mail_outlook

ROOT = Path(__file__).resolve().parent
TAPE = ROOT / "data" / "ctp_tape.jsonl"
FORECAST = ROOT / "data" / "tape_hour_forecast.json"
GRADES = ROOT / "data" / "tape_hour_grades.jsonl"
TZ = ZoneInfo("Asia/Shanghai")
NAMES = ("豆一2701", "豆二2701", "豆粕2701", "菜粕2701", "豆油2701")


def _latest(now: datetime) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not TAPE.exists():
        return latest
    cutoff = now.strftime("%Y-%m-%d %H:%M:%S")
    with TAPE.open() as handle:
        for line in handle:
            row = json.loads(line)
            at = str(row.get("at") or "")
            if row.get("name") in NAMES and at <= cutoff:
                latest[row["name"]] = row
    return latest


def _change_words(points: float) -> str:
    if points > 0:
        return f"上涨 {points:.0f}"
    if points < 0:
        return f"下跌 {abs(points):.0f}"
    return "持平"


def _load_forecast() -> dict | None:
    if not FORECAST.exists():
        return None
    try:
        saved = json.loads(FORECAST.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not saved.get("at") or not isinstance(saved.get("calls"), dict):
        return None
    return saved


def _save_forecast(now: datetime, calls: dict[str, dict]) -> None:
    FORECAST.parent.mkdir(parents=True, exist_ok=True)
    FORECAST.write_text(
        json.dumps({"at": now.strftime("%Y-%m-%d %H:%M"), "calls": calls}, ensure_ascii=False, indent=2)
    )


def _grade_line(now: datetime, name: str, last: float) -> tuple[str, dict | None]:
    """The email line, and the record to keep. A line without a verdict is not kept."""

    saved = _load_forecast()
    if saved is None:
        return "上半小时没有预测", None
    call = saved["calls"].get(name)
    if not call or call.get("side") not in ("涨", "跌", "平") or call.get("price") is None:
        return "上半小时没有预测", None
    then = datetime.strptime(saved["at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    gap = (now - then).total_seconds() / 60
    if gap < 20:
        return "上一封还不到半小时，先不核对", None
    if gap > 50:
        return "上一封隔了休市，不按半小时核对", None
    move = last - float(call["price"])
    side = call["side"]
    points = call.get("points")
    if side == "平":
        hit = abs(move) < 0.5
        predicted = "持平"
    else:
        hit = (side == "涨" and move > 0) or (side == "跌" and move < 0)
        word = "上涨" if side == "涨" else "下跌"
        predicted = word if points is None else f"{word} {abs(float(points)):.0f} 点"
    actual = _change_words(move)
    verdict = "对" if hit else "没对"
    record = {
        "graded_at": now.strftime("%Y-%m-%d %H:%M"),
        "forecast_at": saved["at"],
        "name": name,
        "side": side,
        "hit": hit,
        "points": None if points is None else round(float(points), 2),
        "price_then": float(call["price"]),
        "price_now": float(last),
        "move": round(move, 2),
        "verdict": verdict,
    }
    return f"上半小时预测{predicted}，实际{actual}，{verdict}", record


def _save_grades(records: list[dict]) -> None:
    """Keep every verdict. Only a sent mail grades, so a line here was also mailed."""

    if not records:
        return
    GRADES.parent.mkdir(parents=True, exist_ok=True)
    with GRADES.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _body(now: datetime) -> tuple[str, dict[str, dict], list[dict]]:
    latest = _latest(now)
    views = hour_model.forecast(now)
    ready = [name for name in NAMES if name in latest and latest[name].get("last")]
    if len(ready) < 3:
        raise RuntimeError("行情还不齐")
    lines = [f"{now:%Y-%m-%d %H:%M} 盘面。半小时模型看上半小时，估下半小时。", ""]
    counts = []
    calls: dict[str, dict] = {}
    grades: list[dict] = []
    for name in NAMES:
        row = latest.get(name)
        if row is None or row.get("last") is None:
            lines.append(f"{name}  还没有行情")
            continue
        last = float(row["last"])
        view = views.get(name)
        lines.append(f"{name}  当前 {last:.0f}")
        if not view or "ret" not in view:
            lines.append("这半小时还没走满，半小时模型先不报")
        else:
            lines.append(
                f"上半小时{_change_words(view['ret'])}，振幅 {view['rng']:.0f}，收在 {view['loc'] * 100:.0f}% 处"
            )
            if view.get("side") not in ("涨", "跌", "平"):
                lines.append(f"半小时模型只有 {view.get('n', 0)} 段上半小时，先不报")
            else:
                counts.append(view["n"])
                word = {"涨": "上涨", "跌": "下跌", "平": "持平"}[view["side"]]
                amount = "" if view["side"] == "平" else f" {abs(view['points']):.0f} 点"
                lines.append(f"下半小时预测{word}{amount}，概率 {view['prob'] * 100:.0f}%")
                calls[name] = {
                    "price": last,
                    "side": view["side"],
                    "prob": round(view["prob"], 4),
                    "points": round(view["points"], 2),
                }
        grade_line, record = _grade_line(now, name, last)
        lines.append(grade_line)
        if record is not None:
            grades.append(record)
    lines.append("")
    if counts:
        lines.append(f"半小时模型用已有的 {min(counts)} 到 {max(counts)} 段上半小时盘面来拟合，还在初创。")
    lines.append("它和 14:50 的交易日模型不是一层。没有新的模拟单，也没有实盘。")
    return "\n".join(lines), calls, grades


def main() -> int:
    now = datetime.now(TZ)
    subject = f"期货盘面（{now:%Y-%m-%d %H:%M}）"
    try:
        body, calls, grades = _body(now)
        mail_outlook._send(subject, body)
    except Exception as exc:
        print(f"mail failed {exc}", flush=True)
        return 1
    _save_forecast(now, calls)
    _save_grades(grades)
    print(f"sent {now:%Y-%m-%d %H:%M} 核对 {len(grades)} 条", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
