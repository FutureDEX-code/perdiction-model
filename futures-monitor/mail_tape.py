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
TZ = ZoneInfo("Asia/Shanghai")
NAMES = ("豆一2701", "豆二2701", "豆粕2701", "菜粕2701")


def _latest(now: datetime) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not TAPE.exists():
        return latest
    with TAPE.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("name") in NAMES:
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


def _grade_line(now: datetime, name: str, last: float) -> str:
    saved = _load_forecast()
    if saved is None:
        return "上一小时没有预测"
    call = saved["calls"].get(name)
    if not call or call.get("side") not in ("涨", "跌", "平") or call.get("price") is None:
        return "上一小时没有预测"
    then = datetime.strptime(saved["at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    gap = (now - then).total_seconds() / 60
    if gap < 45:
        return "上一封还不到一小时，先不核对"
    if gap > 100:
        return "上一封隔了休市，不按一小时核对"
    move = last - float(call["price"])
    side = call["side"]
    if side == "平":
        hit = abs(move) < 0.5
        predicted = "持平"
    else:
        hit = (side == "涨" and move > 0) or (side == "跌" and move < 0)
        word = "上涨" if side == "涨" else "下跌"
        points = call.get("points")
        predicted = word if points is None else f"{word} {abs(float(points)):.0f} 点"
    actual = _change_words(move)
    verdict = "对" if hit else "没对"
    return f"上一小时预测{predicted}，实际{actual}，{verdict}"


def _body(now: datetime) -> tuple[str, dict[str, dict]]:
    latest = _latest(now)
    views = hour_model.forecast(now)
    ready = [name for name in NAMES if name in latest and latest[name].get("last")]
    if len(ready) < 3:
        raise RuntimeError("行情还不齐")
    lines = [f"{now:%Y-%m-%d %H:%M} 盘面。小时模型看上一小时，估下一个小时。", ""]
    counts = []
    calls: dict[str, dict] = {}
    for name in NAMES:
        row = latest.get(name)
        if row is None or row.get("last") is None:
            lines.append(f"{name}  还没有行情")
            continue
        last = float(row["last"])
        view = views.get(name)
        lines.append(f"{name}  当前 {last:.0f}")
        if not view or "ret" not in view:
            lines.append("这一小时还没走满，小时模型先不报")
        else:
            lines.append(
                f"这一小时{_change_words(view['ret'])}，振幅 {view['rng']:.0f}，收在 {view['loc'] * 100:.0f}% 处"
            )
            if view.get("side") not in ("涨", "跌", "平"):
                lines.append(f"小时模型只有 {view.get('n', 0)} 段上一小时，先不报")
            else:
                counts.append(view["n"])
                word = {"涨": "上涨", "跌": "下跌", "平": "持平"}[view["side"]]
                amount = "" if view["side"] == "平" else f" {abs(view['points']):.0f} 点"
                lines.append(f"下小时预测{word}{amount}，概率 {view['prob'] * 100:.0f}%")
                calls[name] = {
                    "price": last,
                    "side": view["side"],
                    "prob": round(view["prob"], 4),
                    "points": round(view["points"], 2),
                }
        lines.append(_grade_line(now, name, last))
    lines.append("")
    if counts:
        lines.append(f"小时模型用已有的 {min(counts)} 到 {max(counts)} 段上一小时盘面来拟合，还在初创。")
    lines.append("它和 14:50 的交易日模型不是一层。没有新的模拟单，也没有实盘。")
    return "\n".join(lines), calls


def main() -> int:
    now = datetime.now(TZ)
    subject = f"期货盘面（{now:%Y-%m-%d %H:%M}）"
    try:
        body, calls = _body(now)
        mail_outlook._send(subject, body)
    except Exception as exc:
        print(f"mail failed {exc}", flush=True)
        return 1
    _save_forecast(now, calls)
    print(f"sent {now:%Y-%m-%d %H}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
