#!/usr/bin/env python3
"""Email the latest futures tape. Gmail to the same Gmail. Never the company mailbox."""

from __future__ import annotations

import sys

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from live import hour_model
from jobs import mail_outlook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TAPE = ROOT / "data" / "ctp_tape.jsonl"
FORECAST = ROOT / "data" / "tape_hour_forecast.json"
FORECASTS = ROOT / "data" / "tape_hour_forecasts.jsonl"
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
    with FORECASTS.open("a") as handle:
        handle.write(json.dumps({"at": now.strftime("%Y-%m-%d %H:%M"), "calls": calls}, ensure_ascii=False) + "\n")


def _graded_keys() -> set[tuple[str, str]]:
    if not GRADES.exists():
        return set()
    out = set()
    for line in GRADES.read_text().splitlines():
        row = json.loads(line)
        out.add((str(row.get("forecast_at")), str(row.get("name"))))
    return out


def _pending_forecast(now: datetime, name: str) -> dict | None:
    """Newest call for `name` whose target bar has completed and that is not graded yet."""

    if not FORECASTS.exists():
        return None
    graded = _graded_keys()
    chosen = None
    for line in FORECASTS.read_text().splitlines():
        saved = json.loads(line)
        then = datetime.strptime(saved["at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        if then >= now - timedelta(minutes=20):
            continue
        if _target_end(then) > now:
            continue
        call = (saved.get("calls") or {}).get(name)
        if not call or call.get("side") not in ("涨", "跌", "平") or call.get("price") is None:
            continue
        if (saved["at"], name) in graded:
            continue
        chosen = {"at": saved["at"], "then": then, "call": call}
    return chosen


def _target_end(then: datetime) -> datetime:
    """When the bar a forecast from `then` refers to completes. Sessions are chained."""

    from live.hour_model import SEGMENTS

    ends = sorted(end for _start, end in SEGMENTS)
    minutes = then.hour * 60 + then.minute
    done = max((end for end in ends if end <= minutes), default=None)
    if done is None:
        # Before today's first segment end: the source bar closed last night at 23:00.
        nxt = min(end for end in ends if end > 0)
        return then.replace(hour=nxt // 60, minute=nxt % 60, second=0, microsecond=0)
    later = [end for end in ends if end > done]
    if not later:
        # Source bar is the last night segment; the target is tomorrow's first bar.
        first = ends[0]
        from datetime import timedelta

        return (then + timedelta(days=1)).replace(hour=first // 60, minute=first % 60, second=0, microsecond=0)
    end = later[0]
    return then.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)


def _grade_line(now: datetime, name: str, bars: dict) -> tuple[str, dict | None]:
    """Grade the newest ungraded call against that half-hour bar's open-to-close.

    The quoted price in the previous mail is not the bar. Using it made
    「实际」disagree with the bar line underneath.
    """

    pending = _pending_forecast(now, name)
    if pending is None:
        if not FORECASTS.exists():
            return "上半小时没有预测", None
        latest = json.loads(FORECASTS.read_text().splitlines()[-1])
        then = datetime.strptime(latest["at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        if (now - then).total_seconds() / 60 < 20:
            return "上一封还不到半小时，先不核对", None
        return f"下一段 {_target_end(then):%H:%M} 走完再核对", None
    call = pending["call"]
    then = pending["then"]
    target = _target_end(then)
    bar_end = target.strftime("%Y-%m-%d %H:%M")
    bar = bars.get((name, bar_end))
    gap_min = round((now - then).total_seconds() / 60)
    span = "上半小时" if gap_min <= 40 else f"{pending['at'][11:16]} 那封"
    if bar is None:
        return f"{span}的K线还没聚齐，先不核对", None
    move = float(bar["ret"])
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
        "forecast_at": pending["at"],
        "name": name,
        "side": side,
        "hit": hit,
        "points": None if points is None else round(float(points), 2),
        "bar_end": bar_end,
        "bar_open": round(float(bar["open"]), 2),
        "bar_close": round(float(bar["close"]), 2),
        "move": round(move, 2),
        "verdict": verdict,
    }
    return f"{span}预测{predicted}，实际{actual}，{verdict}", record


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
    views, bars = hour_model.prepare(now)
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
        grade_line, record = _grade_line(now, name, bars)
        lines.append(grade_line)
        if record is not None:
            grades.append(record)
        if not view or "ret" not in view:
            lines.append("这半小时还没走满，半小时模型先不报")
        else:
            # Same bar as the grade line: print that bar's open-to-close, not a second price gap.
            told = record["move"] if record is not None and record.get("bar_end") == view.get("end") else view["ret"]
            lines.append(
                f"上半小时{_change_words(told)}，振幅 {view['rng']:.0f}，收在 {view['loc'] * 100:.0f}% 处"
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
