#!/usr/bin/env python3
"""Daily grading of the combo signals, with the environment frozen at the time.

At 20:30 on day D, yesterday's 14:50 combo call is graded against the
14:55-to-14:55 move. Wrong calls and missed big moves get the full environment
recorded: US bars, fund and seat positions, and the
model votes. On the 1st and 16th, the last half month's records and the
half-hour grades roll up into one review mail.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from models import stack

TZ = ZoneInfo("Asia/Shanghai")
REVIEW = ROOT / "data" / "review"
DAILY = REVIEW / "daily.jsonl"
HH_GRADES = ROOT / "data" / "tape_hour_grades.jsonl"
BIG_MOVE = 1.0  # % — a move this size with 没有方向 counts as a miss worth reviewing


def _position_state(product: str) -> dict | None:
    path = ROOT / "training_dataset" / "position" / f"{product}.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return rows[-1] if rows else None


def _cot_state() -> dict[str, dict]:
    out = {}
    for key, _name, filename in train.COT_MARKETS:
        path = train.COT_DAILY / f"{filename}.json"
        if path.exists():
            rows = json.loads(path.read_text())
            if rows:
                out[key] = rows[-1]
    return out


OLD_FAMILY_NAMES = {"新浪回归": "日线回归", "新浪树": "日线树"}


def _combos_from(payload: dict) -> dict[str, dict]:
    """The combo call per product, recomputed from the four family views if absent."""

    combos = dict(payload.get("combo") or {})
    models = {OLD_FAMILY_NAMES.get(family, family): views for family, views in (payload.get("models") or {}).items()}
    families = ("日线回归", "日线树", "分钟回归", "分钟树")
    for key in train.CONTRACTS:
        if key in combos:
            continue
        rule = train.combo_rule(key)
        probs = {}
        for family in families:
            view = next((v for v in (models.get(family) or []) if v.get("code") == key), None)
            probs[family] = (view or {}).get("prob_up", 0.5)
        if rule == "多数票2":
            lean, ups, downs, mean = stack.majority(probs, need=2)
        elif rule == "平均":
            lean, ups, downs, mean = stack.average(probs)
        elif rule in families:
            mean = probs.get(rule, 0.5)
            lean = train.direction(mean)
            rule = f"跟{rule}"
        else:
            continue
        combos[key] = {"lean": lean, "prob_up": round(mean, 4), "rule": rule}
    return combos


def grade_yesterday() -> list[dict]:
    """Grade the previous trading day's combo calls. Run at 20:30."""

    calendar = sorted(dict(train._cached_rows(train.series_path("A"))))
    today = datetime.now(TZ).date().isoformat()
    earlier = [day for day in calendar if day < today]
    if len(earlier) < 2:
        return []
    signal_day, exit_day = earlier[-2], earlier[-1]
    snap = REVIEW.parent / "compare" / f"{signal_day}_1450.json"
    if not snap.exists():
        return []
    payload = json.loads(snap.read_text())
    combos = _combos_from(payload)
    if not combos:
        return []
    records = []
    for key, combo in combos.items():
        lean = combo.get("lean", "没有方向")
        px = stack.price_map(key)
        if signal_day not in px or exit_day not in px:
            continue
        move_pct = (px[exit_day] - px[signal_day]) / px[signal_day] * 100.0
        side = 1 if lean == "偏涨" else (-1 if lean == "偏跌" else 0)
        if side and side * move_pct > 0:
            verdict = "对"
        elif side and side * move_pct < 0:
            verdict = "错"
        elif not side and abs(move_pct) >= BIG_MOVE:
            verdict = "漏看"
        else:
            verdict = "平"
        inputs = payload.get("inputs") or {}
        us = inputs.get("us") or {}
        record = {
            "signal_day": signal_day,
            "exit_day": exit_day,
            "graded_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
            "product": key,
            "name": train.NAMES[key],
            "combo_rule": combo.get("rule"),
            "lean": lean,
            "prob_up": combo.get("prob_up"),
            "move_pct": round(move_pct, 3),
            "verdict": verdict,
            "env": {
                "us": {name: f"{slot['close']} {slot['pct']:+.2f}%" for name, slot in us.items()},
                "news": inputs.get("news"),
                "news_ema": inputs.get("news_ema"),
                "cot": _cot_state(),
                "seat": _position_state(key) if key == "RM" else None,
            },
            "models": {
                family: {
                    "prob": (view or {}).get("prob_up"),
                    "lean": (view or {}).get("lean"),
                }
                for family in ("日线回归", "日线树", "分钟回归", "分钟树")
                for view in [next((v for v in (payload["models"].get(family) or []) if v.get("code") == key), None)]
            },
        }
        records.append(record)
    if records:
        REVIEW.mkdir(parents=True, exist_ok=True)
        with DAILY.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _load_daily(start: str, end: str) -> list[dict]:
    if not DAILY.exists():
        return []
    out = []
    for line in DAILY.read_text().splitlines():
        row = json.loads(line)
        if start <= row["signal_day"] <= end:
            out.append(row)
    return out


def _load_hh(start: str, end: str) -> list[dict]:
    if not HH_GRADES.exists():
        return []
    out = []
    for line in HH_GRADES.read_text().splitlines():
        row = json.loads(line)
        day = str(row.get("graded_at") or "")[:10]
        if start <= day <= end and row.get("side") in ("涨", "跌"):
            out.append(row)
    return out


def half_month_body(today: str) -> str:
    year, month, day = (int(part) for part in today.split("-"))
    if day == 1:
        start = f"{year}-{month - 1:02d}-16" if month > 1 else f"{year - 1}-12-16"
    else:
        start = f"{year}-{month:02d}-01"
    end = today
    daily = _load_daily(start, end)
    hh = _load_hh(start, end)
    lines = [f"半月复盘（{start} → {end}）。日频综合和半小时两层。", ""]
    lines.append("一、日频综合")
    for key in train.CONTRACTS:
        rows = [row for row in daily if row["product"] == key]
        if not rows:
            continue
        good = sum(1 for row in rows if row["verdict"] == "对")
        bad = sum(1 for row in rows if row["verdict"] == "错")
        miss = sum(1 for row in rows if row["verdict"] == "漏看")
        line = f"{train.NAMES[key]}  判 {len(rows)} 天：对 {good}  错 {bad}  漏看 {miss}"
        worst = [row for row in rows if row["verdict"] in ("错", "漏看")]
        worst.sort(key=lambda row: -abs(row["move_pct"]))
        if worst:
            row = worst[0]
            line += f"  最疼 {row['signal_day']} {row['lean']} 实际 {row['move_pct']:+.2f}%"
        lines.append(line)
    lines.append("")
    misses = [row for row in daily if row["verdict"] in ("错", "漏看")]
    misses.sort(key=lambda row: -abs(row["move_pct"]))
    if misses:
        lines.append("二、值得回看的三天")
        for row in misses[:3]:
            lines.append(f"· {row['signal_day']} {row['name']} 综合{row['lean']}（{row['prob_up']}），实际 {row['move_pct']:+.2f}%（{row['verdict']}）")
            env = row.get("env") or {}
            us = env.get("us") or {}
            if us:
                lines.append("  美盘 " + " ".join(f"{name}{slot}" for name, slot in us.items()))
            votes = row.get("models") or {}
            split = " ".join(f"{fam}{view.get('lean')}" for fam, view in votes.items() if view.get("lean"))
            if split:
                lines.append(f"  当时四模型 {split}")
    lines.append("")
    lines.append("三、半小时层")
    for name in ("豆一2701", "豆二2701", "豆粕2701", "菜粕2701", "豆油2701"):
        rows = [row for row in hh if row.get("name") == name]
        if not rows:
            continue
        hits = sum(1 for row in rows if row.get("hit"))
        lines.append(f"{name}  核对 {len(rows)} 次  命中 {hits / len(rows) * 100:.0f}%")
    return "\n".join(lines)


def half_month_due(now: datetime) -> bool:
    return now.day in (1, 16) and (now.hour, now.minute) >= (20, 35)


def main() -> int:
    from jobs import mail_outlook

    if "--halfmonth" in sys.argv:
        today = datetime.now(TZ).date().isoformat()
        body = half_month_body(today)
        print(body, flush=True)
        mail_outlook._send(f"半月复盘（{today}）", body)
        print("半月复盘已发出", flush=True)
        return 0
    records = grade_yesterday()
    bad = sum(1 for row in records if row["verdict"] == "错")
    miss = sum(1 for row in records if row["verdict"] == "漏看")
    print(f"复盘记录 {len(records)} 条：错 {bad} 漏看 {miss}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
