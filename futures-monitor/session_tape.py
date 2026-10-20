#!/usr/bin/env python3
"""Turn one finished trading day of CTP snapshots into a few numbers.

The day runs from 21:00 the previous evening through the 15:00 close.
train.py does not read this file. The live direction stays on the daily
close until these rows are long enough to test.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TAPE = DATA / "ctp_tape.jsonl"
OUT = DATA / "session_tape.jsonl"
NAMES = ("豆一2701", "豆二2701", "豆粕2701", "菜粕2701", "豆油2701")


def _day_before(day: str) -> str:
    return (datetime.strptime(day, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()


def _saved_days() -> set[str]:
    if not OUT.exists():
        return set()
    days = set()
    for line in OUT.read_text().splitlines():
        if line.strip():
            days.add(json.loads(line)["day"])
    return days


def _rows(day: str) -> list[dict]:
    start = f"{_day_before(day)} 21:00:00"
    end = f"{day} 15:10:00"
    rows = []
    if not TAPE.exists():
        return rows
    with TAPE.open() as handle:
        for line in handle:
            row = json.loads(line)
            if start <= row["at"] <= end and row.get("name") in NAMES:
                rows.append(row)
    return rows


def _last_at(rows: list[dict], name: str, not_before: str) -> dict | None:
    found = None
    for row in rows:
        if row["name"] == name and row["at"] >= not_before:
            found = row
    return found


def _first(rows: list[dict], name: str) -> dict | None:
    for row in rows:
        if row["name"] == name:
            return row
    return None


def _night_volume(rows: list[dict], day: str, name: str) -> float | None:
    cut = f"{day} 02:30:00"
    found = None
    for row in rows:
        if row["name"] == name and row["at"] < cut:
            found = row
    if found is None or found.get("volume") is None:
        return None
    return float(found["volume"])


def summarize(day: str) -> dict | None:
    rows = _rows(day)
    close_from = f"{day} 15:00:00"
    contracts = {}
    for name in NAMES:
        close = _last_at(rows, name, close_from)
        first = _first(rows, name)
        if close is None or first is None:
            return None
        high = float(close["high"])
        low = float(close["low"])
        last = float(close["last"])
        span = high - low
        volume = None if close.get("volume") is None else float(close["volume"])
        oi = None if close.get("oi") is None else float(close["oi"])
        oi0 = None if first.get("oi") is None else float(first["oi"])
        night = _night_volume(rows, day, name)
        contracts[name] = {
            "open": float(close["open"]) if close.get("open") is not None else float(first["last"]),
            "high": high,
            "low": low,
            "close": last,
            "close_loc": None if span <= 0 else round((last - low) / span, 4),
            "giveback_pct": None if high <= 0 else round((high - last) / high * 100.0, 4),
            "volume": volume,
            "night_volume": night,
            "oi": oi,
            "oi_change": None if oi is None or oi0 is None else oi - oi0,
        }
    return {"day": day, "contracts": contracts}


def record(day: str) -> str:
    if day in _saved_days():
        return f"盘面已记 {day}"
    row = summarize(day)
    if row is None:
        return f"盘面还不齐 {day}"
    DATA.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return f"盘面记下 {day}"


def main() -> None:
    import sys

    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(record(day), flush=True)


if __name__ == "__main__":
    main()
