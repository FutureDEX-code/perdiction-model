#!/usr/bin/env python3
"""Append a futures quote snapshot every 5 minutes.

Domestic 豆一 / 豆粕 / 菜粕 and US soy / meal / oil. Used later to study the
model. Does not place orders or change the paper rules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from monitor import fetch_snapshot, now_cn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TAPE = DATA / "tape.jsonl"
INTERVAL = 300


def _quote(q) -> dict:
    return {
        "code": q.code,
        "name": q.name,
        "last": q.last,
        "prev": q.prev,
        "open": q.open,
        "high": q.high,
        "low": q.low,
        "volume": q.volume,
        "oi": q.oi,
        "pct": None if q.pct is None else round(q.pct, 4),
        "date": q.date,
        "time": q.time,
        "ok": q.ok,
        "error": q.error,
    }


def append_once() -> None:
    snap = fetch_snapshot()
    today = now_cn().strftime("%Y-%m-%d")
    dates = {q.date for q in snap.domestic if q.date}
    session = snap.session
    if dates and today not in dates:
        session = "休市，沿用 " + sorted(dates)[-1]
    row = {
        "at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "session": session,
        "domestic": [_quote(q) for q in snap.domestic],
        "overseas": [_quote(q) for q in snap.overseas],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    with TAPE.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"tape {row['at']} {row['session']}", flush=True)


def main() -> None:
    while True:
        try:
            append_once()
        except Exception as exc:
            print(f"tape error {exc}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
