#!/usr/bin/env python3
"""Stitch main-continuous minute files from the roll calendar, not a roll rule.

Reads training_dataset/roll_calendar/<PRODUCT>.json (the one true calendar) and
the per-contract minute files. Writes training_dataset/main_minute/<PRODUCT>.csv
with the live file's 15-column layout. Minutes the feed already wrote, and the
per-contract archive does not have yet, are kept. The session after the
calendar's last 15:00 stays a verbatim tail.

    python3 ingest/stitch_from_calendar.py [P OI ...]
"""

from __future__ import annotations

import csv
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import roll_calendar

PRODUCTS = ("A", "B", "M", "RM", "Y", "P", "OI", "PK")
# Live tails are preserved for anything the CTP feed writes; the three new
# products simply have no tail yet until tonight's session starts.
TRADED = PRODUCTS
SRC = ROOT / "data" / "tushare_min"
OUT = ROOT / "training_dataset" / "main_minute"
HEADER = ["trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "contract", "bid", "ask", "bid_vol", "ask_vol", "pre_close", "pre_settle", "昨收", "昨仓", "昨量", "当日收", "当日仓", "当日量"]


def _contract_rows(product: str, trading: list[str], main: dict[str, str]) -> list[list[str]]:
    bucket: list[tuple[str, list[str]]] = []
    for src in sorted((SRC / product).glob("*.csv")):
        code = src.stem
        with src.open() as handle:
            next(handle, None)
            for line in handle:
                if len(line) < 20:
                    continue
                date, clock = line[:10], line[11:16]
                if clock >= "21:00":
                    index = bisect_right(trading, date)
                    if index >= len(trading):
                        continue
                    target = trading[index]
                    gap = (datetime.fromisoformat(target) - datetime.fromisoformat(date)).days
                    if not (0 < gap <= 15) or main.get(target) != code:
                        continue
                elif "09:00" <= clock <= "15:00":
                    if main.get(date) != code:
                        continue
                else:
                    continue
                parts = line.rstrip("\n").split(",")[:8]
                bucket.append((parts[0], parts + [code]))
    bucket.sort()
    return [row + [""] * (len(HEADER) - len(row)) for _, row in bucket]


def _live_bars(product: str, covered: set[str], last_day: str) -> tuple[list[list[str]], list[str]]:
    """Keep feed bars the per-contract archive does not have yet.

    History is whatever the archive has, plus any minute already written on a
    calendar day the archive skipped. Only the session after the calendar's
    last 15:00 stays a verbatim tail.
    """

    path = OUT / f"{product}.csv"
    if product not in TRADED or not path.exists():
        return [], []
    cutoff = f"{last_day} 15:00"
    history: list[list[str]] = []
    tail: list[str] = []
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            if not line[:4].isdigit():
                continue
            stamp = line[:19]
            if stamp > cutoff:
                tail.append(line.rstrip("\n"))
            elif stamp not in covered:
                history.append(line.rstrip("\n").split(","))
    return history, tail


def stitch(product: str) -> None:
    cal = roll_calendar.load_calendar(product)
    if not cal:
        print(f"{product}: 没有日历，跳过", flush=True)
        return
    trading = sorted(cal)
    main = dict(cal)
    rolls = [(prev, day, cal[prev], cal[day]) for prev, day in zip(trading, trading[1:]) if cal[prev] != cal[day]]
    rows = _contract_rows(product, trading, main)
    last_day = trading[-1]
    extra, tail = _live_bars(product, {row[0] for row in rows}, last_day)
    rows.extend(extra)
    rows.sort(key=lambda row: row[0])
    path = OUT / f"{product}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
        for line in tail:
            handle.write(line + "\n")
    with (OUT / "rolls.csv").open("a", newline="") as handle:
        writer = csv.writer(handle)
        for judged, used_from, old, new in rolls:
            writer.writerow([product, judged, old, new, used_from])
    print(f"{product}: {len(rows)} 行 + 尾盘 {len(tail)} 行，换月 {len(rolls)} 次，{trading[0]} -> {last_day}", flush=True)
    from ingest.roll_reference import apply

    apply(product)


if __name__ == "__main__":
    only = sys.argv[1:]
    (OUT / "rolls.csv").write_text("product,judged_on,old,new,used_from\n")
    for product in PRODUCTS:
        if only and product not in only:
            continue
        stitch(product)
