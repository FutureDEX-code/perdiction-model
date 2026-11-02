#!/usr/bin/env python3
"""Patch the day sessions the 10-day chunking dropped from the per-contract files.

For every data/tushare_min/<PRODUCT>/<CONTRACT>.csv, walk the trading calendar
(Sina daily dates) between the file's first and last day. Any trading day with
zero day-session bars is a hole: refetch that single day from tushare and insert.
Days the contract genuinely did not trade come back empty and are marked checked
in data/tushare_min/<PRODUCT>/.repair so reruns skip them.

    python3 ingest/repair_minute_holes.py [P OI ...]
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from ingest import fetch_tushare_min, pull_main_minutes

PRODUCTS = ("A", "B", "M", "RM", "Y", "P", "OI", "PK")
FIELDS = fetch_tushare_min.FIELDS


def _trading_days() -> list[str]:
    rows = json.loads((train.SINA_DAILY / "A.json").read_text())
    return [str(r["日期"]) for r in rows if str(r["日期"]) >= train.DATA_FLOOR]


def _holes(path: Path, calendar: set[str]) -> tuple[list[str], list[str]]:
    days_with_day = set()
    stamps = []
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            if len(line) < 20:
                continue
            day, hm = line[:10], line[11:16]
            stamps.append(day)
            if "09:00" <= hm <= "15:00":
                days_with_day.add(day)
    if not stamps:
        return [], []
    first, last = min(stamps), max(stamps)
    span = [day for day in calendar if first <= day <= last]
    holes = [day for day in span if day not in days_with_day]
    return holes, [first, last]


def _fetch_day(token: str, product: str, exchange: str, code: str, day: str) -> list[tuple]:
    ts_code = f"{code}.{exchange}"
    rows = []
    for item in fetch_tushare_min.fetch(token, ts_code, f"{day} 00:00:00", f"{day} 23:59:00"):
        record = dict(zip(
            ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount", "oi"),
            item,
        ))
        rows.append(tuple(record[name] for name in FIELDS))
    rows.sort()
    return rows


def _merge(path: Path, fresh: list[tuple]) -> None:
    with path.open() as handle:
        header = handle.readline()
        old = [line.rstrip("\n") for line in handle if line[:4].isdigit()]
    merged = {line.split(",", 1)[0]: line for line in old}
    for bar in fresh:
        stamp = bar[0]
        if stamp not in merged:
            merged[stamp] = ",".join("" if v is None else str(v) for v in bar)
    with path.open("w", newline="") as handle:
        handle.write(header)
        for stamp in sorted(merged):
            handle.write(merged[stamp] + "\n")


def _marks(product: str) -> set[str]:
    path = ROOT / "data" / "tushare_min" / product / ".repair"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _remember(product: str, key: str) -> None:
    path = ROOT / "data" / "tushare_min" / product / ".repair"
    marks = _marks(product)
    marks.add(key)
    path.write_text("\n".join(sorted(marks)) + "\n")


def repair(product: str, calendar: list[str]) -> None:
    folder = ROOT / "data" / "tushare_min" / product
    if not folder.exists():
        return
    token = pull_main_minutes._token()
    exchange = dict(fetch_tushare_min.PRODUCTS)[product]
    days_set = set(calendar)
    done = _marks(product)
    files = sorted(folder.glob("*.csv"))
    print(f"{product}: {len(files)} 个合约", flush=True)
    filled_total = 0
    for index, path in enumerate(files, start=1):
        code = path.stem
        holes, _span = _holes(path, days_set)
        todo = [day for day in holes if f"{code} {day}" not in done]
        if not todo:
            continue
        fresh_all = []
        for day in todo:
            try:
                fresh = _fetch_day(token, product, exchange, code, day)
            except Exception as exc:
                print(f"  {code} {day} 拉取失败 {exc}", flush=True)
                continue
            fresh_all.extend(fresh)
            _remember(product, f"{code} {day}")
            time.sleep(0.12)
        if fresh_all:
            _merge(path, fresh_all)
        filled_total += len(fresh_all)
        print(f"  {index}/{len(files)} {code} 补 {len(todo)} 天，写入 {len(fresh_all)} 根", flush=True)
    print(f"{product} 完成，共补 {filled_total} 根", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    calendar = _trading_days()
    for product in PRODUCTS:
        if only and product not in only:
            continue
        repair(product, calendar)
    print("REPAIR-ALL-DONE", flush=True)
