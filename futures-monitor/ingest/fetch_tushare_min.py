#!/usr/bin/env python3
"""Download 1-minute bars with open interest for the five soybean contracts.

Tushare's main-contract map is not on this token, so each delivery month is
requested on its own. Months that never listed return no rows and are skipped.
The token is read from TUSHARE_TOKEN and is not stored in this file.
"""

from __future__ import annotations

import sys

import csv
import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "tushare_min_history"
# Delivery months that can be the main contract while the daily model is in sample.
# The last month follows today, so a later main such as A2709 stays in the list.
FIRST = date(2016, 9, 1)
HORIZON_MONTHS = 24
PRODUCTS = (
    ("A", "DCE"),
    ("B", "DCE"),
    ("M", "DCE"),
    ("Y", "DCE"),
    ("RM", "ZCE"),
    ("P", "DCE"),
    ("OI", "ZCE"),
    ("PK", "ZCE"),
)
KEEP_FROM = "2016-10-01"
FIELDS = ("trade_time", "open", "high", "low", "close", "vol", "amount", "oi")


def add_months(day: date, count: int) -> date:
    month = day.month - 1 + count
    year = day.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def last_month() -> date:
    return add_months(date.today(), HORIZON_MONTHS)


def months() -> list[date]:
    out = []
    year, month = FIRST.year, FIRST.month
    stop = last_month()
    while (year, month) <= (stop.year, stop.month):
        out.append(date(year, month, 1))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def code_of(product: str, exchange: str, expiry: date) -> str:
    return f"{product}{expiry.year % 100:02d}{expiry.month:02d}.{exchange}"


def call(token: str, params: dict) -> tuple[int, str, list[list]]:
    body = json.dumps(
        {"api_name": "ft_mins", "token": token, "params": params, "fields": ""}
    ).encode()
    request = urllib.request.Request(
        "https://api.tushare.pro",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    rows = (payload.get("data") or {}).get("items") or []
    return int(payload.get("code") or 0), str(payload.get("msg") or ""), rows


def fetch(token: str, ts_code: str, start: str, end: str) -> list[list]:
    last_error = ""
    for attempt in range(4):
        try:
            code, message, rows = call(
                token,
                {
                    "ts_code": ts_code,
                    "freq": "1min",
                    "start_date": start,
                    "end_date": end,
                },
            )
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2 + attempt)
            continue
        if code == 0:
            if len(rows) >= 8000:
                raise RuntimeError(f"{ts_code} {start} 返回 {len(rows)} 行，窗口太大")
            return rows
        last_error = message
        time.sleep(2 + attempt)
    raise RuntimeError(f"{ts_code} {start} {last_error}")


def listed(token: str, ts_code: str, expiry: date) -> bool:
    probes = (add_months(expiry, -6), add_months(expiry, -2), expiry)
    for day in probes:
        # A single calendar day can be a holiday. A week covers the session.
        start = f"{day.isoformat()} 09:00:00"
        end = f"{(day + timedelta(days=6)).isoformat()} 15:00:00"
        if fetch(token, ts_code, start, end):
            return True
        time.sleep(0.12)
    return False


def download_one(token: str, product: str, ts_code: str, expiry: date) -> int:
    folder = OUT / product
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{ts_code.split('.')[0]}.csv"
    done = target.with_suffix(".done")
    if done.exists():
        return int(done.read_text().strip() or "0")
    if target.exists():
        target.unlink()
    start = add_months(expiry, -14)
    stop = expiry.replace(day=25)
    seen: set[str] = set()
    rows: list[tuple] = []
    day = start
    while day <= stop:
        chunk_end = min(day + timedelta(days=9), stop)
        # Contiguous windows: the next chunk starts the day after this one ends,
        # so no day session can fall between chunks.
        start_text = f"{day.isoformat()} 00:00:00"
        end_text = f"{chunk_end.isoformat()} 23:59:00"
        for item in fetch(token, ts_code, start_text, end_text):
            record = dict(zip(
                ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount", "oi"),
                item,
            ))
            stamp = record["trade_time"]
            if stamp < KEEP_FROM or stamp in seen:
                continue
            seen.add(stamp)
            rows.append(tuple(record[name] for name in FIELDS))
        time.sleep(0.12)
        day = chunk_end + timedelta(days=1)
    rows.sort()
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    done.write_text(str(len(rows)))
    return len(rows)


def main() -> None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少 TUSHARE_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "progress.log"
    print("分钟线下载开始", flush=True)
    total = 0
    only = os.environ.get("ONLY_PRODUCT", "").strip()
    products = [item for item in PRODUCTS if not only or item[0] == only]
    if only and not products:
        raise SystemExit(f"没有这个品种 {only}")
    contracts = [(product, exchange, expiry, code_of(product, exchange, expiry))
                 for product, exchange in products for expiry in months()]
    with log.open("a") as handle:
        handle.write(f"\n开始 {datetime.now():%Y-%m-%d %H:%M} {only or '全部'} 候选 {len(contracts)}\n")
    for index, (product, _exchange, expiry, ts_code) in enumerate(contracts, start=1):
        folder = OUT / product
        done = folder / f"{ts_code.split('.')[0]}.done"
        if done.exists():
            count = int(done.read_text().strip() or "0")
            total += count
            continue
        try:
            if not listed(token, ts_code, expiry):
                continue
            count = download_one(token, product, ts_code, expiry)
        except Exception as exc:
            with log.open("a") as handle:
                handle.write(f"失败 {ts_code} {exc}\n")
            print(f"失败 {ts_code} {exc}", flush=True)
            continue
        total += count
        line = f"{index}/{len(contracts)} {ts_code} {count} 行 累计 {total}"
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")
    with log.open("a") as handle:
        handle.write(f"完成 {datetime.now():%Y-%m-%d %H:%M} 累计 {total}\n")
    print(f"完成 累计 {total}", flush=True)


if __name__ == "__main__":
    main()
