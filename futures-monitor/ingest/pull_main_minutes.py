#!/usr/bin/env python3
"""Append minute bars for Sina's current main contract.

History already stitched into the continuous file is left as it is. Only bars
after that file's last timestamp are added, and only for the contract whose
live quote matches the Sina continuous quote.
"""

from __future__ import annotations

import sys

import csv
import os
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from ingest import fetch_tushare_min as tushare

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "tushare_min"
MAIN = ROOT / "training_dataset" / "main_minute"
TOKEN_FILE = ROOT / "data" / "tushare_token"
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
FIELDS = ("trade_time", "open", "high", "low", "close", "vol", "amount", "oi")
EXTRAS = ("bid", "ask", "bid_vol", "ask_vol", "pre_close", "pre_settle")


def _token() -> str:
    found = os.environ.get("TUSHARE_TOKEN", "").strip()
    if found:
        return found
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    raise RuntimeError("缺少 TUSHARE_TOKEN")


def _months(product: str) -> list[str]:
    today = date.today()
    start = today.year * 12 + today.month - 1 - 3
    codes = []
    for offset in range(16):
        index = start + offset
        year, month = index // 12, index % 12 + 1
        codes.append(f"{product}{year % 100:02d}{month:02d}")
    return codes


def _quotes(symbols: list[str]) -> dict[str, list[str]]:
    url = "https://hq.sinajs.cn/list=" + ",".join(f"nf_{item}" for item in symbols)
    request = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
    raw = urllib.request.urlopen(request, timeout=20).read().decode("gbk", errors="replace")
    out = {}
    for line in raw.splitlines():
        if "hq_str_" not in line or '="' not in line:
            continue
        key = line.split("hq_str_", 1)[1].split("=", 1)[0].removeprefix("nf_")
        payload = line.split('="', 1)[1].rsplit('";', 1)[0]
        if payload:
            out[key] = payload.split(",")
    return out


def _num(parts: list[str], index: int) -> float | None:
    if index >= len(parts) or not parts[index]:
        return None
    try:
        return float(parts[index])
    except ValueError:
        return None


def _same_contract(continuous: list[str], contract: list[str]) -> bool:
    # open, high, low, last, volume, open interest, trading day
    for index in (2, 3, 4, 5, 13, 14, 17):
        if index == 17:
            if continuous[index] != contract[index]:
                return False
            continue
        left, right = _num(continuous, index), _num(contract, index)
        if left is None or right is None or abs(left - right) > 1e-6:
            return False
    return True


def current_mains() -> dict[str, str]:
    symbols = [f"{product}0" for product, _exchange in PRODUCTS]
    for product, _exchange in PRODUCTS:
        symbols.extend(_months(product))
    book = _quotes(symbols)
    chosen = {}
    for product, _exchange in PRODUCTS:
        continuous = book.get(f"{product}0")
        if not continuous:
            print(f"{product} 新浪连续合约没有报价", flush=True)
            continue
        hits = [code for code in _months(product) if code in book and _same_contract(continuous, book[code])]
        if len(hits) != 1:
            print(f"{product} 新浪主力没有唯一对上 {hits or '无'}", flush=True)
            continue
        chosen[product] = hits[0]
        print(f"{product} 新浪当前主力 {hits[0]}", flush=True)
    return chosen


def _last_stamp(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open() as handle:
        for line in handle:
            if line[:4].isdigit():
                last = line.split(",", 1)[0]
    return last


def _append(path: Path, rows: list[tuple], header: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def _fetch_contract(token: str, product: str, exchange: str, code: str, start: datetime) -> list[tuple]:
    ts_code = f"{code}.{exchange}"
    end = datetime.now() + timedelta(hours=6)
    rows = []
    seen = set()
    for item in tushare.fetch(token, ts_code, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")):
        record = dict(zip(
            ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount", "oi"),
            item,
        ))
        stamp = str(record["trade_time"])
        if stamp in seen:
            continue
        seen.add(stamp)
        rows.append(tuple(record[name] for name in FIELDS))
    rows.sort()
    return rows


def update() -> list[str]:
    token = _token()
    exchanges = dict(PRODUCTS)
    notes = []
    for product, code in current_mains().items():
        main_path = MAIN / f"{product}.csv"
        last = _last_stamp(main_path)
        if not last:
            notes.append(f"{product} 没有主连文件，跳过")
            print(notes[-1], flush=True)
            continue
        start = datetime.strptime(last, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=1)
        try:
            bars = _fetch_contract(token, product, exchanges[product], code, start)
        except Exception as exc:
            notes.append(f"{product} {code} 拉取失败 {exc}")
            print(notes[-1], flush=True)
            continue
        fresh = [row for row in bars if row[0] > last]
        if not fresh:
            notes.append(f"{product} {code} 没有新的分钟，主连仍到 {last}")
            print(notes[-1], flush=True)
            continue
        raw_path = RAW / product / f"{code}.csv"
        raw_last = _last_stamp(raw_path)
        raw_rows = [row for row in fresh if row[0] > raw_last]
        if raw_rows:
            _append(raw_path, raw_rows, list(FIELDS))
        blank = ("",) * len(EXTRAS)
        _append(main_path, [(*row, code, *blank) for row in fresh], [*FIELDS, "contract", *EXTRAS])
        note = f"{product} {code} 补上 {len(fresh)} 根，{fresh[0][0]} → {fresh[-1][0]}"
        notes.append(note)
        print(note, flush=True)
    return notes


def main() -> int:
    update()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
