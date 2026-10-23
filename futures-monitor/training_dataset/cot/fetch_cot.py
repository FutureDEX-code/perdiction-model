#!/usr/bin/env python3
"""CFTC Commitments of Traders for the CBOT soy complex, stored as weekly records.

Each file under training_dataset/cot/ is one market. A record keeps the fund
(non-commercial) long and short, the commercial long and short, and open
interest. The report lands Friday Beijing time with Tuesday positions, so the
20:30 training job refreshes it; models do not read these files yet.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_dataset" / "cot"
UA = {"User-Agent": "Mozilla/5.0"}
CURRENT = "https://www.cftc.gov/dea/newcot/deafut.txt"
ANNUAL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
MARKETS = {
    "SOYBEANS": "SOYBEANS - CHICAGO BOARD OF TRADE",
    "SOYBEAN_MEAL": "SOYBEAN MEAL - CHICAGO BOARD OF TRADE",
    "SOYBEAN_OIL": "SOYBEAN OIL - CHICAGO BOARD OF TRADE",
}
# Legacy futures-only layout, fields after the name and codes.
I_DATE = 2
I_OI = 7
I_FUND_LONG = 8
I_FUND_SHORT = 9
I_COMM_LONG = 11
I_COMM_SHORT = 12


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _record(parts: list[str]) -> dict | None:
    try:
        day = parts[I_DATE].strip().strip('"')
        fund_long = int(parts[I_FUND_LONG])
        fund_short = int(parts[I_FUND_SHORT])
        return {
            "日期": day,
            "总持仓": int(parts[I_OI]),
            "基金多单": fund_long,
            "基金空单": fund_short,
            "基金净多": fund_long - fund_short,
            "商业多单": int(parts[I_COMM_LONG]),
            "商业空单": int(parts[I_COMM_SHORT]),
        }
    except (ValueError, IndexError):
        return None


def _parse(text: str) -> dict[str, list[dict]]:
    """Rows keyed per market. The current file has one week; the annual file a year."""

    found: dict[str, list[dict]] = {name: [] for name in MARKETS}
    for parts in csv.reader(io.StringIO(text)):
        if len(parts) <= I_COMM_SHORT:
            continue
        label = (parts[0] or "").strip().strip('"')
        for name, wanted in MARKETS.items():
            if label == wanted:
                row = _record(parts)
                if row is not None:
                    found[name].append(row)
    return found


def _merge(path: Path, rows: list[dict]) -> tuple[int, int]:
    kept: dict[str, dict] = {}
    if path.exists():
        for row in json.loads(path.read_text()):
            kept[row["日期"]] = row
    before = len(kept)
    for row in rows:
        kept[row["日期"]] = row
    if len(kept) != before or rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [kept[day] for day in sorted(kept)]
        path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2))
    return before, len(kept)


def fetch_current() -> dict[str, dict]:
    """The newest Tuesday positions. Called by the 20:30 job; harmless any day."""

    text = _get(CURRENT).decode("latin-1", "replace")
    latest: dict[str, dict] = {}
    for name, rows in _parse(text).items():
        if not rows:
            print(f"{name} 这份报告里没有", flush=True)
            continue
        _merge(OUT / f"{name}.json", rows)
        latest[name] = rows[-1]
    return latest


def seed_history(first_year: int = 2016) -> None:
    from datetime import datetime

    last_year = datetime.now().year
    for year in range(first_year, last_year + 1):
        raw = _get(ANNUAL.format(year=year))
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            member = next(item for item in bundle.namelist() if item.lower().endswith(".txt"))
            text = bundle.read(member).decode("latin-1", "replace")
        counts = []
        for name, rows in _parse(text).items():
            if rows:
                _merge(OUT / f"{name}.json", rows)
            counts.append(f"{name} {len(rows)}")
        print(f"{year} 年 " + " ".join(counts), flush=True)


if __name__ == "__main__":
    import sys

    if "--seed" in sys.argv:
        seed_history()
    else:
        got = fetch_current()
        for name, row in got.items():
            print(f"{name} {row['日期']} 基金净多 {row['基金净多']}", flush=True)
