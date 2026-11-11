#!/usr/bin/env python3
"""Stamp a roll day from that contract's own Sina daily bars.

昨收、昨仓、昨量 are the previous day's 15:00 close, open interest, and full-day
volume. 当日收、当日仓、当日量 are the same three numbers for the roll day itself.
Both sides are the daily bar, so volume is comparable.

Sina results, including confirmed empty responses, are cached under
data/sina_contract_daily/. Expired contracts are not fetched again. If Sina
has no usable previous session, the same six numbers come from that
contract's minute file.
"""

from __future__ import annotations

import csv
import json
import time
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

from core import train
from ingest import roll_calendar

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "training_dataset" / "main_minute"
RAW = ROOT / "data" / "tushare_min_history"
CACHE = ROOT / "data" / "sina_contract_daily"
BASIS = ("昨收", "昨仓", "昨量", "当日收", "当日仓", "当日量")
_SINA: dict[str, dict[str, dict]] = {}


def _session_day(days: list[str], dayset: set[str], stamp: str, hm: str) -> str | None:
    if hm >= "21:00":
        index = bisect_right(days, stamp)
        if index >= len(days):
            return None
        target = days[index]
        gap = (datetime.fromisoformat(target) - datetime.fromisoformat(stamp)).days
        return target if 0 < gap <= 15 else None
    if "09:00" <= hm <= "15:00" and stamp in dayset:
        return stamp
    return None


def _take(row: dict, days: list[str], dayset: set[str], slot_book: dict[str, dict]) -> None:
    stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
    if "15:00" < hm < "21:00":
        return
    day = _session_day(days, dayset, stamp, hm)
    if day is None:
        return
    slot = slot_book.get(day)
    if slot is None:
        slot = {"收": None, "仓": None, "量": 0.0}
        slot_book[day] = slot
    vol = row.get("vol") or ""
    if vol not in ("", "None"):
        slot["量"] += float(vol)
    if hm <= "15:00":
        slot["收"] = float(row["close"])
        oi = row.get("oi") or ""
        if oi not in ("", "None"):
            slot["仓"] = float(oi)


def _from_file(path: Path, days: list[str], dayset: set[str], only: str | None) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not path.exists():
        return found
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if only is not None and (row.get("contract") or "") != only:
                continue
            _take(row, days, dayset, found)
    return found


def _own_sessions(product: str, code: str, days: list[str], live: dict[str, dict]) -> dict[str, dict]:
    dayset = set(days)
    found = _from_file(RAW / product / f"{code}.csv", days, dayset, None)
    for day, slot in live.get(code, {}).items():
        if slot.get("收"):
            found[day] = slot
    return found


def _previous(sessions: dict[str, dict], day: str) -> dict | None:
    earlier = [item for item in sessions if item < day and sessions[item].get("收") and sessions[item].get("仓") and sessions[item].get("量")]
    if not earlier:
        return None
    return sessions[max(earlier)]


def roll_days(cal: dict[str, str]) -> dict[str, str]:
    days = sorted(cal)
    return {day: cal[day] for prev, day in zip(days, days[1:]) if cal[prev] != cal[day]}


def _delivery_end(code: str) -> datetime | None:
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) < 4:
        return None
    yy, mm = int(digits[-4:-2]), int(digits[-2:])
    if not 1 <= mm <= 12:
        return None
    year = 2000 + yy if yy < 80 else 1900 + yy
    if mm == 12:
        return datetime(year + 1, 1, 1)
    return datetime(year, mm + 1, 1)


def _listed(code: str, day: str | None = None) -> bool:
    end = _delivery_end(code)
    if end is None:
        return True
    now = datetime.fromisoformat(day) if day else datetime.now()
    return now < end + timedelta(days=30)


def _cache_path(code: str) -> Path:
    return CACHE / f"{code}.json"


def _load_cache(code: str) -> dict[str, dict] | None:
    path = _cache_path(code)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    rows = raw.get("rows") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return None
    return {str(row["日期"]): row for row in rows if row.get("日期")}


def _save_cache(code: str, book: dict[str, dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": [book[day] for day in sorted(book)],
    }
    _cache_path(code).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _fetch_sina(code: str) -> dict[str, dict]:
    try:
        rows = train.fetch_domestic(code)
    except Exception as exc:
        print(f"  {code} 新浪日线没有 {exc}", flush=True)
        return {}
    book = {str(row["日期"]): row for row in rows}
    if not book:
        print(f"  {code} 新浪没有日线", flush=True)
    time.sleep(0.12)
    return book


def _sina(code: str, *, refresh: bool = False) -> dict[str, dict]:
    if not refresh and code in _SINA:
        return _SINA[code]
    previous = _SINA.get(code)
    if previous is None:
        previous = _load_cache(code)
        if previous is not None and not refresh:
            _SINA[code] = previous
            return previous
    book = _fetch_sina(code)
    if not book and previous:
        _SINA[code] = previous
        return previous
    _SINA[code] = book
    _save_cache(code, book)
    return book


def _numbers(row: dict | None) -> tuple[str, str, str] | None:
    if not row:
        return None
    close, oi, vol = row.get("收盘"), row.get("持仓"), row.get("成交量")
    if not close or not oi or not vol or float(oi) <= 0 or float(vol) <= 0:
        return None
    return (str(close), str(oi), str(vol))


def _basis_from_minutes(product: str, code: str, day: str) -> tuple[str, str, str, str, str, str] | None:
    cal = roll_calendar.load_calendar(product)
    sessions = _own_sessions(product, code, sorted(cal), {})
    prev = _previous(sessions, day)
    if prev is None:
        return None
    today = sessions.get(day) or {}
    close, oi, vol = today.get("收"), today.get("仓"), today.get("量")
    return (
        str(prev["收"]),
        str(prev["仓"]),
        str(prev["量"]),
        "" if close is None else str(close),
        "" if oi is None else str(oi),
        "" if not vol else str(vol),
    )


def _basis(code: str, day: str, product: str | None = None) -> tuple[str, str, str, str, str, str] | None:
    """Previous session and the roll day, from that contract's own Sina daily."""

    book = _sina(code)
    if day not in book and _listed(code):
        book = _sina(code, refresh=True)
    dated = sorted(book)
    earlier = [item for item in dated if item < day]
    prev = _numbers(book.get(earlier[-1])) if earlier else None
    today = _numbers(book.get(day))
    if prev is not None and today is not None:
        return (*prev, *today)
    from_minutes = _basis_from_minutes(product, code, day) if product else None
    if prev is not None:
        if today is not None:
            return (*prev, *today)
        if from_minutes and from_minutes[3]:
            return (*prev, from_minutes[3], from_minutes[4], from_minutes[5])
        return (*prev, "", "", "")
    return from_minutes


def for_bar(product: str, trade_time: str, contract: str) -> tuple[str, str, str, str, str, str] | None:
    """Six basis columns for one minute, when that minute belongs to a roll session.

    Bars outside 09:00–15:00 and the night session stay empty. A contract that
    is not the new main that day stays empty too.
    """

    if len(trade_time) < 16 or not contract:
        return None
    cal = roll_calendar.load_calendar(product)
    days = sorted(cal)
    stamp, hm = trade_time[:10], trade_time[11:16]
    day = _session_day(days, set(days), stamp, hm)
    code = roll_days(cal).get(day or "")
    if not day or code != contract:
        return None
    return _basis(code, day, product)


def apply(product: str) -> tuple[int, int]:
    """Fill roll sessions from that contract's Sina daily bars."""

    cal = roll_calendar.load_calendar(product)
    days = sorted(cal)
    rolls = roll_days(cal)
    basis: dict[str, tuple[str, str, str, str, str, str]] = {}
    missing = 0
    by_code: dict[str, list[str]] = {}
    for day, code in rolls.items():
        by_code.setdefault(code, []).append(day)
    for code, listed in by_code.items():
        for day in listed:
            got = _basis(code, day, product)
            if got is None:
                missing += 1
                continue
            basis[day] = got
    path = MAIN / f"{product}.csv"
    dayset = set(days)
    temporary = path.with_suffix(".csv.tmp")
    filled_rows = 0
    with path.open() as src, temporary.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames or [])
        for name in BASIS:
            if name not in fields:
                fields.append(name)
        writer = csv.DictWriter(dst, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in reader:
            for name in BASIS:
                row.setdefault(name, "")
            stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
            day = _session_day(days, dayset, stamp, hm)
            code = rolls.get(day or "")
            if day and code and (row.get("contract") or code) == code and day in basis:
                (row["昨收"], row["昨仓"], row["昨量"], row["当日收"], row["当日仓"], row["当日量"]) = basis[day]
                filled_rows += 1
            elif day not in rolls:
                for name in BASIS:
                    row[name] = ""
            writer.writerow(row)
    temporary.replace(path)
    print(f"{product} 换月 {len(rolls)} 天，写上昨收 {len(basis)} 天，{filled_rows} 根，没有上一节 {missing} 天", flush=True)
    return len(basis), missing


def fill_late(product: str) -> int:
    """Stamp roll-session bars that were appended after the 18:00 pass."""

    path = MAIN / f"{product}.csv"
    if not path.exists():
        return 0
    cal = roll_calendar.load_calendar(product)
    days = sorted(cal)
    dayset = set(days)
    rolls = roll_days(cal)
    cached: dict[str, tuple[str, str, str, str, str, str] | None] = {}
    filled = 0
    temporary = path.with_suffix(".csv.tmp")
    with path.open() as src, temporary.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(dst, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in reader:
            if not (row.get("昨收") or "").strip():
                stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
                day = _session_day(days, dayset, stamp, hm)
                code = rolls.get(day or "")
                contract = row.get("contract") or ""
                if day and code and contract == code:
                    if day not in cached:
                        cached[day] = _basis(code, day, product)
                    got = cached[day]
                    if got:
                        (row["昨收"], row["昨仓"], row["昨量"], row["当日收"], row["当日仓"], row["当日量"]) = got
                        filled += 1
            writer.writerow(row)
    if filled == 0:
        temporary.unlink()
        return 0
    temporary.replace(path)
    print(f"{product} 补上晚到的换月盖章 {filled} 根", flush=True)
    return filled


if __name__ == "__main__":
    import sys

    only = sys.argv[1:]
    for product in roll_calendar.PRODUCTS:
        if only and product not in only:
            continue
        apply(product)
