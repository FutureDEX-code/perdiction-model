#!/usr/bin/env python3
"""Keep the main-continuous minute files. This process is their only writer.

During a session, each finished minute is written from the live quote.
After the close, one pass fills empty turnover from the paid minute download.
Every evening at 20:00, another pass inserts any missing minutes from that
download for the last week of closed sessions, and fills empty turnover.
Price, volume, open interest, and bid/ask stay as the live minute wrote them.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ingest import pull_main_minutes
TZ = ZoneInfo("Asia/Shanghai")
LOG = ROOT / "main-minute-daemon.log"
PIDFILE = ROOT / "data" / "main_minute_daemon.pid"
BACKFILL_MARK = ROOT / "data" / "main_minute_backfill.txt"
HOLEFILL_MARK = ROOT / "data" / "main_minute_holefill.txt"
HOLEFILL_LOOKBACK_DAYS = 7
GAP_ONCE = ROOT / "data" / "main_minute_gap_once.txt"
GAP_START = "2026-10-21 21:00:00"
GAP_END = "2026-10-22 13:29:00"
MAIN = ROOT / "training_dataset" / "main_minute"
LIVE = "http://127.0.0.1:8771/latest"
EXTRAS = ("bid", "ask", "bid_vol", "ask_vol", "pre_close", "pre_settle", "昨收", "昨仓", "昨量", "当日收", "当日仓", "当日量")
HEADER = ["trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "contract", *EXTRAS]
SESSIONS = ((9 * 60, 10 * 60 + 15), (10 * 60 + 30, 11 * 60 + 30), (13 * 60 + 30, 15 * 60), (21 * 60, 23 * 60))
def _minutes(now: datetime) -> int:
    return now.hour * 60 + now.minute


def _in_session(minutes: int) -> bool:
    return any(start <= minutes < end for start, end in SESSIONS)


def _keep_clock(product: str, hm: str) -> bool:
    """Trading minutes only. The closing minute of each session stays.

    Quotes before the open, across the breaks, and between 15:01 and 20:59
    are not minutes. Peanut has no night session. History through 23:30 is
    left in the files; new writes stop at 23:00.
    """

    if "09:00" <= hm <= "10:15" or "10:30" <= hm <= "11:30" or "13:30" <= hm <= "15:00":
        return True
    return product != "PK" and "21:00" <= hm <= "23:00"


_CALENDAR: dict[str, tuple[float, dict[str, str]]] = {}


def _calendar(product: str) -> dict[str, str]:
    path = ROOT / "training_dataset" / "roll_calendar" / f"{product}.json"
    mtime = path.stat().st_mtime if path.exists() else 0.0
    cached = _CALENDAR.get(product)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    book = json.loads(path.read_text()) if path.exists() else {}
    _CALENDAR[product] = (mtime, book)
    return book


def _foreign_day(product: str, trade_time: str, contract: str) -> bool:
    """A day-session bar whose contract is not the main already on the calendar."""

    if len(trade_time) < 16 or not contract:
        return False
    hm = trade_time[11:16]
    if not ("09:00" <= hm <= "15:00"):
        return False
    official = _calendar(product).get(trade_time[:10])
    return bool(official and official != contract)


def _contract_name(code: str) -> str:
    text = code.lower()
    for prefix in ("rm", "oi", "pk"):
        if text.startswith(prefix):
            digits = text[len(prefix) :]
            upper = prefix.upper()
            if len(digits) == 3:
                return upper + "2" + digits
            return upper + digits
    return text.upper()


def _last_stamp(path: Path) -> str:
    last = ""
    with path.open() as handle:
        for line in handle:
            if line[:4].isdigit():
                last = line.split(",", 1)[0]
    return last


def _ensure_columns(path: Path) -> None:
    with path.open() as handle:
        header = handle.readline().strip()
    if "bid" in header.split(","):
        return
    temporary = path.with_suffix(".csv.tmp")
    with path.open() as src, temporary.open("w", newline="") as dst:
        src.readline()
        writer = csv.writer(dst)
        writer.writerow(HEADER)
        for line in src:
            writer.writerow(line.rstrip("\n").split(","))
    temporary.replace(path)
    print(f"{path.name} 已加上买卖盘列", flush=True)


def _ask_live() -> dict:
    with urllib.request.urlopen(LIVE, timeout=3) as response:
        return json.loads(response.read().decode())


_mains_cache: tuple[str, dict[str, str], set[str]] | None = None

SIMNOW_MD = ROOT / "live" / "simnow_md.py"
# Half-hour tape keeps the 2701 contracts of the five traded products. Only the
# main-contract slot rolls; the three new products have no tape slot.
TAPE_INSTRUMENT = {"A": "a2701", "B": "b2701", "M": "m2701", "Y": "y2701", "RM": "RM701"}
ZCE = ("RM", "OI", "PK")
PRODUCTS_ALL = ("A", "B", "M", "Y", "RM", "P", "OI", "PK")


def _instrument_for(product: str, contract: str) -> str:
    if product in ZCE:
        # CTP Zhengzhou ids drop the decade digit: OI2701 -> OI701, PK2701 -> PK701.
        return product + contract[len(product) + 1 :]
    return contract.lower()


def _feed_name(item: str) -> str:
    low = item.lower()
    zce_names = {"rm": "菜粕", "oi": "菜籽油", "pk": "花生"}
    for prefix, name in zce_names.items():
        if low.startswith(prefix):
            digits = low[len(prefix) :]
            full = digits if len(digits) == 4 else "2" + digits
            return f"{name}{full}"
    names = {"a": "豆一", "b": "豆二", "m": "豆粕", "y": "豆油", "p": "棕榈油"}
    return f"{names[low[0]]}{low[1:]}"


def _ensure_subscribed(product: str, contract: str) -> None:
    """A Sina main that is not in the CTP feed means no minutes. Subscribe it and let launchd restart the feed."""

    instrument = _instrument_for(product, contract)
    text = SIMNOW_MD.read_text()
    if f'"{instrument}"' in text:
        return
    match = re.search(r"INSTRUMENTS = \(([^)]*)\)", text)
    names_match = re.search(r"NAMES = \{([^}]*)\}", text)
    if not match or not names_match:
        print(f"{product} 新浪主力是 {contract}，行情文件没认出来，不敢自动订阅", flush=True)
        return
    current = [item.strip().strip('"') for item in match.group(1).split(",") if item.strip()]
    prefix = product if product in ZCE else product.lower()
    tape = TAPE_INSTRUMENT.get(product)
    kept = [item for item in current if not (item.startswith(prefix) and item != tape)]
    kept.append(instrument)
    instruments_line = "INSTRUMENTS = (" + ", ".join(f'"{item}"' for item in kept) + ")"
    names_new = "NAMES = {\n" + "\n".join(f'    "{item.lower()}": "{_feed_name(item)}",' for item in kept) + "\n}"
    text = text[: match.start()] + instruments_line + text[match.end() :]
    names_match = re.search(r"NAMES = \{([^}]*)\}", text)
    text = text[: names_match.start()] + names_new + text[names_match.end() :]
    SIMNOW_MD.write_text(text)
    print(f"{product} 新浪主力换成 {contract}，订阅 {instrument} 并重启行情进程", flush=True)
    subprocess.run(["pkill", "-f", "live.simnow_md"], check=False)


def _session_key(now: datetime) -> str:
    """Main contract is fixed for one session. Night opens a new one; the daytime does too."""

    if _minutes(now) >= 21 * 60:
        return f"{now.date().isoformat()} night"
    return f"{now.date().isoformat()} day"


def _session_opening(now: datetime) -> bool:
    minutes = _minutes(now)
    return (21 * 60 <= minutes < 21 * 60 + 5) or (9 * 60 <= minutes < 9 * 60 + 5)


def _file_contract(product: str) -> str:
    path = MAIN / f"{product}.csv"
    if not path.exists():
        return ""
    contract = ""
    with path.open() as handle:
        for line in handle:
            if not line[:4].isdigit():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) > 8 and parts[8]:
                contract = parts[8]
    return contract


def _mains_for(now: datetime) -> dict[str, str]:
    global _mains_cache
    key = _session_key(now)
    if _mains_cache is not None and _mains_cache[0] == key and not _mains_cache[2]:
        return _mains_cache[1]
    if _mains_cache is not None and _mains_cache[0] == key and not _session_opening(now):
        return _mains_cache[1]
    found = pull_main_minutes.current_mains()
    for product, contract in found.items():
        try:
            _ensure_subscribed(product, contract)
        except Exception as exc:
            print(f"自动订阅 {product} {contract} 失败 {exc}", flush=True)
    mapping: dict[str, str] = {}
    unresolved: set[str] = set()
    for product in PRODUCTS_ALL:
        contract = found.get(product) or ""
        if contract:
            mapping[contract] = product
            continue
        previous = _file_contract(product)
        if not previous:
            print(f"{product} 分不出主力，文件里也没有合约，这一节先不写", flush=True)
            continue
        mapping[previous] = product
        unresolved.add(product)
        print(f"{product} 这一节分不出唯一主力，继续用 {previous}", flush=True)
    if not _session_opening(now):
        unresolved = set()
    _mains_cache = (key, mapping, unresolved)
    if mapping:
        print("这一节的主力 " + " ".join(f"{product} {contract}" for contract, product in mapping.items()), flush=True)
    return mapping


def _roll_basis(product: str, trade_time: str, contract: str) -> tuple[str, str, str, str, str, str]:
    """Same six columns as the 18:00 stamp, for a bar that arrives after it."""

    try:
        from ingest import roll_reference

        got = roll_reference.for_bar(product, trade_time, contract)
    except Exception as exc:
        print(f"{product} 换月盖章没写上 {exc}", flush=True)
        return ("", "", "", "", "", "")
    return got or ("", "", "", "", "", "")


def _append_live(now: datetime) -> None:
    try:
        payload = _ask_live()
    except Exception as exc:
        print(f"实盘进程没应答 {exc}", flush=True)
        return
    mains = _mains_for(now)
    written = 0
    for code, quote in (payload.get("quotes") or {}).items():
        contract = _contract_name(code)
        product = mains.get(contract)
        if product is None:
            continue
        path = MAIN / f"{product}.csv"
        if not path.exists():
            continue
        last = _last_stamp(path)
        rows = []
        for bar in quote.get("minutes") or []:
            stamp = str(bar.get("trade_time") or "")
            if not stamp or stamp <= last or bar.get("close") in (None, 0):
                continue
            if len(stamp) < 16 or not _keep_clock(product, stamp[11:16]):
                continue
            if _foreign_day(product, stamp, contract):
                print(f"{product} {contract} 跳过 {stamp}，日历当天是 {_calendar(product).get(stamp[:10])}", flush=True)
                continue
            basis = _roll_basis(product, stamp, contract)
            rows.append((
                stamp,
                bar.get("open"),
                bar.get("high"),
                bar.get("low"),
                bar.get("close"),
                bar.get("vol") if bar.get("vol") not in (None, 0) else "",
                "",
                bar.get("oi") if bar.get("oi") is not None else "",
                contract,
                bar.get("bid") if bar.get("bid") is not None else "",
                bar.get("ask") if bar.get("ask") is not None else "",
                bar.get("bid_vol") if bar.get("bid_vol") is not None else "",
                bar.get("ask_vol") if bar.get("ask_vol") is not None else "",
                bar.get("pre_close") if bar.get("pre_close") is not None else "",
                bar.get("pre_settle") if bar.get("pre_settle") is not None else "",
                *basis,
            ))
            last = stamp
        if not rows:
            continue
        with path.open("a", newline="") as handle:
            csv.writer(handle).writerows(rows)
        written += len(rows)
        print(f"{product} {contract} 写入 {rows[0][0]} → {rows[-1][0]} {len(rows)} 根", flush=True)
    if written == 0:
        print("这一分钟没有新的主连可写", flush=True)


def _pad(parts: list[str]) -> list[str]:
    if len(parts) < len(HEADER):
        parts = parts + [""] * (len(HEADER) - len(parts))
    return parts[: len(HEADER)]


def _blank(value: str) -> bool:
    return str(value).strip() in ("", "None")


def _merge_download(start: str, end: str, skip: set[str]) -> tuple[int, set[str]]:
    """Fill empty amount only. Prices, volume, open interest, and bid/ask stay as written."""

    token = pull_main_minutes._token()
    exchanges = dict(pull_main_minutes.PRODUCTS)
    changed = 0
    ready: set[str] = set()
    for product, contract in pull_main_minutes.current_mains().items():
        if product in skip:
            continue
        fetched = pull_main_minutes._fetch_contract(
            token,
            product,
            exchanges[product],
            contract,
            datetime.strptime(start, "%Y-%m-%d %H:%M:%S"),
        )
        official = {
            bar[0]: bar
            for bar in fetched
            if start <= bar[0] <= end and bar[6] not in ("", None)
        }
        if not official:
            print(f"{product} {contract} 下载接口还没有 {start} → {end}", flush=True)
            continue
        ready.add(product)
        path = MAIN / f"{product}.csv"
        rows: dict[str, list[str]] = {}
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = _pad(line.rstrip("\n").split(","))
                    rows[parts[0]] = parts
        filled = 0
        for stamp, bar in official.items():
            current = rows.get(stamp)
            if current is None or not _blank(current[6]):
                continue
            current[6] = bar[6]
            filled += 1
        changed += filled
        if filled == 0:
            print(f"{product} {contract} 这段成交额已经有了", flush=True)
            continue
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = _pad(line.rstrip("\n").split(","))
                    rows.setdefault(parts[0], parts)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for stamp in sorted(rows):
                writer.writerow(rows[stamp])
        temporary.replace(path)
        print(f"{product} {contract} 补了成交额 {start} → {end}", flush=True)
    return changed, ready


def _done_marks() -> set[str]:
    if not BACKFILL_MARK.exists():
        return set()
    return {line.strip() for line in BACKFILL_MARK.read_text().splitlines() if line.strip()}


def _remember(key: str) -> None:
    marks = _done_marks()
    marks.add(key)
    BACKFILL_MARK.write_text("\n".join(sorted(marks)) + "\n")


def _backfill_due(now: datetime) -> tuple[str, str, str] | None:
    """After the day close or the night close, one download pass for that session."""

    if now.weekday() >= 5:
        return None
    done = _done_marks()
    today = now.date().isoformat()
    if (now.hour, now.minute) >= (15, 10):
        key = f"{today} day"
        if any(f"{key} {product}" not in done for product in PRODUCTS_ALL):
            return key, f"{today} 09:00:00", f"{today} 15:00:00"
    if (now.hour, now.minute) >= (23, 5):
        key = f"{today} night"
        if any(f"{key} {product}" not in done for product in PRODUCTS_ALL):
            return key, f"{today} 21:00:00", f"{today} 23:00:00"
    return None



def _holefill_marks() -> set[str]:
    if not HOLEFILL_MARK.exists():
        return set()
    return {line.strip() for line in HOLEFILL_MARK.read_text().splitlines() if line.strip()}


def _remember_holefill(key: str) -> None:
    marks = _holefill_marks()
    marks.add(key)
    HOLEFILL_MARK.write_text("\n".join(sorted(marks)) + "\n")


def _holefill_due(now: datetime) -> bool:
    """One pass between 20:00 and 21:00, after the day close and before the night open."""

    if not ((20, 0) <= (now.hour, now.minute) < (21, 0)):
        return False
    return now.date().isoformat() not in _holefill_marks()


def _next_trading_day(days: list[str], stamp_date: str) -> str | None:
    index = bisect_right(days, stamp_date)
    if index >= len(days):
        return None
    target = days[index]
    gap = (datetime.fromisoformat(target) - datetime.fromisoformat(stamp_date)).days
    return target if 0 < gap <= 15 else None


def _bar_contract(product: str, stamp: str, cal: dict[str, str], fallback: str) -> str:
    if len(stamp) < 16:
        return ""
    hm, day = stamp[11:16], stamp[:10]
    if hm >= "21:00":
        target = _next_trading_day(sorted(cal), day)
        return (cal.get(target) if target else "") or fallback
    if "09:00" <= hm <= "15:00":
        return cal.get(day, "") or fallback
    return ""


def _session_stamps(day: datetime, night: bool, product: str) -> list[str]:
    if night:
        if product == "PK" or day.weekday() == 5:
            return []
        ranges = ((21, 0, 23, 0),)
    else:
        ranges = ((9, 0, 10, 15), (10, 30, 11, 30), (13, 30, 15, 0))
    out = []
    for start_hour, start_minute, end_hour, end_minute in ranges:
        current = day.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = day.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        while current <= end:
            out.append(current.strftime("%Y-%m-%d %H:%M:%S"))
            current += timedelta(minutes=1)
    return out


def _needed_contracts(product: str, start: datetime, end: datetime, cal: dict[str, str], fallback: str) -> set[str]:
    found: set[str] = set()
    cursor = start.date()
    last = end.date()
    while cursor <= last:
        text = cursor.isoformat()
        day_end = datetime(cursor.year, cursor.month, cursor.day, 15, 0, 0)
        if start <= day_end <= end:
            if text in cal:
                found.add(cal[text])
            elif cursor.weekday() < 5 and cal and text > max(cal) and fallback:
                # Calendar has not yet recorded this weekday; use the live main.
                found.add(fallback)
        night_end = datetime(cursor.year, cursor.month, cursor.day, 23, 0, 0)
        if product != "PK" and cursor.weekday() != 5 and start <= night_end <= end:
            code = _bar_contract(product, f"{text} 21:00:00", cal, fallback)
            if code:
                found.add(code)
        cursor += timedelta(days=1)
    return found


def _merge_raw(product: str, contract: str, bars: list[tuple]) -> int:
    path = pull_main_minutes.RAW / product / f"{contract}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, tuple] = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = line.rstrip("\n").split(",")
                    rows[parts[0]] = tuple(parts[:8])
    added = 0
    for bar in bars:
        if bar[0] not in rows:
            rows[bar[0]] = tuple(bar[i] if i < len(bar) else "" for i in range(8))
            added += 1
    if added == 0:
        return 0
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(pull_main_minutes.FIELDS))
        for stamp in sorted(rows):
            writer.writerow(rows[stamp])
    temporary.replace(path)
    return added


def _evening_holefill(now: datetime) -> None:
    """Insert missing minutes for closed sessions in the last week, and fill empty amount."""

    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    end = now.replace(hour=15, minute=0, second=0, microsecond=0)
    start = (end - timedelta(days=HOLEFILL_LOOKBACK_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_text = start.strftime("%Y-%m-%d %H:%M:%S")
    end_text = end.strftime("%Y-%m-%d %H:%M:%S")
    print(f"晚上8点补缺失分钟 {start_text} → {end_text}", flush=True)
    token = pull_main_minutes._token()
    exchanges = dict(pull_main_minutes.PRODUCTS)
    try:
        fallbacks = pull_main_minutes.current_mains()
    except Exception as exc:
        print(f"晚上8点分主力失败 {exc}，改用文件里最后的合约", flush=True)
        fallbacks = {}
    for product in PRODUCTS_ALL:
        path = MAIN / f"{product}.csv"
        if not path.exists():
            continue
        cal = _calendar(product)
        fallback = fallbacks.get(product) or _file_contract(product)
        contracts = _needed_contracts(product, start, end, cal, fallback)
        if not contracts:
            print(f"{product} 这段没有可查的合约", flush=True)
            continue
        official: dict[str, tuple] = {}
        fetched_by_code: dict[str, list[tuple]] = {}
        for contract in sorted(contracts):
            try:
                fetched = pull_main_minutes._fetch_contract(
                    token,
                    product,
                    exchanges[product],
                    contract,
                    start,
                )
            except Exception as exc:
                print(f"{product} {contract} 下载失败 {exc}", flush=True)
                continue
            kept = []
            for bar in fetched:
                stamp = bar[0]
                if not (start_text <= stamp <= end_text):
                    continue
                if len(stamp) < 16 or not _keep_clock(product, stamp[11:16]):
                    continue
                owner = _bar_contract(product, stamp, cal, fallback)
                if owner and owner != contract:
                    continue
                kept.append(bar)
                official[stamp] = (bar, contract)
            fetched_by_code[contract] = kept
            if not kept:
                print(f"{product} {contract} 下载接口还没有 {start_text} → {end_text}", flush=True)
        if not official:
            print(f"{product} 下载接口还没有可补的分钟", flush=True)
            continue
        rows: dict[str, list[str]] = {}
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = _pad(line.rstrip("\n").split(","))
                    rows[parts[0]] = parts
        added = 0
        filled = 0
        for stamp, (bar, contract) in official.items():
            current = rows.get(stamp)
            if current is None:
                basis = _roll_basis(product, stamp, contract)
                rows[stamp] = _pad([
                    stamp, bar[1], bar[2], bar[3], bar[4], bar[5], bar[6], bar[7], contract,
                    "", "", "", "", "", "", *basis,
                ])
                added += 1
                continue
            if _blank(current[6]) and bar[6] not in ("", None):
                current[6] = bar[6]
                filled += 1
        raw_added = 0
        for contract, bars in fetched_by_code.items():
            raw_added += _merge_raw(product, contract, bars)
        if added or filled:
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                for stamp in sorted(rows):
                    writer.writerow(rows[stamp])
            temporary.replace(path)
        expected_missing = 0
        cursor = start.date()
        while cursor <= end.date():
            day = datetime(cursor.year, cursor.month, cursor.day)
            for night in (False, True):
                stamps = [item for item in _session_stamps(day, night, product) if start_text <= item <= end_text]
                if not stamps or not any(item in official for item in stamps):
                    continue
                expected_missing += sum(1 for item in stamps if item not in rows)
            cursor += timedelta(days=1)
        extra = f"，接口也缺 {expected_missing} 根" if expected_missing else "，已齐"
        print(
            f"{product} 晚上8点补入缺失 {added} 根，成交额 {filled} 处，原始档案 {raw_added} 根{extra}",
            flush=True,
        )


def _gap_once() -> bool:
    """Insert the one hole left when collection switched to every minute. Bid and ask stay empty."""

    if GAP_ONCE.exists() and GAP_ONCE.read_text().strip() == "done":
        return False
    token = pull_main_minutes._token()
    exchanges = dict(pull_main_minutes.PRODUCTS)
    inserted = 0
    bean_two = False
    for product, contract in pull_main_minutes.current_mains().items():
        fetched = pull_main_minutes._fetch_contract(
            token,
            product,
            exchanges[product],
            contract,
            datetime.strptime(GAP_START, "%Y-%m-%d %H:%M:%S"),
        )
        official = [bar for bar in fetched if GAP_START <= bar[0] <= GAP_END]
        if not official:
            print(f"{product} {contract} 这次切换的空档接口还没有", flush=True)
            continue
        if product == "B":
            bean_two = True
        path = MAIN / f"{product}.csv"
        rows: dict[str, list[str]] = {}
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = _pad(line.rstrip("\n").split(","))
                    rows[parts[0]] = parts
        added = 0
        for bar in official:
            if bar[0] in rows:
                continue
            rows[bar[0]] = _pad([bar[0], bar[1], bar[2], bar[3], bar[4], bar[5], bar[6], bar[7], contract])
            added += 1
        inserted += added
        if added == 0:
            continue
        with path.open() as handle:
            handle.readline()
            for line in handle:
                if line[:4].isdigit():
                    parts = _pad(line.rstrip("\n").split(","))
                    rows.setdefault(parts[0], parts)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for stamp in sorted(rows):
                writer.writerow(rows[stamp])
        temporary.replace(path)
        print(f"{product} {contract} 空档补入 {added} 根，没有买卖盘", flush=True)
    if bean_two:
        GAP_ONCE.write_text("done\n")
        print(f"切换空档已补完，共写入 {inserted} 根，以后不再补", flush=True)
    return bean_two


def drop_foreign_day_bars(product: str) -> int:
    """Drop day-session minutes that belong to another contract than that day's main."""

    path = MAIN / f"{product}.csv"
    if not path.exists():
        return 0
    dropped = 0
    temporary = path.with_suffix(".csv.tmp")
    with path.open() as src, temporary.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=list(reader.fieldnames or []), extrasaction="ignore", restval="")
        writer.writeheader()
        for row in reader:
            stamp = row.get("trade_time") or ""
            if _foreign_day(product, stamp, row.get("contract") or ""):
                dropped += 1
                continue
            writer.writerow(row)
    if dropped == 0:
        temporary.unlink()
        return 0
    temporary.replace(path)
    print(f"{product} 去掉日历之外的日盘分钟 {dropped} 根", flush=True)
    return dropped


def _strip_quotes(path: Path, product: str) -> int:
    """Remove stored quotes that are outside a trading minute. Night history past 23:00 stays."""

    removed = 0
    temporary = path.with_suffix(".csv.tmp")
    with path.open() as src, temporary.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=list(reader.fieldnames or []), extrasaction="ignore", restval="")
        writer.writeheader()
        for row in reader:
            stamp = row.get("trade_time") or ""
            hm = stamp[11:16] if len(stamp) >= 16 else ""
            if hm and not _keep_clock(product, hm) and not (product != "PK" and "23:00" < hm <= "23:30"):
                removed += 1
                continue
            writer.writerow(row)
    if removed == 0:
        temporary.unlink()
        return 0
    temporary.replace(path)
    print(f"{product} 去掉盘外报价 {removed} 根", flush=True)
    return removed


def serve() -> None:
    print(f"main minute daemon {os.getpid()} started", flush=True)
    for product in PRODUCTS_ALL:
        path = MAIN / f"{product}.csv"
        if path.exists():
            _ensure_columns(path)
    seen_minute = ""
    flushed = ""
    while True:
        try:
            now = datetime.now(TZ)
            minute = now.strftime("%Y-%m-%d %H:%M")
            previous = _minutes(now) - 1
            if now.second >= 2 and minute != seen_minute and _in_session(previous):
                seen_minute = minute
                _append_live(now)
            # 14:59 and 15:00 finish after the day window closes. Write them
            # before the 18:00 roll stamp, and drop any quote sitting in between.
            post = ""
            if (now.hour, now.minute) in ((15, 6), (23, 6)):
                post = minute
            if now.second >= 2 and post and post != flushed:
                flushed = post
                _append_live(now)
            if now.minute % 10 == 0 and now.second < 5 and not (GAP_ONCE.exists() and GAP_ONCE.read_text().strip() == "done"):
                print(f"补一次切换空档 {GAP_START} → {GAP_END}", flush=True)
                _gap_once()
            due = _backfill_due(now)
            if due is not None and now.minute % 10 == 0 and now.second < 5:
                key, start, end = due
                print(f"收盘后补缺的字段 {start} → {end}", flush=True)
                done = {item.removeprefix(key + " ") for item in _done_marks() if item.startswith(key + " ")}
                changed, ready = _merge_download(start, end, done)
                for product in ready:
                    _remember(f"{key} {product}")
                if ready:
                    print(f"补入 {changed} 处", flush=True)
                else:
                    print("下载接口还没有这批分钟，过十分钟再试一次", flush=True)
            if _holefill_due(now) and now.minute % 10 == 0 and now.second < 5:
                print("晚上8点检查缺失分钟", flush=True)
                _evening_holefill(now)
                _remember_holefill(now.date().isoformat())
        except Exception as exc:
            print(f"main minute error {exc}", flush=True)
        time.sleep(1)


def daemonize() -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)
    os.chdir(ROOT)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    log = os.open(str(LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open("/dev/null", os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(log, 1)
    os.dup2(log, 2)


if __name__ == "__main__":
    if "--holefill-now" in sys.argv:
        _evening_holefill(datetime.now(TZ))
        raise SystemExit(0)
    if "--daemon" in sys.argv:
        daemonize()
    serve()
