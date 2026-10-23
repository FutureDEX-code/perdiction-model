#!/usr/bin/env python3
"""Keep the main-continuous minute files. This process is their only writer.

During a session, each finished minute is written from the live quote.
After the close, one pass fills empty turnover from the paid minute download.
Price, volume, open interest, and bid/ask stay as the live minute wrote them.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
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
GAP_ONCE = ROOT / "data" / "main_minute_gap_once.txt"
GAP_START = "2026-10-21 21:00:00"
GAP_END = "2026-10-22 13:29:00"
MAIN = ROOT / "training_dataset" / "main_minute"
LIVE = "http://127.0.0.1:8771/latest"
EXTRAS = ("bid", "ask", "bid_vol", "ask_vol", "pre_close", "pre_settle")
HEADER = ["trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "contract", *EXTRAS]
SESSIONS = ((9 * 60, 10 * 60 + 15), (10 * 60 + 30, 11 * 60 + 30), (13 * 60 + 30, 15 * 60), (21 * 60, 23 * 60))
def _minutes(now: datetime) -> int:
    return now.hour * 60 + now.minute


def _in_session(minutes: int) -> bool:
    return any(start <= minutes < end for start, end in SESSIONS)


def _contract_name(code: str) -> str:
    text = code.lower()
    if text.startswith("rm"):
        digits = text[2:]
        if len(digits) == 3:
            return "RM2" + digits
        return "RM" + digits
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
    mapping: dict[str, str] = {}
    unresolved: set[str] = set()
    for product in ("A", "B", "M", "Y", "RM"):
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
        if any(f"{key} {product}" not in done for product in ("A", "B", "M", "Y", "RM")):
            return key, f"{today} 09:00:00", f"{today} 15:00:00"
    if (now.hour, now.minute) >= (23, 5):
        key = f"{today} night"
        if any(f"{key} {product}" not in done for product in ("A", "B", "M", "Y", "RM")):
            return key, f"{today} 21:00:00", f"{today} 23:00:00"
    return None


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


def serve() -> None:
    print(f"main minute daemon {os.getpid()} started", flush=True)
    for product in ("A", "B", "M", "Y", "RM"):
        path = MAIN / f"{product}.csv"
        if path.exists():
            _ensure_columns(path)
    seen_minute = ""
    while True:
        try:
            now = datetime.now(TZ)
            minute = now.strftime("%Y-%m-%d %H:%M")
            previous = _minutes(now) - 1
            if now.second >= 2 and minute != seen_minute and _in_session(previous):
                seen_minute = minute
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
    if "--daemon" in sys.argv:
        daemonize()
    serve()
