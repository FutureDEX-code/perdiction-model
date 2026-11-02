#!/usr/bin/env python3
"""The one true roll calendar: which contract the Sina continuous tracked each day.

Derived by matching the Sina continuous daily bar against single-contract daily
bars. History comes from the local per-contract minute files (data/tushare_min),
the recent tail from Sina's per-contract daily API. One file per product in
training_dataset/roll_calendar/<KEY>.json: {"2026-10-27": "B2612", ...}.

The daily files and the minute stitches both read this calendar. The 18:00
sina_daily_daemon appends every new trading day and, for the five traded
products, a confirmed roll also drives the CTP subscription.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train

CAL = ROOT / "training_dataset" / "roll_calendar"
PRODUCTS = ("A", "B", "M", "RM", "Y", "P", "OI", "PK")
TRADED = PRODUCTS


def _contract_daily_from_minutes(product: str) -> dict[str, dict[str, dict]]:
    """Aggregate data/tushare_min/<product>/*.csv into per-contract daily bars."""

    folder = ROOT / "data" / "tushare_min" / product
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.csv")):
        code = path.stem
        days: dict[str, dict] = {}
        with path.open() as handle:
            for row in csv.DictReader(handle):
                stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
                if hm >= "21:00":
                    continue
                slot = days.get(stamp)
                if slot is None:
                    slot = {"收": None, "量": 0.0, "仓": None}
                    days[stamp] = slot
                slot["收"] = float(row["close"])
                slot["量"] += float(row["vol"] or 0)
                if row["oi"]:
                    slot["仓"] = float(row["oi"])
        out[code] = days
    return out


def _contract_daily_from_sina(codes: list[str]) -> dict[str, dict[str, dict]]:
    """Per-contract daily bars from Sina's own API. Used for the recent tail."""

    out: dict[str, dict[str, dict]] = {}
    for code in codes:
        try:
            rows = train.fetch_domestic(code)
        except Exception as exc:
            print(f"  {code} 新浪单合约日线拉取失败 {exc}", flush=True)
            continue
        out[code] = {
            str(r["日期"]): {"收": float(r["收盘"]), "量": float(r["成交量"] or 0), "仓": float(r["持仓"] or 0)}
            for r in rows
        }
    return out


def _match(bar: dict, candidates: dict[str, dict[str, dict]], day: str) -> str | None:
    """Which candidate's daily bar is this continuous bar?

    The close must line up; the tracked contract's own open interest is the
    decisive tie-break when two delivery months close within a tick of each other.
    """

    close = float(bar["收盘"])
    vol = float(bar.get("成交量") or 0)
    oi = float(bar.get("持仓") or 0)
    scored = []
    for code, days in candidates.items():
        slot = days.get(day)
        if not slot or slot.get("收") is None:
            continue
        diff = abs(slot["收"] - close)
        if diff > 0.5:
            continue
        scored.append((diff, abs((slot.get("仓") or 0) - oi), abs(slot["量"] - vol), code))
    if not scored:
        return None
    scored.sort()
    return scored[0][3]


def _next_months(code: str, count: int = 3) -> list[str]:
    """Candidate delivery months after the given contract code."""

    prefix = code.rstrip("0123456789")
    ym = int(code[len(prefix):])
    year, month = ym // 100, ym % 100
    out = []
    for step in range(1, count + 1):
        index = year * 12 + (month - 1) + step
        out.append(f"{prefix}{index // 12 % 100:02d}{index % 12 + 1:02d}")
    return out


def calendar_path(product: str) -> Path:
    return CAL / f"{product}.json"


def load_calendar(product: str) -> dict[str, str]:
    path = calendar_path(product)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_calendar(product: str, cal: dict[str, str]) -> None:
    CAL.mkdir(parents=True, exist_ok=True)
    path = calendar_path(product)
    path.write_text(json.dumps(dict(sorted(cal.items())), ensure_ascii=False, indent=2))


def backfill(product: str) -> dict[str, str]:
    """Full rebuild of one product's calendar from local minutes plus the Sina tail."""

    sina = [r for r in json.loads((train.SINA_DAILY / f"{product}.json").read_text()) if str(r["日期"]) >= train.DATA_FLOOR]
    candidates = _contract_daily_from_minutes(product)
    # Tail candidates: contracts alive in the last 30 days by their newest bar,
    # plus the next two delivery months of the freshest few. Alphabetical recency
    # would mistake far-month files for the current main.
    freshest = sorted(
        (code for code, days in candidates.items() if days),
        key=lambda code: max(candidates[code]),
        reverse=True,
    )
    cutoff = str(sina[-30]["日期"])
    tail_codes = {code for code in freshest if max(candidates[code]) >= cutoff}
    for code in freshest[:4]:
        tail_codes.update(_next_months(code, 2))
    for code, days in _contract_daily_from_sina(sorted(tail_codes)).items():
        target = candidates.setdefault(code, {})
        for day, slot in days.items():
            target.setdefault(day, slot)
    cal: dict[str, str] = {}
    misses = []
    for bar in sina:
        day = str(bar["日期"])
        hit = _match(bar, candidates, day)
        if hit is None:
            misses.append(day)
            continue
        cal[day] = hit
    # A day nobody can be matched on keeps the previous contract: it is either a
    # Sina phantom bar or a roll the next session will confirm anyway.
    carried = 0
    for day in misses:
        earlier = [known for known in cal if known < day]
        if earlier:
            cal[day] = cal[max(earlier)]
            carried += 1
    cal = dict(sorted(cal.items()))
    save_calendar(product, cal)
    days = sorted(cal)
    rolls = [(day, cal[prev], cal[day]) for prev, day in zip(days, days[1:]) if cal[prev] != cal[day]]
    print(
        f"{product}: {len(cal)} 天 {days[0] if days else '-'} -> {days[-1] if days else '-'} 换月 {len(rolls)} 次"
        + (f" 沿用前约 {carried} 天" if carried else "")
        + (f" 仍缺 {len(misses) - carried} 天" if len(misses) > carried else ""),
        flush=True,
    )
    return cal


def update_today(domestic_rows: dict[str, list[dict]], traded_hook=None) -> list[tuple[str, str, str]]:
    """Catch the calendar up to the freshest Sina daily rows. Returns confirmed rolls.

    Walks every day missing from the calendar, not just today, so a day the API
    could not confirm lands on the next run anyway. traded_hook(product, old, new)
    fires for the five traded products on every confirmed roll.
    """

    rolls = []
    for product in PRODUCTS:
        rows = domestic_rows.get(product) or []
        if not rows:
            continue
        cal = load_calendar(product)
        by_day = {str(r["日期"]): r for r in rows}
        missing = [day for day in sorted(by_day) if day not in cal]
        if not missing:
            continue
        candidates_cache: dict[tuple, dict] = {}
        changed = False
        for day in missing:
            recent = [cal[d] for d in sorted(cal)[-5:]] or []
            codes = tuple(sorted(set(recent + [m for code in recent[-2:] for m in _next_months(code)])))
            if codes not in candidates_cache:
                candidates_cache[codes] = _contract_daily_from_sina(list(codes))
            hit = _match(by_day[day], candidates_cache[codes], day)
            if hit is None:
                print(f"{product} {day} 单合约日线还没出齐，留下次再补", flush=True)
                continue
            cal[day] = hit
            changed = True
            days = sorted(cal)
            index = days.index(day)
            if index > 0 and cal[days[index - 1]] != hit:
                old = cal[days[index - 1]]
                rolls.append((product, old, hit))
                print(f"{product} 换月确认 {day} {old} -> {hit}", flush=True)
                if traded_hook is not None and product in TRADED:
                    traded_hook(product, old, hit)
        if changed:
            save_calendar(product, cal)
    return rolls


if __name__ == "__main__":
    for key in (sys.argv[1:] or PRODUCTS):
        backfill(key)
