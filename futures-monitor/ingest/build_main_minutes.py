#!/usr/bin/env python3
"""SUPERSEDED 2026-10-27. Do not run this against training_dataset/main_minute.

The rule-based roll (schedule / schedule_sina, "East Money rule") was retired.
Main-continuous files are now stitched from the true roll calendar by
ingest/stitch_from_calendar.py, reading training_dataset/roll_calendar/*.json.
Running this file's main()/main_sina() would overwrite those stitches.

The functions stay for reference only. Everything below is frozen history.
"""

from __future__ import annotations

import sys

import csv
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "data" / "tushare_min"
OUT = ROOT / "data" / "tushare_min_main"
PRODUCTS = ("A", "B", "M", "Y", "RM")


def expiry(product: str, code: str) -> tuple[int, int]:
    ym = int(code[len(product) :])
    year, month = 2000 + ym // 100, ym % 100
    return year, month


def shift_month(year: int, month: int, back: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) - back
    return index // 12, index % 12 + 1


def daily_stats(product: str) -> tuple[dict, dict, dict]:
    """End-of-day open interest, day volume, and night volume by calendar date."""

    oi: dict[str, dict[str, float]] = {}
    day_vol: dict[str, dict[str, float]] = {}
    night_vol: dict[str, dict[str, float]] = {}
    for path in sorted((SRC / product).glob("*.csv")):
        code = path.stem
        oi[code] = {}
        day_vol[code] = defaultdict(float)
        night_vol[code] = defaultdict(float)
        with path.open() as handle:
            next(handle, None)
            for line in handle:
                if len(line) < 20:
                    continue
                date, clock = line[:10], line[11:16]
                parts = line.split(",")
                try:
                    vol = float(parts[5])
                    interest = float(parts[7])
                except (IndexError, ValueError):
                    continue
                if clock >= "21:00":
                    night_vol[code][date] += vol
                elif "09:00" <= clock <= "15:00":
                    day_vol[code][date] += vol
                    if interest > 0:
                        oi[code][date] = interest
    return oi, day_vol, night_vol


def choose(cands: list[tuple[float, float, tuple[int, int], str]]) -> str:
    return max(cands)[3]


def schedule(product: str, oi: dict, day_vol: dict, night_vol: dict) -> tuple[list[str], dict[str, str], list[tuple]]:
    trading = sorted({date for days in oi.values() for date in days})
    night_day = {}
    for code, booked in night_vol.items():
        for calendar in booked:
            index = bisect_right(trading, calendar)
            if index >= len(trading):
                continue
            target = trading[index]
            gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
            if 0 < gap <= 15:
                night_day.setdefault(target, []).append((code, calendar))
    volume = {date: {} for date in trading}
    for index, date in enumerate(trading):
        for code in oi:
            total = day_vol[code].get(date, 0.0)
            if index:
                prev = trading[index - 1]
                # Night on the calendar date of the previous session belongs here
                # only when that night was attached to this trading day.
                for night_code, calendar in night_day.get(date, ()):
                    if night_code == code:
                        total += night_vol[code].get(calendar, 0.0)
            volume[date][code] = total
    main: dict[str, str] = {}
    rolls: list[tuple] = []
    current = None
    for index, date in enumerate(trading):
        cands = []
        for code, days in oi.items():
            interest = days.get(date)
            if not interest:
                continue
            cands.append((interest, volume[date].get(code, 0.0), expiry(product, code), code))
        if not cands:
            continue
        winner = choose(cands)
        if current is None:
            current = winner
        else:
            year, month = expiry(product, current)
            limit_year, limit_month = shift_month(year, month, 2)
            month_days = [day for day in trading if day[:7] == f"{limit_year:04d}-{limit_month:02d}"]
            forced = bool(month_days) and date == month_days[-1]
            if forced:
                farther = [item for item in cands if item[2] > expiry(product, current)]
                if farther:
                    winner = choose(farther)
            if expiry(product, winner) > expiry(product, current):
                rolls.append((date, current, winner))
                current = winner
        if index + 1 < len(trading):
            main[trading[index + 1]] = current
    if trading and trading[0] not in main:
        first = choose(
            [
                (days[trading[0]], volume[trading[0]].get(code, 0.0), expiry(product, code), code)
                for code, days in oi.items()
                if trading[0] in days
            ]
        )
        main[trading[0]] = first
    return trading, main, rolls


def schedule_sina(product: str, oi: dict, day_vol: dict, night_vol: dict) -> tuple[list[str], dict[str, str], list[tuple]]:
    """Sina does not publish its roll. This is the East Money rule that matched it.

    A contract becomes the main only when it leads both volume and open interest.
    If the current main leads neither, it gives way next session to the open-interest leader.
    """

    trading = sorted({date for days in oi.values() for date in days})
    night_day: dict[str, list] = {}
    for code, booked in night_vol.items():
        for calendar in booked:
            index = bisect_right(trading, calendar)
            if index >= len(trading):
                continue
            target = trading[index]
            gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
            if 0 < gap <= 15:
                night_day.setdefault(target, []).append((code, calendar))
    volume = {date: {} for date in trading}
    for date in trading:
        for code in oi:
            total = day_vol[code].get(date, 0.0)
            for night_code, calendar in night_day.get(date, ()):
                if night_code == code:
                    total += night_vol[code].get(calendar, 0.0)
            volume[date][code] = total

    def stats(date: str) -> dict[str, tuple]:
        found = {}
        for code, days in oi.items():
            interest = days.get(date)
            if not interest:
                continue
            found[code] = (interest, volume[date].get(code, 0.0), expiry(product, code))
        return found

    def leaders(found: dict[str, tuple]) -> tuple[str, str]:
        max_oi = max(found, key=lambda code: found[code])
        max_vol = max(found, key=lambda code: (found[code][1], found[code][0], found[code][2]))
        return max_oi, max_vol

    main: dict[str, str] = {}
    rolls: list[tuple] = []
    current = None
    for index, date in enumerate(trading):
        found = stats(date)
        if not found:
            continue
        max_oi, max_vol = leaders(found)
        if current is None:
            current = max_oi
        elif max_oi == max_vol and max_oi != current:
            rolls.append((date, current, max_oi))
            current = max_oi
        elif current != max_oi and current != max_vol:
            rolls.append((date, current, max_oi))
            current = max_oi
        if index + 1 < len(trading):
            main[trading[index + 1]] = current
    if trading and trading[0] not in main:
        found = stats(trading[0])
        if found:
            main[trading[0]] = leaders(found)[0]
    return trading, main, rolls


def write_series(product: str, trading: list[str], main: dict[str, str], folder: Path = OUT) -> int:
    night_target = {}
    for index, date in enumerate(trading):
        if index == 0:
            continue
        night_target[trading[index - 1]] = date
    # A night on calendar C maps to the next trading day, which may skip a weekend.
    # Rebuild from the actual night dates while writing.
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{product}.csv"
    rows = 0
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "contract"])
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
                    parts = line.rstrip("\n").split(",")
                    bucket.append((parts[0], parts[:8] + [code]))
        bucket.sort()
        writer.writerows(row for _, row in bucket)
        rows = len(bucket)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roll_path = OUT / "rolls.csv"
    with roll_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product", "judged_on", "old", "new", "used_from"])
        for product in PRODUCTS:
            print(f"统计{product}", flush=True)
            oi, day_vol, night_vol = daily_stats(product)
            trading, used, rolls = schedule(product, oi, day_vol, night_vol)
            print(f"  交易日 {len(trading)} 换月 {len(rolls)}", flush=True)
            for judged, old, new in rolls:
                index = trading.index(judged)
                used_from = trading[index + 1] if index + 1 < len(trading) else ""
                writer.writerow([product, judged, old, new, used_from])
            print(f"  写出分钟", flush=True)
            count = write_series(product, trading, used)
            print(f"  {product}.csv {count} 行", flush=True)
    print(f"已写入 {OUT}", flush=True)


def main_sina() -> None:
    folder = ROOT / "training_dataset" / "main_minute"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "rolls.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product", "judged_on", "old", "new", "used_from"])
        for product in PRODUCTS:
            print(f"统计{product}", flush=True)
            oi, day_vol, night_vol = daily_stats(product)
            trading, used, rolls = schedule_sina(product, oi, day_vol, night_vol)
            print(f"  交易日 {len(trading)} 换月 {len(rolls)}", flush=True)
            for judged, old, new in rolls:
                index = trading.index(judged)
                used_from = trading[index + 1] if index + 1 < len(trading) else ""
                writer.writerow([product, judged, old, new, used_from])
            print("  写出分钟", flush=True)
            count = write_series(product, trading, used, folder)
            print(f"  {product}.csv {count} 行", flush=True)
    print(f"已写入 {folder}", flush=True)


if __name__ == "__main__":
    import sys

    raise SystemExit(
        "这个脚本已被 stitch_from_calendar.py 取代（2026-10-27）。"
        "再跑它会用旧规则覆盖按真值日历拼好的主连文件。"
    )
