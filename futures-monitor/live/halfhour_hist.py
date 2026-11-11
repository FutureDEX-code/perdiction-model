#!/usr/bin/env python3
"""Half-hour regression on historical minutes. Does not touch the live tape model.

Breaks 10:15–10:30 and 11:30–13:30 are not slots. The half hour before a break
is paired with the first half hour after it, so 13:30–14:00 is a target.
"""

from __future__ import annotations

import sys

import csv
import math
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

from core import train
from live.hour_model import _solve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MIN_DIR = ROOT / "data" / "tushare_min_history"
DAY_SLOTS = (9 * 60, 9 * 60 + 30, 10 * 60 + 30, 11 * 60, 13 * 60 + 30, 14 * 60, 14 * 60 + 30)
NIGHT_SLOTS = (21 * 60, 21 * 60 + 30, 22 * 60, 22 * 60 + 30)
# 11:00–11:30 predicts 13:30–14:00. 9:30–10:00 predicts 10:30–11:00.
BREAK_PAIRS = {3, 1}
SPLITS = ("2024-10-20", "2025-10-20")
MIN_BARS = 20
YUAN = 100


def _slot(clock: str, slots: tuple[int, ...]) -> int | None:
    minutes = int(clock[:2]) * 60 + int(clock[3:5])
    for index, start in enumerate(slots):
        if start <= minutes < start + 30:
            return index
    return None


def _empty() -> dict:
    return {"o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "n": 0, "vol": 0.0}


def _add(bar: dict, op: float, high: float, low: float, close: float, vol: float) -> None:
    if bar["n"] == 0:
        bar["o"] = op if op > 0 else close
        bar["h"] = high
        bar["l"] = low
    else:
        bar["h"] = max(bar["h"], high)
        bar["l"] = min(bar["l"], low)
    bar["c"] = close
    bar["n"] += 1
    bar["vol"] += vol


def scan(product: str) -> tuple[dict, dict, dict]:
    day: dict[str, dict[str, list]] = {}
    night: dict[str, dict[str, list]] = {}
    oi: dict[str, dict[str, float]] = {}
    for path in sorted((MIN_DIR / product).glob("*.csv")):
        code = path.stem
        by_day: dict[str, list] = {}
        by_night: dict[str, list] = {}
        oi_day: dict[str, float] = {}
        with path.open(newline="") as handle:
            rows = csv.reader(handle)
            next(rows, None)
            for row in rows:
                if len(row) < 8 or len(row[0]) < 16:
                    continue
                date, clock = row[0][:10], row[0][11:16]
                try:
                    op, high, low, close, vol, interest = (
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                        float(row[7]),
                    )
                except ValueError:
                    continue
                if close <= 0:
                    continue
                if clock == "14:50" and interest > 0:
                    oi_day[date] = interest
                if clock >= "21:00":
                    index = _slot(clock, NIGHT_SLOTS)
                    book = by_night
                else:
                    index = _slot(clock, DAY_SLOTS)
                    book = by_day
                if index is None:
                    continue
                bars = book.setdefault(date, [_empty() for _ in range(8)])
                _add(bars[index], op, high, low, close, vol)
        if by_day:
            day[code] = by_day
        if by_night:
            night[code] = by_night
        if oi_day:
            oi[code] = oi_day
    return day, night, oi


def _ym(product: str, code: str) -> int:
    return int(code[len(product) :])


def samples_for(product: str) -> list[dict]:
    day, night, oi = scan(product)
    trading = sorted({date for days in oi.values() for date in days})
    night_day: dict[str, str] = {}
    for code, booked in night.items():
        for calendar in booked:
            index = bisect_right(trading, calendar)
            if index >= len(trading):
                continue
            target = trading[index]
            gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
            if 0 < gap <= 15:
                night_day[calendar] = target
    mains: dict[str, str] = {}
    for index, date in enumerate(trading):
        if index == 0:
            continue
        prev = trading[index - 1]
        best = None
        for code, booked in oi.items():
            interest = booked.get(prev) or 0.0
            rank = (interest, _ym(product, code))
            if interest > 0 and (best is None or rank > best[0]):
                best = (rank, code)
        if best is not None:
            mains[date] = best[1]
    out = []
    attached: dict[str, str] = {}
    for calendar, target in night_day.items():
        if target not in attached or calendar > attached[target]:
            attached[target] = calendar

    def usable(bar: dict | None) -> bool:
        return bar is not None and bar["n"] >= MIN_BARS and bar["vol"] > 0 and bar["o"] > 0

    def feat(bar: dict) -> dict:
        span = bar["h"] - bar["l"]
        loc = 0.5 if span <= 0 else (bar["c"] - bar["l"]) / span
        return {"ret": bar["c"] - bar["o"], "rng": span, "loc": loc}

    for date, code in mains.items():
        bars = (day.get(code) or {}).get(date)
        if not bars:
            continue
        for index in range(len(DAY_SLOTS) - 1):
            prev, nxt = bars[index], bars[index + 1]
            if not usable(prev) or not usable(nxt):
                continue
            out.append({**feat(prev), "y": nxt["c"] - nxt["o"], "day": date, "break": index in BREAK_PAIRS})
        calendar = attached.get(date)
        nbars = (night.get(code) or {}).get(calendar) if calendar else None
        if not nbars:
            continue
        for index in range(len(NIGHT_SLOTS) - 1):
            prev, nxt = nbars[index], nbars[index + 1]
            if not usable(prev) or not usable(nxt):
                continue
            out.append({**feat(prev), "y": nxt["c"] - nxt["o"], "day": date, "break": False})
    return out


def fit_predict(train_rows: list[dict], test_rows: list[dict]) -> dict | None:
    if len(train_rows) < 100 or not test_rows:
        return None
    cols = ("ret", "rng", "loc")
    mean = {col: sum(row[col] for row in train_rows) / len(train_rows) for col in cols}
    scale = {}
    for col in cols:
        var = sum((row[col] - mean[col]) ** 2 for row in train_rows) / len(train_rows)
        scale[col] = math.sqrt(var) or 1.0

    def zrow(row: dict) -> list[float]:
        return [1.0, *[(row[col] - mean[col]) / scale[col] for col in cols]]

    size = 4
    xtx = [[0.0] * size for _ in range(size)]
    xty = [0.0] * size
    for row in train_rows:
        zed = zrow(row)
        for i in range(size):
            xty[i] += zed[i] * row["y"]
            for j in range(size):
                xtx[i][j] += zed[i] * zed[j]
    for i in range(1, size):
        xtx[i][i] += 1.0
    beta = _solve(xtx, xty)
    orders = wins = breaks = break_wins = 0
    pnl = 0.0
    ys = []
    yh = []
    for row in test_rows:
        points = sum(weight * value for weight, value in zip(beta, zrow(row)))
        ys.append(row["y"])
        yh.append(points)
        side = 1 if points > 0.5 else -1 if points < -0.5 else 0
        if side == 0:
            continue
        this = side * row["y"] * YUAN
        pnl += this
        orders += 1
        hit = this > 0
        wins += hit
        if row["break"]:
            breaks += 1
            break_wins += hit
    mean_y = sum(ys) / len(ys)
    mean_h = sum(yh) / len(yh)
    cov = sum((y - mean_y) * (h - mean_h) for y, h in zip(ys, yh))
    sy = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    sh = math.sqrt(sum((h - mean_h) ** 2 for h in yh))
    corr = cov / (sy * sh) if sy and sh else 0.0
    return {
        "train": len(train_rows),
        "test": len(test_rows),
        "corr": corr,
        "orders": orders,
        "wins": wins,
        "pnl": pnl,
        "breaks": breaks,
        "break_wins": break_wins,
    }


def main() -> None:
    print("休息时段不进样本。9:30 预测 10:30 那节，11:00 预测 13:30 那节。", flush=True)
    for product in train.CONTRACTS:
        rows = [row for row in samples_for(product) if row["day"] >= "2017-01-01"]
        print(f"\n{train.NAMES[product]} 样本 {len(rows)} 其中跨休息 {sum(row['break'] for row in rows)}", flush=True)
        for split in SPLITS:
            train_rows = [row for row in rows if row["day"] < split]
            test_rows = [row for row in rows if row["day"] >= split]
            result = fit_predict(train_rows, test_rows)
            if result is None:
                print(f"  {split} 之后 样本不够", flush=True)
                continue
            hit = result["wins"] / result["orders"] if result["orders"] else 0.0
            bhit = result["break_wins"] / result["breaks"] if result["breaks"] else 0.0
            print(
                f"  {split} 之后 相关 {result['corr']:+.3f} "
                f"下单{result['orders']} 对{result['wins']} ({hit * 100:.1f}%) "
                f"毛利{result['pnl'] / 10000:+.2f}万 "
                f"跨休息下单{result['breaks']} 对{result['break_wins']} ({bhit * 100:.1f}%) "
                f"训练{result['train']} 测试{result['test']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
