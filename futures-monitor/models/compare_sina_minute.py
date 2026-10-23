#!/usr/bin/env python3
"""Add the Sina-rule continuous minutes on top of the daily model.

Base features stay the Sina daily ones. The extra columns are that day's
night move, day move, open-interest change, volume change, and price/OI
attitude. Money uses the live rule: enter at 14:55, hold through a flat
signal, close only when the direction reverses, on the same contract.
"""

from __future__ import annotations

import sys

import csv
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core import train
from models import trade1455_model as t55

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAIN = train.MAIN_MINUTE
SPLITS = ("2023-10-20", "2024-10-20", "2025-10-20")
YUAN = 100
VEC = {"A": "x", "M": "x", "RM": "x_rm", "B": "x_b", "Y": "x_y"}
BASE = {
    "A": train.FEATURE_NAMES,
    "M": train.FEATURE_NAMES,
    "RM": train.RM_FEATURES,
    "B": train.B_FEATURES,
    "Y": train.Y_FEATURES,
}
WATCH = {
    "A": ("A", "M", "RM"),
    "M": ("A", "M", "RM"),
    "RM": ("M", "RM"),
    "B": ("M", "RM", "B"),
    "Y": ("A", "M", "RM", "Y"),
}
PARAMS = {
    "objective": "binary",
    "max_depth": 2,
    "num_leaves": 4,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 0,
    "num_threads": 1,
}


def extra_names(key: str) -> tuple[str, ...]:
    names = []
    for item in WATCH[key]:
        name = train.NAMES[item]
        names.extend((f"{name}夜盘", f"{name}日盘", f"{name}持仓变化", f"{name}成交量变化", f"{name}价仓"))
    return tuple(names)


def load_path(product: str) -> dict[str, dict]:
    nights: dict[str, list] = {}
    days: dict[str, list] = {}
    with (MAIN / f"{product}.csv").open() as handle:
        for row in csv.DictReader(handle):
            date, clock = row["trade_time"][:10], row["trade_time"][11:16]
            bar = (
                clock,
                float(row["open"]),
                float(row["close"]),
                float(row["vol"]),
                float(row["oi"]),
                row["contract"],
            )
            if clock >= "21:00":
                nights.setdefault(date, []).append(bar)
            elif "09:00" <= clock <= "15:00":
                days.setdefault(date, []).append(bar)
    trading = sorted(days)
    out = {}
    for date in trading:
        index = None
        bars = list(days[date])
        # night belonging to this trading day was on an earlier calendar date
        out[date] = bars
    for calendar, bars in nights.items():
        index = bisect_right(trading, calendar)
        if index >= len(trading):
            continue
        target = trading[index]
        gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
        if 0 < gap <= 15:
            out[target] = bars + out.get(target, [])
    features = {}
    prev_oi = None
    prev_vol = None
    for date in trading:
        bars = out.get(date) or []
        if not bars:
            continue
        night = [bar for bar in bars if bar[0] >= "21:00"]
        day = [bar for bar in bars if "09:00" <= bar[0] <= "14:50"]
        px1450 = next((bar[2] for bar in reversed(day) if bar[0] == "14:50"), None)
        oi1450 = next((bar[4] for bar in reversed(day) if bar[0] == "14:50" and bar[4] > 0), None)
        contract = day[-1][5] if day else bars[-1][5]
        vol = sum(bar[3] for bar in bars)
        num = den = 0.0
        prev = None
        for bar in bars:
            if bar[3] <= 0:
                continue
            if prev is not None:
                change = bar[2] - prev[0]
                oi_change = bar[4] - prev[1]
                if change != 0 and oi_change != 0:
                    num += (1.0 if change > 0 else -1.0) * oi_change
                    den += abs(oi_change)
            prev = (bar[2], bar[4])
        night_ret = 0.0
        if len(night) >= 2 and night[0][1]:
            night_ret = (night[-1][2] - night[0][1]) / night[0][1] * 100.0
        day_ret = 0.0
        if day and px1450 and day[0][1]:
            day_ret = (px1450 - day[0][1]) / day[0][1] * 100.0
        oi_chg = 0.0 if not prev_oi or not oi1450 else (oi1450 - prev_oi) / prev_oi * 100.0
        vol_chg = 0.0 if not prev_vol or not vol else (vol - prev_vol) / prev_vol * 100.0
        features[date] = {
            "夜盘": night_ret,
            "日盘": day_ret,
            "持仓变化": oi_chg,
            "成交量变化": vol_chg,
            "价仓": num / den if den else 0.0,
            "contract": contract,
        }
        if oi1450:
            prev_oi = oi1450
        if vol:
            prev_vol = vol
    return features


def hold(events, px) -> tuple[int, int, float]:
    pnl = 0.0
    closes = wins = 0
    pos = None
    for ev in events:
        side = ev["side"]
        if pos is None:
            if side == 0 or ev["entry"] is None:
                continue
            pos = ev
            continue
        if side == 0 or side == pos["side"]:
            continue
        exit_px = px.get(pos["code"], {}).get(ev["day"])
        if exit_px is None:
            continue
        this = pos["side"] * (exit_px - pos["entry"]) * YUAN
        pnl += this
        closes += 1
        wins += this > 0
        pos = None if ev["entry"] is None else ev
    if pos is not None and events:
        exit_px = px.get(pos["code"], {}).get(events[-1]["day"])
        if exit_px is not None and exit_px != pos["entry"]:
            this = pos["side"] * (exit_px - pos["entry"]) * YUAN
            pnl += this
            closes += 1
            wins += this > 0
    return closes, wins, pnl


def side_of(prob: float) -> int:
    return 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0


def main() -> None:
    paths = {product: load_path(product) for product in train.CONTRACTS}
    print("读取14:55", flush=True)
    px1455 = {product: t55.scan_1455(product) for product in train.CONTRACTS}
    raw = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in (*train.DOMESTIC, *train.OVERSEAS)
    }
    rows = [row for row in train.samples(raw, train.news_scores()) if row["day"] >= "2017-01-01"]
    common = sorted(set(raw["A"]) & set(raw["M"]) & set(raw["RM"]))
    prev_of = {day: prev for prev, day in zip(common, common[1:])}

    def named_extra(signal: str, key: str) -> dict[str, float] | None:
        named = {}
        for item in WATCH[key]:
            slot = paths[item].get(signal)
            if slot is None:
                return None
            name = train.NAMES[item]
            for suffix in ("夜盘", "日盘", "持仓变化", "成交量变化", "价仓"):
                named[f"{name}{suffix}"] = slot[suffix]
        return named

    totals = {split: {"回归": 0.0, "回归加分钟": 0.0, "树": 0.0, "树加分钟": 0.0} for split in SPLITS}
    for key in train.CONTRACTS:
        vec = VEC[key]
        base_names = tuple(BASE[key])
        full_names = base_names + extra_names(key)
        usable = []
        for row in rows:
            signal = prev_of.get(row["day"])
            if signal is None or not row.get(vec) or key not in row["y"]:
                continue
            extra = named_extra(signal, key)
            if extra is None:
                continue
            code = paths[key][signal]["contract"]
            entry = px1455[key].get(code, {}).get(signal)
            usable.append((row, signal, extra, code, entry))
        print(f"\n{train.NAMES[key]} 可对齐 {len(usable)} 天", flush=True)
        for split in SPLITS:
            train_rows = [item for item in usable if item[0]["day"] < split]
            test_rows = [item for item in usable if item[0]["day"] >= split]

            def fit(names):
                matrix = []
                for row, _signal, extra, _code, _entry in train_rows:
                    values = {**dict(zip(base_names, row[vec])), **extra}
                    matrix.append([values[name] for name in names])
                mean, scale = train.standardize(matrix)
                xs = [train.apply_scale(row, mean, scale) for row in matrix]
                ys = [row["y"][key] for row, *_ in train_rows]
                ws = [train.sample_weight(row["move"][key]) for row, *_ in train_rows]
                weight, bias = train.fit(xs, ys, ws)
                booster = lgb.train(
                    PARAMS,
                    lgb.Dataset(np.array(xs), label=np.array(ys), weight=np.array(ws)),
                    num_boost_round=100,
                )
                return mean, scale, weight, bias, booster

            def events(names, mean, scale, weight, bias, booster, kind):
                made = []
                for row, signal, extra, code, entry in test_rows:
                    values = {**dict(zip(base_names, row[vec])), **extra}
                    vector = train.apply_scale([values[name] for name in names], mean, scale)
                    prob = train.predict_prob(vector, weight, bias) if kind == "logit" else float(booster.predict([vector])[0])
                    made.append({"day": signal, "side": side_of(prob), "code": code, "entry": entry})
                return made

            base = fit(base_names)
            full = fit(full_names)
            line = [f"  {split}"]
            for label, names, packed, kind in (
                ("回归", base_names, base, "logit"),
                ("回归加分钟", full_names, full, "logit"),
                ("树", base_names, base, "tree"),
                ("树加分钟", full_names, full, "tree"),
            ):
                mean, scale, weight, bias, booster = packed
                closes, wins, pnl = hold(events(names, mean, scale, weight, bias, booster, kind), px1455[key])
                totals[split][label] += pnl
                line.append(f"{label}{pnl / 10000:+.2f}万/{closes}笔")
            print(" ".join(line), flush=True)
    print("\n合计", flush=True)
    for split in SPLITS:
        print(split, " ".join(f"{name}{pnl / 10000:+.2f}万" for name, pnl in totals[split].items()), flush=True)


if __name__ == "__main__":
    main()
