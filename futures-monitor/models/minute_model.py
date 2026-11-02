#!/usr/bin/env python3
"""A side model. It does not read or write data/models/.

Label is one real contract, from today's 14:50 to the next session's 14:50.
Features are the live model's features, plus that session's night move, day
move, open-interest change, and the minute-by-minute price/OI attitude.
Weights go to data/models_1450/.
"""

from __future__ import annotations

import sys

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MIN_DIR = ROOT / "data" / "tushare_min"
OUT = ROOT / "data" / "models_1450"
SPLITS = ("2023-10-20", "2024-10-20", "2025-10-20")
KEEP = "2017-01-01"
YUAN_PER_POINT = 100  # 10 lots

# Minute features added on top of the live feature list, per contract.
MINUTE_OF = {
    "A": ("A", "M", "RM"),
    "M": ("A", "M", "RM"),
    "RM": ("M", "RM"),
    "B": ("M", "RM", "B"),
    "Y": ("A", "M", "RM", "Y"),
}
BASE_OF = {
    "A": train.FEATURE_NAMES,
    "M": train.FEATURE_NAMES,
    "RM": train.RM_FEATURES,
    "B": train.B_FEATURES,
    "Y": train.Y_FEATURES,
}
VECTOR = {"A": "x", "M": "x", "RM": "x_rm", "B": "x_b", "Y": "x_y"}


def _agg() -> dict:
    return {
        "open": None,
        "first_px": None,
        "first_oi": None,
        "prev_px": None,
        "prev_oi": None,
        "att_num": 0.0,
        "att_den": 0.0,
        "vol": 0.0,
        "px1450": None,
        "oi1450": None,
    }


def _feed(agg: dict, op: float, px: float, vol: float, oi: float, clock: str) -> None:
    if agg["prev_px"] is not None and vol > 0:
        change = px - agg["prev_px"]
        oi_change = oi - agg["prev_oi"]
        if change != 0 and oi_change != 0:
            agg["att_num"] += (1.0 if change > 0 else -1.0) * oi_change
            agg["att_den"] += abs(oi_change)
    if vol > 0 and agg["open"] is None:
        agg["open"] = op if op > 0 else px
        agg["first_px"] = px
        agg["first_oi"] = oi
    if vol > 0:
        agg["prev_px"] = px
        agg["prev_oi"] = oi
    agg["vol"] += vol
    if clock == "14:50":
        agg["px1450"] = px
        agg["oi1450"] = oi


def _pct(start: float | None, end: float | None) -> float:
    if not start or end is None:
        return 0.0
    return (end - start) / start * 100.0


def _attitude(num: float, den: float) -> float:
    return num / den if den else 0.0


def scan_product(product: str) -> tuple[dict, dict, dict]:
    """Per contract, per day: 14:50 price, oi, and the session path into that close."""

    px: dict[str, dict[str, float]] = {}
    oi: dict[str, dict[str, float]] = {}
    session: dict[str, dict[str, dict]] = {}
    folder = MIN_DIR / product
    for path in sorted(folder.glob("*.csv")):
        code = path.stem
        day: dict[str, dict] = {}
        night: dict[str, dict] = {}
        with path.open(newline="") as handle:
            rows = csv.reader(handle)
            next(rows, None)
            for row in rows:
                if len(row) < 8 or len(row[0]) < 16:
                    continue
                date, clock = row[0][:10], row[0][11:16]
                try:
                    op, price, vol, interest = float(row[1]), float(row[4]), float(row[5]), float(row[7])
                except ValueError:
                    continue
                if clock >= "21:00":
                    _feed(night.setdefault(date, _agg()), op, price, vol, interest, clock)
                elif "09:00" <= clock <= "15:00":
                    _feed(day.setdefault(date, _agg()), op, price, vol, interest, clock)
        trading = sorted(date for date, agg in day.items() if agg["px1450"] is not None and agg["oi1450"])
        night_for: dict[str, dict] = {}
        for calendar, agg in night.items():
            index = bisect_right(trading, calendar)
            if index >= len(trading):
                continue
            target = trading[index]
            gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
            if gap > 15:
                continue
            if target not in night_for or calendar > night_for[target][0]:
                night_for[target] = (calendar, agg)
        px[code] = {}
        oi[code] = {}
        session[code] = {}
        for date in trading:
            agg = day[date]
            booked = night_for.get(date)
            night_agg = booked[1] if booked else None
            num = agg["att_num"]
            den = agg["att_den"]
            if night_agg is not None:
                num += night_agg["att_num"]
                den += night_agg["att_den"]
                if (
                    night_agg["prev_px"] is not None
                    and agg["first_px"] is not None
                ):
                    change = agg["first_px"] - night_agg["prev_px"]
                    oi_change = agg["first_oi"] - night_agg["prev_oi"]
                    if change != 0 and oi_change != 0:
                        num += (1.0 if change > 0 else -1.0) * oi_change
                        den += abs(oi_change)
            px[code][date] = agg["px1450"]
            oi[code][date] = agg["oi1450"]
            session[code][date] = {
                "oi": agg["oi1450"],
                "vol": agg["vol"] + (night_agg["vol"] if night_agg else 0.0),
                "night": _pct(
                    night_agg["open"] if night_agg else None,
                    night_agg["prev_px"] if night_agg else None,
                ),
                "day": _pct(agg["open"], agg["px1450"]),
                "att": _attitude(num, den),
            }
    return px, oi, session


def _ym(product: str, code: str) -> int:
    return int(code[len(product) :])


def build_series(product: str, px: dict, oi: dict, session: dict) -> dict:
    by_date: dict[str, list] = defaultdict(list)
    for code, days in session.items():
        expiry = _ym(product, code)
        for date, rec in days.items():
            by_date[date].append((rec["oi"], rec["vol"], expiry, code, rec))
    dates = sorted(by_date)
    ret: dict[str, float | None] = {}
    night: dict[str, float] = {}
    day: dict[str, float] = {}
    oichg: dict[str, float] = {}
    attitude: dict[str, float] = {}
    main: dict[str, str] = {}
    for index, date in enumerate(dates):
        _interest, _vol, _expiry, code, rec = max(by_date[date])
        main[date] = code
        if index == 0:
            ret[date] = None
            oichg[date] = 0.0
        else:
            prev = dates[index - 1]
            start = px[code].get(prev)
            ret[date] = (px[code][date] - start) / start * 100.0 if start else None
            start_oi = oi[code].get(prev)
            oichg[date] = (oi[code][date] - start_oi) / start_oi * 100.0 if start_oi else 0.0
        night[date] = rec["night"]
        day[date] = rec["day"]
        attitude[date] = rec["att"]
    level = 1000.0
    levels: list[float] = []
    used: list[str] = []
    for date in dates:
        if ret[date] is None:
            continue
        level *= 1.0 + ret[date] / 100.0
        levels.append(level)
        used.append(date)
    swing = dict(zip(used, train.swing_path(levels)))
    return {
        "dates": dates,
        "main": main,
        "ret": ret,
        "swing": swing,
        "night": night,
        "day": day,
        "oi": oichg,
        "att": attitude,
        "px": px,
    }


def minute_names(keys: tuple[str, ...]) -> tuple[str, ...]:
    names = []
    for key in keys:
        name = train.NAMES[key]
        names.extend((f"{name}夜盘", f"{name}日盘", f"{name}持仓变化", f"{name}价仓"))
    return tuple(names)


def build_rows(products: dict, overseas: dict, news: dict) -> list[dict]:
    common = sorted(
        set(products["A"]["swing"]) & set(products["M"]["swing"]) & set(products["RM"]["swing"])
    )
    common = [day for day in common if day >= KEEP]
    overseas_days = {key: sorted(overseas[key]) for key in train.OVERSEAS}
    cot = train.cot_series()
    seats = train.seat_series()
    rows = []
    dates = {key: products[key]["dates"] for key in products}
    for prev, day in zip(common, common[1:]):
        if any(train.latest_on_or_before(overseas_days[key], prev) is None for key in train.OVERSEAS):
            continue
        if any(products[key]["ret"][prev] is None for key in ("A", "M", "RM")):
            continue
        named = {
            "豆一昨涨跌": products["A"]["ret"][prev],
            "豆粕昨涨跌": products["M"]["ret"][prev],
            "菜粕昨涨跌": products["RM"]["ret"][prev],
            "美豆隔夜": train.overseas_session(overseas["S"], prev, day),
            "美豆粕隔夜": train.overseas_session(overseas["SM"], prev, day),
            "美豆油隔夜": train.overseas_session(overseas["BO"], prev, day),
            "豆一这段": products["A"]["swing"][prev],
            "豆粕这段": products["M"]["swing"][prev],
            "菜粕这段": products["RM"]["swing"][prev],
        }
        named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
        for key in ("B", "Y"):
            if prev in products[key]["swing"] and products[key]["ret"][prev] is not None:
                named[f"{train.NAMES[key]}昨涨跌"] = products[key]["ret"][prev]
                named[f"{train.NAMES[key]}这段"] = products[key]["swing"][prev]
        for key in products:
            named.update(train.extra_named(key, prev))
        named.update(train.cot_named(cot, prev))
        named.update(train.mood_named(cot, seats, prev))
        for key in products:
            if prev not in products[key]["night"]:
                continue
            name = train.NAMES[key]
            named[f"{name}夜盘"] = products[key]["night"][prev]
            named[f"{name}日盘"] = products[key]["day"][prev]
            named[f"{name}持仓变化"] = products[key]["oi"][prev]
            named[f"{name}价仓"] = products[key]["att"][prev]
        row = {"day": day, "signal": prev, "named": named, "y": {}, "move": {}, "entry": {}, "exit": {}}
        for key in products:
            code = products[key]["main"].get(prev)
            if code is None:
                continue
            entry = products[key]["px"][code].get(prev)
            exit_px = products[key]["px"][code].get(day)
            if not entry or exit_px is None:
                continue
            row["y"][key] = 1 if exit_px > entry else 0
            row["move"][key] = (exit_px - entry) / entry * 100.0
            row["entry"][key] = entry
            row["exit"][key] = exit_px
        rows.append(row)
    return rows


def features_for(key: str) -> tuple[str, ...]:
    return tuple(BASE_OF[key]) + minute_names(MINUTE_OF[key])


def fit_money(rows: list[dict], key: str, names: tuple[str, ...], split: str) -> dict | None:
    usable = []
    for row in rows:
        if key not in row["y"]:
            continue
        if any(name not in row["named"] for name in names):
            continue
        usable.append(row)
    train_rows = [row for row in usable if row["day"] < split]
    test_rows = [row for row in usable if row["day"] >= split]
    if len(train_rows) < 100 or not test_rows:
        return None
    mean, scale = train.standardize([[row["named"][name] for name in names] for row in train_rows])
    weight, bias = train.fit(
        [train.apply_scale([row["named"][name] for name in names], mean, scale) for row in train_rows],
        [row["y"][key] for row in train_rows],
        [train.sample_weight(row["move"][key]) for row in train_rows],
    )
    pnl = 0.0
    orders = wins = 0
    for row in test_rows:
        prob = train.predict_prob(
            train.apply_scale([row["named"][name] for name in names], mean, scale),
            weight,
            bias,
        )
        side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
        if side == 0:
            continue
        this = side * (row["exit"][key] - row["entry"][key]) * YUAN_PER_POINT
        pnl += this
        orders += 1
        wins += this > 0
    return {
        "orders": orders,
        "wins": wins,
        "pnl": pnl,
        "train": len(train_rows),
        "test": len(test_rows),
        "weight": weight,
        "bias": bias,
        "mean": mean,
        "scale": scale,
    }


def original_money(rows: list[dict], raw: dict, key: str, split: str) -> dict | None:
    vec = VECTOR[key]
    usable = [row for row in rows if row.get(vec) and key in row["y"] and row["day"] >= KEEP]
    train_rows = [row for row in usable if row["day"] < split]
    test_rows = [row for row in usable if row["day"] >= split]
    if len(train_rows) < 100 or not test_rows:
        return None
    mean, scale = train.standardize([row[vec] for row in train_rows])
    weight, bias = train.fit(
        [train.apply_scale(row[vec], mean, scale) for row in train_rows],
        [row["y"][key] for row in train_rows],
        [train.sample_weight(row["move"][key]) for row in train_rows],
    )
    pnl = 0.0
    orders = wins = 0
    for row in test_rows:
        prob = train.predict_prob(train.apply_scale(row[vec], mean, scale), weight, bias)
        side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
        if side == 0:
            continue
        exit_px = raw[key][row["day"]]
        entry = exit_px / (1.0 + row["move"][key] / 100.0)
        this = side * (exit_px - entry) * YUAN_PER_POINT
        pnl += this
        orders += 1
        wins += this > 0
    return {"orders": orders, "wins": wins, "pnl": pnl, "train": len(train_rows), "test": len(test_rows)}


def _print(title: str, result: dict | None) -> None:
    if result is None:
        print(f"{title} 样本不够", flush=True)
        return
    print(
        f"{title} 下单{result['orders']} 赢{result['wins']} "
        f"毛利{result['pnl'] / 10000:+.2f}万 训练{result['train']} 测试{result['test']}",
        flush=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    products = {}
    for product in train.CONTRACTS:
        print(f"读取{train.NAMES[product]}分钟", flush=True)
        px, oi, session = scan_product(product)
        products[product] = build_series(product, px, oi, session)
        print(f"  {train.NAMES[product]} {len(products[product]['dates'])} 个交易日", flush=True)
    overseas = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in train.OVERSEAS
    }
    domestic = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in train.DOMESTIC
    }
    news = train.news_scores()
    rows = build_rows(products, overseas, news)
    print(f"新样本 {len(rows)} 末天 {rows[-1]['day'] if rows else '-'}", flush=True)
    old_rows = [row for row in train.samples(domestic | overseas, news) if row["day"] >= KEEP]
    report = {"folder": str(OUT), "live_folder": str(train.MODELS), "splits": {}}
    saved = False
    for split in SPLITS:
        print(f"\n==== {split} 之后 ====", flush=True)
        report["splits"][split] = {}
        for key in train.CONTRACTS:
            names = features_for(key)
            base_names = tuple(BASE_OF[key])
            old = original_money(old_rows, domestic, key, split)
            base = fit_money(rows, key, base_names, split)
            minute = fit_money(rows, key, names, split)
            _print(f"{train.NAMES[key]} 现在的模型", old)
            _print(f"{train.NAMES[key]} 只换14:50", base)
            _print(f"{train.NAMES[key]} 14:50加分钟和持仓", minute)
            report["splits"][split][key] = {
                "live": None if old is None else {k: old[k] for k in ("orders", "wins", "pnl")},
                "close_1450": None if base is None else {k: base[k] for k in ("orders", "wins", "pnl")},
                "minute": None if minute is None else {k: minute[k] for k in ("orders", "wins", "pnl")},
            }
            if split == SPLITS[-1] and minute is not None:
                body = {
                    "code": key,
                    "name": train.NAMES[key],
                    "split": f"训练到 {split} 之前，不接入实盘",
                    "label": "同一张合约的14:50到下一个交易日14:50，有方向持有一天",
                    "feature_names": list(names),
                    "weight": minute["weight"],
                    "bias": minute["bias"],
                    "mean": minute["mean"],
                    "scale": minute["scale"],
                    "test_orders": minute["orders"],
                    "test_wins": minute["wins"],
                    "test_pnl": minute["pnl"],
                }
                (OUT / f"{key}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2))
                saved = True
    if saved:
        (OUT / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
