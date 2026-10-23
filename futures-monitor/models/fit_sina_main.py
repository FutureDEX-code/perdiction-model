#!/usr/bin/env python3
"""Refit the two 14:50 models on the Sina-matched continuous minutes.

14:50 is the close. 14:55 is the fill. A missing bar uses the next minute.
Extra inputs are the night move, the day move, open-interest change, volume
change, and the price/OI attitude. Live data/models/ is not written.
"""

from __future__ import annotations

import csv
import json
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAIN = train.MAIN_MINUTE
TAPE = ROOT / "data" / "ctp_tape.jsonl"
LOGIT_OUT = ROOT / "data" / "models_1450"
TREE_OUT = ROOT / "data" / "models_tree_1450"
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
TAPE_CODE = {"豆一2701": "A", "豆二2701": "B", "豆粕2701": "M", "菜粕2701": "RM", "豆油2701": "Y"}


def _at_or_after(bars: list[tuple], start: str) -> tuple | None:
    for bar in bars:
        if start <= bar[0] <= "15:00":
            return bar
    return None


def load_days(product: str) -> dict[str, dict]:
    nights: dict[str, list] = {}
    days: dict[str, list] = {}
    with (MAIN / f"{product}.csv").open() as handle:
        for row in csv.DictReader(handle):
            date, clock = row["trade_time"][:10], row["trade_time"][11:16]
            bar = (
                clock,
                float(row["open"]),
                float(row["close"]),
                float(row["vol"] or 0),
                float(row["oi"] or 0),
                row["contract"],
            )
            if clock >= "21:00":
                nights.setdefault(date, []).append(bar)
            elif "09:00" <= clock <= "15:00":
                days.setdefault(date, []).append(bar)
    trading = sorted(days)
    attached: dict[str, list] = {date: [] for date in trading}
    for calendar, bars in nights.items():
        index = bisect_right(trading, calendar)
        if index >= len(trading):
            continue
        target = trading[index]
        gap = (datetime.fromisoformat(target) - datetime.fromisoformat(calendar)).days
        if 0 < gap <= 15:
            attached[target].extend(bars)
    out = {}
    for date in trading:
        day_bars = days[date]
        close = _at_or_after(day_bars, "14:50")
        entry = _at_or_after(day_bars, "14:55")
        if close is None or entry is None:
            continue
        night = attached[date]
        path = [bar for bar in night if bar[3] > 0] + [bar for bar in day_bars if bar[0] <= close[0] and bar[3] > 0]
        num = den = 0.0
        prev = None
        for bar in path:
            if prev is not None:
                change = bar[2] - prev[0]
                oi_change = bar[4] - prev[1]
                if change != 0 and oi_change != 0:
                    num += (1.0 if change > 0 else -1.0) * oi_change
                    den += abs(oi_change)
            prev = (bar[2], bar[4])
        night_ret = 0.0
        traded_night = [bar for bar in night if bar[3] > 0]
        if len(traded_night) >= 2 and traded_night[0][1]:
            night_ret = (traded_night[-1][2] - traded_night[0][1]) / traded_night[0][1] * 100.0
        day_ret = (close[2] - day_bars[0][1]) / day_bars[0][1] * 100.0 if day_bars[0][1] else 0.0
        vol = sum(bar[3] for bar in night) + sum(bar[3] for bar in day_bars if bar[0] <= close[0])
        out[date] = {
            "close": close[2],
            "entry": entry[2],
            "oi": close[4],
            "vol": vol,
            "night": night_ret,
            "day": day_ret,
            "att": num / den if den else 0.0,
            "contract": close[5],
        }
    dates = sorted(out)
    for prev, date in zip(dates, dates[1:]):
        out[date]["oi_chg"] = (out[date]["oi"] - out[prev]["oi"]) / out[prev]["oi"] * 100.0 if out[prev]["oi"] else 0.0
        out[date]["vol_chg"] = (out[date]["vol"] - out[prev]["vol"]) / out[prev]["vol"] * 100.0 if out[prev]["vol"] else 0.0
    if dates:
        out[dates[0]]["oi_chg"] = 0.0
        out[dates[0]]["vol_chg"] = 0.0
    return out


def extra_names(key: str) -> tuple[str, ...]:
    names = []
    for item in WATCH[key]:
        name = train.NAMES[item]
        names.extend((f"{name}夜盘", f"{name}日盘", f"{name}持仓变化", f"{name}成交量变化", f"{name}价仓"))
    return tuple(names)


def minute_names(key: str, family: str) -> tuple[str, ...]:
    """Base minute columns plus the side dish the holdout picked for this product."""

    return train._variant_names(names_for(key, True), key, train.variant_for(key, family))


def build_rows(days: dict[str, dict], overseas: dict, news: dict) -> list[dict]:
    closes = {key: {day: slot["close"] for day, slot in book.items()} for key, book in days.items()}
    domestic = {key: train.returns(closes[key]) for key in days}
    swings = {
        key: dict(zip(sorted(closes[key]), train.swing_path([closes[key][day] for day in sorted(closes[key])])))
        for key in days
    }
    common = sorted(set(swings["A"]) & set(swings["M"]) & set(swings["RM"]))
    common = [day for day in common if day >= "2017-01-01"]
    overseas_days = {key: sorted(overseas[key]) for key in train.OVERSEAS}
    cot = train.cot_series()
    seats = train.seat_series()
    alpha = 2 / (3 + 1)
    ema = 0.0
    rows = []
    for prev, day in zip(common, common[1:]):
        if any(train.latest_on_or_before(overseas_days[key], prev) is None for key in train.OVERSEAS):
            continue
        if any(prev not in domestic[key] or prev not in swings[key] for key in ("A", "M", "RM")):
            continue
        raw_news = train.news_before(news, prev, day)
        ema = alpha * raw_news + (1 - alpha) * ema
        named = {
            "豆一昨涨跌": domestic["A"][prev],
            "豆粕昨涨跌": domestic["M"][prev],
            "菜粕昨涨跌": domestic["RM"][prev],
            "美豆隔夜": train.overseas_session(overseas["S"], prev, day),
            "美豆粕隔夜": train.overseas_session(overseas["SM"], prev, day),
            "美豆油隔夜": train.overseas_session(overseas["BO"], prev, day),
            "豆一这段": swings["A"][prev],
            "豆粕这段": swings["M"][prev],
            "菜粕这段": swings["RM"][prev],
            "新闻倾向": raw_news,
            "新闻三日": ema,
        }
        named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
        for key in ("B", "Y"):
            if prev in domestic.get(key, {}) and prev in swings.get(key, {}):
                named[f"{train.NAMES[key]}昨涨跌"] = domestic[key][prev]
                named[f"{train.NAMES[key]}这段"] = swings[key][prev]
        for key in days:
            named.update(train.extra_named(key, prev))
        named.update(train.cot_named(cot, prev))
        named.update(train.mood_named(cot, seats, prev))
        ok = True
        for key in days:
            slot = days[key].get(prev)
            if slot is None:
                ok = False
                break
            name = train.NAMES[key]
            named[f"{name}夜盘"] = slot["night"]
            named[f"{name}日盘"] = slot["day"]
            named[f"{name}持仓变化"] = slot["oi_chg"]
            named[f"{name}成交量变化"] = slot["vol_chg"]
            named[f"{name}价仓"] = slot["att"]
        if not ok:
            continue
        row = {"day": day, "signal": prev, "named": named, "y": {}, "move": {}, "entry": {}, "exit": {}}
        for key in days:
            if prev not in days[key] or day not in days[key]:
                continue
            entry = days[key][prev]["entry"]
            exit_px = days[key][day]["entry"]
            if not entry or not exit_px:
                continue
            row["y"][key] = 1 if exit_px > entry else 0
            row["move"][key] = (exit_px - entry) / entry * 100.0
            row["entry"][key] = entry
            row["exit"][key] = exit_px
        rows.append(row)
    return rows


def names_for(key: str, extra: bool) -> tuple[str, ...]:
    base = tuple(BASE[key])
    return base + extra_names(key) if extra else base


def usable(rows: list[dict], key: str, extra: bool) -> list[dict]:
    names = names_for(key, extra)
    return [row for row in rows if key in row["y"] and all(name in row["named"] for name in names)]


def names_for(key: str, extra: bool) -> tuple[str, ...]:
    base = tuple(BASE[key])
    return base + extra_names(key) if extra else base


def hold(rows: list[dict], key: str, names, mean, scale, weight, bias, booster, kind: str) -> tuple[int, float]:
    pnl = 0.0
    closes = 0
    pos = None
    for row in rows:
        vector = train.apply_scale([row["named"][name] for name in names], mean, scale)
        prob = train.predict_prob(vector, weight, bias) if kind == "logit" else float(booster.predict([vector])[0])
        side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
        entry = row["entry"].get(key)
        if pos is None:
            if side == 0 or entry is None:
                continue
            pos = (side, entry)
            continue
        if side == 0 or side == pos[0] or entry is None:
            continue
        pnl += pos[0] * (entry - pos[1]) * YUAN
        closes += 1
        pos = (side, entry)
    if pos is not None and rows:
        last = rows[-1]["exit"].get(key)
        if last is not None and last != pos[1]:
            pnl += pos[0] * (last - pos[1]) * YUAN
            closes += 1
    return closes, pnl


def _stored_spec(folder: Path, key: str) -> dict:
    path = folder / f"{key}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("train_params") or {}


def fit_pair(rows: list[dict], key: str, extra: bool):
    """Each family takes the side dish the holdout picked for this product."""

    if extra:
        tree_names = tuple(minute_names(key, "分钟树"))
        logit_names = tuple(minute_names(key, "分钟回归"))
    else:
        tree_names = tuple(names_for(key, extra))
        logit_names = tree_names
    ys = [row["y"][key] for row in rows]
    ws = [train.sample_weight(row["move"][key]) for row in rows]
    logit_spec = _stored_spec(LOGIT_OUT, key)
    tree_spec = _stored_spec(TREE_OUT, key)
    matrix_l = [[row["named"][name] for name in logit_names] for row in rows]
    mean_l, scale_l = train.standardize(matrix_l)
    xs_l = [train.apply_scale(row, mean_l, scale_l) for row in matrix_l]
    weight, bias = train.fit(
        xs_l, ys, ws,
        lr=float(logit_spec.get("lr", 0.08)),
        steps=int(logit_spec.get("steps", 500)),
        l2=float(logit_spec.get("l2", 0.2)),
    )
    matrix_t = [[row["named"][name] for name in tree_names] for row in rows]
    mean_t, scale_t = train.standardize(matrix_t)
    xs_t = [train.apply_scale(row, mean_t, scale_t) for row in matrix_t]
    params = dict(PARAMS)
    params["max_depth"] = int(tree_spec.get("max_depth", params["max_depth"]))
    params["num_leaves"] = int(tree_spec.get("num_leaves", 2 ** params["max_depth"]))
    params["learning_rate"] = float(tree_spec.get("learning_rate", params["learning_rate"]))
    params["min_child_samples"] = int(tree_spec.get("min_child_samples", params["min_child_samples"]))
    params["lambda_l2"] = float(tree_spec.get("lambda_l2", params["lambda_l2"]))
    rounds = int(tree_spec.get("num_boost_round", 100))
    booster = lgb.train(params, lgb.Dataset(np.array(xs_t), label=np.array(ys), weight=np.array(ws)), num_boost_round=rounds)
    names = {"logit": logit_names, "tree": tree_names}
    mean = {"logit": mean_l, "tree": mean_t}
    scale = {"logit": scale_l, "tree": scale_t}
    return names, mean, scale, weight, bias, booster


def hold(rows: list[dict], key: str, names, mean, scale, weight, bias, booster, kind: str) -> tuple[int, float]:
    pnl = 0.0
    closes = 0
    pos = None
    for row in rows:
        vector = train.apply_scale([row["named"][name] for name in names], mean, scale)
        prob = train.predict_prob(vector, weight, bias) if kind == "logit" else float(booster.predict([vector])[0])
        side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
        entry = row["entry"].get(key)
        if pos is None:
            if side == 0 or entry is None:
                continue
            pos = (side, entry)
            continue
        if side == 0 or side == pos[0] or entry is None:
            continue
        this = pos[0] * (entry - pos[1]) * YUAN
        pnl += this
        closes += 1
        pos = (side, entry)
    if pos is not None and rows:
        last = rows[-1]["exit"].get(key)
        if last is not None and last != pos[1]:
            pnl += pos[0] * (last - pos[1]) * YUAN
            closes += 1
    return closes, pnl


def save(key: str, names, mean, scale, weight, bias, booster) -> None:
    trained_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    rule = "主连分钟。14:50 那根为收盘，缺了往后取。14:55 那根为成交价，缺了往后取。同向或没方向继续持有，只有反向才平仓。"
    body = {
        "code": key,
        "name": train.NAMES[key],
        "trained_at": trained_at,
        "source": "training_dataset/main_minute",
        "price_close": "14:50",
        "price_fill": "14:55",
        "rule": rule,
        "direction_up": 0.55,
        "direction_down": 0.45,
    }
    LOGIT_OUT.mkdir(parents=True, exist_ok=True)
    TREE_OUT.mkdir(parents=True, exist_ok=True)
    logit = {
        **body,
        "kind": "logit",
        "feature_names": list(names["logit"]),
        "mean": mean["logit"],
        "scale": scale["logit"],
        "weight": weight,
        "bias": bias,
        "train_params": _stored_spec(LOGIT_OUT, key),
    }
    (LOGIT_OUT / f"{key}.json").write_text(json.dumps(logit, ensure_ascii=False, indent=2))
    booster.save_model(str(TREE_OUT / f"{key}.txt"))
    tree = {
        **body,
        "kind": "tree",
        "feature_names": list(names["tree"]),
        "mean": mean["tree"],
        "scale": scale["tree"],
        "model": f"{key}.txt",
        "train_params": _stored_spec(TREE_OUT, key),
    }
    (TREE_OUT / f"{key}.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2))


def sina_baseline(raw, news, days, split: str) -> dict[str, tuple[int, float]]:
    """Daily Sina model, same 14:55 hold, so the extra minute columns are the only change."""

    rows = [row for row in train.samples(raw, news) if row["day"] >= "2017-01-01"]
    common = sorted(set(raw["A"]) & set(raw["M"]) & set(raw["RM"]))
    prev_of = {day: prev for prev, day in zip(common, common[1:])}
    out = {}
    for key in train.CONTRACTS:
        vec = VEC[key]
        usable = []
        for row in rows:
            signal = prev_of.get(row["day"])
            if signal is None or not row.get(vec) or key not in row["y"] or signal not in days[key] or row["day"] not in days[key]:
                continue
            usable.append((row, days[key][signal]["entry"], days[key][row["day"]]["entry"]))
        train_rows = [item for item in usable if item[0]["day"] < split]
        test_rows = [item for item in usable if item[0]["day"] >= split]
        xs = [item[0][vec] for item in train_rows]
        mean, scale = train.standardize(xs)
        scaled = [train.apply_scale(row, mean, scale) for row in xs]
        ys = [item[0]["y"][key] for item in train_rows]
        ws = [train.sample_weight(item[0]["move"][key]) for item in train_rows]
        weight, bias = train.fit(scaled, ys, ws)
        booster = lgb.train(PARAMS, lgb.Dataset(np.array(scaled), label=np.array(ys), weight=np.array(ws)), num_boost_round=100)
        tree_p = booster.predict(np.array([train.apply_scale(item[0][vec], mean, scale) for item in test_rows]))
        for kind, probs in (
            ("回归", [train.predict_prob(train.apply_scale(item[0][vec], mean, scale), weight, bias) for item in test_rows]),
            ("树", tree_p),
        ):
            pnl = 0.0
            closes = 0
            pos = None
            for item, prob in zip(test_rows, probs):
                side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
                entry = item[1]
                if pos is None:
                    if side:
                        pos = (side, entry)
                    continue
                if side == 0 or side == pos[0]:
                    continue
                pnl += pos[0] * (entry - pos[1]) * YUAN
                closes += 1
                pos = (side, entry)
            out[f"{kind}:{key}"] = (closes, pnl)
    return out


def main() -> None:
    days = {product: load_days(product) for product in train.CONTRACTS}
    overseas = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in train.OVERSEAS
    }
    news = train.news_scores()
    rows = build_rows(days, overseas, news)
    print(f"样本 {len(rows)} 末天 {rows[-1]['day']}", flush=True)
    if "--refit" in sys.argv:
        for key in train.CONTRACTS:
            got = usable(rows, key, True)
            names, mean, scale, weight, bias, booster = fit_pair(got, key, True)
            save(key, names, mean, scale, weight, bias, booster)
            print(f"写入 {train.NAMES[key]} {len(got)} 天", flush=True)
        print("主连分钟模型已重新训练", flush=True)
        return
    report = {}
    for key in train.CONTRACTS:
        got = usable(rows, key, True)
        names, mean, scale, weight, bias, booster = fit_pair(got, key, True)
        save(key, names, mean, scale, weight, bias, booster)
        print(f"写入 {train.NAMES[key]} {len(got)} 天", flush=True)
        for split in SPLITS:
            train_rows = [row for row in got if row["day"] < split]
            test_rows = [row for row in got if row["day"] >= split]
            full = fit_pair(usable(train_rows, key, True), key, True)
            base = fit_pair(usable(train_rows, key, False), key, False)
            full_logit = hold(test_rows, key, full[0]["logit"], full[1]["logit"], full[2]["logit"], full[3], full[4], full[5], "logit")
            full_tree = hold(test_rows, key, full[0]["tree"], full[1]["tree"], full[2]["tree"], full[3], full[4], full[5], "tree")
            base_logit = hold(test_rows, key, base[0]["logit"], base[1]["logit"], base[2]["logit"], base[3], base[4], base[5], "logit")
            base_tree = hold(test_rows, key, base[0]["tree"], base[1]["tree"], base[2]["tree"], base[3], base[4], base[5], "tree")
            report.setdefault(split, {})[key] = {
                "回归14:50": base_logit[1],
                "回归加分钟": full_logit[1],
                "树14:50": base_tree[1],
                "树加分钟": full_tree[1],
            }
            print(
                f"  {split} 回归 {base_logit[1]/10000:+.2f}→加分钟 {full_logit[1]/10000:+.2f}"
                f"  树 {base_tree[1]/10000:+.2f}→加分钟 {full_tree[1]/10000:+.2f}",
                flush=True,
            )
    if "--refit" not in sys.argv:
        raw = {
            key: {day: close for day, close in train._cached_rows(train.series_path(key))}
            for key in (*train.DOMESTIC, *train.OVERSEAS)
        }
        print("\n对照新浪日线，同一套 14:55 持有规则", flush=True)
        for split in SPLITS:
            base = sina_baseline(raw, news, days, split)
            total = sum(pnl for _, pnl in base.values())
            print(f"  {split} 新浪回归+树合计 {total/10000:+.2f}万", flush=True)
    (LOGIT_OUT / "compare_sina_main.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("已写入", LOGIT_OUT, TREE_OUT, flush=True)


if __name__ == "__main__":
    main()
