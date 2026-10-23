#!/usr/bin/env python3
"""14:50 signal, 14:55 fill, same contract until the direction changes.

Logistic weights go to data/models_1450/. Depth-2 LightGBM goes to
data/models_tree_1450/. The saved fit uses every cached day. compare_1455.json
is the holdout, trained only before each split. data/models/ is untouched.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

from models import minute_model
from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOGIT_OUT = ROOT / "data" / "models_1450"
TREE_OUT = ROOT / "data" / "models_tree_1450"
SPLITS = ("2023-10-20", "2024-10-20", "2025-10-20")
YUAN = 100
TREE_PARAMS = {
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
ROUNDS = 100


TAPE = ROOT / "data" / "ctp_tape.jsonl"


def _tape_code(instrument: str) -> tuple[str, str] | None:
    inst = instrument.lower()
    if inst.startswith("rm") and inst[2:].isdigit():
        return "RM", "RM" + inst[2:]
    if inst[:1] in "abmy" and inst[1:].isdigit():
        return inst[0].upper(), inst.upper()
    return None


def fold_tape(
    px: dict[str, dict],
    oi: dict[str, dict],
    session: dict[str, dict],
    px1455: dict[str, dict],
) -> None:
    """Add days the minute files do not have, using the live tape.

    Exchange time is `update`. A morning replay of the night session is skipped.
    Tushare days stay as they are.
    """

    if not TAPE.exists():
        return
    day_aggs: dict[tuple[str, str, str], dict] = {}
    night_aggs: dict[tuple[str, str, str], dict] = {}
    fills: dict[tuple[str, str, str], float] = {}
    with TAPE.open() as handle:
        for line in handle:
            if "trading_day" not in line:
                continue
            row = json.loads(line)
            parsed = _tape_code(str(row.get("instrument") or ""))
            if parsed is None or parsed[0] not in px:
                continue
            product, code = parsed
            update = str(row.get("update") or "")
            at = str(row.get("at") or "")
            trading_day = str(row.get("trading_day") or "")
            if len(update) < 8 or len(at) < 16 or len(trading_day) != 8:
                continue
            clock, minute = update[:8], update[:5]
            at_date, at_clock = at[:10], at[11:19]
            day = f"{trading_day[:4]}-{trading_day[4:6]}-{trading_day[6:8]}"
            try:
                price = float(row["last"])
                interest = float(row.get("oi") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue
            if clock >= "21:00:00" and at_clock < "21:00:00":
                continue
            key = (product, code, day)
            if "09:00:00" <= clock < "15:00:00" and at_date == day:
                minute_model._feed(day_aggs.setdefault(key, minute_model._agg()), price, price, 1.0, interest, minute)
                if "14:55:00" <= clock < "14:56:00":
                    fills[key] = price
            elif clock >= "21:00:00" and at_date < day:
                minute_model._feed(night_aggs.setdefault(key, minute_model._agg()), price, price, 1.0, interest, minute)
    for (product, code, day), agg in day_aggs.items():
        if not agg["px1450"] or not agg["oi1450"]:
            continue
        book = px[product].setdefault(code, {})
        if day in book:
            continue
        night = night_aggs.get((product, code, day))
        num = agg["att_num"] + (night["att_num"] if night else 0.0)
        den = agg["att_den"] + (night["att_den"] if night else 0.0)
        book[day] = agg["px1450"]
        oi[product].setdefault(code, {})[day] = agg["oi1450"]
        session[product].setdefault(code, {})[day] = {
            "oi": agg["oi1450"],
            "vol": agg["vol"] + (night["vol"] if night else 0.0),
            "night": minute_model._pct(night["open"] if night else None, night["prev_px"] if night else None),
            "day": minute_model._pct(agg["open"], agg["px1450"]),
            "att": minute_model._attitude(num, den),
        }
    for (product, code, day), price in fills.items():
        px1455[product].setdefault(code, {}).setdefault(day, price)


def scan_1455(product: str) -> dict[str, dict[str, float]]:
    book: dict[str, dict[str, float]] = {}
    for path in sorted((minute_model.MIN_DIR / product).glob("*.csv")):
        days: dict[str, float] = {}
        with path.open() as handle:
            next(handle, None)
            for line in handle:
                if len(line) < 20 or line[11:16] != "14:55":
                    continue
                parts = line.split(",")
                try:
                    close = float(parts[4])
                except (IndexError, ValueError):
                    continue
                if close > 0:
                    days[line[:10]] = close
        if days:
            book[path.stem] = days
    return book


def attach_1455(rows: list[dict], px1455: dict[str, dict[str, dict[str, float]]]) -> None:
    """Replace the 14:50 label with the same contract's 14:55 to next 14:55."""

    for row in rows:
        row["code"] = {}
        for key in list(row["y"]):
            code = row["code_1450"].get(key)
            entry = px1455[key].get(code, {}).get(row["signal"]) if code else None
            exit_px = px1455[key].get(code, {}).get(row["day"]) if code else None
            if not entry or exit_px is None:
                row["y"].pop(key, None)
                row["move"].pop(key, None)
                row["entry"].pop(key, None)
                row["exit"].pop(key, None)
                continue
            row["code"][key] = code
            row["entry"][key] = entry
            row["exit"][key] = exit_px
            row["y"][key] = 1 if exit_px > entry else 0
            row["move"][key] = (exit_px - entry) / entry * 100.0


def remember_contract(rows: list[dict], products: dict) -> None:
    for row in rows:
        row["code_1450"] = {
            key: products[key]["main"].get(row["signal"]) for key in products
        }


def usable(rows: list[dict], key: str) -> list[dict]:
    names = minute_model.features_for(key)
    return [
        row for row in rows
        if key in row["y"] and key in row.get("code", {}) and all(name in row["named"] for name in names)
    ]


def fit_logit(rows: list[dict], key: str):
    names = minute_model.features_for(key)
    mean, scale = train.standardize([[row["named"][name] for name in names] for row in rows])
    xs = [train.apply_scale([row["named"][name] for name in names], mean, scale) for row in rows]
    ys = [row["y"][key] for row in rows]
    ws = [train.sample_weight(row["move"][key]) for row in rows]
    weight, bias = train.fit(xs, ys, ws)
    return weight, bias, mean, scale


def fit_tree(rows: list[dict], key: str, mean: list[float], scale: list[float]):
    names = minute_model.features_for(key)
    xs = [train.apply_scale([row["named"][name] for name in names], mean, scale) for row in rows]
    ys = [row["y"][key] for row in rows]
    ws = [train.sample_weight(row["move"][key]) for row in rows]
    return lgb.train(
        TREE_PARAMS,
        lgb.Dataset(np.array(xs), label=np.array(ys), weight=np.array(ws)),
        num_boost_round=ROUNDS,
    )


def probs_of(rows: list[dict], key: str, kind: str, model, mean, scale) -> list[float]:
    names = minute_model.features_for(key)
    out = []
    for row in rows:
        vector = train.apply_scale([row["named"][name] for name in names], mean, scale)
        if kind == "logit":
            weight, bias = model
            out.append(train.predict_prob(vector, weight, bias))
        else:
            out.append(float(model.predict([vector])[0]))
    return out


def hold_money(rows: list[dict], key: str, probs: list[float], px1455: dict[str, dict[str, float]]) -> dict:
    """Same contract from its 14:55 until the direction is gone or reversed."""

    pnl = 0.0
    closes = wins = holds = missed = 0
    position = None
    for row, prob in zip(rows, probs):
        side = 1 if prob >= 0.55 else -1 if prob <= 0.45 else 0
        if position is None:
            if side == 0:
                continue
            position = {"side": side, "code": row["code"][key], "entry": row["entry"][key]}
            continue
        if side == position["side"]:
            holds += 1
            continue
        exit_px = px1455.get(position["code"], {}).get(row["signal"])
        if exit_px is None:
            missed += 1
            continue
        this = position["side"] * (exit_px - position["entry"]) * YUAN
        pnl += this
        closes += 1
        wins += this > 0
        position = None if side == 0 else {"side": side, "code": row["code"][key], "entry": row["entry"][key]}
    if position is not None and rows:
        exit_px = px1455.get(position["code"], {}).get(rows[-1]["day"])
        if exit_px is not None:
            this = position["side"] * (exit_px - position["entry"]) * YUAN
            pnl += this
            closes += 1
            wins += this > 0
        else:
            missed += 1
    return {"closes": closes, "wins": wins, "holds": holds, "missed": missed, "pnl": pnl}


def save(folder: Path, key: str, kind: str, mean, scale, model) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    body = {
        "code": key,
        "name": train.NAMES[key],
        "kind": kind,
        "trained_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M"),
        "price": "14:55",
        "rule": "14:50 给方向，14:55 按当天主力开仓。下一天同向继续拿着同一张合约；没有方向就在 14:55 平仓；方向反了就平掉并按新主力开仓。",
        "feature_names": list(minute_model.features_for(key)),
        "mean": mean,
        "scale": scale,
        "direction_up": 0.55,
        "direction_down": 0.45,
    }
    if kind == "logit":
        weight, bias = model
        body["weight"] = weight
        body["bias"] = bias
    else:
        model.save_model(str(folder / f"{key}.txt"))
        body["model"] = f"{key}.txt"
    (folder / f"{key}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2))


def load_rows() -> tuple[dict, dict, list[dict]]:
    px: dict[str, dict] = {}
    oi: dict[str, dict] = {}
    session: dict[str, dict] = {}
    px1455: dict[str, dict] = {}
    for product in train.CONTRACTS:
        print(f"读取{train.NAMES[product]}", flush=True)
        px[product], oi[product], session[product] = minute_model.scan_product(product)
        px1455[product] = scan_1455(product)
    fold_tape(px, oi, session, px1455)
    products = {
        product: minute_model.build_series(product, px[product], oi[product], session[product])
        for product in train.CONTRACTS
    }
    overseas = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in train.OVERSEAS
    }
    rows = minute_model.build_rows(products, overseas, train.news_scores())
    remember_contract(rows, products)
    attach_1455(rows, px1455)
    print(f"样本 {len(rows)} 末天 {max(row['day'] for row in rows)}", flush=True)
    return products, px1455, rows


def refit() -> None:
    """Daily fit of both minute models. No holdout, no mail."""

    _products, _px, rows = load_rows()
    for key in train.CONTRACTS:
        got = usable(rows, key)
        weight, bias, mean, scale = fit_logit(got, key)
        booster = fit_tree(got, key, mean, scale)
        save(LOGIT_OUT, key, "logit", mean, scale, (weight, bias))
        save(TREE_OUT, key, "tree", mean, scale, booster)
        print(f"写入 {train.NAMES[key]} {len(got)} 天", flush=True)
    print("分钟模型已按当天收盘重新训练", flush=True)


def main() -> None:
    products, px1455, rows = load_rows()
    del products
    report = {"price": "14:55", "same_contract": True, "logit": {}, "tree": {}}
    for key in train.CONTRACTS:
        got = usable(rows, key)
        weight, bias, mean, scale = fit_logit(got, key)
        booster = fit_tree(got, key, mean, scale)
        save(LOGIT_OUT, key, "logit", mean, scale, (weight, bias))
        save(TREE_OUT, key, "tree", mean, scale, booster)
        print(f"写入 {train.NAMES[key]} {len(got)} 天", flush=True)
        for split in SPLITS:
            train_rows = [row for row in got if row["day"] < split]
            test_rows = [row for row in got if row["day"] >= split]
            logit_w, logit_b, logit_mean, logit_scale = fit_logit(train_rows, key)
            tree = fit_tree(train_rows, key, logit_mean, logit_scale)
            logit_pnl = hold_money(
                test_rows, key, probs_of(test_rows, key, "logit", (logit_w, logit_b), logit_mean, logit_scale), px1455[key]
            )
            tree_pnl = hold_money(
                test_rows, key, probs_of(test_rows, key, "tree", tree, logit_mean, logit_scale), px1455[key]
            )
            report["logit"].setdefault(split, {})[key] = logit_pnl
            report["tree"].setdefault(split, {})[key] = tree_pnl
            print(
                f"  {split} 回归 {logit_pnl['closes']}笔 赢{logit_pnl['wins']} {logit_pnl['pnl'] / 10000:+.2f}万"
                f"  树 {tree_pnl['closes']}笔 赢{tree_pnl['wins']} {tree_pnl['pnl'] / 10000:+.2f}万"
                f"  缺价 {logit_pnl['missed']}/{tree_pnl['missed']}",
                flush=True,
            )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (LOGIT_OUT / "compare_1455.json").write_text(text)
    (TREE_OUT / "compare_1455.json").write_text(text)
    stale = LOGIT_OUT / "compare.json"
    if stale.exists():
        stale.unlink()
    print("已写入", LOGIT_OUT, TREE_OUT, flush=True)


if __name__ == "__main__":
    if "--refit" in sys.argv:
        refit()
    else:
        main()
