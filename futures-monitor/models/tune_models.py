#!/usr/bin/env python3
"""Pick training settings on two past years, then check last year once.

Dataset is all history. The training set for a test window is only the days
before that window. Settings are chosen on 2023-10-20→2024-10-20 and
2024-10-20→2025-10-20. Last year is reported after the choice, not used to pick.
"""

from __future__ import annotations

import sys

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from models import fit_sina_main as minute
from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
YUAN = 100
VEC = {"A": "x", "M": "x", "RM": "x_rm", "B": "x_b", "Y": "x_y"}
VAL = (("2023-10-20", "2024-10-20"), ("2024-10-20", "2025-10-20"))
TEST = ("2025-10-20", "2099-01-01")
LOGIT_GRID = [
    {"lr": lr, "steps": steps, "l2": l2}
    for lr in (0.05, 0.08)
    for steps in (300, 500)
    for l2 in (0.05, 0.2, 1.0)
]
TREE_GRID = [
    {"max_depth": depth, "num_leaves": 2 ** depth, "learning_rate": lr, "num_boost_round": rounds,
     "min_child_samples": leaf, "lambda_l2": l2}
    for depth in (2, 3)
    for lr in (0.05, 0.1)
    for rounds in (80, 200)
    for leaf in (30, 80)
    for l2 in (0.3, 3.0)
]


def fit_logit(xs, ys, ws, spec):
    dim = len(xs[0])
    weight = [0.0] * dim
    bias = 0.0
    lr = spec["lr"]
    total = sum(ws) or 1.0
    for _ in range(spec["steps"]):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y, wi in zip(xs, ys, ws):
            p = train._sigmoid(bias + sum(weight[j] * x[j] for j in range(dim)))
            err = (p - y) * wi
            for j in range(dim):
                grad_w[j] += err * x[j]
            grad_b += err
        for j in range(dim):
            weight[j] -= lr * (grad_w[j] / total + spec["l2"] * weight[j])
        bias -= lr * grad_b / total
    return weight, bias


def pnl_of(rows, probs, key, prices):
    total = 0.0
    orders = 0
    for row, prob in zip(rows, probs):
        if 0.45 < prob < 0.55:
            continue
        side = 1 if prob >= 0.55 else -1
        entry, exit_px = prices(row, key)
        if not entry or not exit_px:
            continue
        total += side * (exit_px - entry) * YUAN
        orders += 1
    return orders, total


def sina_prices(raw):
    def prices(row, key):
        exit_px = raw[key][row["day"]]
        entry = exit_px / (1.0 + row["move"][key] / 100.0)
        return entry, exit_px
    return prices


def minute_prices(row, key):
    return row["entry"].get(key), row["exit"].get(key)


def split_rows(rows, start, end):
    train_rows = [row for row in rows if row["day"] < start]
    test_rows = [row for row in rows if start <= row["day"] < end]
    return train_rows, test_rows


def matrix_of(rows, key, names):
    return [[row["named"][name] for name in names] if "named" in row else row[VEC[key]] for row in rows]


def search_family(label, rows_for, names_for, prices, kind):
    print(f"\n{label}", flush=True)
    chosen = {}
    for key in train.CONTRACTS:
        rows = rows_for(key)
        names = names_for(key)
        grid = LOGIT_GRID if kind == "logit" else TREE_GRID
        best = None
        for spec in grid:
            score = 0.0
            for start, end in VAL:
                train_rows, test_rows = split_rows(rows, start, end)
                if len(train_rows) < 100 or not test_rows:
                    continue
                probs = predict(train_rows, test_rows, names, spec, kind)
                _, total = pnl_of(test_rows, probs, key, prices)
                score += total
            if best is None or score > best[0]:
                best = (score, spec)
        _, test_rows = split_rows(rows, *TEST)
        train_rows, _ = split_rows(rows, TEST[0], TEST[1])
        probs = predict(train_rows, test_rows, names, best[1], kind)
        orders, last = pnl_of(test_rows, probs, key, prices)
        base_probs = predict(train_rows, test_rows, names, default_spec(kind), kind)
        _, base_last = pnl_of(test_rows, base_probs, key, prices)
        chosen[key] = {"spec": best[1], "val": best[0], "last": last, "base_last": base_last, "orders": orders}
        print(
            f"  {train.NAMES[key]} 挑选 {best[0]/10000:+.1f}万 去年 {last/10000:+.1f}万"
            f"（原参数 {base_last/10000:+.1f}万） {best[1]}",
            flush=True,
        )
    return chosen


def default_spec(kind):
    if kind == "logit":
        return {"lr": 0.08, "steps": 500, "l2": 0.2}
    return {"max_depth": 2, "num_leaves": 4, "learning_rate": 0.05, "num_boost_round": 100,
            "min_child_samples": 50, "lambda_l2": 1.0}


def predict(train_rows, test_rows, names, spec, kind):
    raw_x = [[row["named"][name] for name in names] if names else row[VEC[train.CONTRACTS[0]]] for row in train_rows]
    # names is None for sina vectors
    if names is None:
        key_vec = train_rows  # placeholder
    xs = vectors(train_rows, names)
    mean, scale = train.standardize(xs)
    scaled = [train.apply_scale(row, mean, scale) for row in xs]
    ys = [row["y_label"] for row in train_rows]
    ws = [row["w"] for row in train_rows]
    test_x = [train.apply_scale(row, mean, scale) for row in vectors(test_rows, names)]
    if kind == "logit":
        weight, bias = fit_logit(scaled, ys, ws, spec)
        return [train.predict_prob(row, weight, bias) for row in test_x]
    params = {k: spec[k] for k in ("max_depth", "num_leaves", "learning_rate", "min_child_samples", "lambda_l2")}
    params.update({"objective": "binary", "verbosity": -1, "seed": 0, "num_threads": 1})
    booster = lgb.train(params, lgb.Dataset(np.array(scaled), label=np.array(ys), weight=np.array(ws)),
                        num_boost_round=spec["num_boost_round"])
    return list(booster.predict(np.array(test_x)))


def vectors(rows, names):
    if names is None:
        return [row["x"] for row in rows]
    return [[row["named"][name] for name in names] for row in rows]


def pack_sina(rows, key):
    vec = VEC[key]
    packed = []
    for row in rows:
        if not row.get(vec) or key not in row["y"]:
            continue
        packed.append({"day": row["day"], "x": row[vec], "y_label": row["y"][key], "w": train.sample_weight(row["move"][key]),
                       "move": row["move"], "y": row["y"]})
    return packed


def pack_minute(rows, key, extra):
    names = minute_names(key, extra)
    packed = []
    for row in rows:
        if key not in row["y"] or any(name not in row["named"] for name in names):
            continue
        packed.append({**row, "y_label": row["y"][key], "w": train.sample_weight(row["move"][key]), "x": None})
    return packed, names


def minute_names(key, extra):
    return fit_sina_main_names(key, extra)


def fit_sina_main_names(key, extra):
    from models import fit_sina_main
    return fit_sina_main.names_for(key, extra)


def main() -> None:
    import fit_sina_main
    raw = {k: {d: c for d, c in train._cached_rows(train.series_path(k))} for k in (*train.DOMESTIC, *train.OVERSEAS)}
    sina = [row for row in train.samples(raw, train.news_scores()) if row["day"] >= "2017-01-01"]
    days = {product: fit_sina_main.load_days(product) for product in train.CONTRACTS}
    overseas = {k: raw[k] for k in train.OVERSEAS}
    minute_rows = fit_sina_main.build_rows(days, overseas, train.news_scores())
    print(f"新浪样本 {len(sina)} 分钟样本 {len(minute_rows)}", flush=True)

    families = {}
    families["新浪回归"] = search_family(
        "新浪回归", lambda key: pack_sina(sina, key), lambda key: None, sina_prices(raw), "logit")
    families["新浪树"] = search_family(
        "新浪树", lambda key: pack_sina(sina, key), lambda key: None, sina_prices(raw), "tree")
    for extra, title in ((False, "分钟只有收盘"), (True, "分钟加持仓成交")):
        families[title + "回归"] = search_family(
            title + "回归", lambda key, extra=extra: pack_minute(minute_rows, key, extra)[0],
            lambda key, extra=extra: fit_sina_main.names_for(key, extra), minute_prices, "logit")
        families[title + "树"] = search_family(
            title + "树", lambda key, extra=extra: pack_minute(minute_rows, key, extra)[0],
            lambda key, extra=extra: fit_sina_main.names_for(key, extra), minute_prices, "tree")
    (ROOT / "data" / "tune_choice.json").write_text(json.dumps(families, ensure_ascii=False, indent=2, default=str))
    print("已写入 data/tune_choice.json", flush=True)


def sina_prices(raw):
    def prices(row, key):
        exit_px = raw[key][row["day"]]
        return exit_px / (1.0 + row["move"][key] / 100.0), exit_px
    return prices


if __name__ == "__main__":
    main()
