#!/usr/bin/env python3
"""Pick one ensemble rule per product from the four chosen models.

Reads <product>/主配菜对比.json, refits those four menus on 2022-2025, and
scores 多数票2/3, 平均, and each single model on the 2026 holdout. Writes
<product>/综合对比.json with the best rule.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN_FROM = "2022-01-01"
TRAIN_TO = "2025-12-31"
YUAN = 100

PRODUCTS = {
    "A": ("豆一", ["豆一昨涨跌", "豆一这段", "美豆隔夜", "美豆粕隔夜", "美豆油隔夜"],
          ["豆一夜盘", "豆一日盘", "豆一价仓"], ["豆一振幅", "豆一位置", "豆一量变化", "豆一仓变化"],
          ["豆二这段", "豆二仓变化"], ["美豆基金净占比", "美豆基金占比变动", "美豆粕基金净占比", "美豆粕基金占比变动", "美豆油基金净占比", "美豆油基金占比变动"], []),
    "B": ("豆二", ["豆二昨涨跌", "豆二这段", "美豆隔夜", "美豆粕隔夜", "美豆油隔夜"],
          ["豆二夜盘", "豆二日盘", "豆二价仓"], ["豆二振幅", "豆二位置", "豆二量变化", "豆二仓变化"],
          ["豆粕这段", "豆粕仓变化"], ["美豆基金净占比", "美豆基金占比变动", "美豆粕基金净占比", "美豆粕基金占比变动", "美豆油基金净占比", "美豆油基金占比变动"], []),
    "M": ("豆粕", ["豆粕昨涨跌", "豆粕这段", "美豆隔夜", "美豆粕隔夜", "美豆油隔夜"],
          ["豆粕夜盘", "豆粕日盘", "豆粕价仓"], ["豆粕振幅", "豆粕位置", "豆粕量变化", "豆粕仓变化"],
          ["豆粕菜粕价差这段", "菜粕仓变化"], ["美豆基金净占比", "美豆基金占比变动", "美豆粕基金净占比", "美豆粕基金占比变动", "美豆油基金净占比", "美豆油基金占比变动"], []),
    "Y": ("豆油", ["豆油昨涨跌", "豆油这段", "美豆隔夜", "美豆粕隔夜", "美豆油隔夜"],
          ["豆油夜盘", "豆油日盘", "豆油价仓"], ["豆油振幅", "豆油位置", "豆油量变化", "豆油仓变化"],
          ["豆棕价差这段", "菜油价差这段", "棕榈油仓变化"], ["美豆基金净占比", "美豆基金占比变动", "美豆粕基金净占比", "美豆粕基金占比变动", "美豆油基金净占比", "美豆油基金占比变动"], []),
    "RM": ("菜粕", ["菜粕昨涨跌", "菜粕这段"],
           ["菜粕夜盘", "菜粕日盘", "菜粕价仓"], ["菜粕振幅", "菜粕位置", "菜粕量变化", "菜粕仓变化"],
           ["豆粕这段", "菜粕豆粕价差这段", "豆粕仓变化"], ["油菜籽基金净占比", "油菜籽基金占比变动"], ["菜粕席位净占比", "菜粕席位净变动"]),
    "P": ("棕榈油", ["棕榈油昨涨跌", "棕榈油这段", "马棕油隔夜"],
          ["棕榈油夜盘", "棕榈油日盘", "棕榈油价仓"], ["棕榈油振幅", "棕榈油位置", "棕榈油量变化", "棕榈油仓变化"],
          ["豆油这段", "豆棕价差这段", "豆油仓变化"], [], []),
    "OI": ("菜籽油", ["菜籽油昨涨跌", "菜籽油这段"],
           ["菜籽油夜盘", "菜籽油日盘", "菜籽油价仓"], ["菜籽油振幅", "菜籽油位置", "菜籽油量变化", "菜籽油仓变化"],
           ["豆油这段", "菜油价差这段", "豆油仓变化"], ["油菜籽基金净占比", "油菜籽基金占比变动"], ["菜籽油席位净占比", "菜籽油席位净变动"]),
    "PK": ("花生", ["花生昨涨跌", "花生这段"],
           ["花生夜盘", "花生日盘", "花生价仓"], ["花生振幅", "花生位置", "花生量变化", "花生仓变化"],
           ["豆粕这段", "豆油这段", "豆粕仓变化"], [], ["花生席位净占比", "花生席位净变动"]),
}
MODELS = ("日线逻辑回归", "日线树", "分钟逻辑回归", "分钟树")


def menu_cols(product: str, menu: str, family: str) -> list[str]:
    _name, main_day, min_extra, side1, ref, funds, seats = PRODUCTS[product]
    main_min = list(main_day) + min_extra
    side2_day = ref + funds + seats
    side2_min = ref + seats
    if menu == "主菜":
        return main_day if family.startswith("日线") else main_min
    if menu == "主菜+配菜一":
        return (main_day + side1) if family.startswith("日线") else (main_min + side1)
    if menu == "主菜+配菜二":
        return (main_day + side2_day) if family.startswith("日线") else (main_min + side2_min)
    return (main_day + side1 + side2_day) if family.startswith("日线") else (main_min + side1 + side2_min)


def sgn(p: float) -> int:
    return 1 if p >= 0.55 else -1 if p <= 0.45 else 0


def score(days, signals, labels):
    pos = 0
    pnl = 0.0
    directed = hits = 0
    for day in days:
        lab = labels[day]
        entry, exit_px = float(lab["入场价"]), float(lab["出场价"])
        signal = signals[day]
        up = exit_px > entry
        if signal != 0:
            directed += 1
            hits += int((signal == 1 and up) or (signal == -1 and not up))
        held = pos if signal == 0 else signal
        pnl += held * (exit_px - entry) * YUAN
        pos = held
    return directed, hits, pnl


def predict(rows, cols, spec, kind, product):
    tr = [r for r in rows if TRAIN_FROM <= r["日期"] <= TRAIN_TO]
    te = [r for r in rows if r["日期"] > TRAIN_TO]
    if not te:
        return {}
    matrix = [[float(r[n]) for n in cols] for r in tr]
    mean, scale = train.standardize(matrix)
    xs = [train.apply_scale(row, mean, scale) for row in matrix]
    ys = [r["y"] for r in tr]
    ws = [train.sample_weight(r["move"]) for r in tr]
    if kind == "logit":
        w, b = train.fit(xs, ys, ws, lr=spec["lr"], steps=spec["steps"], l2=spec["l2"])
        return {r["日期"]: train.predict_prob(train.apply_scale([float(r[n]) for n in cols], mean, scale), w, b) for r in te}
    params = {k: spec[k] for k in ("max_depth", "num_leaves", "learning_rate", "min_child_samples", "lambda_l2")}
    params.update({"objective": "binary", "verbosity": -1, "seed": 0, "num_threads": 1})
    booster = lgb.train(params, lgb.Dataset(np.array(xs), label=np.array(ys), weight=np.array(ws)), num_boost_round=spec["num_boost_round"])
    return {r["日期"]: float(booster.predict([train.apply_scale([float(r[n]) for n in cols], mean, scale)])[0]) for r in te}


def pick(product: str) -> dict:
    rows = json.loads((HERE / product / "主配菜对比.json").read_text())
    chosen = {}
    for family in MODELS:
        cands = [r for r in rows if r["模型"] == family]
        if not cands:
            continue
        chosen[family] = max(cands, key=lambda r: (r["利润"], r["准确率"]))
    labels = {r["日期"]: r for r in csv.DictReader((HERE / product / "标签.csv").open())}
    feat = {r["日期"]: r for r in csv.DictReader((HERE / product / "特征.csv").open())}
    joined = []
    for day, row in feat.items():
        lab = labels.get(day)
        if not lab or day < TRAIN_FROM:
            continue
        item = dict(row)
        item["y"] = int(float(lab["涨跌"]))
        item["move"] = float(lab["涨跌幅"])
        joined.append(item)

    probs = {}
    for family, row in chosen.items():
        cols = menu_cols(product, row["菜单"], family)
        data = [r for r in joined if all(str(r.get(n, "")).strip() != "" for n in cols)]
        kind = "tree" if family.endswith("树") else "logit"
        probs[family] = predict(data, cols, row["参数"], kind, product)

    common = sorted(set.intersection(*(set(p) for p in probs.values()))) if len(probs) == 4 else sorted(set.union(*(set(p) for p in probs.values())))
    names = [n for n in MODELS if n in probs]
    signals = {}
    for need in (2, 3):
        sig = {}
        for day in common:
            votes = [sgn(probs[n][day]) for n in names]
            ups = sum(v == 1 for v in votes)
            downs = sum(v == -1 for v in votes)
            sig[day] = 1 if ups >= need and ups > downs else -1 if downs >= need and downs > ups else 0
        signals[f"多数票{need}"] = sig
    avg = {day: sgn(sum(probs[n][day] for n in names) / len(names)) for day in common}
    signals["平均"] = avg
    for n in names:
        signals[f"只跟{n}"] = {day: sgn(probs[n][day]) for day in common}

    results = []
    for title, sig in signals.items():
        d, h, pnl = score(common, sig, labels)
        results.append({"综合": title, "有方向": d, "正确": h, "准确率": round(h / d * 100, 1) if d else 0, "利润": round(pnl)})
    best = max(results, key=lambda r: (r["利润"], r["准确率"]))
    payload = {"品种": PRODUCTS[product][0], "四个模型": {k: {"菜单": v["菜单"], "准确率": v["准确率"], "利润": v["利润"]} for k, v in chosen.items()}, "比较": results, "选定": best}
    (HERE / product / "综合对比.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    import lightgbm as lgb  # noqa: F401

    for product in sys.argv[1:] or PRODUCTS:
        payload = pick(product)
        print(product, payload["选定"])
