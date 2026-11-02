#!/usr/bin/env python3
"""Train one product's four models on 2022-01-01..2025-12-31, validate on 2026.

Hyperparameters are searched on the two one-year windows inside the training
segment, by reverse-trade PnL. The 2026 holdout never touches the choice.
Writes <product>/主配菜对比.json with the picked menu per model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN_FROM = "2022-01-01"
TRAIN_TO = "2025-12-31"
YUAN = 100
VAL = (("2023-10-01", "2024-10-01"), ("2024-10-01", "2025-10-01"))
SPANS = (("2026-01-01", "2026-04-30"), ("2026-05-01", "2026-08-31"), ("2026-09-01", "2026-10-31"))
LOGIT_GRID = [{"lr": lr, "steps": steps, "l2": l2} for lr in (0.05, 0.08) for steps in (300, 500) for l2 in (0.05, 0.2, 1.0)]
TREE_GRID = [
    {"max_depth": depth, "num_leaves": 2 ** depth, "learning_rate": lr, "num_boost_round": rounds, "min_child_samples": leaf, "lambda_l2": l2}
    for depth in (2, 3) for lr in (0.05, 0.1) for rounds in (80, 200) for leaf in (30, 80) for l2 in (0.3, 3.0)
]


def sgn(p: float) -> int:
    return 1 if p >= 0.55 else -1 if p <= 0.45 else 0


def score_probs(rows, probs, labels, start=None, end=None):
    pos = 0
    pnl = 0.0
    directed = hits = 0
    for row, prob in zip(rows, probs):
        if start and not (start <= row["日期"] <= end):
            continue
        lab = labels[row["日期"]]
        entry, exit_px = float(lab["入场价"]), float(lab["出场价"])
        signal = sgn(prob)
        up = exit_px > entry
        if signal != 0:
            directed += 1
            hits += int((signal == 1 and up) or (signal == -1 and not up))
        held = pos if signal == 0 else signal
        pnl += held * (exit_px - entry) * YUAN
        pos = held
    return directed, hits, pnl


def predict(tr, te, cols, spec, kind):
    matrix = [[float(r[n]) for n in cols] for r in tr]
    mean, scale = train.standardize(matrix)
    xs = [train.apply_scale(row, mean, scale) for row in matrix]
    ys = [r["y"] for r in tr]
    ws = [train.sample_weight(r["move"]) for r in tr]
    test_x = [train.apply_scale([float(r[n]) for n in cols], mean, scale) for r in te]
    if kind == "logit":
        w, b = train.fit(xs, ys, ws, lr=spec["lr"], steps=spec["steps"], l2=spec["l2"])
        return [train.predict_prob(row, w, b) for row in test_x]
    params = {k: spec[k] for k in ("max_depth", "num_leaves", "learning_rate", "min_child_samples", "lambda_l2")}
    params.update({"objective": "binary", "verbosity": -1, "seed": 0, "num_threads": 1})
    booster = lgb.train(params, lgb.Dataset(np.array(xs), label=np.array(ys), weight=np.array(ws)), num_boost_round=spec["num_boost_round"])
    return [float(booster.predict([row])[0]) for row in test_x]


def run(product: str, menus: dict, models: tuple) -> list[dict]:
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

    def usable(cols):
        return [r for r in joined if all(str(r.get(n, "")).strip() != "" for n in cols)]

    rows_out = []
    for name, which, kind, grid in models:
        for menu, pair in menus.items():
            cols = pair[which]
            data = usable(cols)
            best = None
            for spec in grid:
                total = 0.0
                ok = True
                for start, end in VAL:
                    tr = [r for r in data if TRAIN_FROM <= r["日期"] < start]
                    va = [r for r in data if start <= r["日期"] < end]
                    if len(tr) < 100 or not va:
                        ok = False
                        break
                    _, _, pnl = score_probs(va, predict(tr, va, cols, spec, kind), labels)
                    total += pnl
                if ok and (best is None or total > best[0]):
                    best = (total, spec)
            tr = [r for r in data if TRAIN_FROM <= r["日期"] <= TRAIN_TO]
            te = [r for r in data if r["日期"] > TRAIN_TO]
            if not te:
                continue
            probs = predict(tr, te, cols, best[1], kind)
            d, h, pnl = score_probs(te, probs, labels)
            acc = h / d * 100 if d else 0
            seg = [round(score_probs(te, probs, labels, start, end)[2] / 10000) for start, end in SPANS]
            rows_out.append({"模型": name, "菜单": menu, "有方向": d, "正确": h, "准确率": round(acc, 1), "利润": round(pnl), "参数": best[1], "段": seg})
    (HERE / product / "主配菜对比.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n")
    return rows_out


def main() -> None:
    product = sys.argv[1]
    menus = json.loads(sys.argv[2])
    models = (
        ("日线逻辑回归", 0, "logit", LOGIT_GRID),
        ("日线树", 0, "tree", TREE_GRID),
        ("分钟逻辑回归", 1, "logit", LOGIT_GRID),
        ("分钟树", 1, "tree", TREE_GRID),
    )
    rows = run(product, menus, models)
    for row in rows:
        print(f"{row['模型']:<10}{row['菜单']:<12}{row['有方向']:>6}{row['正确']:>6}{row['准确率']:>7}%{row['利润'] / 10000:>8.2f}万  {row['段']}")


if __name__ == "__main__":
    main()
