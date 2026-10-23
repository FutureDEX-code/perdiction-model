#!/usr/bin/env python3
"""Shallow LightGBM next to the live logistic. It does not write data/models/.

Same features, same magnitude weights, same 45–55 band. Depth is fixed at 2.
The files in data/models_tree/ are fit on all cached history. The compare file
is the holdout, trained only on days before each split.
"""

from __future__ import annotations

import sys

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "models_tree"
SPLITS = ("2023-10-20", "2024-10-20", "2025-10-20")
KEEP = "2017-01-01"
YUAN = 100
VEC = {"A": "x", "M": "x", "RM": "x_rm", "B": "x_b", "Y": "x_y"}
NAMES = {
    "A": train.FEATURE_NAMES,
    "M": train.FEATURE_NAMES,
    "RM": train.RM_FEATURES,
    "B": train.B_FEATURES,
    "Y": train.Y_FEATURES,
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
ROUNDS = 100


def _usable(rows: list[dict], key: str) -> list[dict]:
    vec = VEC[key]
    return [row for row in rows if row.get(vec) and key in row["y"] and row["day"] >= KEEP]


def _fit(rows: list[dict], key: str) -> tuple[lgb.Booster, list[float], list[float]]:
    vec = VEC[key]
    mean, scale = train.standardize([row[vec] for row in rows])
    matrix = np.array([train.apply_scale(row[vec], mean, scale) for row in rows])
    labels = np.array([row["y"][key] for row in rows])
    weights = np.array([train.sample_weight(row["move"][key]) for row in rows])
    booster = lgb.train(PARAMS, lgb.Dataset(matrix, label=labels, weight=weights), num_boost_round=ROUNDS)
    return booster, mean, scale


def _money(rows: list[dict], raw: dict[str, dict[str, float]], key: str, split: str) -> dict:
    usable = _usable(rows, key)
    train_rows = [row for row in usable if row["day"] < split]
    test_rows = [row for row in usable if row["day"] >= split]
    booster, mean, scale = _fit(train_rows, key)
    vec = VEC[key]
    matrix = np.array([train.apply_scale(row[vec], mean, scale) for row in test_rows])
    probs = booster.predict(matrix)
    pnl = 0.0
    orders = wins = 0
    for row, prob in zip(test_rows, probs):
        if 0.45 < prob < 0.55:
            continue
        side = 1 if prob >= 0.55 else -1
        exit_px = raw[key][row["day"]]
        entry = exit_px / (1.0 + row["move"][key] / 100.0)
        this = side * (exit_px - entry) * YUAN
        pnl += this
        orders += 1
        wins += this > 0
    return {"orders": orders, "wins": wins, "pnl": pnl, "train": len(train_rows), "test": len(test_rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in (*train.DOMESTIC, *train.OVERSEAS)
    }
    rows = train.samples(raw, train.news_scores())
    trained_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    last_day = max(row["day"] for row in rows)
    for key in train.CONTRACTS:
        usable = _usable(rows, key)
        booster, mean, scale = _fit(usable, key)
        booster.save_model(str(OUT / f"{key}.txt"))
        body = {
            "code": key,
            "name": train.NAMES[key],
            "kind": "lightgbm",
            "trained_at": trained_at,
            "trained_through": last_day,
            "note": "不接入实盘。每天 14:51 仍用 data/models 里的逻辑回归。",
            "params": {**PARAMS, "num_boost_round": ROUNDS},
            "direction_up": 0.55,
            "direction_down": 0.45,
            "feature_names": list(NAMES[key]),
            "mean": mean,
            "scale": scale,
            "rows": len(usable),
        }
        (OUT / f"{key}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2))
        print(f"写入 {train.NAMES[key]} {len(usable)} 天", flush=True)
    compare = {"folder": str(OUT), "live_folder": str(train.MODELS), "splits": {}}
    for split in SPLITS:
        compare["splits"][split] = {}
        print(f"\n{split} 之后", flush=True)
        total = 0.0
        for key in train.CONTRACTS:
            result = _money(rows, raw, key, split)
            total += result["pnl"]
            compare["splits"][split][key] = result
            print(
                f"  {train.NAMES[key]} 下单{result['orders']} 赢{result['wins']} {result['pnl'] / 10000:+.2f}万",
                flush=True,
            )
        print(f"  合计 {total / 10000:+.2f}万", flush=True)
        compare["splits"][split]["total"] = total
    (OUT / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2))
    print(f"\n已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
