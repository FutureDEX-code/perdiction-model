#!/usr/bin/env python3
"""Proper half-hour models on the full minute history, one pair per product.

Replaces the three-feature on-the-fly fit for the 30-minute layer. Bars are the
chained session segments (including the 10:00-10:15 stub); the label is the
next bar's direction. Training is all-history for production, or cut off for
holdout validation. Weights land in data/models_hh/ and data/models_hh_tree/.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from live.hour_model import SEGMENTS, SESSIONS

LOGIT_OUT = ROOT / "data" / "models_hh"
TREE_OUT = ROOT / "data" / "models_hh_tree"
FEATURES = ("这段", "振幅", "位置", "量变化", "仓变化", "上一段", "今日累计", "夜盘", "美豆隔夜")
MIN_VOL_TICKS = 1


def _slot_end(day: str, end_min: int) -> str:
    return f"{day} {end_min // 60:02d}:{end_min % 60:02d}:00"


def _session_index(minutes: int) -> int | None:
    for idx, (start, end) in enumerate(SESSIONS):
        if start <= minutes < end:
            return idx
    return None


def load_bars(product: str) -> list[dict]:
    """Chained half-hour bars from the main-continuous minute file."""

    path = train.MAIN_MINUTE / f"{product}.csv"
    if not path.exists():
        return []
    stamps: dict[str, dict] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            stamp = row["trade_time"]
            day, hm = stamp[:10], stamp[11:16]
            minutes = int(hm[:2]) * 60 + int(hm[3:])
            idx = _session_index(minutes)
            if idx is None or not row["close"]:
                continue
            for start, end in SEGMENTS:
                if start <= minutes < end:
                    key = f"{day} {start // 60:02d}:{start % 60:02d}"
                    bar = stamps.get(key)
                    if bar is None:
                        bar = {"slot": key, "o": None, "h": None, "l": None, "c": None, "vol": 0.0, "oi": None, "night": 1 if start >= 21 * 60 else 0}
                        stamps[key] = bar
                    o, h, lo, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                    bar["o"] = o if bar["o"] is None else bar["o"]
                    bar["h"] = c if bar["h"] is None else max(bar["h"], h)
                    bar["l"] = c if bar["l"] is None else min(bar["l"], lo)
                    bar["c"] = c
                    bar["vol"] += float(row["vol"] or 0)
                    if row["oi"]:
                        bar["oi"] = float(row["oi"])
                    break
    return [stamps[key] for key in sorted(stamps) if stamps[key]["o"] is not None]


def build_rows(product: str, overseas: dict[str, dict[str, float]]) -> list[dict]:
    bars = load_bars(product)
    if not bars:
        return []
    closes = dict(train._cached_rows(train.series_path(product)))
    days = sorted(closes)
    prev_close = {day: closes[days[i - 1]] for i, day in enumerate(days) if i > 0}
    us = train.returns(overseas["S"])
    us_days = sorted(us)
    # 同时段均量：每个 slot 的全历史平均，标准化量变化
    slot_vol: dict[str, list[float]] = {}
    for bar in bars:
        slot_vol.setdefault(bar["slot"][6:], []).append(bar["vol"])
    slot_mean = {slot: sum(vols) / len(vols) for slot, vols in slot_vol.items() if vols}
    rows = []
    prev_oi = None
    for i, bar in enumerate(bars):
        day = bar["slot"][:10]
        base = prev_close.get(day)
        today_ret = (bar["c"] - base) / base * 100.0 if base else 0.0
        ret = 0.0 if not bar["o"] else (bar["c"] - bar["o"]) / bar["o"] * 100.0
        span = 0.0 if not bar["o"] else (bar["h"] - bar["l"]) / bar["o"] * 100.0
        loc = 0.5 if bar["h"] <= bar["l"] else (bar["c"] - bar["l"]) / (bar["h"] - bar["l"])
        mean_vol = slot_mean.get(bar["slot"][6:], 0.0)
        vol_chg = 0.0 if not mean_vol else (bar["vol"] / mean_vol - 1.0) * 100.0
        oi = bar["oi"]
        oi_chg = 0.0 if not oi or not prev_oi else (oi / prev_oi - 1.0) * 100.0
        prev_oi = oi or prev_oi
        prev_ret = rows[-1]["这段"] if rows else 0.0
        us_day = train.latest_on_or_before(us_days, day)
        us_ret = us.get(us_day, 0.0) if us_day else 0.0
        nxt = bars[i + 1] if i + 1 < len(bars) else None
        if nxt is None:
            continue
        rows.append(
            {
                "day": day,
                "slot": bar["slot"],
                "这段": ret,
                "振幅": span,
                "位置": loc,
                "量变化": vol_chg,
                "仓变化": oi_chg,
                "上一段": prev_ret,
                "今日累计": today_ret,
                "夜盘": float(bar["night"]),
                "美豆隔夜": us_ret,
                "y": 1 if nxt["c"] > bar["c"] else 0,
                "move": (nxt["c"] - bar["c"]) / bar["c"] * 100.0 if bar["c"] else 0.0,
                "exit": nxt["c"],
                "entry": bar["c"],
            }
        )
    return rows


def fit_logit(rows: list[dict], spec: dict):
    matrix = [[row[name] for name in FEATURES] for row in rows]
    mean, scale = train.standardize(matrix)
    weight, bias = train.fit(
        [train.apply_scale(row, mean, scale) for row in matrix],
        [row["y"] for row in rows],
        [train.sample_weight(row["move"]) for row in rows],
        lr=float(spec.get("lr", 0.08)),
        steps=int(spec.get("steps", 500)),
        l2=float(spec.get("l2", 0.2)),
    )
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias}


def fit_tree(rows: list[dict], spec: dict):
    matrix = [[row[name] for name in FEATURES] for row in rows]
    mean, scale = train.standardize(matrix)
    params = {
        "objective": "binary",
        "verbosity": -1,
        "seed": 0,
        "num_threads": 1,
        "max_depth": int(spec.get("max_depth", 2)),
        "num_leaves": int(spec.get("num_leaves", 4)),
        "learning_rate": float(spec.get("learning_rate", 0.05)),
        "min_child_samples": int(spec.get("min_child_samples", 50)),
        "lambda_l2": float(spec.get("lambda_l2", 1.0)),
    }
    booster = lgb.train(
        params,
        lgb.Dataset(
            np.array([train.apply_scale(row, mean, scale) for row in matrix]),
            label=np.array([row["y"] for row in rows]),
            weight=np.array([train.sample_weight(row["move"]) for row in rows]),
        ),
        num_boost_round=int(spec.get("num_boost_round", 100)),
    )
    return {"mean": mean, "scale": scale, "booster": booster}


def score(fit: dict, row: dict, kind: str) -> float:
    vec = train.apply_scale([row[name] for name in FEATURES], fit["mean"], fit["scale"])
    if kind == "logit":
        return train.predict_prob(vec, fit["weight"], fit["bias"])
    return float(fit["booster"].predict([vec])[0])


def save(product: str, logit: dict, tree: dict, rows: int) -> None:
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    LOGIT_OUT.mkdir(parents=True, exist_ok=True)
    TREE_OUT.mkdir(parents=True, exist_ok=True)
    body = {
        "code": product,
        "name": train.NAMES[product],
        "trained_at": trained_at,
        "source": "training_dataset/main_minute 半小时段",
        "feature_names": list(FEATURES),
        "rows": rows,
        "direction_up": 0.55,
        "direction_down": 0.45,
    }
    (LOGIT_OUT / f"{product}.json").write_text(json.dumps({**body, "kind": "logit", **logit}, ensure_ascii=False, indent=2))
    tree_body = {**body, "kind": "tree", "mean": tree["mean"], "scale": tree["scale"], "model": f"{product}.txt"}
    (TREE_OUT / f"{product}.json").write_text(json.dumps(tree_body, ensure_ascii=False, indent=2))
    tree["booster"].save_model(str(TREE_OUT / f"{product}.txt"))


def main() -> None:
    overseas = {key: dict(train._cached_rows(train.series_path(key))) for key in train.OVERSEAS}
    for product in train.CONTRACTS:
        rows = build_rows(product, overseas)
        logit = fit_logit(rows, {})
        tree = fit_tree(rows, {})
        save(product, logit, tree, len(rows))
        print(f"{train.NAMES[product]} 半小时两族已训 {len(rows)} 段", flush=True)


if __name__ == "__main__":
    main()
