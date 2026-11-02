#!/usr/bin/env python3
"""Second layer for one product: learn how much to trust each of the four models.

The four base probabilities are produced walk-forward: for the 2019 block the
base models are fit on data before 2019, and so on. The meta logistic never
sees an in-sample probability. Label is the 14:55-to-next-14:55 direction,
priced by the same fallback chain as the validation (exact bar, next bar,
previous bar, daily close).

The meta model lives in the product's own directory: data/<KEY>/综合.json.
"""

from __future__ import annotations

import sys

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core import train
from models import fit_sina_main as minute

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FAMILIES = ("日线回归", "日线树", "分钟回归", "分钟树")
FIRST_BLOCK_YEAR = 2019


def price_map(product: str) -> dict[str, float]:
    import csv

    daily_close = dict(train._cached_rows(train.series_path(product)))
    exact, after, before = {}, {}, {}
    with (train.MAIN_MINUTE / f"{product}.csv").open() as handle:
        for row in csv.DictReader(handle):
            stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
            if not row["close"]:
                continue
            if hm == "14:55":
                exact[stamp] = float(row["close"])
            elif "14:55" < hm <= "15:00":
                after.setdefault(stamp, float(row["close"]))
            elif hm < "14:55":
                before[stamp] = float(row["close"])
    days = set(daily_close) | set(exact) | set(after) | set(before)
    return {day: exact.get(day) or after.get(day) or before.get(day) or daily_close.get(day) for day in days if exact.get(day) or after.get(day) or before.get(day) or daily_close.get(day)}


def _fit_logit(rows, vec_of, y_of, mv_of, spec):
    mean, scale = train.standardize([vec_of(r) for r in rows])
    weight, bias = train.fit(
        [train.apply_scale(vec_of(r), mean, scale) for r in rows],
        [y_of(r) for r in rows],
        [train.sample_weight(mv_of(r)) for r in rows],
        lr=float(spec.get("lr", 0.08)),
        steps=int(spec.get("steps", 500)),
        l2=float(spec.get("l2", 0.2)),
    )
    return {"kind": "logit", "mean": mean, "scale": scale, "weight": weight, "bias": bias}


def _fit_tree(rows, vec_of, y_of, mv_of, spec):
    mean, scale = train.standardize([vec_of(r) for r in rows])
    params = {
        "objective": "binary", "verbosity": -1, "seed": 0, "num_threads": 1,
        "max_depth": int(spec.get("max_depth", 2)),
        "num_leaves": int(spec.get("num_leaves", 4)),
        "learning_rate": float(spec.get("learning_rate", 0.05)),
        "min_child_samples": int(spec.get("min_child_samples", 50)),
        "lambda_l2": float(spec.get("lambda_l2", 1.0)),
    }
    booster = lgb.train(
        params,
        lgb.Dataset(
            np.array([train.apply_scale(vec_of(r), mean, scale) for r in rows]),
            label=np.array([y_of(r) for r in rows]),
            weight=np.array([train.sample_weight(mv_of(r)) for r in rows]),
        ),
        num_boost_round=int(spec.get("num_boost_round", 100)),
    )
    return {"kind": "tree", "mean": mean, "scale": scale, "booster": booster}


def _score(fit: dict, vec: list[float]) -> float:
    scaled = train.apply_scale(vec, fit["mean"], fit["scale"])
    if fit["kind"] == "logit":
        return train.predict_prob(scaled, fit["weight"], fit["bias"])
    return float(fit["booster"].predict([scaled])[0])


def _spec(product: str, folder: str) -> dict:
    path = ROOT / "data" / folder / f"{product}.json"
    if path.exists():
        return json.loads(path.read_text()).get("train_params") or {}
    return {}


def fit_bases(product: str, sina_rows, min_rows, before: str) -> dict:
    """Fit the four chosen variants for one product, using rows strictly before `before`.

    A family with too little history is skipped; downstream treats it as 0.5.
    """

    sina_train = [r for r in sina_rows if r["day"] < before and product in r["y"]]
    min_train = [r for r in min_rows if r["day"] < before and product in r["y"]]
    names_dl = train.sina_feature_names(product, "日线回归")
    names_dt = train.sina_feature_names(product, "日线树")
    names_ml = tuple(minute.minute_names(product, "分钟回归"))
    names_mt = tuple(minute.minute_names(product, "分钟树"))
    fits = {}
    sina_dl = [r for r in sina_train if all(n in r["named"] for n in names_dl)]
    sina_dt = [r for r in sina_train if all(n in r["named"] for n in names_dt)]
    min_l = [r for r in min_train if all(n in r["named"] for n in names_ml)]
    min_t = [r for r in min_train if all(n in r["named"] for n in names_mt)]
    if len(sina_dl) >= 200:
        fits["日线回归"] = (
            _fit_logit(sina_dl, lambda r: [r["named"][n] for n in names_dl], lambda r: r["y"][product], lambda r: r["move"][product], _spec(product, "models")),
            names_dl, "sina",
        )
    if len(sina_dt) >= 200:
        fits["日线树"] = (
            _fit_tree(sina_dt, lambda r: [r["named"][n] for n in names_dt], lambda r: r["y"][product], lambda r: r["move"][product], _spec(product, "models_tree")),
            names_dt, "sina",
        )
    if len(min_l) >= 200:
        fits["分钟回归"] = (
            _fit_logit(min_l, lambda r: [r["named"][n] for n in names_ml], lambda r: r["y"][product], lambda r: r["move"][product], _spec(product, "models_1450")),
            names_ml, "minute",
        )
    if len(min_t) >= 200:
        fits["分钟树"] = (
            _fit_tree(min_t, lambda r: [r["named"][n] for n in names_mt], lambda r: r["y"][product], lambda r: r["move"][product], _spec(product, "models_tree_1450")),
            names_mt, "minute",
        )
    return fits


def block_probs(fits: dict, sina_by_anchor: dict, min_by_signal: dict, days: list[str]) -> dict[str, dict]:
    out = {}
    for day in days:
        probs = {}
        for family, (fit, names, source) in fits.items():
            row = sina_by_anchor.get(day) if source == "sina" else min_by_signal.get(day)
            if row is None or any(n not in row["named"] for n in names):
                continue
            probs[family] = _score(fit, [row["named"][n] for n in names])
        out[day] = probs
    return out


def meta_path(product: str) -> Path:
    return ROOT / "data" / product / "综合.json"


def save_meta(product: str, mean, scale, weight, bias, trained_at: str) -> None:
    path = meta_path(product)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "code": product,
        "name": train.NAMES[product],
        "kind": "logit",
        "feature_names": list(FAMILIES),
        "mean": mean,
        "scale": scale,
        "weight": weight,
        "bias": bias,
        "trained_at": trained_at,
        "direction_up": 0.55,
        "direction_down": 0.45,
    }, ensure_ascii=False, indent=2))


def load_meta(product: str) -> dict | None:
    path = meta_path(product)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def meta_prob(meta: dict, probs: dict[str, float]) -> float:
    vector = [float(probs.get(family, 0.5)) for family in FAMILIES]
    return train.predict_prob(train.apply_scale(vector, meta["mean"], meta["scale"]), meta["weight"], meta["bias"])


def majority(probs: dict[str, float], need: int = 2) -> tuple[str, int, int, float]:
    """Follow a direction when at least `need` models agree and they outnumber the other side."""

    ups = sum(1 for family in FAMILIES if probs.get(family, 0.5) >= 0.55)
    downs = sum(1 for family in FAMILIES if probs.get(family, 0.5) <= 0.45)
    mean = sum(probs.get(family, 0.5) for family in FAMILIES) / len(FAMILIES)
    if ups >= need and ups > downs:
        return "偏涨", ups, downs, mean
    if downs >= need and downs > ups:
        return "偏跌", ups, downs, mean
    return "没有方向", ups, downs, mean


def average(probs: dict[str, float]) -> tuple[str, int, int, float]:
    """Mean of the four probabilities, direction from the 45–55 band on the mean."""

    mean = sum(probs.get(family, 0.5) for family in FAMILIES) / len(FAMILIES)
    ups = sum(1 for family in FAMILIES if probs.get(family, 0.5) >= 0.55)
    downs = sum(1 for family in FAMILIES if probs.get(family, 0.5) <= 0.45)
    if mean >= 0.55:
        return "偏涨", ups, downs, mean
    if mean <= 0.45:
        return "偏跌", ups, downs, mean
    return "没有方向", ups, downs, mean


def build(product: str, until: str | None = None):
    """Walk-forward base probabilities, then the meta fit. Returns (meta, rows)."""

    series = train.load_series()
    news = train.news_scores()
    common = sorted(set(train.session_returns("A", series["A"])) & set(train.session_returns("M", series["M"])) & set(train.session_returns("RM", series["RM"])))
    prev_of = {day: common[i - 1] for i, day in enumerate(common) if i > 0}
    sina_rows = [row for row in train.samples(series, news) if row["day"] >= "2017-01-01"]
    days = {p: minute.load_days(p) for p in ("A", "M", "RM")}
    min_rows = minute.build_rows(days, {k: series[k] for k in train.OVERSEAS}, news)
    sina_by_anchor = {prev_of[r["day"]]: r for r in sina_rows}
    min_by_signal = {r["signal"]: r for r in min_rows}
    calendar = sorted(dict(train._cached_rows(train.series_path(product))))
    next_day = {day: calendar[i + 1] for i, day in enumerate(calendar[:-1])}
    px = price_map(product)
    last = until or calendar[-1]

    rows = []
    for year in range(FIRST_BLOCK_YEAR, int(last[:4]) + 1):
        start = f"{year}-01-01"
        end = min(f"{year}-12-31", last)
        if start >= end:
            continue
        block_days = [day for day in calendar if start <= day <= end and day in next_day]
        if not block_days:
            continue
        fits = fit_bases(product, sina_rows, min_rows, start)
        probs = block_probs(fits, sina_by_anchor, min_by_signal, block_days)
        for day in block_days:
            exit_day = next_day[day]
            if day not in px or exit_day not in px:
                continue
            rows.append({
                "day": day,
                "exit": exit_day,
                "probs": probs.get(day) or {},
                "y": 1 if px[exit_day] > px[day] else 0,
                "move": (px[exit_day] - px[day]) / px[day] * 100.0,
            })
        print(f"综合 {train.NAMES[product]} 块 {year} 完成 {len(block_days)} 天", flush=True)
    usable = [row for row in rows if row["day"] <= (until or "9999")]
    mean, scale = train.standardize([[row["probs"].get(f, 0.5) for f in FAMILIES] for row in usable])
    weight, bias = train.fit(
        [train.apply_scale([row["probs"].get(f, 0.5) for f in FAMILIES], mean, scale) for row in usable],
        [row["y"] for row in usable],
        [train.sample_weight(row["move"]) for row in usable],
        lr=0.08, steps=500, l2=0.2,
    )
    trained_at = train.datetime.now().strftime("%Y-%m-%d %H:%M")
    save_meta(product, mean, scale, weight, bias, trained_at)
    meta = {"mean": mean, "scale": scale, "weight": weight, "bias": bias}
    return meta, rows
