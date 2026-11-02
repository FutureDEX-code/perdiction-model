#!/usr/bin/env python3
"""Email the depth-2 tree outlook. The logistic mail is unchanged.

Run with the project virtualenv: it is the interpreter that has LightGBM.
The fit uses the same daily closes the logistic just saved. It does not read
the minute files.
"""

from __future__ import annotations

import sys

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core import train
from jobs.mail_outlook import (
    SENDER,
    TO,
    _forbid_company,
    _google_enabled,
    _locate,
    _outbox_count,
    _pull_from_bin,
    _send,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "models_tree"
SENT = ROOT / "data" / "futures_tree_emailed.txt"
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


def _series() -> dict[str, dict[str, float]]:
    """Closes train.py just wrote, including a 14:50 that replaced today's 15:00."""

    return {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in (*train.DOMESTIC, *train.OVERSEAS)
    }


def _named(series: dict[str, dict[str, float]]) -> dict[str, float]:
    domestic = {key: train.session_returns(key, series[key]) for key in train.DOMESTIC if key in series}
    last = train.session_day(series)
    named = {
        "豆一昨涨跌": domestic["A"][last],
        "豆粕昨涨跌": domestic["M"][last],
        "菜粕昨涨跌": domestic["RM"][last],
    }
    cutoff = train._day_after(last)
    for name, key in zip(("美豆隔夜", "美豆粕隔夜", "美豆油隔夜"), train.OVERSEAS):
        named[name] = train.overseas_session(series[key], last, cutoff)
    tape = train.domestic_tape(series, last)
    for name, value in zip(("豆一这段", "豆粕这段", "菜粕这段"), tape):
        named[name] = value
    if "B" in domestic and last in domestic["B"]:
        named["豆二昨涨跌"] = domestic["B"][last]
        named["豆二这段"] = train.domestic_tape(series, last, keys=("B",))[0]
    if "Y" in domestic and last in domestic["Y"]:
        named["豆油昨涨跌"] = domestic["Y"][last]
        named["豆油这段"] = train.domestic_tape(series, last, keys=("Y",))[0]
    named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
    for key in train.DOMESTIC:
        named.update(train.extra_named(key, last))
    named.update(train.cot_named(train.cot_series(), last))
    named.update(train.mood_named(train.cot_series(), train.seat_series(), last))
    return named


def _fit_views(series: dict[str, dict[str, float]], named: dict[str, float]) -> list[dict]:
    rows = [row for row in train.samples(series) if row["day"] >= "2017-01-01"]
    views = []
    OUT.mkdir(parents=True, exist_ok=True)
    for key in train.CONTRACTS:
        names = train.sina_feature_names(key, "日线树")
        usable = [row for row in rows if key in row["y"]]
        spec = {}
        meta = OUT / f"{key}.json"
        if meta.exists():
            spec = json.loads(meta.read_text()).get("train_params") or {}
        params = dict(PARAMS)
        params["max_depth"] = int(spec.get("max_depth", params["max_depth"]))
        params["num_leaves"] = int(spec.get("num_leaves", 2 ** params["max_depth"]))
        params["learning_rate"] = float(spec.get("learning_rate", params["learning_rate"]))
        params["min_child_samples"] = int(spec.get("min_child_samples", params["min_child_samples"]))
        params["lambda_l2"] = float(spec.get("lambda_l2", params["lambda_l2"]))
        rounds = int(spec.get("num_boost_round", ROUNDS))
        matrix_raw = [[row["named"][name] for name in names] for row in usable]
        mean, scale = train.standardize(matrix_raw)
        matrix = np.array([train.apply_scale(row, mean, scale) for row in matrix_raw])
        labels = np.array([row["y"][key] for row in usable])
        weights = np.array([train.sample_weight(row["move"][key]) for row in usable])
        booster = lgb.train(params, lgb.Dataset(matrix, label=labels, weight=weights), num_boost_round=rounds)
        booster.save_model(str(OUT / f"{key}.txt"))
        body = json.loads(meta.read_text()) if meta.exists() else {"code": key, "name": train.NAMES[key]}
        body.update(
            {
                "kind": "lightgbm",
                "trained_at": train.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "source": "sina-daily",
                "feature_names": list(names),
                "mean": mean,
                "scale": scale,
                "train_params": {
                    "max_depth": params["max_depth"],
                    "num_leaves": params["num_leaves"],
                    "learning_rate": params["learning_rate"],
                    "num_boost_round": rounds,
                    "min_child_samples": params["min_child_samples"],
                    "lambda_l2": params["lambda_l2"],
                },
                "direction_up": 0.55,
                "direction_down": 0.45,
            }
        )
        meta.write_text(json.dumps(body, ensure_ascii=False, indent=2))
        vector = train.apply_scale([float(named[name]) for name in names], mean, scale)
        prob = float(booster.predict([vector])[0])
        views.append(
            {
                "code": key,
                "name": train.NAMES[key],
                "prob_up": round(prob, 4),
                "lean": train.direction(prob),
            }
        )
    return views


def _body(as_of: str, views: list[dict]) -> str:
    lines = [f"树模型。基于{as_of} 14:50 的价格，下一交易日：", ""]
    for item in views:
        lines.append(f"{item['name']} 上涨概率 {item['prob_up'] * 100:.1f}%  {item['lean']}")
    lines.append("")
    lines.append("45%到55%之间按规则不下新单。这封是树模型，回归那封照旧。")
    return "\n".join(lines)


def send_tree() -> str:
    series = _series()
    as_of = train.session_day(series)
    sent = SENT.read_text().strip() if SENT.exists() else ""
    if sent == as_of:
        return f"already sent {as_of}"
    if not _google_enabled():
        raise RuntimeError("Mail 里的 Gmail 是停用的，信寄不出去")
    _forbid_company(SENDER)
    _forbid_company(TO)
    views = _fit_views(series, _named(series))
    subject = f"下一交易日树模型（基于{as_of}收盘）"
    if _outbox_count(subject) == 0 and _locate(subject) == "no":
        _send(subject, _body(as_of, views))
    for _ in range(8):
        place = _locate(subject)
        if place == "Bin":
            _pull_from_bin(subject)
            place = _locate(subject)
        if place in {"INBOX", "Sent Mail", "All Mail"}:
            SENT.write_text(as_of + "\n")
            (OUT / "outlook.json").write_text(
                json.dumps({"as_of": as_of, "views": views}, ensure_ascii=False, indent=2)
            )
            return f"sent {as_of}"
        import time

        time.sleep(2)
    raise RuntimeError("树模型的信还没出现在收件箱")


def main() -> int:
    try:
        print(send_tree(), flush=True)
    except Exception as exc:
        print(f"mail failed {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
