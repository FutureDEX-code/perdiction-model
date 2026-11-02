#!/usr/bin/env python3
"""Fit 豆一 only, without the news column. The latest year is not used to fit."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from jobs import mail_tree
from models import fit_sina_main as minute
from no_news.run import fit_family, one_window
from path_train.run import records_for

KEY = "A"


def main() -> None:
    series = train.load_series()
    news = train.news_scores()
    sina_rows = [row for row in train.samples(series, news) if row["day"] >= "2017-01-01" and KEY in row["y"]]
    last = sina_rows[-1]["day"]
    cutoff = f"{int(last[:4]) - 1}{last[4:]}"
    print(f"豆一 拟合到 {cutoff} 之前，{cutoff} → {last} 当未知", flush=True)
    days = {product: minute.load_days(product) for product in train.CONTRACTS}
    min_rows = minute.build_rows(days, {key: series[key] for key in train.OVERSEAS}, news)
    minute_by_anchor = {row["signal"]: row for row in min_rows}
    records = records_for(KEY, sina_rows, series[KEY])
    scored = one_window(KEY, sina_rows, min_rows, minute_by_anchor, records, cutoff, None)
    combo = scored["去掉新闻"]["综合"]
    print(
        f"综合 {combo['净盈亏'] / 10000:.2f} 万  方向 {combo['方向命中率'] * 100:.1f}%"
        f"  {combo['方向命中']}/{combo['有方向天数']}  持仓 {combo['持仓命中率'] * 100:.1f}%",
        flush=True,
    )
    for family, item in scored["去掉新闻"]["家族"].items():
        hit = item["方向命中率"]
        hit_text = "—" if hit is None else f"{hit * 100:.1f}%"
        print(f"  {family} {item['净盈亏'] / 10000:.2f} 万  方向 {hit_text}", flush=True)

    trained_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    split = f"未知从{cutoff}起，拟合只用更早的日子，不含新闻"
    daily_names = train.sina_feature_names(KEY, "日线回归")
    daily = fit_family(sina_rows, daily_names, KEY, "logit", cutoff, train.saved_train_params(KEY), "sina")
    fit, names, _source = daily
    train.save_contract(
        KEY,
        {
            "weight": fit["weight"],
            "bias": fit["bias"],
            "mean": fit["mean"],
            "scale": fit["scale"],
            "feature_names": list(names),
            "train_params": train.saved_train_params(KEY) or {"lr": 0.08, "steps": 500, "l2": 0.2},
            "test_accuracy": combo["方向命中率"],
        },
        trained_at=trained_at,
        split=split,
    )

    tree_names = train.sina_feature_names(KEY, "日线树")
    tree_rows = [row for row in sina_rows if row["day"] < cutoff and all(name in row["named"] for name in tree_names)]
    spec = {}
    meta_path = mail_tree.OUT / f"{KEY}.json"
    if meta_path.exists():
        spec = json.loads(meta_path.read_text()).get("train_params") or {}
    params = dict(mail_tree.PARAMS)
    params["max_depth"] = int(spec.get("max_depth", params["max_depth"]))
    params["num_leaves"] = int(spec.get("num_leaves", 2 ** params["max_depth"]))
    params["learning_rate"] = float(spec.get("learning_rate", params["learning_rate"]))
    params["min_child_samples"] = int(spec.get("min_child_samples", params["min_child_samples"]))
    params["lambda_l2"] = float(spec.get("lambda_l2", params["lambda_l2"]))
    rounds = int(spec.get("num_boost_round", mail_tree.ROUNDS))
    raw = [[row["named"][name] for name in tree_names] for row in tree_rows]
    mean, scale = train.standardize(raw)
    matrix = np.array([train.apply_scale(row, mean, scale) for row in raw])
    booster = lgb.train(
        params,
        lgb.Dataset(
            matrix,
            label=np.array([row["y"][KEY] for row in tree_rows]),
            weight=np.array([train.sample_weight(row["move"][KEY]) for row in tree_rows]),
        ),
        num_boost_round=rounds,
    )
    booster.save_model(str(mail_tree.OUT / f"{KEY}.txt"))
    body = json.loads(meta_path.read_text()) if meta_path.exists() else {"code": KEY, "name": train.NAMES[KEY]}
    body.update(
        {
            "kind": "lightgbm",
            "trained_at": trained_at,
            "source": "sina-daily",
            "split": split,
            "feature_names": list(tree_names),
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
    meta_path.write_text(json.dumps(body, ensure_ascii=False, indent=2))

    minute_train = [row for row in min_rows if row["day"] < cutoff and KEY in row["y"]]
    names, mean, scale, weight, bias, booster = minute.fit_pair(minute_train, KEY, True)
    minute.save(KEY, names, mean, scale, weight, bias, booster)
    for folder in (minute.LOGIT_OUT, minute.TREE_OUT):
        path = folder / f"{KEY}.json"
        saved = json.loads(path.read_text())
        saved["split"] = split
        saved["trained_at"] = trained_at
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))

    report = {
        "品种": "豆一",
        "不含": ["新闻倾向", "新闻三日"],
        "未知从": cutoff,
        "未知到": last,
        "综合": scored["去掉新闻"]["综合"],
        "家族": scored["去掉新闻"]["家族"],
    }
    out = Path(__file__).resolve().parent / "豆一最近一年.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写入豆一四个模型，未知段成绩在 {out}", flush=True)


if __name__ == "__main__":
    main()
