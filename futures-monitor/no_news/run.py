#!/usr/bin/env python3
"""Same live models with the news column removed. Does not write data/models/.

Two score windows. Models for a window are fit only on earlier days.
Combo rule, 45/55 band, and 只反向平 stay as they are in 特征.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from models import fit_sina_main as minute
from models import stack
from path_train.run import records_for, run_path, score_combo

OUT = Path(__file__).resolve().parent
NEWS = ("新闻倾向", "新闻三日")
FAMILIES = stack.FAMILIES
WINDOWS = (
    ("2024-10-20", "2025-10-20"),
    ("2025-10-20", None),
)


def without(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if name not in NEWS)


def fit_family(rows: list[dict], names: tuple[str, ...], product: str, kind: str, before: str, spec: dict, source: str):
    usable = [row for row in rows if row["day"] < before and product in row.get("y", {}) and all(name in row["named"] for name in names)]
    if len(usable) < 200:
        return None
    vec = lambda row, names=names: [row["named"][name] for name in names]
    y = lambda row, product=product: row["y"][product]
    move = lambda row, product=product: row["move"][product]
    fit = stack._fit_tree(usable, vec, y, move, spec) if kind == "tree" else stack._fit_logit(usable, vec, y, move, spec)
    return fit, names, source


def probs_for(fits: dict, rows: list[dict], minute_by_anchor: dict) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        minute_named = (minute_by_anchor.get(row["anchor"]) or {}).get("named") or {}
        slot = {}
        for family, packed in fits.items():
            if packed is None:
                continue
            fit, names, source = packed
            named = row["named"] if source == "sina" else minute_named
            if any(name not in named for name in names):
                continue
            slot[family] = stack._score(fit, [named[name] for name in names])
        if slot:
            table[row["anchor"]] = slot
    return table


def one_window(product: str, sina_rows, min_rows, minute_by_anchor: dict, records, lo: str, hi: str | None) -> dict:
    rule = train.combo_rule(product)
    score_rows = [row for row in records if row["anchor"] >= lo and (hi is None or row["anchor"] < hi)]
    out = {"规则": rule, "样本天数": len(score_rows), "家族": {}}
    for label, drop in (("有新闻", False), ("去掉新闻", True)):
        fits = {}
        for family, kind, source, folder in (
            ("日线回归", "logit", sina_rows, "models"),
            ("日线树", "tree", sina_rows, "models_tree"),
            ("分钟回归", "logit", min_rows, "models_1450"),
            ("分钟树", "tree", min_rows, "models_tree_1450"),
        ):
            tag = "sina" if source is sina_rows else "minute"
            names = train.sina_feature_names(product, family) if tag == "sina" else tuple(minute.minute_names(product, family))
            if drop:
                names = without(names)
            fits[family] = fit_family(source, names, product, kind, lo, stack._spec(product, folder), tag)
        table = probs_for(fits, score_rows, minute_by_anchor)
        path = score_combo(score_rows, table, rule, 0.55, 0.45, 2, None)
        families = {}
        for family, packed in fits.items():
            if packed is None:
                continue
            alone = {anchor: {family: probs[family]} for anchor, probs in table.items() if family in probs}
            families[family] = run_path(
                score_rows,
                lambda row, alone=alone, family=family: (
                    1 if alone.get(row["anchor"], {}).get(family, 0.5) >= 0.55
                    else -1 if alone.get(row["anchor"], {}).get(family, 0.5) <= 0.45
                    else 0
                ),
            )
        out[label] = {"综合": path, "家族": {name: {"净盈亏": item["净盈亏"], "方向命中率": item["方向命中率"], "持仓命中率": item["持仓命中率"], "有方向天数": item["有方向天数"]} for name, item in families.items()}}
    base = out["有新闻"]["综合"]["净盈亏"]
    alt = out["去掉新闻"]["综合"]["净盈亏"]
    out["综合差额万元"] = round((alt - base) / 10000.0, 2)
    return out


def main() -> None:
    print("读数据", flush=True)
    series = train.load_series()
    news = train.news_scores()
    sina_rows = [row for row in train.samples(series, news) if row["day"] >= "2017-01-01"]
    days = {product: minute.load_days(product) for product in train.CONTRACTS}
    min_rows = minute.build_rows(days, {key: series[key] for key in train.OVERSEAS}, news)
    minute_by_anchor = {row["signal"]: row for row in min_rows}
    report = {"说明": "只去掉新闻倾向或新闻三日。综合规则、45/55、只反向平不变。不写 data/models。", "窗口": {}}
    for lo, hi in WINDOWS:
        label = f"{lo} → {hi or '最新'}"
        print(label, flush=True)
        block = {}
        for product in train.CONTRACTS:
            print(f"  {train.NAMES[product]}", flush=True)
            records = records_for(product, sina_rows, series[product])
            block[product] = one_window(product, sina_rows, min_rows, minute_by_anchor, records, lo, hi)
            item = block[product]
            print(
                f"    有新闻 {item['有新闻']['综合']['净盈亏']/10000:.2f} 万"
                f"  去掉 {item['去掉新闻']['综合']['净盈亏']/10000:.2f} 万"
                f"  差额 {item['综合差额万元']:+.2f} 万",
                flush=True,
            )
        report["窗口"][label] = block
    (OUT / "结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写入 {OUT / '结果.json'}", flush=True)


if __name__ == "__main__":
    main()
