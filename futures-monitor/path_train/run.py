#!/usr/bin/env python3
"""Train once against the live position path. Does not write data/models/.

The score is the 只反向平 path: a direction opens or reverses, no direction
keeps the current position. Ten lots, ten tons. Fee is 2 yuan per lot per side,
so an open costs 20 yuan and a reverse costs 40.

Parameters are chosen on anchors in [2022-10-20, 2024-10-20), with models fit
only on outcome days before 2022-10-20. The chosen recipe is refit on outcome
days before 2024-10-20 and scored once on [2024-10-20, 2025-10-20). Later days
are reported and are not used to choose anything.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train
from models import fit_sina_main as minute
from models import stack

OUT = Path(__file__).resolve().parent
SELECT_LO = "2022-10-20"
SELECT_HI = "2024-10-20"
SCORE_HI = "2025-10-20"
YUAN = 100
FEE_SIDE = 20.0  # 10 lots * 2 yuan
BANDS = ((0.52, 0.48), (0.55, 0.45), (0.58, 0.42), (0.62, 0.38))
TAUS = (0.02, 0.05, 0.10)
FAMILIES = stack.FAMILIES
POLICY = {"Y": "日线回归", "RM": "日线树", "B": "日线树"}


def _sigmoid(z: float) -> float:
    if z > 20:
        return 1.0
    if z < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def records_for(product: str, sina_rows: list[dict], px: dict[str, float]) -> list[dict]:
    """One decision per sample. Features are known at the anchor close; the move ends on row['day']."""

    common = sorted({row["day"] for row in sina_rows})
    prev_of = {day: common[i - 1] for i, day in enumerate(common) if i}
    out = []
    for row in sina_rows:
        if product not in row.get("y", {}):
            continue
        anchor = prev_of.get(row["day"])
        if anchor is None or anchor not in px or row["day"] not in px:
            continue
        out.append(
            {
                "anchor": anchor,
                "day": row["day"],
                "named": row["named"],
                "y": row["y"][product],
                "move": row["move"][product],
                "entry": px[anchor],
                "exit": px[row["day"]],
            }
        )
    return out


def window(rows: list[dict], lo: str, hi: str | None) -> list[dict]:
    if hi is None:
        return [row for row in rows if row["anchor"] >= lo]
    return [row for row in rows if lo <= row["anchor"] < hi]


def run_path(rows: list[dict], signal_of) -> dict:
    pos = 0
    pnl = 0.0
    fees = 0.0
    flips = 0
    directed = 0
    signal_hits = 0
    held_days = 0
    held_hits = 0
    for row in rows:
        signal = signal_of(row)
        new = pos if signal == 0 else signal
        if new != pos:
            fee = FEE_SIDE * abs(new - pos)
            fees += fee
            pnl -= fee
            flips += 1
        up = row["exit"] > row["entry"]
        if signal != 0:
            directed += 1
            signal_hits += int((signal == 1 and up) or (signal == -1 and not up))
        if new != 0:
            held_days += 1
            held_hits += int((new == 1 and up) or (new == -1 and not up))
        pnl += new * (row["exit"] - row["entry"]) * YUAN
        pos = new
    return {
        "净盈亏": round(pnl, 0),
        "手续费": round(fees, 0),
        "反手或开仓": flips,
        "有方向天数": directed,
        "方向命中": signal_hits,
        "方向命中率": round(signal_hits / directed, 4) if directed else None,
        "持仓天数": held_days,
        "持仓命中": held_hits,
        "持仓命中率": round(held_hits / held_days, 4) if held_days else None,
        "样本天数": len(rows),
        "期末仓位": pos,
    }


def single_signal(prob: float, up: float, down: float) -> int:
    if prob >= up:
        return 1
    if prob <= down:
        return -1
    return 0


def combo_signal(probs: dict[str, float], rule: str, up: float, down: float, need: int, drop: str | None) -> int:
    used = {family: probs.get(family, 0.5) for family in FAMILIES if family != drop}
    if rule == "多数票2":
        ups = sum(prob >= up for prob in used.values())
        downs = sum(prob <= down for prob in used.values())
        if ups >= need and ups > downs:
            return 1
        if downs >= need and downs > ups:
            return -1
        return 0
    if rule == "平均":
        if not used:
            return 0
        return single_signal(sum(used.values()) / len(used), up, down)
    return single_signal(used.get(rule, probs.get(rule, 0.5)), up, down)


def fit_logit(rows: list[dict], names: tuple[str, ...], weights: list[float]) -> dict:
    mean, scale = train.standardize([[row["named"][name] for name in names] for row in rows])
    xs = [train.apply_scale([row["named"][name] for name in names], mean, scale) for row in rows]
    weight, bias = train.fit(xs, [row["y"] for row in rows], weights)
    return {"kind": "logit", "mean": mean, "scale": scale, "weight": weight, "bias": bias, "names": names}


def fit_tree(rows: list[dict], names: tuple[str, ...], weights: list[float], spec: dict) -> dict:
    mean, scale = train.standardize([[row["named"][name] for name in names] for row in rows])
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
            np.array([train.apply_scale([row["named"][name] for name in names], mean, scale) for row in rows]),
            label=np.array([row["y"] for row in rows]),
            weight=np.array(weights, dtype=float),
        ),
        num_boost_round=int(spec.get("num_boost_round", 100)),
    )
    return {"kind": "tree", "mean": mean, "scale": scale, "booster": booster, "names": names}


def predict_one(model: dict, row: dict) -> float | None:
    names = model["names"]
    if any(name not in row["named"] for name in names):
        return None
    scaled = train.apply_scale([row["named"][name] for name in names], model["mean"], model["scale"])
    if model["kind"] == "logit":
        return train.predict_prob(scaled, model["weight"], model["bias"])
    return float(model["booster"].predict([scaled])[0])


def base_weights(rows: list[dict]) -> list[float]:
    return [train.sample_weight(row["move"]) for row in rows]


def cost_weights(rows: list[dict]) -> list[float]:
    out = []
    for row in rows:
        notional = row["entry"] * YUAN
        hurdle = FEE_SIDE / notional * 100.0 if notional else 0.0
        out.append(0.0 if abs(row["move"]) <= hurdle else min(abs(row["move"]), 2.0))
    return out


def reweighted(rows: list[dict], model: dict, weights: list[float]) -> list[float]:
    """Double a training day when the position it would hold loses more than one fee."""

    pos = 0
    adjusted = list(weights)
    for index, row in enumerate(rows):
        prob = predict_one(model, row)
        signal = 0 if prob is None else single_signal(prob, 0.55, 0.45)
        new = pos if signal == 0 else signal
        held = new * (row["exit"] - row["entry"]) * YUAN
        if held < -FEE_SIDE:
            adjusted[index] = min(adjusted[index] * 2.0 if adjusted[index] else 2.0, 4.0)
        pos = new
    return adjusted


def soft_fit(rows: list[dict], names: tuple[str, ...], tau: float, warm: dict) -> dict:
    """Gradient steps on a soft position path. The hard rule is only used at scoring time."""

    mean, scale = warm["mean"], warm["scale"]
    xs = [train.apply_scale([row["named"][name] for name in names], mean, scale) for row in rows]
    rets = [(row["exit"] - row["entry"]) / row["entry"] for row in rows]
    fee_fracs = [FEE_SIDE / (row["entry"] * YUAN) for row in rows]
    dim = len(xs[0])
    weight = list(warm["weight"])
    bias = warm["bias"]
    lr = 0.03
    for _ in range(60):
        poss = []
        gates = []
        dirs = []
        ps = []
        pos = 0.0
        for x in xs:
            z = bias + sum(weight[j] * x[j] for j in range(dim))
            prob = _sigmoid(z)
            centered = prob - 0.5
            direction = math.tanh(centered / tau)
            gate = _sigmoid((abs(centered) - 0.05) / tau)
            new = gate * direction + (1.0 - gate) * pos
            ps.append(prob)
            gates.append(gate)
            dirs.append(direction)
            poss.append((pos, new))
            pos = new
        grad_w = [0.0] * dim
        grad_b = 0.0
        carry = 0.0
        for index in reversed(range(len(xs))):
            old, new = poss[index]
            prob = ps[index]
            gate = gates[index]
            direction = dirs[index]
            centered = prob - 0.5
            change_sign = 0.0 if new == old else (1.0 if new > old else -1.0)
            d_new = -rets[index] + fee_fracs[index] * change_sign + carry
            d_gate = d_new * (direction - old)
            d_dir = d_new * gate
            d_old = d_new * (1.0 - gate)
            d_centered = d_dir * (1.0 - direction * direction) / tau
            sign_c = 0.0 if centered == 0 else (1.0 if centered > 0 else -1.0)
            d_centered += d_gate * gate * (1.0 - gate) / tau * sign_c
            d_z = d_centered * prob * (1.0 - prob)
            if d_z > 1.0:
                d_z = 1.0
            elif d_z < -1.0:
                d_z = -1.0
            grad_b += d_z
            x = xs[index]
            for j in range(dim):
                grad_w[j] += d_z * x[j]
            carry = d_old * 0.95
        n = float(len(xs))
        for j in range(dim):
            step = lr * (grad_w[j] / n + 0.05 * weight[j])
            if step > 0.05:
                step = 0.05
            elif step < -0.05:
                step = -0.05
            weight[j] -= step
        bias -= max(-0.05, min(0.05, lr * grad_b / n))
    return {"kind": "logit", "mean": mean, "scale": scale, "weight": weight, "bias": bias, "names": names}


def usable(rows: list[dict], names: tuple[str, ...], before: str) -> list[dict]:
    return [row for row in rows if row["day"] < before and all(name in row["named"] for name in names)]


def best_band(rows: list[dict], probs: dict[str, float]) -> tuple[tuple[float, float], dict]:
    winner = None
    chosen = BANDS[1]
    for up, down in BANDS:
        result = run_path(rows, lambda row, up=up, down=down: single_signal(probs.get(row["anchor"], 0.5), up, down))
        if winner is None or result["净盈亏"] > winner["净盈亏"]:
            winner = result
            chosen = (up, down)
    return chosen, winner


def prob_map(model: dict, rows: list[dict]) -> dict[str, float]:
    found = {}
    for row in rows:
        prob = predict_one(model, row)
        if prob is not None:
            found[row["anchor"]] = prob
    return found


def family_probs(fits: dict, rows: list[dict]) -> dict[str, dict[str, float]]:
    """anchor -> family -> probability, from the four stack fits."""

    out: dict[str, dict[str, float]] = {}
    for row in rows:
        slot = {}
        for family, (fit, names, _source) in fits.items():
            if any(name not in row["named"] for name in names):
                continue
            scaled = train.apply_scale([row["named"][name] for name in names], fit["mean"], fit["scale"])
            if fit["kind"] == "logit":
                slot[family] = train.predict_prob(scaled, fit["weight"], fit["bias"])
            else:
                slot[family] = float(fit["booster"].predict([scaled])[0])
        if slot:
            out[row["anchor"]] = slot
    return out


def score_combo(rows: list[dict], table: dict[str, dict[str, float]], rule: str, up: float, down: float, need: int, drop: str | None) -> dict:
    def signal(row, up=up, down=down, need=need, drop=drop):
        return combo_signal(table.get(row["anchor"], {}), rule, up, down, need, drop)

    return run_path(rows, signal)


def search_combo(rows: list[dict], table: dict[str, dict[str, float]], rule: str) -> tuple[dict, dict]:
    winner = None
    chosen = {"上沿": 0.55, "下沿": 0.45, "票数": 2, "去掉": None}
    needs = (2, 3) if rule == "多数票2" else (2,)
    drops = [None]
    if rule == "平均":
        drops = [None, *FAMILIES]
    for up, down in BANDS:
        for need in needs:
            for drop in drops:
                result = score_combo(rows, table, rule, up, down, need, drop)
                if winner is None or result["净盈亏"] > winner["净盈亏"]:
                    winner = result
                    chosen = {"上沿": up, "下沿": down, "票数": need, "去掉": drop}
    return chosen, winner


def policy_models(product: str, train_rows: list[dict], names: tuple[str, ...], kind: str) -> list[tuple[str, dict]]:
    spec = stack._spec(product, "models_tree" if kind == "tree" else "models")
    models = [("现行权重", fit_tree(train_rows, names, base_weights(train_rows), spec) if kind == "tree" else fit_logit(train_rows, names, base_weights(train_rows)))]
    cost = cost_weights(train_rows)
    if any(cost):
        cost_model = fit_tree(train_rows, names, cost, spec) if kind == "tree" else fit_logit(train_rows, names, cost)
        models.append(("成本权重", cost_model))
        doubled = reweighted(train_rows, cost_model, cost)
        models.append(("成本权重再加重", fit_tree(train_rows, names, doubled, spec) if kind == "tree" else fit_logit(train_rows, names, doubled)))
    warm = models[0][1] if kind == "logit" else fit_logit(train_rows, names, base_weights(train_rows))
    for tau in TAUS:
        models.append((f"软仓位 τ={tau}", soft_fit(train_rows, names, tau, warm)))
    return models


def pick_policy(product: str, rows: list[dict], names: tuple[str, ...], kind: str) -> tuple[str, tuple[float, float]]:
    train_rows = usable(rows, names, SELECT_LO)
    select_rows = window(rows, SELECT_LO, SELECT_HI)
    print(f"  {train.NAMES[product]} 选参 训练 {len(train_rows)} 天 验证 {len(select_rows)} 天", flush=True)
    best_name = "现行权重"
    chosen_band = (0.55, 0.45)
    best_pnl = None
    for name, model in policy_models(product, train_rows, names, kind):
        probs = prob_map(model, select_rows)
        band, result = best_band(select_rows, probs)
        print(f"    {name} 带 {band[0]:.2f}/{band[1]:.2f} 选参窗 {result['净盈亏'] / 10000:.2f} 万", flush=True)
        if best_pnl is None or result["净盈亏"] > best_pnl:
            best_pnl = result["净盈亏"]
            best_name = name
            chosen_band = band
    return best_name, chosen_band


def refit_policy(product: str, rows: list[dict], names: tuple[str, ...], kind: str, recipe: str) -> dict:
    train_rows = usable(rows, names, SELECT_HI)
    spec = stack._spec(product, "models_tree" if kind == "tree" else "models")
    if recipe == "现行权重":
        weights = base_weights(train_rows)
        model = fit_tree(train_rows, names, weights, spec) if kind == "tree" else fit_logit(train_rows, names, weights)
    elif recipe == "成本权重":
        weights = cost_weights(train_rows)
        model = fit_tree(train_rows, names, weights, spec) if kind == "tree" else fit_logit(train_rows, names, weights)
    elif recipe == "成本权重再加重":
        weights = cost_weights(train_rows)
        first = fit_tree(train_rows, names, weights, spec) if kind == "tree" else fit_logit(train_rows, names, weights)
        model = fit_tree(train_rows, names, reweighted(train_rows, first, weights), spec) if kind == "tree" else fit_logit(train_rows, names, reweighted(train_rows, first, weights))
    else:
        tau = float(recipe.split("=")[1])
        warm = fit_logit(train_rows, names, base_weights(train_rows))
        model = soft_fit(train_rows, names, tau, warm)
    return model


def pack(result: dict) -> dict:
    body = dict(result)
    body["净盈亏万元"] = round(result["净盈亏"] / 10000.0, 2)
    return body


def main() -> None:
    print("读日线和新闻", flush=True)
    series = train.load_series()
    news = train.news_scores()
    sina_rows = [row for row in train.samples(series, news) if row["day"] >= "2017-01-01"]
    print("读分钟线", flush=True)
    days = {product: minute.load_days(product) for product in train.CONTRACTS}
    min_rows = minute.build_rows(days, {key: series[key] for key in train.OVERSEAS}, news)
    report = {
        "费用": "2 元/手/边，10 手。开仓 20 元，反手 40 元。",
        "选参窗": f"{SELECT_LO} → {SELECT_HI}（不含结束日）",
        "计分窗": f"{SELECT_HI} → {SCORE_HI}（不含结束日）",
        "品种": {},
    }
    for product in train.CONTRACTS:
        print(f"拟合 {train.NAMES[product]}", flush=True)
        px = series[product]
        rows = records_for(product, sina_rows, px)
        rule = train.combo_rule(product)
        select_rows = window(rows, SELECT_LO, SELECT_HI)
        score_rows = window(rows, SELECT_HI, SCORE_HI)
        later_rows = window(rows, SCORE_HI, None)
        early = stack.fit_bases(product, sina_rows, min_rows, SELECT_LO)
        final = stack.fit_bases(product, sina_rows, min_rows, SELECT_HI)
        early_table = family_probs(early, select_rows)
        final_table = family_probs(final, score_rows + later_rows)
        current = {"上沿": 0.55, "下沿": 0.45, "票数": 2, "去掉": None, "规则": rule}
        baseline_score = score_combo(score_rows, final_table, rule, 0.55, 0.45, 2, None)
        baseline_later = score_combo(later_rows, final_table, rule, 0.55, 0.45, 2, None)
        if product in POLICY:
            kind = "logit" if POLICY[product] == "日线回归" else "tree"
            names = train.sina_feature_names(product, POLICY[product])
            recipe, band = pick_policy(product, rows, names, kind)
            model = refit_policy(product, rows, names, kind, recipe)
            probs = prob_map(model, score_rows + later_rows)
            chosen = {"配方": recipe, "上沿": band[0], "下沿": band[1], "模型": POLICY[product]}
            trained_score = run_path(score_rows, lambda row, band=band: single_signal(probs.get(row["anchor"], 0.5), band[0], band[1]))
            trained_later = run_path(later_rows, lambda row, band=band: single_signal(probs.get(row["anchor"], 0.5), band[0], band[1]))
        else:
            chosen, _select = search_combo(select_rows, early_table, rule)
            chosen = {"配方": "底座不改，只调综合", "规则": rule, **chosen}
            print(
                f"  {train.NAMES[product]} 选中 上沿 {chosen['上沿']} 下沿 {chosen['下沿']} 票数 {chosen['票数']} 去掉 {chosen['去掉'] or '无'}",
                flush=True,
            )
            trained_score = score_combo(score_rows, final_table, rule, chosen["上沿"], chosen["下沿"], chosen["票数"], chosen["去掉"])
            trained_later = score_combo(later_rows, final_table, rule, chosen["上沿"], chosen["下沿"], chosen["票数"], chosen["去掉"])
        entry = {
            "现行规则": current,
            "选中": chosen,
            "计分窗现行": pack(baseline_score),
            "计分窗路径训练": pack(trained_score),
            "计分窗差额万元": round((trained_score["净盈亏"] - baseline_score["净盈亏"]) / 10000.0, 2),
            "计分窗之后现行": pack(baseline_later),
            "计分窗之后路径训练": pack(trained_later),
            "之后差额万元": round((trained_later["净盈亏"] - baseline_later["净盈亏"]) / 10000.0, 2),
        }
        report["品种"][product] = entry
        print(
            f"  {train.NAMES[product]} 计分窗 现行 {entry['计分窗现行']['净盈亏万元']:.2f} 万"
            f"  路径 {entry['计分窗路径训练']['净盈亏万元']:.2f} 万"
            f"  差额 {entry['计分窗差额万元']:+.2f} 万",
            flush=True,
        )
    (OUT / "结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写入 {OUT / '结果.json'}", flush=True)


if __name__ == "__main__":
    main()
