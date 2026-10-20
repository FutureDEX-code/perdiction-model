#!/usr/bin/env python3
"""Paper ledger for the next-session futures rule. Nothing is sent.

Ten lots. Long only at 55% or above, short only at 45% or below.
Enter at the prior close, leave at the next 15:00 close.
"""

from __future__ import annotations

import json
from pathlib import Path

import train

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LEDGER = DATA / "paper_trades.json"
LOTS = 10
TONS = 10
NAMES = {"A": "豆一", "B": "豆二", "M": "豆粕", "RM": "菜粕", "Y": "豆油"}


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return json.loads(LEDGER.read_text())


def _save(rows: list[dict]) -> None:
    LEDGER.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def _pnl(side: str, entry: float, exit_px: float) -> float:
    sign = 1.0 if side == "多" else -1.0
    return round(sign * (exit_px - entry) * TONS * LOTS, 2)


def simulate_month(month: str) -> list[dict]:
    """One calendar month. The model is fit only on days before that month."""

    series = {
        key: {day: float(close) for day, close in json.loads((DATA / f"{key}.json").read_text())}
        for key in ("A", "M", "RM", "S", "SM", "BO")
    }
    rows = [row for row in train.samples(series, train.news_scores()) if row["day"] >= "2017-01-01"]
    start = f"{month}-01"
    end = f"{month}-31"
    train_rows = [row for row in rows if row["day"] < start]
    test = [row for row in rows if start <= row["day"] <= end]
    prev_close = {}
    for key in ("A", "M", "RM"):
        days = sorted(series[key])
        for left, right in zip(days, days[1:]):
            prev_close[(key, right)] = series[key][left]
    models = {}
    for key in ("A", "M", "RM"):
        mean, scale = train.standardize([row["x"] for row in train_rows])
        weight, bias = train.fit(
            [train.apply_scale(row["x"], mean, scale) for row in train_rows],
            [row["y"][key] for row in train_rows],
            [train.sample_weight(row["move"][key]) for row in train_rows],
        )
        models[key] = (mean, scale, weight, bias)
    booked = []
    for row in test:
        for key in ("A", "M", "RM"):
            mean, scale, weight, bias = models[key]
            prob = train.predict_prob(train.apply_scale(row["x"], mean, scale), weight, bias)
            if prob >= 0.55:
                side = "多"
            elif prob <= 0.45:
                side = "空"
            else:
                continue
            entry = prev_close[(key, row["day"])]
            exit_px = series[key][row["day"]]
            booked.append(
                {
                    "code": key,
                    "name": NAMES[key],
                    "day": row["day"],
                    "side": side,
                    "prob": round(prob, 4),
                    "lots": LOTS,
                    "entry": entry,
                    "exit": exit_px,
                    "pnl": _pnl(side, entry, exit_px),
                }
            )
    return booked


def record_outlook(view: dict, series: dict[str, dict[str, float]]) -> list[str]:
    """Settle finished paper days, then write the next session from this outlook."""

    rows = _load()
    notes = []
    for row in rows:
        if row.get("pnl") is not None:
            continue
        later = sorted(day for day in series[row["code"]] if day > row["signal_day"])
        if not later:
            continue
        exit_day = later[0]
        exit_px = series[row["code"]][exit_day]
        row["day"] = exit_day
        row["exit"] = exit_px
        row["pnl"] = _pnl(row["side"], row["entry"], exit_px)
        notes.append(
            f"模拟结算 {row['name']} {exit_day} {row['side']} {row['pnl']:+.0f} 元"
        )
    signal = view["as_of"]
    rows = [row for row in rows if not (row.get("pnl") is None and row.get("signal_day") == signal)]
    overseas = view.get("overseas_pct") or []
    overseas_note = ""
    if len(overseas) == 3:
        overseas_note = f"  参考美豆 {overseas[0]:+.2f}% 美豆粕 {overseas[1]:+.2f}% 美豆油 {overseas[2]:+.2f}%"
    for item in view["views"]:
        if item["code"] not in NAMES:
            continue
        if item["lean"] not in ("偏涨", "偏跌"):
            continue
        side = "多" if item["lean"] == "偏涨" else "空"
        entry = series[item["code"]][signal]
        rows.append(
            {
                "code": item["code"],
                "name": item["name"],
                "signal_day": signal,
                "day": None,
                "side": side,
                "prob": item["prob_up"],
                "lots": LOTS,
                "entry": entry,
                "exit": None,
                "pnl": None,
            }
        )
        notes.append(
            f"模拟下单 {item['name']} {side} {LOTS}手  信号 {signal} 收盘 {entry:.0f}  "
            f"上涨概率 {item['prob_up']*100:.1f}%{overseas_note}"
        )
    _save(rows)
    return notes


def _summary(booked: list[dict]) -> str:
    lines = []
    total = 0.0
    wins = 0
    for key in ("A", "B", "M", "RM", "Y"):
        xs = [row for row in booked if row["code"] == key]
        pnl = sum(row["pnl"] for row in xs)
        won = sum(1 for row in xs if row["pnl"] > 0)
        total += pnl
        wins += won
        lines.append(f"{NAMES[key]} {len(xs)} 笔 赢 {won} 盈亏 {pnl:+,.0f} 元")
    lines.append(f"合计 {len(booked)} 笔 赢 {wins} 盈亏 {total:+,.0f} 元")
    return "\n".join(lines)


if __name__ == "__main__":
    month = "2026-09"
    booked = simulate_month(month)
    print(_summary(booked))
    for row in booked:
        print(
            f"{row['day']} {row['name']} {row['side']} {row['lots']}手 "
            f"@{row['entry']:.0f}→{row['exit']:.0f} {row['pnl']:+,.0f} 元  {row['prob']*100:.1f}%"
        )
