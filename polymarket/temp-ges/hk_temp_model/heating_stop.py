#!/usr/bin/env python3
"""Heating plus clock time as a Hong Kong stop. Recorded 2026-10-15. Not a live rule.

Idea: if the afternoon heating stays weak and the held Yes degree is still
unprinted, sell before settlement instead of losing the whole five dollars.
Sunny afternoons can still rise, so a high or rising heating factor should
not be sold.

Bar for changing the live model: losing less counts. The sale does not have
to raise total profit. It still must not give away a later winner such as
6 Oct Yes 28°C just to trim a few cents off a ticket the market has already
marked near zero. This file only replays. decide.py does not call it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CASES = Path(__file__).resolve().parent / "data" / "cases.jsonl"
PAPER = Path(__file__).resolve().parent / "data" / "paper_trades.json"


def _hour(stamp: str) -> float:
    hm = stamp.split()[1]
    hh, mm = (int(part) for part in hm.split(":")[:2])
    return hh + mm / 60.0


def _load_series() -> dict[str, list[tuple]]:
    series: dict[str, list[tuple]] = defaultdict(list)
    if not CASES.exists():
        return series
    with CASES.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("city") != "hk" or row.get("kind") != "forecast":
                continue
            stamp = row.get("observed_at") or ""
            if "HKT" not in stamp:
                continue
            inputs = row.get("inputs") or {}
            outputs = row.get("outputs") or {}
            series[stamp.split()[0]].append(
                (
                    _hour(stamp),
                    inputs.get("hko_max_c"),
                    outputs.get("heating"),
                    row.get("bids") or {},
                )
            )
    return series


def _degree(label: str) -> float | None:
    text = label.replace("°C", "").strip()
    if text[:1].isdigit() and "or" not in text and "-" not in text:
        return float(text.split()[0])
    return None


def _at(points: list[tuple], hour: float) -> tuple | None:
    earlier = [row for row in points if row[0] <= hour]
    return earlier[-1] if earlier else None


def replay() -> None:
    series = _load_series()
    paper = json.loads(PAPER.read_text()) if PAPER.exists() else []
    rows = []
    print("香港 Yes，14:00 还没印到这一档")
    print(f"{'日期':<12}{'档':<8}{'到底':>8}{'12点加热':>8}{'14点加热':>8}{'差':>6}{'14点买价':>8}")
    for row in paper:
        if row.get("city") != "hk" or row.get("side") != "YES" or row.get("pnl") is None:
            continue
        degree = _degree(row["label"])
        points = series.get(row["day"])
        if degree is None or not points:
            continue
        noon = _at(points, 12)
        at14 = _at(points, 14)
        if at14 is None or at14[1] is None or at14[1] >= degree:
            continue
        bid = (at14[3] or {}).get(row["label"])
        heat12 = None if noon is None else noon[2]
        heat14 = at14[2]
        gap = degree - at14[1]
        held = row["pnl"]
        sold = None if bid is None else round(float(row["shares"]) * float(bid) - float(row["usd"]), 2)
        rows.append((row["day"], row["label"], held, heat12, heat14, gap, bid, sold))
        print(
            f"{row['day']:<12}{row['label']:<8}{held:+8.2f}"
            f"{'' if heat12 is None else f'{heat12:8.2f}'}{'' if heat14 is None else f'{heat14:8.2f}'}"
            f"{gap:6.1f}{'' if bid is None else f'{float(bid):8.2f}'}"
        )
    print(f"样本 {len(rows)} 笔。")
    # Sell only when noon-to-14:00 heating stays under 0.40 and a bid still exists.
    weak = [row for row in rows if row[3] is not None and row[4] is not None and max(row[3], row[4]) < 0.40 and row[7] is not None]
    rising = [row for row in rows if row[3] is not None and row[4] is not None and row[4] > row[3] + 0.05]
    weak_delta = sum(row[7] - row[2] for row in weak)
    rising_delta = sum((row[7] if row[7] is not None else row[2]) - row[2] for row in rising)
    print(f"加热中午到14点都低于0.40：{len(weak)} 笔，提前卖相对拿到结算 {weak_delta:+.2f} 美元")
    print(f"加热从中午到14点还在上升：{len(rising)} 笔，若在14点卖掉，相对拿到结算 {rising_delta:+.2f} 美元")
    print("优化标准包括少亏损，不要求总盈亏变高。")
    print("加热低于0.40的亏损单，14点买价已经接近0，少亏大约0.05美元。")
    print("要再多捞亏损单，就会卖掉加热还在上升、后来赚钱的单。10月6日 Yes 28°C 是 +17.73。")
    print("未达改进现模型的标准。decide.py 不使用本文件。")


if __name__ == "__main__":
    replay()
