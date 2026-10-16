#!/usr/bin/env python3
"""Hold-the-wave model for 豆一 / 豆粕 / 菜粕.

A rally is confirmed when the close rises 3% from a low. A decline is
confirmed when the close falls 3% from a high. Bounces inside that band do
not count. Entry is the confirming close. Exit is the close that confirms
the opposite turn. The daily bar is the 15:00 close.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAMES = {"A": "豆一", "M": "豆粕", "RM": "菜粕"}
REV = 0.03


def waves(days: list[str], closes: list[float], rev: float = REV) -> dict:
    """Completed trades plus the wave still open on the last close."""

    done: list[dict] = []
    if len(closes) < 2:
        return {"done": done, "open": None}
    trough_i = 0
    peak_i = 0
    mode = "seek_trough"
    entry_i = None
    side = None
    for i in range(1, len(closes)):
        if mode == "seek_trough":
            if closes[i] <= closes[trough_i]:
                trough_i = i
            elif closes[trough_i] and (closes[i] - closes[trough_i]) / closes[trough_i] >= rev:
                if side == "空" and entry_i is not None:
                    done.append(_trade(days, closes, entry_i, i, "空", trough_i))
                peak_i = i
                entry_i = i
                side = "多"
                mode = "seek_peak"
        elif closes[i] >= closes[peak_i]:
            peak_i = i
        elif closes[peak_i] and (closes[peak_i] - closes[i]) / closes[peak_i] >= rev:
            if side == "多" and entry_i is not None:
                done.append(_trade(days, closes, entry_i, i, "多", peak_i))
            trough_i = i
            entry_i = i
            side = "空"
            mode = "seek_trough"
    open_trade = None
    if side and entry_i is not None:
        last = len(closes) - 1
        if side == "多":
            stop = closes[peak_i] * (1 - rev)
            anchor_i = trough_i
        else:
            stop = closes[trough_i] * (1 + rev)
            anchor_i = peak_i
        open_trade = {
            "side": side,
            "entry_day": days[entry_i],
            "entry": closes[entry_i],
            "anchor_day": days[anchor_i],
            "anchor": closes[anchor_i],
            "extreme_day": days[peak_i if side == "多" else trough_i],
            "extreme": closes[peak_i if side == "多" else trough_i],
            "last_day": days[last],
            "last": closes[last],
            "exit_price": stop,
            "held_days": last - entry_i,
            "open_pct": (closes[last] - closes[entry_i]) / closes[entry_i] * 100.0
            if side == "多"
            else (closes[entry_i] - closes[last]) / closes[entry_i] * 100.0,
        }
    return {"done": done, "open": open_trade}


def _trade(days, closes, entry_i, exit_i, side, extreme_i) -> dict:
    entry = closes[entry_i]
    exit_px = closes[exit_i]
    pnl = (exit_px - entry) / entry * 100.0 if side == "多" else (entry - exit_px) / entry * 100.0
    return {
        "side": side,
        "entry_day": days[entry_i],
        "exit_day": days[exit_i],
        "days": exit_i - entry_i,
        "pnl": pnl,
        "extreme_day": days[extreme_i],
        "extreme": closes[extreme_i],
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def report(key: str) -> None:
    rows = json.loads((DATA / f"{key}.json").read_text())
    rows = [row for row in rows if row[0] >= "2017-01-01"]
    days = [row[0] for row in rows]
    closes = [row[1] for row in rows]
    book = waves(days, closes)
    done = book["done"]
    wins = [row for row in done if row["pnl"] > 0]
    print(f"\n{NAMES[key]}  已走完 {len(done)} 段")
    if done:
        print(
            f"  赢 {len(wins)}/{len(done)}  "
            f"中位收益 {_median([row['pnl'] for row in done]):+.1f}%  "
            f"中位持有 {_median([row['days'] for row in done]):.0f} 个交易日"
        )
    cur = book["open"]
    if not cur:
        print("  现在没有持仓")
        return
    print(
        f"  现在{cur['side']}  入场 {cur['entry_day']} 15:00 收盘 {cur['entry']:.0f}"
    )
    print(
        f"  这段从 {cur['anchor_day']} 的 {cur['anchor']:.0f} 确认，"
        f"极值 {cur['extreme_day']} {cur['extreme']:.0f}"
    )
    print(
        f"  最近收盘 {cur['last_day']} {cur['last']:.0f}  "
        f"浮动 {cur['open_pct']:+.1f}%  已持有 {cur['held_days']} 个交易日"
    )
    print(
        f"  离场：收盘触及 {cur['exit_price']:.0f} 的那个交易日 15:00。"
        f"在这之前不走"
    )


def main() -> None:
    print("回头 3% 确认一段。入场和离场都在当日 15:00 收盘。")
    for key in ("A", "M", "RM"):
        report(key)


if __name__ == "__main__":
    main()
