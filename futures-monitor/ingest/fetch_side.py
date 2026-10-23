#!/usr/bin/env python3
"""Download the side data the daily model can actually backtest.

Domestic soybean oil, daily open interest, and the Jan/May/Sep curve.
The 5-minute file is only the short window Sina still serves. It is stored,
not used as a training feature.
"""

from __future__ import annotations

import sys

import json
from pathlib import Path

from core import train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
BARS = DATA / "bars"
CURVE = DATA / "contracts"
MINUTE = DATA / "minute"


def _daily_bars(symbol: str) -> list[dict]:
    url = (
        "https://stock.finance.sina.com.cn/futures/api/jsonp.php/var%20_/"
        f"InnerFuturesNewService.getDailyKLine?symbol={symbol}"
    )
    text = train._get(url)
    start, end = text.find("(["), text.rfind("])")
    if start < 0:
        return []
    bars = json.loads(text[start + 1 : end + 1])
    return [row for row in bars if row.get("d") and row.get("c")]


def _save_bars(path: Path, bars: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bars, ensure_ascii=False))


def main() -> None:
    BARS.mkdir(parents=True, exist_ok=True)
    CURVE.mkdir(parents=True, exist_ok=True)
    MINUTE.mkdir(parents=True, exist_ok=True)
    months = ("01", "05", "09")
    for prefix in ("M", "Y", "A", "B", "RM"):
        kept = 0
        for year in range(2016, 2028):
            for month in months:
                symbol = f"{prefix}{year % 100:02d}{month}"
                try:
                    bars = _daily_bars(symbol)
                except Exception as exc:
                    print(f"{symbol} 失败 {exc}", flush=True)
                    continue
                if len(bars) < 20:
                    continue
                _save_bars(CURVE / f"{symbol}.json", bars)
                kept += 1
        print(f"{prefix} 曲线合约 {kept}", flush=True)
    for key, symbol in (("A", "A0"), ("B", "B0"), ("M", "M0"), ("RM", "RM0"), ("Y", "Y0")):
        url = (
            "https://stock2.finance.sina.com.cn/futures/api/json.php/"
            f"InnerFuturesNewService.getFewMinLine?symbol={symbol}&type=5"
        )
        try:
            rows = json.loads(train._get(url))
        except Exception as exc:
            print(f"{symbol} 5分钟失败 {exc}", flush=True)
            continue
        if not rows:
            print(f"{symbol} 5分钟空", flush=True)
            continue
        _save_bars(MINUTE / f"{key}_5m.json", rows)
        print(f"{symbol} 5分钟 {rows[0]['d']} → {rows[-1]['d']}  {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
