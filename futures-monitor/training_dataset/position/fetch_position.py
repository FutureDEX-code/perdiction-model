#!/usr/bin/env python3
"""Exchange member position ranks, stored per product under training_dataset/position/.

菜粕、菜籽油、花生 are covered: the Zhengzhou exchange publishes a daily text
table anyone can read. The Dalian exchange blocks this network, so
豆一/豆二/豆粕/豆油/棕榈油 wait for a Tushare fut_holding permission on the token.

A record keeps the top-20 long and short totals with their day changes.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_dataset" / "position"
UA = {"User-Agent": "Mozilla/5.0"}
CZCE_TXT = "http://www.czce.com.cn/cn/DFSStaticFiles/Future/{year}/{day8}/FutureDataHolding.txt"
CZCE_PRODUCTS = {"RM": "菜粕RM", "OI": "菜油OI", "PK": "花生PK"}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _number(text: str) -> int:
    return int(text.replace(",", "").strip() or "0")


def _parse_variety(text: str, marker: str) -> dict | None:
    """The 合计 row of one variety block: volume, top-20 long, top-20 short, with changes."""

    head = f"品种：{marker}"
    start = text.find(head)
    if start < 0:
        return None
    end = text.find("品种：", start + len(head))
    block = text[start:end if end > 0 else len(text)]
    day = ""
    if "日期：" in block[:80]:
        day = block.split("日期：", 1)[1].strip()[:10]
    for line in block.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if cells and cells[0].startswith("合计") and len(cells) >= 10:
            long_ = _number(cells[5])
            short = _number(cells[8])
            long_chg = _number(cells[6])
            short_chg = _number(cells[9])
            return {
                "日期": day,
                "前20成交": _number(cells[2]),
                "前20多单": long_,
                "前20空单": short,
                "净多": long_ - short,
                "多单增减": long_chg,
                "空单增减": short_chg,
                "净变动": long_chg - short_chg,
            }
    return None


def fetch_day(day: str) -> dict[str, dict]:
    """One trading day, every covered product. Missing days raise."""

    day8 = day.replace("-", "")
    text = _get(CZCE_TXT.format(year=day[:4], day8=day8))
    out = {}
    for product, marker in CZCE_PRODUCTS.items():
        row = _parse_variety(text, marker)
        if row is not None:
            out[product] = row
    return out


def _merge(path: Path, rows: list[dict]) -> None:
    kept: dict[str, dict] = {}
    if path.exists():
        for row in json.loads(path.read_text()):
            kept[row["日期"]] = row
    for row in rows:
        kept[row["日期"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([kept[d] for d in sorted(kept)], ensure_ascii=False, indent=2))


def update(day: str | None = None) -> dict[str, dict]:
    """The 20:30 job calls this. Exchange tables land after the close."""

    from datetime import datetime
    from zoneinfo import ZoneInfo

    day = day or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    got = fetch_day(day)
    for product, row in got.items():
        _merge(OUT / f"{product}.json", [row])
    return got


def _have(product: str) -> set[str]:
    path = OUT / f"{product}.json"
    if not path.exists():
        return set()
    return {row["日期"] for row in json.loads(path.read_text())}


def seed(days: list[str]) -> None:
    """Fill days still missing from any covered product. Already-stored days are skipped."""

    have = {product: _have(product) for product in CZCE_PRODUCTS}
    pending: dict[str, list[dict]] = {product: [] for product in CZCE_PRODUCTS}
    total = 0
    asked = 0
    for index, day in enumerate(days, start=1):
        needed = [product for product in CZCE_PRODUCTS if product != "PK" or day >= "2021-02-01"]
        if all(day in have[product] for product in needed):
            continue
        asked += 1
        try:
            got = fetch_day(day)
        except Exception:
            continue
        for product, row in got.items():
            if row["日期"] in have[product]:
                continue
            pending[product].append(row)
            have[product].add(row["日期"])
            total += 1
        if asked % 100 == 0:
            for product, rows in pending.items():
                if rows:
                    _merge(OUT / f"{product}.json", rows)
                    pending[product] = []
            print(f"{index}/{len(days)} 新写入 {total}", flush=True)
        time.sleep(0.12)
    for product, rows in pending.items():
        if rows:
            _merge(OUT / f"{product}.json", rows)
    print(f"补历史完成 新写入 {total} 条", flush=True)


if __name__ == "__main__":
    import sys

    if "--seed" in sys.argv:
        sys.path.insert(0, str(ROOT))
        from core import train

        days = [day for day, _close in train._cached_rows(train.series_path("RM")) if day >= "2016-10-01"]
        seed(days)
    else:
        for product, row in update().items():
            print(f"{product} {row['日期']} 净多 {row['净多']} 变动 {row['净变动']:+d}", flush=True)
