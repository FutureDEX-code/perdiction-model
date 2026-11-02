#!/usr/bin/env python3
"""Build one aligned feature table per product. Does not touch the live models.

Each row is a 14:50 decision. Price, volume, and open interest come from the
main-minute file only. On a roll day the file already carries that contract's
own previous close, open interest, and volume. Other days use the previous
session in the same file. A missing trading day is not bridged. Overseas and
COT values are the latest already published by that decision day.
"""

from __future__ import annotations

import csv
import json
import sys
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train

HERE = Path(__file__).resolve().parent
PRODUCTS = ("A", "B", "M", "RM", "Y", "P", "OI", "PK")
NAMES = {"A": "豆一", "B": "豆二", "M": "豆粕", "RM": "菜粕", "Y": "豆油", "P": "棕榈油", "OI": "菜籽油", "PK": "花生"}
# Cross-product links. A spread is the swing of one roll-adjusted price index
# divided by the other, not the difference of two swing columns.
# 菜籽油不是单列的一条，和豆油共用菜油价差：豆油指数除以菜油指数。
# 这段: other products whose own swing is attached.
# 价差: (numerator, denominator, column). The level is the roll-adjusted
# price index of the numerator divided by the denominator.
LINKS = {
    "M": {"这段": (), "价差": (("M", "RM", "豆粕菜粕价差这段"),)},
    "RM": {"这段": ("M",), "价差": (("RM", "M", "菜粕豆粕价差这段"),)},
    "Y": {"这段": (), "价差": (("Y", "P", "豆棕价差这段"), ("Y", "OI", "菜油价差这段"))},
    "P": {"这段": ("Y",), "价差": (("Y", "P", "豆棕价差这段"),)},
    "A": {"这段": ("B",), "价差": ()},
    "B": {"这段": ("M",), "价差": ()},
    "PK": {"这段": ("M", "Y"), "价差": ()},
    "OI": {"这段": ("Y",), "价差": (("Y", "OI", "菜油价差这段"),)},
}
# One domestic open-interest change in the second side dish. The liquid
# crush contract is meal; soybean No.1 watches No.2, oils watch each other.
OI_REF = {"A": "B", "B": "M", "M": "RM", "RM": "M", "Y": "P", "P": "Y", "PK": "M", "OI": "Y"}
US = {"S": "美豆隔夜", "SM": "美豆粕隔夜", "BO": "美豆油隔夜"}
SOY = ("A", "B", "M", "Y")
CANOLA = ("RM", "OI")
SEAT = {"RM": "菜粕", "OI": "菜籽油", "PK": "花生"}


def _calendar(product: str) -> dict[str, str]:
    path = ROOT / "training_dataset" / "roll_calendar" / f"{product}.json"
    return json.loads(path.read_text())


def _trading_days(cal: dict[str, str]) -> list[str]:
    return sorted(cal)


def _session_day(days: list[str], stamp: str, hm: str) -> str | None:
    if hm >= "21:00":
        index = bisect_right(days, stamp)
        if index >= len(days):
            return None
        target = days[index]
        gap = (datetime.fromisoformat(target) - datetime.fromisoformat(stamp)).days
        return target if 0 < gap <= 15 else None
    if "09:00" <= hm <= "15:00" and stamp in set(days):
        return stamp
    return None


def _blank(value: str | None) -> bool:
    return value is None or str(value).strip() in ("", "None")


def _main_sessions(product: str, days: list[str]) -> dict[str, dict]:
    """14:50 snapshot of the tracked contract, plus the 14:55 price and intraday path."""

    dayset = set(days)
    sessions: dict[str, dict] = {}
    path = ROOT / "training_dataset" / "main_minute" / f"{product}.csv"
    with path.open() as handle:
        for row in csv.DictReader(handle):
            stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
            day = stamp if stamp in dayset and "09:00" <= hm <= "15:00" else None
            if day is None and hm >= "21:00":
                day = _session_day(days, stamp, hm)
            if day is None or _blank(row.get("close")):
                continue
            price = float(row["close"])
            slot = sessions.get(day)
            if slot is None:
                slot = {
                    "合约": row.get("contract") or "",
                    "开": float(row["open"]),
                    "高": price,
                    "低": price,
                    "收": None,
                    "收1455": None,
                    "合约1455": "",
                    "量": 0.0,
                    "仓": 0.0,
                    "路径": [],
                    "夜开": None,
                    "夜收": None,
                    "日开": None,
                    "昨收": None,
                    "昨仓": None,
                    "昨量": None,
                    "当日收": None,
                    "当日仓": None,
                    "当日量": None,
                }
                sessions[day] = slot
            if hm >= "21:00":
                if slot["夜开"] is None:
                    slot["夜开"] = float(row["open"])
                slot["夜收"] = price
            elif "09:00" <= hm <= "14:50" and slot["日开"] is None:
                slot["日开"] = float(row["open"])
            if hm >= "21:00" or hm <= "14:50":
                slot["高"] = max(slot["高"], float(row["high"]))
                slot["低"] = min(slot["低"], float(row["low"]))
                if hm <= "14:50":
                    slot["收"] = price
                if not _blank(row.get("vol")):
                    slot["量"] += float(row["vol"])
                if not _blank(row.get("oi")):
                    slot["仓"] = float(row["oi"])
                if not _blank(row.get("vol")) and float(row["vol"]) > 0:
                    slot["路径"].append((price, float(row["oi"] or 0)))
            if slot["收1455"] is None and "14:55" <= hm <= "15:00":
                slot["收1455"] = price
                slot["合约1455"] = row.get("contract") or ""
            if row.get("contract"):
                slot["合约"] = row["contract"]
            if row.get("昨收") and row.get("昨仓") and row.get("昨量"):
                slot["昨收"] = float(row["昨收"])
                slot["昨仓"] = float(row["昨仓"])
                slot["昨量"] = float(row["昨量"])
            if row.get("当日收"):
                slot["当日收"] = float(row["当日收"])
            if row.get("当日仓"):
                slot["当日仓"] = float(row["当日仓"])
            if row.get("当日量"):
                slot["当日量"] = float(row["当日量"])
    return sessions


def _closes(key: str) -> dict[str, float]:
    rows = json.loads(train.series_path(key).read_text())
    return {str(row["日期"]): float(row["收盘"]) for row in rows if row.get("收盘")}


def _cot_points(filename: str) -> list[tuple[str, float, float]]:
    path = ROOT / "training_dataset" / "cot" / filename
    pts = []
    prev = None
    for row in json.loads(path.read_text()):
        oi = row.get("总持仓") or 0
        if not oi:
            continue
        ratio = row["基金净多"] / oi * 100.0
        change = 0.0 if prev is None else ratio - prev
        available = (datetime.strptime(row["日期"], "%Y-%m-%d").date() + timedelta(days=4)).isoformat()
        pts.append((available, ratio, change))
        prev = ratio
    return pts


def _asof(points: list[tuple[str, float, float]], day: str, strict: bool) -> tuple[float, float]:
    ratio = change = 0.0
    for stamp, r, c in points:
        if (stamp >= day) if strict else (stamp > day):
            break
        ratio, change = r, c
    return ratio, change


def _seat_points(product: str) -> list[tuple[str, float, float]]:
    path = ROOT / "training_dataset" / "position" / f"{product}.json"
    pts = []
    for row in json.loads(path.read_text()):
        total = (row.get("前20多单") or 0) + (row.get("前20空单") or 0)
        if not total:
            continue
        pts.append((row["日期"], (row.get("净多") or 0) / total * 100.0, float(row.get("净变动") or 0)))
    return pts


def _previous(sessions: dict[str, dict], day: str, days: list[str]) -> dict | None:
    """Previous session in the main-minute file. A hole means no row."""

    earlier = [item for item in days if item < day and sessions.get(item) and sessions[item].get("收") is not None]
    if not earlier:
        return None
    prev_day = earlier[-1]
    if any(prev_day < item < day for item in days):
        return None
    slot = sessions[prev_day]
    return {"收": slot["收"], "量": slot["量"], "仓": slot["仓"]}


def _same_contract(slot: dict, nxt: dict, day: str, next_day: str, cal: dict[str, str]) -> bool:
    """A label may only join two 14:55 prices of one contract. A roll is left blank."""

    entry = slot.get("合约1455") or ""
    exit_code = nxt.get("合约1455") or ""
    if entry and exit_code:
        return entry == exit_code
    entry = entry or cal.get(day) or ""
    exit_code = exit_code or cal.get(next_day) or ""
    return bool(entry and exit_code and entry == exit_code)


def _oi_change(sessions: dict[str, dict], day: str, days: list[str]) -> float | None:
    """Same open-interest change as this product's own column, including roll days."""

    slot = sessions.get(day)
    if slot is None or _bad_session(slot):
        return None
    prev = _previous(sessions, day, days)
    if slot.get("昨收") and slot.get("昨仓") and slot.get("昨量"):
        prev = {"收": slot["昨收"], "仓": slot["昨仓"], "量": slot["昨量"]}
    if prev is None or not prev.get("仓"):
        return None
    oi = slot["当日仓"] if slot.get("当日仓") else slot["仓"]
    if not oi or oi <= 0 or prev["仓"] <= 0:
        return None
    return (oi / prev["仓"] - 1.0) * 100.0


def _ret(now: float | None, prev: float | None) -> float | None:
    if not now or not prev:
        return None
    return (now - prev) / prev * 100.0


def _attitude(path: list[tuple[float, float]]) -> float:
    num = den = 0.0
    prev = None
    for price, oi in path:
        if prev is not None:
            change = price - prev[0]
            oi_change = oi - prev[1]
            if change != 0 and oi_change != 0:
                num += (1.0 if change > 0 else -1.0) * oi_change
                den += abs(oi_change)
        prev = (price, oi)
    return num / den if den else 0.0


def _bad_session(slot: dict) -> str:
    """Drop a day with no price, or a limit lock that did not trade. A quiet minute stays."""

    if slot.get("收") is None or slot.get("高") is None or slot.get("低") is None:
        return "无收盘"
    volume = slot["当日量"] if slot.get("当日量") else slot.get("量")
    oi = slot["当日仓"] if slot.get("当日仓") else slot.get("仓")
    if volume is None or volume <= 0 or oi is None or oi <= 0:
        return "零成交"
    if slot["高"] <= slot["低"] and volume <= 0:
        return "零成交"
    return ""


def _own_features(day: str, slot: dict, prev: dict | None, swing: float) -> dict[str, float] | None:
    if slot.get("昨收") and slot.get("昨仓") and slot.get("昨量"):
        prev = {"收": slot["昨收"], "仓": slot["昨仓"], "量": slot["昨量"]}
    if prev is None or slot.get("收") is None or prev.get("收") is None:
        return None
    close = slot["当日收"] if slot.get("当日收") else slot["收"]
    volume = slot["当日量"] if slot.get("当日量") else slot["量"]
    oi = slot["当日仓"] if slot.get("当日仓") else slot["仓"]
    if volume <= 0 or prev["量"] <= 0 or oi <= 0 or prev["仓"] <= 0:
        return None
    name = ""
    yesterday = _ret(close, prev["收"])
    if yesterday is None:
        return None
    high, low = slot["高"], slot["低"]
    amplitude = (high - low) / prev["收"] * 100.0
    position = (close - low) / (high - low) if high > low else 0.5
    night = _ret(slot["夜收"], slot["夜开"]) or 0.0
    day_open = slot["日开"] if slot["日开"] is not None else slot["开"]
    return {
        f"{name}昨涨跌": yesterday,
        f"{name}这段": swing,
        f"{name}振幅": amplitude,
        f"{name}位置": position,
        f"{name}量变化": (volume / prev["量"] - 1.0) * 100.0,
        f"{name}仓变化": (oi / prev["仓"] - 1.0) * 100.0,
        f"{name}夜盘": night,
        f"{name}日盘": _ret(slot["收"], day_open) or 0.0,
        f"{name}价仓": _attitude(slot["路径"]),
    }


def build(product: str, swings_by_product: dict[str, dict[str, float]], sessions_by_product: dict[str, dict[str, dict]], spreads: dict[str, dict[str, float]]) -> None:
    cal = _calendar(product)
    days = _trading_days(cal)
    sessions = sessions_by_product[product]
    us = {key: _closes(key) for key in ("S", "SM", "BO", "FCPO")}
    soy_cot = {key: _cot_points(f"{filename}.json") for key, _name, filename in train.COT_MARKETS}
    canola = _cot_points("CANOLA.json")
    seats = {key: _seat_points(key) for key in SEAT}
    folder = HERE / product
    folder.mkdir(parents=True, exist_ok=True)
    watched = OI_REF[product]
    watched_days = _trading_days(_calendar(watched))
    feature_rows = []
    label_rows = []
    for index, day in enumerate(days):
        slot = sessions.get(day)
        if slot is None or _bad_session(slot):
            continue
        code = cal.get(day) or slot.get("合约") or ""
        prev = _previous(sessions, day, days)
        own = _own_features(day, slot, prev, swings_by_product[product].get(day, 0.0))
        if own is None:
            continue
        # Rename the blank prefix to this product.
        name = NAMES[product]
        renamed = {}
        for key, value in own.items():
            renamed[name + key] = value
        row = {"日期": day, "合约": code}
        row.update(renamed)
        link = LINKS[product]
        for other in link["这段"]:
            row[f"{NAMES[other]}这段"] = swings_by_product[other].get(day, "")
        for _top, _bottom, label in link["价差"]:
            row[label] = spreads.get(label, {}).get(day, "")
        watched_change = _oi_change(sessions_by_product[watched], day, watched_days)
        row[f"{NAMES[watched]}仓变化"] = "" if watched_change is None else round(watched_change, 6)
        if product in SOY:
            nxt = days[index + 1] if index + 1 < len(days) else "9999-99-99"
            for key, label in US.items():
                row[label] = round(train.overseas_session(us[key], day, nxt), 6)
        if product == "P":
            nxt = days[index + 1] if index + 1 < len(days) else "9999-99-99"
            row["马棕油隔夜"] = round(train.overseas_session(us["FCPO"], day, nxt), 6)
        if product in CANOLA:
            ratio, change = _asof(canola, day, strict=False)
            row["油菜籽基金净占比"] = round(ratio, 6)
            row["油菜籽基金占比变动"] = round(change, 6)
        if product in SOY:
            for key, _label, _file in train.COT_MARKETS:
                ratio, change = _asof(soy_cot[key], day, strict=False)
                row[{"S": "美豆基金净占比", "SM": "美豆粕基金净占比", "BO": "美豆油基金净占比"}[key]] = round(ratio, 6)
                row[{"S": "美豆基金占比变动", "SM": "美豆粕基金占比变动", "BO": "美豆油基金占比变动"}[key]] = round(change, 6)
        if product in SEAT:
            ratio, change = _asof(seats[product], day, strict=True)
            row[f"{SEAT[product]}席位净占比"] = round(ratio, 6)
            row[f"{SEAT[product]}席位净变动"] = round(change, 6)
        feature_rows.append(row)
        if index + 1 < len(days):
            next_day = days[index + 1]
            nxt = sessions.get(next_day)
            if slot.get("收1455") and nxt and nxt.get("收1455") and _same_contract(slot, nxt, day, next_day, cal):
                move = (nxt["收1455"] - slot["收1455"]) / slot["收1455"] * 100.0
                label_rows.append({
                    "日期": day,
                    "合约": slot.get("合约1455") or cal.get(day) or "",
                    "入场价": slot["收1455"],
                    "出场价": nxt["收1455"],
                    "涨跌": 1 if nxt["收1455"] > slot["收1455"] else 0,
                    "涨跌幅": round(move, 6),
                })
    _write(folder / "特征.csv", feature_rows, _feature_fields(product))
    _write(folder / "标签.csv", label_rows)
    print(f"{NAMES[product]} 特征 {len(feature_rows)} 行  标签 {len(label_rows)} 行", flush=True)


def _feature_fields(product: str) -> list[str]:
    """Column order is the menu: main, then how the session traded, then other markets."""

    name = NAMES[product]
    fields = ["日期", "合约", f"{name}昨涨跌", f"{name}这段"]
    if product in SOY:
        fields.extend(US.values())
    if product == "P":
        fields.append("马棕油隔夜")
    fields.extend((f"{name}夜盘", f"{name}日盘", f"{name}价仓"))
    fields.extend(f"{name}{suffix}" for suffix in ("振幅", "位置", "量变化", "仓变化"))
    for other in LINKS[product]["这段"]:
        fields.append(f"{NAMES[other]}这段")
    for _top, _bottom, label in LINKS[product]["价差"]:
        fields.append(label)
    fields.append(f"{NAMES[OI_REF[product]]}仓变化")
    if product in SOY:
        for key, _label, _file in train.COT_MARKETS:
            title = {"S": "美豆", "SM": "美豆粕", "BO": "美豆油"}[key]
            fields.extend((f"{title}基金净占比", f"{title}基金占比变动"))
    if product in CANOLA:
        fields.extend(("油菜籽基金净占比", "油菜籽基金占比变动"))
    if product in SEAT:
        fields.extend((f"{SEAT[product]}席位净占比", f"{SEAT[product]}席位净变动"))
    return fields


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    if fields is None:
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _index(sessions: dict[str, dict], days: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Roll-adjusted price index and its swing. A missing session starts a new segment."""

    level = 100.0
    levels: list[float] = []
    kept: list[str] = []
    segments: list[list[int]] = []
    current: list[int] = []
    prev_day = None
    swing: dict[str, float] = {}
    index: dict[str, float] = {}
    for day in days:
        slot = sessions.get(day)
        if slot is None or _bad_session(slot):
            continue
        if slot.get("昨收") and slot.get("当日收"):
            ret = _ret(float(slot["当日收"]), float(slot["昨收"]))
        elif slot.get("昨收"):
            ret = _ret(slot["收"], float(slot["昨收"]))
        else:
            prev_slot = _previous(sessions, day, days)
            ret = _ret(slot["收"], prev_slot["收"] if prev_slot else None)
        if ret is None:
            continue
        if prev_day is not None and any(prev_day < item < day for item in days):
            if current:
                segments.append(current)
            current = []
            level = 100.0
        level *= 1.0 + ret / 100.0
        current.append(len(kept))
        levels.append(level)
        kept.append(day)
        index[day] = level
        swing[day + "|涨跌"] = round(ret, 6)
        prev_day = day
    if current:
        segments.append(current)
    for segment in segments:
        path = train.swing_path([levels[i] for i in segment])
        for i, value in zip(segment, path):
            swing[kept[i]] = round(value, 6)
    return index, swing


def _spread_swing(index_top: dict[str, float], index_bottom: dict[str, float], days_top: list[str], days_bottom: list[str]) -> dict[str, float]:
    """Swing of the top index divided by the bottom index. A hole in either calendar breaks it."""

    both = sorted(set(index_top) & set(index_bottom))
    universe = set(days_top) | set(days_bottom)
    segments: list[list[str]] = []
    current: list[str] = []
    prev = None
    for day in both:
        if prev is not None and any(prev < item < day for item in universe):
            if current:
                segments.append(current)
            current = []
        current.append(day)
        prev = day
    if current:
        segments.append(current)
    out: dict[str, float] = {}
    for segment in segments:
        if len(segment) < 2:
            continue
        levels = [index_top[day] / index_bottom[day] for day in segment]
        for day, value in zip(segment, train.swing_path(levels)):
            out[day] = round(value, 6)
    return out


def main() -> None:
    only = sys.argv[1:]
    sessions_by_product = {}
    swings_by_product = {}
    indexes: dict[str, dict[str, float]] = {}
    calendars: dict[str, list[str]] = {}
    for product in PRODUCTS:
        if only and product not in only:
            continue
        cal = _calendar(product)
        days = _trading_days(cal)
        calendars[product] = days
        print(f"整理{NAMES[product]}", flush=True)
        sessions_by_product[product] = _main_sessions(product, days)
        indexes[product], swings_by_product[product] = _index(sessions_by_product[product], days)
    wanted = {(top, bottom, label) for product in PRODUCTS for top, bottom, label in LINKS[product]["价差"]}
    spreads = {
        label: _spread_swing(indexes[top], indexes[bottom], calendars[top], calendars[bottom])
        for top, bottom, label in sorted(wanted)
    }
    for product in PRODUCTS:
        if only and product not in only:
            continue
        build(product, swings_by_product, sessions_by_product, spreads)


if __name__ == "__main__":
    main()
