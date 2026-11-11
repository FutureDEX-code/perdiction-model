#!/usr/bin/env python3
"""Train next-session direction for 豆一 / 豆二 / 豆粕 / 菜粕 / 豆油.

The inputs are shared: previous close-to-close returns, the latest finished
CBOT session, and the swing. The news score is kept on disk but is not a
feature. Each contract then has its own
logistic model, fit only on whether that contract's next close is up, and
stored in its own file under data/models/. Changing one file does not rewrite
the other two. data/model.json is rebuilt from those three files for the board.

No third-party packages. Daily bars live in training_dataset/; models and runtime state stay under data/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.request
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
SINA_DAILY = ROOT / "training_dataset" / "sina_daily"
US_DAILY = ROOT / "training_dataset" / "us_daily"
MAIN_MINUTE = ROOT / "training_dataset" / "main_minute"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
DOMESTIC = {"A": "A0", "B": "B0", "M": "M0", "RM": "RM0", "Y": "Y0"}
# Download-only symbols: archived next to the five, never read by training.
SINA_EXTRA = {"P": "P0", "OI": "OI0", "PK": "PK0"}
EXTRA_NAMES = {"P": "棕榈油", "OI": "菜籽油", "PK": "花生"}
OVERSEAS = {"S": "S", "SM": "SM", "BO": "BO"}
# Archive-only overseas series: fetched alongside, never read by current features.
OVERSEAS_EXTRA = {"FCPO": "FCPO"}
NAMES = {"A": "豆一", "B": "豆二", "M": "豆粕", "RM": "菜粕", "Y": "豆油"}
CONTRACTS = ("A", "M", "RM", "B", "Y")


def series_path(key: str) -> Path:
    """Domestic closes live with the Sina daily service. US closes live with the US daily set."""

    if key in DOMESTIC or key in SINA_EXTRA:
        return SINA_DAILY / f"{key}.json"
    if key in OVERSEAS or key in OVERSEAS_EXTRA:
        return US_DAILY / f"{key}.json"
    raise KeyError(f"没有日线品种 {key}")
DAILY_EXTRA = ("振幅", "位置", "量变化", "仓变化")
COT_NAMES = (
    "美豆基金净占比",
    "美豆基金占比变动",
    "美豆粕基金净占比",
    "美豆粕基金占比变动",
    "美豆油基金净占比",
    "美豆油基金占比变动",
)
RM_SEAT_NAMES = ("菜粕席位净占比", "菜粕席位净变动")


def _extra_names(*keys: str) -> tuple[str, ...]:
    return tuple(f"{NAMES[key]}{suffix}" for key in keys for suffix in DAILY_EXTRA)


FEATURE_NAMES = (
    "豆一昨涨跌",
    "豆粕昨涨跌",
    "菜粕昨涨跌",
    "美豆隔夜",
    "美豆粕隔夜",
    "美豆油隔夜",
    "豆一这段",
    "豆粕这段",
    "菜粕这段",
)
# 豆二、豆油再加自己的两列。豆二和菜粕另用一组：去掉豆一，新闻用三日平滑，菜粕再用价差波段。
B_FEATURES = (
    "豆粕昨涨跌",
    "菜粕昨涨跌",
    "美豆隔夜",
    "美豆粕隔夜",
    "美豆油隔夜",
    "豆粕这段",
    "菜粕这段",
    "豆二昨涨跌",
    "豆二这段",
)
RM_FEATURES = (
    "豆粕昨涨跌",
    "菜粕昨涨跌",
    "美豆隔夜",
    "美豆粕隔夜",
    "美豆油隔夜",
    "豆粕这段",
    "菜粕价差波段",
)
Y_FEATURES = FEATURE_NAMES + ("豆油昨涨跌", "豆油这段")
MODELS = DATA / "models"
# The news-scrape sentiment pipeline was retired 2026-10-27. No production model
# used 新闻倾向/新闻三日; the LLM poll at data/llm_poll is its successor.
# The four public names below stay as empty stubs so old call sites keep running.


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _daily_number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def fetch_domestic(symbol: str) -> list[dict]:
    """Full daily bar. Models that only need the close still read it through _cached_rows."""

    url = (
        "https://stock.finance.sina.com.cn/futures/api/jsonp.php/var%20_"
        f"{symbol}=/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
    )
    text = _get(url)
    start, end = text.find("(["), text.rfind("])")
    bars = json.loads(text[start + 1 : end + 1])
    out = []
    for row in bars:
        if not row.get("c"):
            continue
        out.append(
            {
                "日期": row["d"],
                "开盘": _daily_number(row, "o"),
                "最高": _daily_number(row, "h"),
                "最低": _daily_number(row, "l"),
                "收盘": float(row["c"]),
                "成交量": _daily_number(row, "v"),
                "持仓": _daily_number(row, "p"),
                "结算": _daily_number(row, "s"),
            }
        )
    return out


_CTP_TO_KEY = {"豆一2701": "A", "豆二2701": "B", "豆粕2701": "M", "菜粕2701": "RM", "豆油2701": "Y"}
_CLOSE_1450 = DATA / "close_1450.json"


def _load_1450() -> dict[str, dict[str, float]]:
    if not _CLOSE_1450.exists():
        return {}
    raw = json.loads(_CLOSE_1450.read_text())
    return {day: {key: float(px) for key, px in prices.items()} for day, prices in raw.items()}


def _store_1450(day: str, prices: dict[str, float]) -> None:
    saved = _load_1450()
    saved[day] = {key: prices[key] for key in DOMESTIC}
    _CLOSE_1450.parent.mkdir(parents=True, exist_ok=True)
    _CLOSE_1450.write_text(json.dumps(saved, ensure_ascii=False, indent=2))


def _ctp_1450_ticks(day: str) -> dict[str, list[float]]:
    """Prices whose exchange time is inside the 14:50 minute. Night quotes are excluded.

    A morning replay of 22:59 still has an update time after 14:50 as text, so the
    window is closed at 14:51. The last price in the minute is that day's close.
    """

    path = DATA / "ctp_tape.jsonl"
    if not path.exists():
        return {}
    found: dict[str, list[float]] = {}
    prefix = day + " "
    with path.open() as handle:
        for line in handle:
            if day not in line:
                continue
            row = json.loads(line)
            if not str(row.get("at", "")).startswith(prefix):
                continue
            update = str(row.get("update") or "")
            if update < "14:50:00" or update >= "14:51:00":
                continue
            key = _CTP_TO_KEY.get(row.get("name"))
            if key is None:
                continue
            try:
                last = float(row["last"])
            except (KeyError, TypeError, ValueError):
                continue
            if last <= 0:
                continue
            found.setdefault(key, []).append(last)
    return found


def _ctp_1450(day: str) -> dict[str, float] | None:
    """Last price in the 14:50 minute, once every domestic contract has one."""

    ticks = _ctp_1450_ticks(day)
    if any(key not in ticks for key in DOMESTIC):
        return None
    return {key: ticks[key][-1] for key in DOMESTIC}


def _repair_1450() -> dict[str, dict[str, float]]:
    """Replace a saved close that never traded in the 14:50 minute.

    A price that did print in that minute stays, including an earlier tick from
    the same minute. Days the tape does not cover stay as saved.
    """

    saved = _load_1450()
    changed = False
    for day, prices in saved.items():
        ticks = _ctp_1450_ticks(day)
        for key, px in list(prices.items()):
            seen = ticks.get(key)
            if not seen or px in seen:
                continue
            prices[key] = seen[-1]
            changed = True
    if changed:
        _CLOSE_1450.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
    return saved


def _fill_session_close(symbol: str, rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Kept for the saved 14:50 file. The Sina daily series does not use it.

    Those days stay on the exchange 15:00 close. 14:50 is read from the minute files.
    """

    key = {"A0": "A", "B0": "B", "M0": "M", "RM0": "RM", "Y0": "Y"}.get(symbol)
    if key is None:
        return rows
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    today = now.date().isoformat()
    saved = _repair_1450()
    # Wait until 14:51 so the stored close is the last price of the 14:50 minute.
    if (now.hour, now.minute) >= (14, 51) and today not in saved:
        got = _ctp_1450(today)
        if got is not None:
            _store_1450(today, got)
            saved = _load_1450()
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for day, px in rows:
        if day in saved and key in saved[day]:
            px = saved[day][key]
        out.append((day, px))
        seen.add(day)
    if today in saved and key in saved[today] and today not in seen:
        out.append((today, saved[today][key]))
    return out


def fetch_overseas(symbol: str) -> list[dict]:
    """Full US daily bar from Sina. Volume, open interest, and settlement are not filled there."""

    url = (
        "https://stock2.finance.sina.com.cn/futures/api/json.php/"
        f"GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={symbol}"
    )
    rows = json.loads(_get(url))
    out = []
    for row in rows:
        if not row.get("close"):
            continue
        out.append(
            {
                "日期": row["date"],
                "开盘": _daily_number(row, "open"),
                "最高": _daily_number(row, "high"),
                "最低": _daily_number(row, "low"),
                "收盘": float(row["close"]),
            }
        )
    return out


DATA_FLOOR = "2016-10-01"


def bar_ok(row: dict) -> bool:
    """A domestic daily bar that can enter training.

    Price-only series (CBOT, FCPO) have no volume field and stay. A bar is
    dropped when volume or open interest is missing or zero, when high equals
    low, or when the open or close sits outside the high-low range. Those are
    holiday stubs and no-trade days, not limit locks with real turnover.
    """

    if "成交量" not in row and "持仓" not in row:
        return True
    open_, high, low, close = row.get("开盘"), row.get("最高"), row.get("最低"), row.get("收盘")
    if None in (open_, high, low, close) or not close or not low:
        return False
    if high <= low or not (low <= open_ <= high and low <= close <= high):
        return False
    volume, oi = row.get("成交量"), row.get("持仓")
    return bool(volume) and volume > 0 and bool(oi) and oi > 0


def _cached_rows(path: Path) -> list[tuple[str, float]]:
    """Close series. Accepts the full daily bar and the older date/close pair."""

    raw = json.loads(path.read_text())
    out = []
    for item in raw:
        if isinstance(item, dict):
            if not bar_ok(item):
                continue
            day = item.get("日期") or item.get("d")
            close = item.get("收盘", item.get("c"))
            if str(day) >= DATA_FLOOR:
                out.append((str(day), float(close)))
            continue
        if str(item[0]) >= DATA_FLOOR:
            out.append((str(item[0]), float(item[1])))
    return out


def load_series() -> dict[str, dict[str, float]]:
    """Read the five Sina daily files. Fetch only the US bars.

    sina_daily_daemon.py is the only writer of the domestic daily files.
    """

    DATA.mkdir(parents=True, exist_ok=True)
    series: dict[str, dict[str, float]] = {}
    for key in DOMESTIC:
        path = series_path(key)
        if not path.exists():
            raise SystemExit(f"没有新浪日线 {NAMES[key]}")
        series[key] = {day: close for day, close in _cached_rows(path)}
    for key, symbol in OVERSEAS.items():
        path = series_path(key)
        try:
            rows = fetch_overseas(symbol)
            if not rows:
                raise RuntimeError("空行情")
        except Exception as exc:
            if not path.exists():
                raise
            print(f"{NAMES.get(key, key)} 行情拉取失败，用缓存 {exc}", flush=True)
            series[key] = {day: close for day, close in _cached_rows(path)}
            continue
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        series[key] = {item["日期"]: float(item["收盘"]) for item in rows}
    for key, symbol in OVERSEAS_EXTRA.items():
        path = series_path(key)
        try:
            rows = fetch_overseas(symbol)
            if not rows:
                raise RuntimeError("空行情")
        except Exception as exc:
            if not path.exists():
                continue
            print(f"{key} 行情拉取失败，用缓存 {exc}", flush=True)
            continue
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return series


def returns(closes: dict[str, float]) -> dict[str, float]:
    days = sorted(closes)
    out = {}
    for prev, day in zip(days, days[1:]):
        if closes[prev]:
            out[day] = (closes[day] - closes[prev]) / closes[prev] * 100.0
    return out


_ROLL_BASIS: dict[str, dict[str, dict]] = {}
_CAL_DAYS: dict[str, list[str]] = {}


def _calendar_days(key: str) -> list[str]:
    if key not in _CAL_DAYS:
        from ingest import roll_calendar

        _CAL_DAYS[key] = sorted(roll_calendar.load_calendar(key))
    return _CAL_DAYS[key]


def roll_basis(key: str) -> dict[str, dict]:
    """Own-contract previous session stamped on roll days in the main-minute file."""

    if key in _ROLL_BASIS:
        return _ROLL_BASIS[key]
    found: dict[str, dict] = {}
    path = MAIN_MINUTE / f"{key}.csv"
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
                if not ("09:00" <= hm <= "15:00") or stamp in found:
                    continue
                if not (row.get("昨收") and row.get("昨仓") and row.get("昨量")):
                    continue
                slot = {
                    "昨收": float(row["昨收"]),
                    "昨仓": float(row["昨仓"]),
                    "昨量": float(row["昨量"]),
                }
                for name in ("当日收", "当日仓", "当日量"):
                    slot[name] = float(row[name]) if row.get(name) else None
                found[stamp] = slot
    _ROLL_BASIS[key] = found
    return found


def _crosses_hole(key: str, prev: str, day: str) -> bool:
    days = _calendar_days(key)
    if not days:
        return False
    index = bisect_right(days, prev)
    return index < len(days) and days[index] < day


def session_returns(key: str, closes: dict[str, float]) -> dict[str, float]:
    """Close-to-close returns. A roll day uses that contract's own previous close.

    A missing trading day is not bridged. Noise bars never enter `closes`.
    """

    basis = roll_basis(key)
    out = {}
    ordered = sorted(closes)
    for prev, day in zip(ordered, ordered[1:]):
        if not closes[prev] or _crosses_hole(key, prev, day):
            continue
        slot = basis.get(day)
        if slot and slot.get("昨收"):
            now = slot.get("当日收") or closes[day]
            out[day] = (now - slot["昨收"]) / slot["昨收"] * 100.0
        else:
            out[day] = (closes[day] - closes[prev]) / closes[prev] * 100.0
    return out


def session_swing(key: str, closes: dict[str, float]) -> dict[str, float]:
    """Swing on a same-contract return index, so a roll gap never moves it."""

    rets = session_returns(key, closes)
    level = 100.0
    levels = []
    kept = []
    for day in sorted(closes):
        if day not in rets:
            continue
        level *= 1.0 + rets[day] / 100.0
        levels.append(level)
        kept.append(day)
    return dict(zip(kept, swing_path(levels)))


def _without_news(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if "新闻" not in name)


COT_DAILY = ROOT / "training_dataset" / "cot"
POSITION_DAILY = ROOT / "training_dataset" / "position"
COT_MARKETS = (("S", "美豆", "SOYBEANS"), ("SM", "美豆粕", "SOYBEAN_MEAL"), ("BO", "美豆油", "SOYBEAN_OIL"))


def _cached_bars(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return [
        row for row in raw
        if isinstance(row, dict) and bar_ok(row) and str(row.get("日期") or row.get("d") or "") >= DATA_FLOOR
    ]


def _extra_from(prev: dict | None, bar: dict) -> dict[str, float]:
    """振幅/位置/量变化/仓变化 of one daily bar against the previous one."""

    out = {"振幅": 0.0, "位置": 0.5, "量变化": 0.0, "仓变化": 0.0}
    high, low, close = bar.get("最高"), bar.get("最低"), bar.get("收盘")
    if high is None or low is None or not close:
        return out
    prev_close = (prev or {}).get("收盘")
    if prev_close:
        out["振幅"] = (high - low) / prev_close * 100.0
    if high > low:
        out["位置"] = (close - low) / (high - low)
    vol, prev_vol = bar.get("成交量"), (prev or {}).get("成交量")
    if vol and prev_vol:
        out["量变化"] = (vol / prev_vol - 1.0) * 100.0
    oi, prev_oi = bar.get("持仓"), (prev or {}).get("持仓")
    if oi and prev_oi:
        out["仓变化"] = (oi / prev_oi - 1.0) * 100.0
    return out


def _minute_bar(key: str, day: str, night_of: str | None = None) -> dict | None:
    """A trading day from the minute files: last night 21:00-23:00 plus today's day session."""

    path = MAIN_MINUTE / f"{key}.csv"
    if not path.exists():
        return None

    def _part(stamp: str, hm: str) -> bool:
        if stamp == day and hm <= "15:00":
            return True
        return night_of is not None and stamp == night_of and hm >= "21:00"

    op = hi = lo = cl = None
    vol = 0.0
    oi = None
    found = False
    with path.open() as handle:
        for row in csv.DictReader(handle):
            stamp, hm = row["trade_time"][:10], row["trade_time"][11:16]
            if not _part(stamp, hm) or not row["close"]:
                continue
            found = True
            o, h, lo_, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            if op is None:
                op = o
            hi = h if hi is None else max(hi, h)
            lo = lo_ if lo is None else min(lo, lo_)
            cl = c
            vol += float(row["vol"] or 0)
            if row["oi"]:
                oi = float(row["oi"])
    if not found:
        return None
    return {"日期": day, "开盘": op, "最高": hi, "最低": lo, "收盘": cl, "成交量": vol or None, "持仓": oi}


_BARS_CACHE: dict[str, list[dict]] = {}


def _roll_extra(extra: dict[str, float], basis: dict | None, bar: dict) -> dict[str, float]:
    """On a roll day, volume, open interest, and amplitude use the new contract."""

    if not basis or not basis.get("昨收") or not basis.get("昨仓") or not basis.get("昨量"):
        return extra
    high, low = bar.get("最高"), bar.get("最低")
    if high is not None and low is not None:
        extra["振幅"] = (high - low) / basis["昨收"] * 100.0
    if basis.get("当日量"):
        extra["量变化"] = (basis["当日量"] / basis["昨量"] - 1.0) * 100.0
    if basis.get("当日仓"):
        extra["仓变化"] = (basis["当日仓"] / basis["昨仓"] - 1.0) * 100.0
    return extra


def extra_named(key: str, anchor: str) -> dict[str, float]:
    """The four daily extras at an anchor day.

    Training rows find the anchor in the daily file. Live afternoons synthesize
    it from minute bars because the exchange daily bar lands after 18:00.
    Bars are cached per process; the jobs that score or fit are short-lived.
    """

    bars = _BARS_CACHE.get(key)
    if bars is None:
        bars = _cached_bars(series_path(key))
        _BARS_CACHE[key] = bars
    basis = roll_basis(key).get(anchor)
    prev = None
    for bar in bars:
        if bar.get("日期") == anchor:
            extra = _roll_extra(_extra_from(prev, bar), basis, bar)
            return {f"{NAMES[key]}{suffix}": value for suffix, value in extra.items()}
        prev = bar
    bar = _minute_bar(key, anchor, night_of=bars[-1]["日期"] if bars else None)
    if bar is not None and bars:
        extra = _roll_extra(_extra_from(bars[-1], bar), basis, bar)
        return {f"{NAMES[key]}{suffix}": value for suffix, value in extra.items()}
    return {f"{NAMES[key]}{suffix}": value for suffix, value in (("振幅", 0.0), ("位置", 0.5), ("量变化", 0.0), ("仓变化", 0.0))}


def cot_series() -> dict[str, list[tuple[str, str, float, float]]]:
    """(available_from, report_day, 净占比, 占比变动). Reports publish the Saturday after."""

    out: dict[str, list[tuple[str, str, float, float]]] = {}
    for key, _name, filename in COT_MARKETS:
        path = COT_DAILY / f"{filename}.json"
        pts: list[tuple[str, str, float, float]] = []
        if path.exists():
            prev_ratio = None
            for row in json.loads(path.read_text()):
                oi = row.get("总持仓") or 0
                if not oi:
                    continue
                ratio = row["基金净多"] / oi * 100.0
                change = 0.0 if prev_ratio is None else ratio - prev_ratio
                available = (datetime.strptime(row["日期"], "%Y-%m-%d").date() + timedelta(days=4)).isoformat()
                pts.append((available, row["日期"], ratio, change))
                prev_ratio = ratio
        out[key] = pts
    return out


def cot_named(cot: dict[str, list[tuple[str, str, float, float]]], anchor: str) -> dict[str, float]:
    named = {}
    for key, name, _filename in COT_MARKETS:
        ratio = change = 0.0
        for available, _day, r, c in cot.get(key) or []:
            if available > anchor:
                break
            ratio, change = r, c
        named[f"{name}基金净占比"] = ratio
        named[f"{name}基金占比变动"] = change
    return named


def seat_series() -> list[tuple[str, float, float]]:
    path = POSITION_DAILY / "RM.json"
    pts: list[tuple[str, float, float]] = []
    if path.exists():
        for row in json.loads(path.read_text()):
            total = (row.get("前20多单") or 0) + (row.get("前20空单") or 0)
            if not total:
                continue
            pts.append((row["日期"], (row.get("净多") or 0) / total * 100.0, float(row.get("净变动") or 0)))
    return pts


def seat_named(seats: list[tuple[str, float, float]], anchor: str) -> dict[str, float]:
    """Published after the close. The anchor sees the table of the day before."""

    ratio = change = 0.0
    for day, r, c in seats:
        if day >= anchor:
            break
        ratio, change = r, c
    return {"菜粕席位净占比": ratio, "菜粕席位净变动": change}


MOOD_NAMES = ("市场情绪", "情绪变动")
EXTRA_WATCH = {"A": ("A", "M", "RM"), "M": ("A", "M", "RM"), "RM": ("M", "RM"), "B": ("M", "RM", "B"), "Y": ("A", "M", "RM", "Y")}


def mood_named(cot: dict[str, list], seats: list[tuple[str, float, float]], anchor: str) -> dict[str, float]:
    """Half CBOT fund balance, half Zhengzhou seat balance, each aligned by its own clock."""

    us = cot_named(cot, anchor)
    us_ratio = (us["美豆基金净占比"] + us["美豆粕基金净占比"] + us["美豆油基金净占比"]) / 3.0
    us_change = (us["美豆基金占比变动"] + us["美豆粕基金占比变动"] + us["美豆油基金占比变动"]) / 3.0
    ratio = change = 0.0
    prev_ratio = None
    for day, r, _c in seats:
        if day >= anchor:
            break
        if prev_ratio is not None:
            change = r - prev_ratio
        prev_ratio = r
        ratio = r
    return {"市场情绪": (us_ratio + ratio) / 2.0, "情绪变动": (us_change + change) / 2.0}


# Products without their own 特征.json keep what production already runs.
DEFAULT_VARIANTS = {"日线回归": "旧", "日线树": "旧", "分钟回归": "+CFTC", "分钟树": "旧"}


def feature_choice(key: str) -> dict:
    path = DATA / key / "特征.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def variant_for(key: str, family: str) -> str:
    return feature_choice(key).get(family) or DEFAULT_VARIANTS.get(family, "旧")


def combo_rule(key: str) -> str:
    """How the product's four models combine into one call. Empty means no combo line."""

    return feature_choice(key).get("综合", "")


def hold_rule(key: str) -> str:
    """Close rule for the combo signal. 只反向平 won on all five products' holdout."""

    return feature_choice(key).get("平仓", "只反向平")


def _variant_names(base: tuple[str, ...], key: str, variant: str) -> tuple[str, ...]:
    names = list(base)
    if variant in ("+日线四件", "全新"):
        names += [f"{NAMES[k]}{suffix}" for k in EXTRA_WATCH[key] for suffix in DAILY_EXTRA]
    if variant in ("+情绪", "全新"):
        names += list(MOOD_NAMES)
    if variant == "+CFTC":
        names += list(COT_NAMES)
    return tuple(names)


def sina_feature_names(key: str, family: str) -> tuple[str, ...]:
    base = {"A": FEATURE_NAMES, "M": FEATURE_NAMES, "RM": RM_FEATURES, "B": B_FEATURES, "Y": Y_FEATURES}[key]
    return _without_news(_variant_names(base, key, variant_for(key, family)))


def swing_path(closes: list[float], rev: float = 0.03) -> list[float]:
    """Signed percent of the unfinished swing at each close.

    A bounce ends a decline only after a rally of `rev` from the low.
    A pullback ends a rally only after a drop of `rev` from the high.
    """

    out = [0.0] * len(closes)
    if len(closes) < 2:
        return out
    trough_i = 0
    peak_i = 0
    mode = "seek_trough"
    saw_peak = False
    for i in range(1, len(closes)):
        if mode == "seek_trough":
            if closes[i] <= closes[trough_i]:
                trough_i = i
            elif closes[trough_i] and (closes[i] - closes[trough_i]) / closes[trough_i] >= rev:
                peak_i = i
                mode = "seek_peak"
                saw_peak = True
        elif closes[i] >= closes[peak_i]:
            peak_i = i
        elif closes[peak_i] and (closes[peak_i] - closes[i]) / closes[peak_i] >= rev:
            trough_i = i
            mode = "seek_trough"
        if mode == "seek_peak":
            base = closes[trough_i]
        elif saw_peak:
            base = closes[peak_i]
        else:
            base = closes[0]
        out[i] = (closes[i] - base) / base * 100.0 if base else 0.0
    return out


def domestic_tape(
    series: dict[str, dict[str, float]],
    day: str | None = None,
    keys: tuple[str, ...] = ("A", "M", "RM"),
) -> list[float]:
    """Swing percent using closes up to `day`. Default order is 豆一, 豆粕, 菜粕."""

    tape = []
    for key in keys:
        swing = session_swing(key, series[key])
        days = [item for item in sorted(swing) if day is None or item <= day]
        tape.append(swing[days[-1]] if days else 0.0)
    return tape


def latest_on_or_before(days: list[str], day: str) -> str | None:
    lo, hi = 0, len(days)
    while lo < hi:
        mid = (lo + hi) // 2
        if days[mid] <= day:
            lo = mid + 1
        else:
            hi = mid
    return days[lo - 1] if lo else None


def news_scores() -> dict[str, float]:
    """Stub. The news pipeline is retired; see the note by MODELS."""

    return {}


def fetch_news() -> dict[str, float]:
    """Stub. Nothing is scraped anymore."""

    return {}


def news_before(scores: dict[str, float], prev: str, day: str) -> float:
    """Stub. Always neutral."""

    return 0.0


def news_ema_as_of(series: dict[str, dict[str, float]], scores: dict[str, float], as_of: str) -> float:
    """Stub. Always neutral."""

    return 0.0


def samples(series: dict[str, dict[str, float]], news: dict[str, float] | None = None) -> list[dict]:
    domestic = {key: session_returns(key, series[key]) for key in DOMESTIC if key in series}
    overseas = {key: returns(series[key]) for key in OVERSEAS}
    overseas_days = {key: sorted(overseas[key]) for key in OVERSEAS}
    swings = {key: session_swing(key, series[key]) for key in domestic}
    common = sorted(set(domestic["A"]) & set(domestic["M"]) & set(domestic["RM"]))
    cot = cot_series()
    seats = seat_series()
    rows = []
    for i in range(1, len(common)):
        day = common[i]
        prev = common[i - 1]
        if any(latest_on_or_before(overseas_days[key], prev) is None for key in OVERSEAS):
            continue
        named = {
            "豆一昨涨跌": domestic["A"][prev],
            "豆粕昨涨跌": domestic["M"][prev],
            "菜粕昨涨跌": domestic["RM"][prev],
        }
        for key in OVERSEAS:
            named[{"S": "美豆隔夜", "SM": "美豆粕隔夜", "BO": "美豆油隔夜"}[key]] = overseas_session(series[key], prev, day)
        named["豆一这段"] = swings["A"][prev]
        named["豆粕这段"] = swings["M"][prev]
        named["菜粕这段"] = swings["RM"][prev]
        named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
        has_b = day in domestic.get("B", {}) and prev in domestic.get("B", {}) and prev in swings.get("B", {})
        has_y = day in domestic.get("Y", {}) and prev in domestic.get("Y", {}) and prev in swings.get("Y", {})
        if has_b:
            named["豆二昨涨跌"] = domestic["B"][prev]
            named["豆二这段"] = swings["B"][prev]
        if has_y:
            named["豆油昨涨跌"] = domestic["Y"][prev]
            named["豆油这段"] = swings["Y"][prev]
        for key in DOMESTIC:
            if prev in series.get(key, {}):
                named.update(extra_named(key, prev))
        named.update(cot_named(cot, prev))
        named.update(mood_named(cot, seats, prev))
        label = {key: 1 if domestic[key][day] > 0 else 0 for key in ("A", "M", "RM")}
        move = {key: domestic[key][day] for key in ("A", "M", "RM")}
        row = {
            "day": day,
            "y": label,
            "move": move,
            "named": named,
            "x": [named[name] for name in FEATURE_NAMES],
            "x_rm": [named[name] for name in RM_FEATURES],
        }
        if has_b:
            row["y"]["B"] = 1 if domestic["B"][day] > 0 else 0
            row["move"]["B"] = domestic["B"][day]
            row["x_b"] = [named[name] for name in B_FEATURES]
        if has_y:
            row["y"]["Y"] = 1 if domestic["Y"][day] > 0 else 0
            row["move"]["Y"] = domestic["Y"][day]
            row["x_y"] = [named[name] for name in Y_FEATURES]
        rows.append(row)
    return rows


def _sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def standardize(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    dim = len(rows[0])
    mean = [sum(row[j] for row in rows) / len(rows) for j in range(dim)]
    var = [
        sum((row[j] - mean[j]) ** 2 for row in rows) / len(rows) for j in range(dim)
    ]
    scale = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, scale


def apply_scale(row: list[float], mean: list[float], scale: list[float]) -> list[float]:
    return [(row[j] - mean[j]) / scale[j] for j in range(len(row))]


def sample_weight(move_pct: float) -> float:
    """A day's lesson is its move. Quiet days count less; 2% caps a crash."""

    return min(max(abs(move_pct), 0.05), 2.0)


def fit(
    xs: list[list[float]],
    ys: list[int],
    ws: list[float] | None = None,
    *,
    lr: float = 0.08,
    steps: int = 500,
    l2: float = 0.2,
) -> tuple[list[float], float]:
    dim = len(xs[0])
    weight = [0.0] * dim
    bias = 0.0
    if ws is None:
        ws = [1.0] * len(xs)
    total = sum(ws)
    for _ in range(steps):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y, wi in zip(xs, ys, ws):
            p = _sigmoid(bias + sum(weight[j] * x[j] for j in range(dim)))
            err = (p - y) * wi
            for j in range(dim):
                grad_w[j] += err * x[j]
            grad_b += err
        for j in range(dim):
            weight[j] -= lr * (grad_w[j] / total + l2 * weight[j])
        bias -= lr * grad_b / total
    return weight, bias


def predict_prob(x: list[float], weight: list[float], bias: float) -> float:
    return _sigmoid(bias + sum(weight[j] * x[j] for j in range(len(x))))


def _vector(row: dict, key: str) -> list[float] | None:
    if key == "B":
        return row.get("x_b")
    if key == "Y":
        return row.get("x_y")
    if key == "RM":
        return row.get("x_rm")
    return row["x"]


def evaluate(key: str, train: list[dict], test: list[dict]) -> dict:
    train = [row for row in train if _vector(row, key) is not None and key in row["y"]]
    test = [row for row in test if _vector(row, key) is not None and key in row["y"]]
    mean, scale = standardize([_vector(row, key) for row in train])
    xs = [apply_scale(_vector(row, key), mean, scale) for row in train]
    ys = [row["y"][key] for row in train]
    ws = [sample_weight(row["move"][key]) for row in train]
    weight, bias = fit(xs, ys, ws)
    base = sum(ys) / len(ys)
    hits = 0
    order_hits = 0
    order_n = 0
    up_moves: list[float] = []
    down_moves: list[float] = []
    for row in test:
        prob = predict_prob(apply_scale(_vector(row, key), mean, scale), weight, bias)
        guess = 1 if prob >= 0.5 else 0
        hits += int(guess == row["y"][key])
        (up_moves if guess == 1 else down_moves).append(row["move"][key])
        if prob >= 0.55 or prob <= 0.45:
            order_n += 1
            order_hits += int(guess == row["y"][key])
    majority = max(base, 1 - base)
    majority_hits = sum(int((1 if base >= 0.5 else 0) == row["y"][key]) for row in test)
    return {
        "name": NAMES[key],
        "train": len(train),
        "test": len(test),
        "accuracy": hits / len(test),
        "order_accuracy": order_hits / order_n if order_n else 0.0,
        "order_n": order_n,
        "majority": majority_hits / len(test),
        "up_avg": sum(up_moves) / len(up_moves) if up_moves else 0.0,
        "down_avg": sum(down_moves) / len(down_moves) if down_moves else 0.0,
        "up_n": len(up_moves),
        "down_n": len(down_moves),
        "majority_rate": majority,
        "weight": weight,
        "bias": bias,
        "mean": mean,
        "scale": scale,
    }


def direction(prob: float) -> str:
    """A call inside 45–55% is not a direction. That band was a coin flip last year."""

    if prob >= 0.55:
        return "偏涨"
    if prob <= 0.45:
        return "偏跌"
    return "没有方向"


def model_path(key: str) -> Path:
    return MODELS / f"{key}.json"


def save_contract(
    key: str,
    model: dict,
    *,
    trained_at: str,
    split: str,
) -> None:
    """Write one contract. The other files are left as they are."""

    MODELS.mkdir(parents=True, exist_ok=True)
    names = list(
        model.get("feature_names")
        or {"B": B_FEATURES, "Y": Y_FEATURES, "RM": RM_FEATURES}.get(key, FEATURE_NAMES)
    )
    body = {
        "code": key,
        "name": NAMES[key],
        "trained_at": trained_at,
        "split": split,
        "feature_names": names,
        "weight": model["weight"],
        "bias": model["bias"],
        "mean": model["mean"],
        "scale": model["scale"],
        "test_accuracy": model.get("test_accuracy", model.get("accuracy")),
        "majority": model.get("majority"),
        "train_params": model.get("train_params") or {"lr": 0.08, "steps": 500, "l2": 0.2},
    }
    model_path(key).write_text(json.dumps(body, ensure_ascii=False, indent=2))


def saved_train_params(key: str) -> dict:
    path = model_path(key)
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("train_params") or {}


def save_common(trained_at: str, split: str) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    body = {
        "trained_at": trained_at,
        "split": split,
        "feature_names": list(FEATURE_NAMES),
        "direction_up": 0.55,
        "direction_down": 0.45,
        "contracts": list(CONTRACTS),
    }
    (MODELS / "common.json").write_text(json.dumps(body, ensure_ascii=False, indent=2))


def load_models() -> dict[str, dict]:
    """Each contract file is one model. A missing file falls back to model.json."""

    found: dict[str, dict] = {}
    for key in CONTRACTS:
        path = model_path(key)
        if path.exists():
            found[key] = json.loads(path.read_text())
    if len(found) == len(CONTRACTS):
        return found
    legacy = json.loads((DATA / "model.json").read_text())
    for key in CONTRACTS:
        found.setdefault(key, legacy["models"][key])
    return found


def write_bundle(trained_at: str, split: str) -> None:
    """Board file derived from the three contract files. Not the source of truth."""

    models = load_models()
    payload = {
        "trained_at": trained_at,
        "split": split,
        "feature_names": list(FEATURE_NAMES),
        "models": {
            key: {
                "weight": models[key]["weight"],
                "bias": models[key]["bias"],
                "mean": models[key]["mean"],
                "scale": models[key]["scale"],
                "test_accuracy": models[key].get("test_accuracy"),
                "majority": models[key].get("majority"),
            }
            for key in CONTRACTS
        },
    }
    (DATA / "model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def views_from_features(feat: list[float] | dict[str, float], models: dict[str, dict]) -> list[dict]:
    named = feat if isinstance(feat, dict) else None
    views = []
    for key in CONTRACTS:
        if key not in models:
            continue
        model = models[key]
        if named is None:
            vector = list(feat)
        else:
            names = model.get("feature_names") or list(FEATURE_NAMES)
            vector = [float(named[name]) for name in names]
        if len(vector) != len(model["weight"]):
            continue
        prob = predict_prob(apply_scale(vector, model["mean"], model["scale"]), model["weight"], model["bias"])
        views.append(
            {
                "code": key,
                "name": NAMES[key],
                "prob_up": round(prob, 4),
                "lean": direction(prob),
            }
        )
    return views


def _us_end(closes: dict[str, float], before: str | None) -> tuple[list[str], str | None]:
    days = sorted(closes)
    if not days:
        return days, None
    if before is None:
        return days, days[-1]
    earlier = [day for day in days if day < before]
    return days, earlier[-1] if earlier else None


def overseas_session(closes: dict[str, float], as_of: str, before: str | None = None) -> float:
    """Last finished US session, one day only. This is what the model trades on.

    美豆 is not an order. A holiday does not add the earlier sessions into this
    number; those days are already past, and the weights were fit on one session.
    """

    days, end = _us_end(closes, before)
    if end is None:
        return 0.0
    prev = [day for day in days if day < end]
    if not prev or not closes[prev[-1]]:
        return 0.0
    return (closes[end] - closes[prev[-1]]) / closes[prev[-1]] * 100.0


def overseas_gap(closes: dict[str, float], as_of: str, before: str | None = None) -> float:
    """Whole move since the last China close. Kept for the log, not as a feature."""

    days, end = _us_end(closes, before)
    start = latest_on_or_before(days, as_of)
    if start is None or end is None or not closes[start]:
        return 0.0
    return (closes[end] - closes[start]) / closes[start] * 100.0


def session_day(series: dict[str, dict[str, float]]) -> str:
    """Last China close. All three contracts must share it. 15:00 does not move."""

    lasts = {key: max(series[key]) for key in DOMESTIC}
    if len(set(lasts.values())) != 1:
        detail = " ".join(f"{NAMES[key]}{lasts[key]}" for key in DOMESTIC)
        raise SystemExit(f"内地日线没有到齐 {detail}")
    return next(iter(lasts.values()))


def _day_after(day: str) -> str:
    return (datetime.strptime(day, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


def outlook(series: dict[str, dict[str, float]], models: dict[str, dict]) -> dict:
    domestic = {key: session_returns(key, series[key]) for key in DOMESTIC if key in series}
    last = session_day(series)
    named = {
        "豆一昨涨跌": domestic["A"][last],
        "豆粕昨涨跌": domestic["M"][last],
        "菜粕昨涨跌": domestic["RM"][last],
    }
    # Same bound as training: the US bar dated after this China close is not a feature.
    cutoff = _day_after(last)
    overseas_pct = [overseas_session(series[key], last, cutoff) for key in OVERSEAS]
    overseas_gap_pct = [overseas_gap(series[key], last, cutoff) for key in OVERSEAS]
    for name, value in zip(("美豆隔夜", "美豆粕隔夜", "美豆油隔夜"), overseas_pct):
        named[name] = value
    tape = domestic_tape(series, last)
    for name, value in zip(("豆一这段", "豆粕这段", "菜粕这段"), tape):
        named[name] = value
    if "B" in domestic and last in domestic["B"]:
        named["豆二昨涨跌"] = domestic["B"][last]
        named["豆二这段"] = domestic_tape(series, last, keys=("B",))[0]
    if "Y" in domestic and last in domestic["Y"]:
        named["豆油昨涨跌"] = domestic["Y"][last]
        named["豆油这段"] = domestic_tape(series, last, keys=("Y",))[0]
    named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
    for key in DOMESTIC:
        named.update(extra_named(key, last))
    named.update(cot_named(cot_series(), last))
    named.update(mood_named(cot_series(), seat_series(), last))
    feat = [named[name] for name in FEATURE_NAMES]
    return {
        "as_of": last,
        "features_pct": [round(v, 4) for v in feat],
        "tape_pct": [round(v, 4) for v in tape],
        "overseas_pct": [round(v, 4) for v in overseas_pct],
        "overseas_gap_pct": [round(v, 4) for v in overseas_gap_pct],
        "views": views_from_features(named, models),
    }


def _print_block(title: str, train: list[dict], test: list[dict]) -> dict[str, dict]:
    print(title)
    print(f"{'品种':<6}{'下单命中':>10}{'下单天数':>8}{'全样本':>10}{'多数类':>10}{'看涨时均涨跌':>14}{'看跌时均涨跌':>14}")
    models = {}
    for key in CONTRACTS:
        result = evaluate(key, train, test)
        models[key] = result
        print(
            f"{result['name']:<6}{result['order_accuracy']*100:9.1f}%"
            f"{result['order_n']:8d}"
            f"{result['accuracy']*100:9.1f}%"
            f"{result['majority']*100:9.1f}%"
            f"{result['up_avg']:+13.3f}%"
            f"{result['down_avg']:+13.3f}%"
        )
    return models


def main() -> int:
    series = load_series()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if (now.hour, now.minute) >= (14, 50):
        today = now.date().isoformat()
        last = session_day(series)
        if last != today:
            raise SystemExit(f"今天的15:00收盘还没有到 {last}")
        print(
            "日线收盘 "
            + " ".join(f"{NAMES[key]}{series[key][today]:.0f}" for key in DOMESTIC),
            flush=True,
        )
    rows = [row for row in samples(series) if row["day"] >= "2017-01-01"]
    year_start = f"{int(rows[-1]['day'][:4]) - 1}{rows[-1]['day'][4:]}"
    train = [row for row in rows if row["day"] < year_start]
    test = [row for row in rows if row["day"] >= year_start]
    if len(train) < 200 or len(test) < 50:
        raise SystemExit(f"样本不够 train={len(train)} test={len(test)}")
    print(f"过去一年 {test[0]['day']} → {test[-1]['day']}  训练只用更早的 {len(train)} 天")
    _print_block("不含新闻", train, test)
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    models = {}

    def _fit_key(key: str, source: list[dict], vec: str, feature_names: tuple[str, ...]) -> None:
        spec = saved_train_params(key)
        lr = float(spec.get("lr", 0.08))
        steps = int(spec.get("steps", 500))
        l2 = float(spec.get("l2", 0.2))
        mean, scale = standardize([row[vec] for row in source])
        weight, bias = fit(
            [apply_scale(row[vec], mean, scale) for row in source],
            [row["y"][key] for row in source],
            [sample_weight(row["move"][key]) for row in source],
            lr=lr,
            steps=steps,
            l2=l2,
        )
        models[key] = {
            "weight": weight,
            "bias": bias,
            "mean": mean,
            "scale": scale,
            "accuracy": None,
            "majority": None,
            "feature_names": list(feature_names),
            "train_params": {"lr": lr, "steps": steps, "l2": l2},
        }
        save_contract(key, models[key], trained_at=trained_at, split=year_start)

    for key in ("A", "M"):
        _fit_key(key, rows, "x", FEATURE_NAMES)
    rm_rows = [row for row in rows if row.get("x_rm") is not None and "RM" in row["y"]]
    if len(rm_rows) >= 200:
        _fit_key("RM", rm_rows, "x_rm", RM_FEATURES)
    b_rows = [row for row in rows if row.get("x_b") is not None and "B" in row["y"]]
    if len(b_rows) >= 200:
        _fit_key("B", b_rows, "x_b", B_FEATURES)
    y_rows = [row for row in rows if row.get("x_y") is not None and "Y" in row["y"]]
    if len(y_rows) >= 200:
        _fit_key("Y", y_rows, "x_y", Y_FEATURES)
    save_common(trained_at, year_start)
    write_bundle(trained_at, year_start)
    view = outlook(series, load_models())
    (DATA / "outlook.json").write_text(json.dumps(view, ensure_ascii=False, indent=2))
    print(f"下一交易日（基于 {view['as_of']} 收盘）")
    for item in view["views"]:
        print(f"  {item['name']}: {item['lean']}  上涨概率 {item['prob_up']*100:.1f}%")
    from models import paper_book

    for note in paper_book.record_outlook(view, series):
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
