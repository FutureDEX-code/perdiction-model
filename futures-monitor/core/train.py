#!/usr/bin/env python3
"""Train next-session direction for 豆一 / 豆二 / 豆粕 / 菜粕 / 豆油.

The inputs are shared: previous close-to-close returns, the latest finished
CBOT session, the swing, and the news score. Each contract then has its own
logistic model, fit only on whether that contract's next close is up, and
stored in its own file under data/models/. Changing one file does not rewrite
the other two. data/model.json is rebuilt from those three files for the board.

No third-party packages. Cached klines live in data/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.request
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
NEWS_DAILY = ROOT / "training_dataset" / "news"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
DOMESTIC = {"A": "A0", "B": "B0", "M": "M0", "RM": "RM0", "Y": "Y0"}
OVERSEAS = {"S": "S", "SM": "SM", "BO": "BO"}
NAMES = {"A": "豆一", "B": "豆二", "M": "豆粕", "RM": "菜粕", "Y": "豆油"}
CONTRACTS = ("A", "M", "RM", "B", "Y")


def series_path(key: str) -> Path:
    """Domestic closes live with the Sina daily service. US closes live with the US daily set."""

    if key in DOMESTIC:
        return SINA_DAILY / f"{key}.json"
    if key in OVERSEAS:
        return US_DAILY / f"{key}.json"
    return DATA / f"{key}.json"
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
    "新闻倾向",
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
    "新闻三日",
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
    "新闻三日",
)
Y_FEATURES = FEATURE_NAMES + ("豆油昨涨跌", "豆油这段")
MODELS = DATA / "models"
NEWS_WORDS = ("豆粕", "菜粕", "大豆", "美豆", "菜籽")
BULL = ("上涨", "走强", "偏强", "减产", "干旱", "短缺", "去库", "提振", "大涨", "供应紧张", "霜冻")
BEAR = ("下跌", "走弱", "偏弱", "丰产", "增产", "到港", "累库", "供应宽松", "大跌", "抛储", "收割顺利")
# Only mainstream financial media, official bodies, and futures-broker research
# desks count as news. Anything else the search returns is dropped on the floor.
NEWS_MEDIA = frozenset(
    {
        # mainstream financial press
        "证券时报", "证券时报网", "财联社", "每日经济新闻", "界面新闻", "新华财经",
        "第一财经", "期货日报", "期货日报网", "上海证券报", "上海证券报·中国证券网",
        "21世纪经济报道", "证券日报", "证券日报之声", "经济日报", "中国新闻网",
        "人民日报", "人民网", "人民财讯", "大河财立方", "中国证券报",
        "中国证券报·中证金牛座", "中国证券报·中证网", "东方财富研究中心",
        "东方财富Choice数据", "东方财富证券", "澎湃新闻", "经济参考报", "经济参考网",
        "深圳商报·读创", "新华社", "新华网", "中国基金报", "华夏时报", "华夏时报网",
        "中国经营报", "中国经营网", "南方财经网", "券商中国", "蓝鲸财经", "上观新闻",
        "央广财经", "新京报", "国际金融报", "南方都市报", "中新经纬", "红星资本局",
        "时代财经", "时代周报", "北京商报", "科创板日报", "中国网财经", "中宏网",
        "广州日报", "大众网", "上游新闻", "湾财ND", "央视财经", "央视新闻",
        "央视新闻客户端", "数据宝", "理财周刊-财事汇", "财中社", "中国财富通",
        # official bodies
        "农业农村部", "国家粮食和物资储备局", "海关总署", "商务部", "财政部",
        "财政部网站", "大连商品交易所", "郑州商品交易所",
        # futures-broker research desks
        "广发期货", "五矿期货", "混沌天成期货", "国投期货", "上海中期期货",
        "东吴期货", "中财期货", "正信期货", "南华期货", "中州期货",
        "申银万国期货", "创元期货", "光大期货", "瑞达期货", "国信期货",
    }
)


def _news_media_ok(name: str) -> bool:
    return bool(name) and name in NEWS_MEDIA


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


def _cached_rows(path: Path) -> list[tuple[str, float]]:
    """Close series. Accepts the full daily bar and the older date/close pair."""

    raw = json.loads(path.read_text())
    out = []
    for item in raw:
        if isinstance(item, dict):
            day = item.get("日期") or item.get("d")
            close = item.get("收盘", item.get("c"))
            out.append((str(day), float(close)))
            continue
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
    return series


def returns(closes: dict[str, float]) -> dict[str, float]:
    days = sorted(closes)
    out = {}
    for prev, day in zip(days, days[1:]):
        if closes[prev]:
            out[day] = (closes[day] - closes[prev]) / closes[prev] * 100.0
    return out


COT_DAILY = ROOT / "training_dataset" / "cot"
POSITION_DAILY = ROOT / "training_dataset" / "position"
COT_MARKETS = (("S", "美豆", "SOYBEANS"), ("SM", "美豆粕", "SOYBEAN_MEAL"), ("BO", "美豆油", "SOYBEAN_OIL"))


def _cached_bars(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return [row for row in raw if isinstance(row, dict)]


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
    prev = None
    for bar in bars:
        if bar.get("日期") == anchor:
            extra = _extra_from(prev, bar)
            return {f"{NAMES[key]}{suffix}": value for suffix, value in extra.items()}
        prev = bar
    bar = _minute_bar(key, anchor, night_of=bars[-1]["日期"] if bars else None)
    if bar is not None and bars:
        extra = _extra_from(bars[-1], bar)
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
    return _variant_names(base, key, variant_for(key, family))


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
        days = [item for item in sorted(series[key]) if day is None or item <= day]
        tape.append(swing_path([series[key][item] for item in days])[-1] if days else 0.0)
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


def _score_blob(blob: str) -> float:
    return sum(blob.count(w) for w in BULL) - sum(blob.count(w) for w in BEAR)


def _news_load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return rows if isinstance(rows, list) else []


def _news_append(path: Path, rows: list[dict]) -> int:
    """Append articles not yet in the day's file. Same code is stored once."""

    kept = _news_load(path)
    have = {str(row.get("code")) for row in kept}
    added = 0
    for row in rows:
        if str(row.get("code")) in have:
            continue
        kept.append(row)
        added += 1
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    return added


def _news_day_files() -> list[Path]:
    if not NEWS_DAILY.exists():
        return []
    return sorted(
        path
        for path in NEWS_DAILY.glob("*.json")
        if path.name != "scores.json" and len(path.stem) == 10 and path.stem[4] == "-"
    )


def _news_rescore_files() -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in _news_day_files():
        total = 0.0
        for row in _news_load(path):
            blob = f"{row.get('title') or ''} {row.get('content') or ''}"
            total += _score_blob(blob)
        if total or path.stem not in scores:
            scores[path.stem] = scores.get(path.stem, 0.0) + total
    return scores


def fetch_news() -> dict[str, float]:
    """Pull Eastmoney articles into training_dataset/news/<day>.json, one record each.

    The day score is the same word count as before, now recomputed from the stored
    articles so the text stays on disk.
    """

    NEWS_DAILY.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for keyword in ("豆粕", "菜粕", "大豆", "美豆"):
        page = 1
        while page <= 40:
            param = {
                "uid": "",
                "keyword": keyword,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "time",
                        "pageIndex": page,
                        "pageSize": 50,
                        "preTag": "",
                        "postTag": "",
                    }
                },
            }
            url = (
                "https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param="
                + urllib.request.quote(json.dumps(param, ensure_ascii=False))
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            body = json.loads(text[text.find("(") + 1 : text.rfind(")")])
            rows = (body.get("result") or {}).get("cmsArticleWebOld") or []
            if not rows:
                break
            batch: dict[str, list[dict]] = {}
            for row in rows:
                code = str(row.get("code") or "")
                if not code or code in seen:
                    continue
                seen.add(code)
                media = str(row.get("mediaName") or "")
                if not _news_media_ok(media):
                    continue
                blob = f"{row.get('title') or ''} {row.get('content') or ''}"
                if not any(word in blob for word in NEWS_WORDS):
                    continue
                day = str(row.get("date") or "")[:10]
                if len(day) != 10:
                    continue
                batch.setdefault(day, []).append(
                    {
                        "code": code,
                        "date": str(row.get("date") or ""),
                        "title": str(row.get("title") or ""),
                        "content": str(row.get("content") or ""),
                        "media": media,
                        "keyword": keyword,
                        "score": _score_blob(blob),
                    }
                )
            for day, items in batch.items():
                _news_append(NEWS_DAILY / f"{day}.json", items)
            if len(rows) < 50:
                break
            page += 1
    scores = _news_rescore_files()
    (NEWS_DAILY / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, sort_keys=True, indent=2))
    return scores


def news_before(scores: dict[str, float], prev: str, day: str) -> float:
    # Headlines from the previous session and the gap before today's open.
    return sum(value for date, value in scores.items() if prev <= date < day)


def news_ema_as_of(series: dict[str, dict[str, float]], scores: dict[str, float], as_of: str) -> float:
    """Three-day EMA of the news score, through the close on `as_of`.

    Days without a finished US session are skipped, the same way samples are.
    """

    common = sorted(set(series["A"]) & set(series["M"]) & set(series["RM"]))
    overseas = {key: returns(series[key]) for key in OVERSEAS if key in series}
    overseas_days = {key: sorted(overseas[key]) for key in overseas}
    alpha = 2 / (3 + 1)
    ema = 0.0
    for prev, day in zip(common, common[1:]):
        if prev > as_of:
            break
        if overseas and any(latest_on_or_before(overseas_days[key], prev) is None for key in OVERSEAS):
            continue
        end = day if day <= as_of else "9999-99-99"
        ema = alpha * news_before(scores, prev, end) + (1 - alpha) * ema
        if prev == as_of:
            break
    if common and common[-1] == as_of:
        if not overseas or all(latest_on_or_before(overseas_days[key], as_of) is not None for key in OVERSEAS):
            ema = alpha * news_before(scores, as_of, "9999-99-99") + (1 - alpha) * ema
    return ema


def samples(series: dict[str, dict[str, float]], news: dict[str, float] | None = None) -> list[dict]:
    domestic = {key: returns(series[key]) for key in DOMESTIC if key in series}
    overseas = {key: returns(series[key]) for key in OVERSEAS}
    overseas_days = {key: sorted(overseas[key]) for key in OVERSEAS}
    swings = {
        key: dict(zip(sorted(series[key]), swing_path([series[key][day] for day in sorted(series[key])])))
        for key in domestic
    }
    common = sorted(set(domestic["A"]) & set(domestic["M"]) & set(domestic["RM"]))
    cot = cot_series()
    seats = seat_series()
    rows = []
    news_ema = 0.0
    news_alpha = 2 / (3 + 1)
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
        raw_news = news_before(news or {}, prev, day)
        news_ema = news_alpha * raw_news + (1 - news_alpha) * news_ema
        named["新闻倾向"] = raw_news
        named["新闻三日"] = news_ema
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


def news_scores() -> dict[str, float]:
    """Day scores from the stored articles. data/news.json is read only as a fallback."""

    if _news_day_files():
        return _news_rescore_files()
    path = DATA / "news.json"
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}


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
    domestic = {key: returns(series[key]) for key in DOMESTIC if key in series}
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
    named["新闻倾向"] = news_before(news_scores(), last, "9999-99-99")
    named["新闻三日"] = news_ema_as_of(series, news_scores(), last)
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
    news = fetch_news()
    news_days = sorted(news)
    print(
        f"新闻 {len(news)} 天有条目  {news_days[0] if news_days else '无'} → {news_days[-1] if news_days else '无'}"
    )
    rows = [row for row in samples(series, news) if row["day"] >= "2017-01-01"]
    year_start = f"{int(rows[-1]['day'][:4]) - 1}{rows[-1]['day'][4:]}"
    train = [row for row in rows if row["day"] < year_start]
    test = [row for row in rows if row["day"] >= year_start]
    if len(train) < 200 or len(test) < 50:
        raise SystemExit(f"样本不够 train={len(train)} test={len(test)}")
    price_train = [{**row, "x": row["x"][:-1]} for row in train]
    price_test = [{**row, "x": row["x"][:-1]} for row in test]
    print(f"过去一年 {test[0]['day']} → {test[-1]['day']}  训练只用更早的 {len(train)} 天")
    _print_block("只看行情", price_train, price_test)
    news_rows = [row for row in rows if row["x"][-1] != 0]
    if len(news_rows) >= 30:
        walked = []
        months = sorted({row["day"][:7] for row in news_rows})
        for month in months:
            prior = [row for row in rows if row["day"] < f"{month}-01"]
            month_rows = [row for row in news_rows if row["day"].startswith(month)]
            if len(prior) < 200 or not month_rows:
                continue
            for key in CONTRACTS:
                usable = [row for row in prior if _vector(row, key) is not None and key in row["y"]]
                month_ok = [row for row in month_rows if _vector(row, key) is not None and key in row["y"]]
                if len(usable) < 200 or not month_ok:
                    continue
                result = evaluate(key, prior, month_rows)
                walked.append((key, result["order_accuracy"], result["order_n"]))
        print(f"新闻按月往前推（报道从 {news_rows[0]['day']} 起，当月不用当月的新闻拟合）")
        print(f"{'品种':<6}{'下单命中':>10}{'下单天数':>8}")
        for key in CONTRACTS:
            picked = [item for item in walked if item[0] == key and item[2]]
            total = sum(item[2] for item in picked)
            hits = sum(item[1] * item[2] for item in picked)
            print(f"{NAMES[key]:<6}{(hits / total if total else 0)*100:9.1f}%{total:8d}")
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
