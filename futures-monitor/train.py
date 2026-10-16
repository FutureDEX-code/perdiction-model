#!/usr/bin/env python3
"""Train next-session direction for 豆一 / 豆二 / 豆粕 / 菜粕.

The inputs are shared: previous close-to-close returns, the latest finished
CBOT session, the swing, and the news score. Each contract then has its own
logistic model, fit only on whether that contract's next close is up, and
stored in its own file under data/models/. Changing one file does not rewrite
the other two. data/model.json is rebuilt from those three files for the board.

No third-party packages. Cached klines live in data/.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
DOMESTIC = {"A": "A0", "B": "B0", "M": "M0", "RM": "RM0"}
OVERSEAS = {"S": "S", "SM": "SM", "BO": "BO"}
NAMES = {"A": "豆一", "B": "豆二", "M": "豆粕", "RM": "菜粕"}
CONTRACTS = ("A", "M", "RM", "B")
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
# 豆二单独多两个自己的价格。豆一、豆粕、菜粕的 10 个输入不改，已有权重才对得上。
B_FEATURES = FEATURE_NAMES + ("豆二昨涨跌", "豆二这段")
MODELS = DATA / "models"
NEWS_WORDS = ("豆粕", "菜粕", "大豆", "美豆", "菜籽")
BULL = ("上涨", "走强", "偏强", "减产", "干旱", "短缺", "去库", "提振", "大涨", "供应紧张", "霜冻")
BEAR = ("下跌", "走弱", "偏弱", "丰产", "增产", "到港", "累库", "供应宽松", "大跌", "抛储", "收割顺利")


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_domestic(symbol: str) -> list[tuple[str, float]]:
    url = (
        "https://stock.finance.sina.com.cn/futures/api/jsonp.php/var%20_"
        f"{symbol}=/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
    )
    text = _get(url)
    start, end = text.find("(["), text.rfind("])")
    bars = json.loads(text[start + 1 : end + 1])
    rows = [(row["d"], float(row["c"])) for row in bars if row.get("c")]
    return _fill_session_close(symbol, rows)


_CTP_TO_KEY = {"豆一2701": "A", "豆二2701": "B", "豆粕2701": "M", "菜粕2701": "RM"}
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


def _ctp_1450(day: str) -> dict[str, float] | None:
    """First CTP price at or after 14:50. That is today's close for the model."""

    path = DATA / "ctp_tape.jsonl"
    if not path.exists():
        return None
    found: dict[str, float] = {}
    prefix = day + " "
    with path.open() as handle:
        for line in handle:
            if day not in line:
                continue
            row = json.loads(line)
            if not str(row.get("at", "")).startswith(prefix):
                continue
            if str(row.get("update") or "") < "14:50:00":
                continue
            key = _CTP_TO_KEY.get(row.get("name"))
            if key is None or key in found:
                continue
            try:
                last = float(row["last"])
            except (KeyError, TypeError, ValueError):
                continue
            if last <= 0:
                continue
            found[key] = last
            if len(found) == len(DOMESTIC):
                break
    if len(found) < len(DOMESTIC):
        return None
    return found


def _fill_session_close(symbol: str, rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """14:50 is the model close from the day it is saved. Earlier days stay on the file.

    A later official 15:00 print does not replace a saved 14:50 close.
    """

    key = {"A0": "A", "B0": "B", "M0": "M", "RM0": "RM"}.get(symbol)
    if key is None:
        return rows
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    today = now.date().isoformat()
    saved = _load_1450()
    if (now.hour, now.minute) >= (14, 50) and today not in saved:
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


def fetch_overseas(symbol: str) -> list[tuple[str, float]]:
    url = (
        "https://stock2.finance.sina.com.cn/futures/api/json.php/"
        f"GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={symbol}"
    )
    rows = json.loads(_get(url))
    return [(row["date"], float(row["close"])) for row in rows if row.get("close")]


def load_series() -> dict[str, dict[str, float]]:
    DATA.mkdir(parents=True, exist_ok=True)
    series: dict[str, dict[str, float]] = {}
    for key, symbol in {**DOMESTIC, **OVERSEAS}.items():
        path = DATA / f"{key}.json"
        fetcher = fetch_domestic if key in DOMESTIC else fetch_overseas
        rows = fetcher(symbol)
        path.write_text(json.dumps(rows))
        series[key] = {day: close for day, close in rows}
    return series


def returns(closes: dict[str, float]) -> dict[str, float]:
    days = sorted(closes)
    out = {}
    for prev, day in zip(days, days[1:]):
        if closes[prev]:
            out[day] = (closes[day] - closes[prev]) / closes[prev] * 100.0
    return out


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


def fetch_news() -> dict[str, float]:
    """Eastmoney headlines scored +1/-1 per bullish/bearish word. Same-day score."""

    scores: dict[str, float] = {}
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
            for row in rows:
                code = str(row.get("code") or "")
                if code in seen:
                    continue
                seen.add(code)
                blob = f"{row.get('title') or ''} {row.get('content') or ''}"
                if not any(word in blob for word in NEWS_WORDS):
                    continue
                day = str(row.get("date") or "")[:10]
                if len(day) != 10:
                    continue
                scores[day] = scores.get(day, 0.0) + sum(blob.count(w) for w in BULL) - sum(blob.count(w) for w in BEAR)
            if len(rows) < 50:
                break
            page += 1
    (DATA / "news.json").write_text(json.dumps(scores, ensure_ascii=False))
    return scores


def news_before(scores: dict[str, float], prev: str, day: str) -> float:
    # Headlines from the previous session and the gap before today's open.
    return sum(value for date, value in scores.items() if prev <= date < day)


def samples(series: dict[str, dict[str, float]], news: dict[str, float] | None = None) -> list[dict]:
    domestic = {key: returns(series[key]) for key in DOMESTIC if key in series}
    overseas = {key: returns(series[key]) for key in OVERSEAS}
    overseas_days = {key: sorted(overseas[key]) for key in OVERSEAS}
    swings = {
        key: dict(zip(sorted(series[key]), swing_path([series[key][day] for day in sorted(series[key])])))
        for key in domestic
    }
    common = sorted(set(domestic["A"]) & set(domestic["M"]) & set(domestic["RM"]))
    b_ret = domestic.get("B", {})
    b_swing = swings.get("B", {})
    rows = []
    for i in range(1, len(common)):
        day = common[i]
        prev = common[i - 1]
        feat = [domestic[key][prev] for key in ("A", "M", "RM")]
        if any(latest_on_or_before(overseas_days[key], prev) is None for key in OVERSEAS):
            continue
        for key in OVERSEAS:
            feat.append(overseas_session(series[key], prev, day))
        feat.extend(swings[key][prev] for key in ("A", "M", "RM"))
        feat.append(news_before(news or {}, prev, day))
        label = {key: 1 if domestic[key][day] > 0 else 0 for key in ("A", "M", "RM")}
        move = {key: domestic[key][day] for key in ("A", "M", "RM")}
        row = {"day": day, "x": feat, "y": label, "move": move}
        if day in b_ret and prev in b_ret and prev in b_swing:
            row["y"]["B"] = 1 if b_ret[day] > 0 else 0
            row["move"]["B"] = b_ret[day]
            row["x_b"] = feat + [b_ret[prev], b_swing[prev]]
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


def fit(xs: list[list[float]], ys: list[int]) -> tuple[list[float], float]:
    dim = len(xs[0])
    weight = [0.0] * dim
    bias = 0.0
    lr = 0.08
    for _ in range(500):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(bias + sum(weight[j] * x[j] for j in range(dim)))
            err = p - y
            for j in range(dim):
                grad_w[j] += err * x[j]
            grad_b += err
        n = len(xs)
        for j in range(dim):
            weight[j] -= lr * (grad_w[j] / n + 0.2 * weight[j])
        bias -= lr * grad_b / n
    return weight, bias


def predict_prob(x: list[float], weight: list[float], bias: float) -> float:
    return _sigmoid(bias + sum(weight[j] * x[j] for j in range(len(x))))


def _vector(row: dict, key: str) -> list[float] | None:
    if key == "B":
        return row.get("x_b")
    return row["x"]


def evaluate(key: str, train: list[dict], test: list[dict]) -> dict:
    train = [row for row in train if _vector(row, key) is not None and key in row["y"]]
    test = [row for row in test if _vector(row, key) is not None and key in row["y"]]
    mean, scale = standardize([_vector(row, key) for row in train])
    xs = [apply_scale(_vector(row, key), mean, scale) for row in train]
    ys = [row["y"][key] for row in train]
    weight, bias = fit(xs, ys)
    base = sum(ys) / len(ys)
    hits = 0
    up_moves: list[float] = []
    down_moves: list[float] = []
    for row in test:
        prob = predict_prob(apply_scale(_vector(row, key), mean, scale), weight, bias)
        guess = 1 if prob >= 0.5 else 0
        hits += int(guess == row["y"][key])
        (up_moves if guess == 1 else down_moves).append(row["move"][key])
    majority = max(base, 1 - base)
    majority_hits = sum(int((1 if base >= 0.5 else 0) == row["y"][key]) for row in test)
    return {
        "name": NAMES[key],
        "train": len(train),
        "test": len(test),
        "accuracy": hits / len(test),
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
    names = list(model.get("feature_names") or (B_FEATURES if key == "B" else FEATURE_NAMES))
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
    }
    model_path(key).write_text(json.dumps(body, ensure_ascii=False, indent=2))


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
    named["新闻倾向"] = news_before(news_scores(), last, "9999-99-99")
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
    print(f"{'品种':<6}{'方向命中':>10}{'多数类':>10}{'看涨时均涨跌':>14}{'看跌时均涨跌':>14}")
    models = {}
    for key in CONTRACTS:
        result = evaluate(key, train, test)
        models[key] = result
        print(
            f"{result['name']:<6}{result['accuracy']*100:9.1f}%"
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
            raise SystemExit(f"两点五十的价格还没有到齐 {last}")
        print(
            "两点五十收盘 "
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
                walked.append((key, result["accuracy"], result["test"], result["up_avg"], result["down_avg"]))
        print(f"新闻按月往前推（报道从 {news_rows[0]['day']} 起，当月不用当月的新闻拟合）")
        print(f"{'品种':<6}{'方向命中':>10}{'天数':>8}")
        for key in CONTRACTS:
            picked = [item for item in walked if item[0] == key]
            total = sum(item[2] for item in picked)
            hits = sum(item[1] * item[2] for item in picked)
            print(f"{NAMES[key]:<6}{(hits / total if total else 0)*100:9.1f}%{total:8d}")
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    models = {}
    for key in ("A", "M", "RM"):
        mean, scale = standardize([row["x"] for row in rows])
        weight, bias = fit([apply_scale(row["x"], mean, scale) for row in rows], [row["y"][key] for row in rows])
        models[key] = {
            "weight": weight,
            "bias": bias,
            "mean": mean,
            "scale": scale,
            "accuracy": None,
            "majority": None,
            "feature_names": list(FEATURE_NAMES),
        }
        save_contract(key, models[key], trained_at=trained_at, split=year_start)
    b_rows = [row for row in rows if row.get("x_b") is not None and "B" in row["y"]]
    if len(b_rows) >= 200:
        mean, scale = standardize([row["x_b"] for row in b_rows])
        weight, bias = fit(
            [apply_scale(row["x_b"], mean, scale) for row in b_rows],
            [row["y"]["B"] for row in b_rows],
        )
        models["B"] = {
            "weight": weight,
            "bias": bias,
            "mean": mean,
            "scale": scale,
            "accuracy": None,
            "majority": None,
            "feature_names": list(B_FEATURES),
        }
        save_contract("B", models["B"], trained_at=trained_at, split=year_start)
    save_common(trained_at, year_start)
    write_bundle(trained_at, year_start)
    view = outlook(series, load_models())
    (DATA / "outlook.json").write_text(json.dumps(view, ensure_ascii=False, indent=2))
    print(f"下一交易日（基于 {view['as_of']} 收盘）")
    for item in view["views"]:
        print(f"  {item['name']}: {item['lean']}  上涨概率 {item['prob_up']*100:.1f}%")
    import paper_book

    for note in paper_book.record_outlook(view, series):
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
