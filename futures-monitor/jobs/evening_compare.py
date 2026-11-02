#!/usr/bin/env python3
"""At 20:30, predict again with the models already on disk.

New inputs since 14:50 are scored. The weights are not refit. The afternoon
snapshot stays the direction already given. This script does not replace that
mail and does not write a new paper trade. China night session opens at 21:00,
so the new inputs are the US daily bar and the position reports.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb

from models import fit_sina_main as minute
from jobs import mail_tree
import training_dataset.sina_daily.sina_daily_daemon as sina_daily_daemon
from core import train
from jobs.mail_outlook import (
    SENDER,
    TO,
    _forbid_company,
    _google_enabled,
    _locate,
    _outbox_count,
    _pull_from_bin,
    _send,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMPARE = ROOT / "data" / "compare"
TZ = ZoneInfo("Asia/Shanghai")
US_NAMES = (("S", "美豆"), ("SM", "美豆粕"), ("BO", "美豆油"))
FAMILIES = ("日线回归", "日线树", "分钟回归", "分钟树")


def _series_from_disk() -> dict[str, dict[str, float]]:
    return {
        key: {day: close for day, close in train._cached_rows(train.series_path(key))}
        for key in (*train.DOMESTIC, *train.OVERSEAS)
    }


def _minute_close_today(key: str, today: str) -> float | None:
    path = train.MAIN_MINUTE / f"{key}.csv"
    if not path.exists():
        return None
    close = None
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["trade_time"][:10] == today and row["close"]:
                close = float(row["close"])
    return close


def _proxy_today(series: dict[str, dict[str, float]], today: str) -> dict[str, dict[str, float]]:
    """The 15:00 daily bar is only pulled after 18:00. The 14:50 mail stands in with minute closes."""

    if datetime.now(TZ).weekday() >= 5 or train.session_day(series) == today:
        return series
    proxied = {key: dict(book) for key, book in series.items()}
    missing = []
    for key in train.DOMESTIC:
        close = _minute_close_today(key, today)
        if close is None:
            missing.append(train.NAMES[key])
            continue
        proxied[key][today] = close
    if missing:
        raise SystemExit(f"今天的15:00收盘还没有到 {train.session_day(series)}，分钟线也没有 {'、'.join(missing)}")
    print("今天日线还没到，先用分钟线收盘当今天", flush=True)
    return proxied


def _inputs(series: dict[str, dict[str, float]], news: dict[str, float], minute_day: str) -> dict:
    last = train.session_day(series)
    cutoff = train._day_after(last)
    us = {}
    for key, name in US_NAMES:
        days = sorted(series[key])
        bar = days[-1]
        us[name] = {
            "day": bar,
            "close": series[key][bar],
            "pct": round(train.overseas_session(series[key], last, cutoff), 4),
        }
    return {
        "as_of": last,
        "domestic": {train.NAMES[key]: series[key][last] for key in train.DOMESTIC},
        "us": us,
        "news": train.news_before(news, last, "9999-99-99"),
        "news_ema": round(train.news_ema_as_of(series, news, last), 4),
        "minute_as_of": minute_day,
    }


def minute_forward(days: dict[str, dict], overseas: dict, news: dict) -> tuple[str, dict]:
    """Features of the last minute day, the same way a training row is built."""

    closes = {key: {day: slot["close"] for day, slot in book.items()} for key, book in days.items()}
    domestic = {key: train.session_returns(key, closes[key]) for key in days}
    swings = {key: train.session_swing(key, closes[key]) for key in days}
    common = [day for day in sorted(set(swings["A"]) & set(swings["M"]) & set(swings["RM"])) if day >= "2017-01-01"]
    overseas_days = {key: sorted(overseas[key]) for key in train.OVERSEAS}
    for prev, day in zip(common, common[1:]):
        if any(train.latest_on_or_before(overseas_days[key], prev) is None for key in train.OVERSEAS):
            continue
        if any(prev not in domestic[key] or prev not in swings[key] for key in ("A", "M", "RM")):
            continue
    last = common[-1]
    cutoff = train._day_after(last)
    named = {
        "豆一昨涨跌": domestic["A"][last],
        "豆粕昨涨跌": domestic["M"][last],
        "菜粕昨涨跌": domestic["RM"][last],
        "美豆隔夜": train.overseas_session(overseas["S"], last, cutoff),
        "美豆粕隔夜": train.overseas_session(overseas["SM"], last, cutoff),
        "美豆油隔夜": train.overseas_session(overseas["BO"], last, cutoff),
        "豆一这段": swings["A"][last],
        "豆粕这段": swings["M"][last],
        "菜粕这段": swings["RM"][last],
    }
    named["菜粕价差波段"] = named["菜粕这段"] - named["豆粕这段"]
    for key in ("B", "Y"):
        if last in domestic.get(key, {}) and last in swings.get(key, {}):
            named[f"{train.NAMES[key]}昨涨跌"] = domestic[key][last]
            named[f"{train.NAMES[key]}这段"] = swings[key][last]
    for key in days:
        named.update(train.extra_named(key, last))
    named.update(train.cot_named(train.cot_series(), last))
    named.update(train.mood_named(train.cot_series(), train.seat_series(), last))
    for key, book in days.items():
        slot = book.get(last)
        if slot is None:
            continue
        name = train.NAMES[key]
        named[f"{name}夜盘"] = slot["night"]
        named[f"{name}日盘"] = slot["day"]
        named[f"{name}持仓变化"] = slot["oi_chg"]
        named[f"{name}成交量变化"] = slot["vol_chg"]
        named[f"{name}价仓"] = slot["att"]
    return last, named


def _view(key: str, prob: float) -> dict:
    return {
        "code": key,
        "name": train.NAMES[key],
        "prob_up": round(prob, 4),
        "lean": train.direction(prob),
    }


def _score_saved(folder: Path, named: dict, kind: str) -> list[dict]:
    views = []
    for key in train.CONTRACTS:
        model = json.loads((folder / f"{key}.json").read_text())
        names = model["feature_names"]
        if any(name not in named for name in names):
            continue
        vector = train.apply_scale([float(named[name]) for name in names], model["mean"], model["scale"])
        if kind == "logit":
            prob = train.predict_prob(vector, model["weight"], model["bias"])
        else:
            booster = lgb.Booster(model_file=str(folder / f"{key}.txt"))
            prob = float(booster.predict([vector])[0])
        views.append(_view(key, prob))
    return views


def score_all(series: dict[str, dict[str, float]], days: dict[str, dict], news: dict[str, float]) -> dict:
    minute_day, named = minute_forward(days, {key: series[key] for key in train.OVERSEAS}, news)
    tree_named = mail_tree._named(series)
    models = {
        "日线回归": train.outlook(series, train.load_models())["views"],
        "日线树": _score_saved(ROOT / "data" / "models_tree", tree_named, "tree"),
        "分钟回归": _score_saved(minute.LOGIT_OUT, named, "logit"),
        "分钟树": _score_saved(minute.TREE_OUT, named, "tree"),
    }
    for family, views in models.items():
        for view in views:
            view["variant"] = train.variant_for(view["code"], family)
    payload = {
        "inputs": _inputs(series, news, minute_day),
        "models": models,
    }
    from models import stack

    combos = {}
    for key in train.CONTRACTS:
        rule = train.combo_rule(key)
        probs = {family: (_by_code(models[family]).get(key) or {}).get("prob_up", 0.5) for family in FAMILIES}
        votes = False
        if rule == "多数票2":
            lean, ups, downs, mean = stack.majority(probs, need=2)
            votes = True
        elif rule == "平均":
            lean, ups, downs, mean = stack.average(probs)
        elif rule in FAMILIES:
            mean = probs.get(rule, 0.5)
            lean = train.direction(mean)
            ups = downs = 0
            rule = f"跟{rule}"
            votes = False
        else:
            continue
        combos[key] = {"lean": lean, "ups": ups, "downs": downs, "prob_up": round(mean, 4), "rule": rule, "votes": votes}
    if combos:
        payload["combo"] = combos
    return payload


def _load_days() -> dict[str, dict]:
    return {product: minute.load_days(product) for product in train.CONTRACTS}


def refresh() -> None:
    """Bring today's closes, US bars, news, and minute bars in. Do not fit."""

    series = train.load_series()
    now = datetime.now(TZ)
    if (now.hour, now.minute) >= (14, 50):
        today = now.date().isoformat()
        series = _proxy_today(series, today)
        print(
            "日线收盘 " + " ".join(f"{train.NAMES[key]}{series[key][today]:.0f}" for key in train.DOMESTIC),
            flush=True,
        )
    view = train.outlook(series, train.load_models())
    (train.DATA / "outlook.json").write_text(json.dumps(view, ensure_ascii=False, indent=2))
    from models import paper_book

    for note in paper_book.record_outlook(view, series):
        print(note, flush=True)
    print(f"数据集已更新到 {view['as_of']}，模型没有重训", flush=True)


def snapshot() -> Path:
    series = _proxy_today(_series_from_disk(), datetime.now(TZ).date().isoformat())
    news = train.news_scores()
    payload = score_all(series, _load_days(), news)
    payload["taken_at"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    payload["role"] = "14:50"
    COMPARE.mkdir(parents=True, exist_ok=True)
    path = COMPARE / f"{payload['inputs']['as_of']}_1450.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"下午方向已存 {path.name}", flush=True)
    return path


def _refit_logit(series: dict[str, dict[str, float]], news: dict[str, float]) -> None:
    rows = [row for row in train.samples(series, news) if row["day"] >= "2017-01-01"]
    trained_at = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    split = "all-history"
    sample = train.MODELS / "A.json"
    if sample.exists():
        split = json.loads(sample.read_text()).get("split") or split
    for key in train.CONTRACTS:
        vec_names = train.sina_feature_names(key, "日线回归")
        usable = [row for row in rows if key in row["y"]]
        spec = train.saved_train_params(key)
        lr = float(spec.get("lr", 0.08))
        steps = int(spec.get("steps", 500))
        l2 = float(spec.get("l2", 0.2))
        mean, scale = train.standardize([[row["named"][name] for name in vec_names] for row in usable])
        weight, bias = train.fit(
            [train.apply_scale([row["named"][name] for name in vec_names], mean, scale) for row in usable],
            [row["y"][key] for row in usable],
            [train.sample_weight(row["move"][key]) for row in usable],
            lr=lr,
            steps=steps,
            l2=l2,
        )
        train.save_contract(
            key,
            {
                "weight": weight,
                "bias": bias,
                "mean": mean,
                "scale": scale,
                "feature_names": list(vec_names),
                "train_params": {"lr": lr, "steps": steps, "l2": l2},
            },
            trained_at=trained_at,
            split=split,
        )
    train.save_common(trained_at, split)
    train.write_bundle(trained_at, split)


def _refit_minute(series: dict[str, dict[str, float]], news: dict[str, float], days: dict[str, dict]) -> None:
    overseas = {key: series[key] for key in train.OVERSEAS}
    rows = minute.build_rows(days, overseas, news)
    for key in train.CONTRACTS:
        got = minute.usable(rows, key, True)
        names, mean, scale, weight, bias, booster = minute.fit_pair(got, key, True)
        minute.save(key, names, mean, scale, weight, bias, booster)


def _changes(before: dict, after: dict) -> list[str]:
    lines = []
    moved = [
        f"{name} {before['domestic'][name]} → {after['domestic'][name]}"
        for name in before["domestic"]
        if before["domestic"][name] != after["domestic"][name]
    ]
    if moved:
        lines.append("内地收盘有变：" + "，".join(moved))
    else:
        lines.append("内地收盘没有变。夜盘 21:00 才开，这段没有新的内地成交。")
    for name in ("美豆", "美豆粕", "美豆油"):
        old, new = before["us"][name], after["us"][name]
        if old["day"] == new["day"] and old["close"] == new["close"]:
            lines.append(f"{name}日线没有新收盘，仍是 {old['day']} {old['close']}，隔夜 {old['pct']:+.2f}%")
        else:
            lines.append(
                f"{name} {old['day']} {old['close']} 隔夜 {old['pct']:+.2f}%"
                f" → {new['day']} {new['close']} 隔夜 {new['pct']:+.2f}%"
            )
    if before["news"] == after["news"] and before["news_ema"] == after["news_ema"]:
        lines.append(f"新闻没有新条目，倾向仍是 {before['news']}")
    else:
        lines.append(
            f"新闻倾向 {before['news']} → {after['news']}，三日 {before['news_ema']} → {after['news_ema']}"
        )
    if before["minute_as_of"] == after["minute_as_of"]:
        lines.append(f"分钟线主连仍停在 {before['minute_as_of']}，这段没有新的分钟线。")
    else:
        lines.append(f"分钟线主连 {before['minute_as_of']} → {after['minute_as_of']}")
    return lines


def _by_code(views: list[dict]) -> dict[str, dict]:
    return {item["code"]: item for item in views}


def _prev_day(as_of: str) -> str:
    earlier = [day for day, _close in train._cached_rows(train.series_path("A")) if day < as_of]
    return earlier[-1] if earlier else as_of


def _minute_span(prev: str) -> str:
    start = f"{prev} 20:30:00"
    path = train.MAIN_MINUTE / "A.csv"
    first = last = ""
    count = 0
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line[:4].isdigit() or line[:19] <= start:
                    continue
                stamp = line.split(",", 1)[0]
                if not first:
                    first = stamp
                last = stamp
                count += 1
    if not count:
        return "这段没有新的分钟线。"
    return f"分钟线 {first} → {last}，豆一这段 {count} 根。夜盘 21:00 才开，所以 20:30 不会比 15:00 多出内地成交。"


def _position_line(lean: str, key: str) -> str:
    hold = train.hold_rule(key)
    if lean == "偏涨":
        return "  持仓参考：空仓→开多  持多→持有  持空→平仓反手开多"
    if lean == "偏跌":
        return "  持仓参考：空仓→开空  持空→持有  持多→平仓反手开空"
    if hold == "没方向就平":
        return "  持仓参考：空仓→不动  持多→平仓  持空→平仓"
    return "  持仓参考：空仓→不动  持多→持有  持空→持有"


def _direction_lines(payload: dict) -> list[str]:
    models = payload["models"]
    combos = payload.get("combo") or {}
    lines = []
    for key in train.CONTRACTS:
        lines.append(train.NAMES[key])
        for family in FAMILIES:
            item = _by_code(models[family]).get(key)
            if item is None:
                lines.append(f"  {family}  没有")
                continue
            variant = item.get("variant")
            tag = f"（{variant}）" if variant and variant != "旧" else ""
            lines.append(f"  {family}{tag}  {item['prob_up'] * 100:.1f}%  {item['lean']}")
        combo = combos.get(key)
        if combo:
            if combo.get("votes"):
                lines.append(f"  综合（{combo['rule']}）  {combo['ups']}涨{combo['downs']}跌  {combo['prob_up'] * 100:.1f}%  {combo['lean']}")
            else:
                lines.append(f"  综合（{combo['rule']}）  {combo['prob_up'] * 100:.1f}%  {combo['lean']}")
            lines.append(_position_line(combo["lean"], key))
        lines.append("")
    return lines


OLD_FAMILY_NAMES = {"新浪回归": "日线回归", "新浪树": "日线树"}


def _body(as_of: str, before: dict, after: dict) -> str:
    # Snapshots from before the rename still carry the old family names.
    before["models"] = {OLD_FAMILY_NAMES.get(family, family): views for family, views in before["models"].items()}
    prev = _prev_day(as_of)
    lines = [
        f"20:30 方向。下一交易日。",
        f"新信息从上一交易日 {prev} 20:30 到这次 {as_of} 20:30。模型不重训，用现有权重再预测一次。",
        "这封和今天 14:50 的方向对照。开仓仍按 14:50 那封。",
        "",
        _minute_span(prev),
        "相对 14:50 又变了的：",
    ]
    lines.extend(_changes(before["inputs"], after["inputs"]))
    lines.append("")
    flipped = []
    for key in train.CONTRACTS:
        lines.append(train.NAMES[key])
        for family in FAMILIES:
            old = _by_code(before["models"][family]).get(key)
            new = _by_code(after["models"][family]).get(key)
            if old is None or new is None:
                lines.append(f"  {family}  这次没有方向可对照")
                continue
            lines.append(
                f"  {family}  {old['prob_up'] * 100:.1f}% {old['lean']}"
                f" → {new['prob_up'] * 100:.1f}% {new['lean']}"
            )
            if old["lean"] != new["lean"]:
                flipped.append(f"{train.NAMES[key]}{family} {old['lean']} → {new['lean']}")
        combo_new = (after.get("combo") or {}).get(key)
        if combo_new:
            combo_old = (before.get("combo") or {}).get(key)
            if combo_old:
                if combo_new.get("votes"):
                    lines.append(
                        f"  综合（{combo_new['rule']}）  {combo_old['ups']}涨{combo_old['downs']}跌 {combo_old['lean']}"
                        f" → {combo_new['ups']}涨{combo_new['downs']}跌 {combo_new['lean']}"
                    )
                else:
                    lines.append(
                        f"  综合（{combo_new['rule']}）  {combo_old['prob_up'] * 100:.1f}% {combo_old['lean']}"
                        f" → {combo_new['prob_up'] * 100:.1f}% {combo_new['lean']}"
                    )
                if combo_old["lean"] != combo_new["lean"]:
                    flipped.append(f"{train.NAMES[key]}综合 {combo_old['lean']} → {combo_new['lean']}")
            else:
                if combo_new.get("votes"):
                    lines.append(f"  综合（{combo_new['rule']}）  {combo_new['ups']}涨{combo_new['downs']}跌  {combo_new['lean']}")
                else:
                    lines.append(f"  综合（{combo_new['rule']}）  {combo_new['prob_up'] * 100:.1f}%  {combo_new['lean']}")
            lines.append(_position_line(combo_new["lean"], key))
        lines.append("")
    if flipped:
        lines.append("方向变了：" + "，".join(flipped))
    else:
        lines.append("20 个模型的方向都没有变。")
    lines.append("45%到55%之间仍然不算有方向。")
    return "\n".join(lines)


def _mail(subject: str, body: str) -> None:
    if not _google_enabled():
        raise RuntimeError("Mail 里的 Gmail 是停用的，信寄不出去")
    _forbid_company(SENDER)
    _forbid_company(TO)
    if _outbox_count(subject) == 0 and _locate(subject) == "no":
        _send(subject, body)
    for _ in range(8):
        place = _locate(subject)
        if place == "Bin":
            _pull_from_bin(subject)
            place = _locate(subject)
        if place in {"INBOX", "Sent Mail", "All Mail"}:
            return
        import time

        time.sleep(2)
    raise RuntimeError("对照信还没出现在收件箱")


def _ensure_sina_daily() -> None:
    """Training reads the daily files. A missing close is fetched by the daily daemon's pull."""

    today = datetime.now(TZ).date().isoformat()
    try:
        cached = {key: {day: close for day, close in train._cached_rows(train.series_path(key))} for key in train.DOMESTIC}
        ready = train.session_day(cached) == today
    except SystemExit:
        ready = False
    if ready:
        print(f"新浪日线已有 {today}", flush=True)
        return
    print("新浪日线还没有今天的收盘，调用日线进程去拉", flush=True)
    ok, detail = sina_daily_daemon.pull()
    print(detail, flush=True)
    if not ok:
        raise SystemExit(detail)


def evening() -> None:
    _ensure_sina_daily()
    series = _series_from_disk()
    as_of = train.session_day(series)
    snap_path = COMPARE / f"{as_of}_1450.json"
    if not snap_path.exists():
        raise SystemExit(f"没有下午的方向存档 {snap_path.name}")
    before = json.loads(snap_path.read_text())
    out_path = COMPARE / f"{as_of}_2030.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    else:
        print("拉取美盘", flush=True)
        series = train.load_series()
        news = {}
        try:
            from training_dataset.cot import fetch_cot

            cot = fetch_cot.fetch_current()
            for name, row in cot.items():
                print(f"COT {name} {row['日期']} 基金净多 {row['基金净多']}", flush=True)
        except Exception as exc:
            print(f"CFTC 持仓拉取失败，沿用已有文件 {exc}", flush=True)
        try:
            from training_dataset.position import fetch_position

            for product, row in fetch_position.update(as_of).items():
                print(f"持仓排名 {train.NAMES[product]} {row['日期']} 前20净多 {row['净多']}", flush=True)
        except Exception as exc:
            print(f"持仓排名拉取失败，沿用已有文件 {exc}", flush=True)
        days = _load_days()
        print("沿用已有的 20 个模型，不再重训", flush=True)
        payload = score_all(series, days, news)
        payload["taken_at"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        payload["role"] = "20:30"
        COMPARE.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    body = _body(as_of, before, payload)
    print(body, flush=True)
    _mail(f"下一交易日方向（20:30，基于{as_of}，相对14:50）", body)
    print(f"20:30 方向已发出 {as_of}", flush=True)


def mail_1450() -> None:
    """One afternoon mail for all 20 models. The 20:30 mail is the second."""

    series = _proxy_today(_series_from_disk(), datetime.now(TZ).date().isoformat())
    as_of = train.session_day(series)
    path = COMPARE / f"{as_of}_1450.json"
    if not path.exists():
        snapshot()
    payload = json.loads(path.read_text())
    sent_path = ROOT / "data" / "futures_1450_emailed.txt"
    if sent_path.exists() and sent_path.read_text().strip() == as_of:
        print(f"already sent {as_of}", flush=True)
        return
    prev = _prev_day(as_of)
    lines = [
        f"14:50 方向。下一交易日，基于{as_of}收盘。",
        "",
    ]
    lines.extend(_direction_lines(payload))
    lines.append("45%到55%之间不下新单。")
    lines.append(f"20:30 会用 {prev} 20:30 到当时的新信息再预测一次，模型不重训，方向和差别另发一封。")
    body = "\n".join(lines)
    print(body, flush=True)
    _mail(f"下一交易日方向（14:50，基于{as_of}）", body)
    sent_path.write_text(as_of + "\n")
    print(f"14:50 方向已发出 {as_of}", flush=True)


def main() -> int:
    if "--snapshot" in sys.argv:
        snapshot()
        return 0
    if "--refresh" in sys.argv:
        refresh()
        return 0
    if "--mail1450" in sys.argv:
        mail_1450()
        return 0
    evening()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
