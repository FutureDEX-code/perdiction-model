#!/usr/bin/env python3
"""News-only model votes. Asked at 18:00. Never a training feature.

Ask from headlines only. Do not read the board or feed prices into the question.
Score later, from that day's 14:55 to the next session's 14:55. Watch through 2026-11-26.
The 20:30 fit does not call this.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
POLL = ROOT / "data" / "llm_poll"
RULES = POLL / "规则.json"
MINUTE = ROOT / "training_dataset" / "main_minute"
PRODUCTS = (("A", "豆一"), ("B", "豆二"), ("M", "豆粕"), ("RM", "菜粕"), ("Y", "豆油"))
WATCH_UNTIL = "2026-11-26"


def _rules() -> dict:
    if not RULES.exists():
        return {}
    return json.loads(RULES.read_text())


def _price(key: str, day: str, hm: str = "14:55") -> float | None:
    path = MINUTE / f"{key}.csv"
    if not path.exists():
        return None
    stamp = f"{day} {hm}:00"
    found = None
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("trade_time") == stamp and row.get("close"):
                found = float(row["close"])
    return found


def _next_day(key: str, day: str) -> str | None:
    path = MINUTE / f"{key}.csv"
    if not path.exists():
        return None
    days = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            trade = row.get("trade_time") or ""
            if trade.endswith("14:55:00"):
                days.append(trade[:10])
    later = sorted({item for item in days if item > day})
    return later[0] if later else None


def grade_file(path: Path) -> str | None:
    """Fill hits once both 14:55 prices exist. Returns a one-line summary."""

    payload = json.loads(path.read_text())
    if payload.get("graded"):
        return None
    score_from = str(payload.get("score_from") or "")
    day = score_from[:10]
    if len(day) != 10:
        return None
    votes = payload.get("news_votes") or []
    if not votes:
        return None
    nxt = _next_day("A", day)
    if nxt is None:
        return None
    actual = {}
    for key, name in PRODUCTS:
        left = _price(key, day)
        right = _price(key, nxt)
        if left is None or right is None or right == left:
            return None
        actual[name] = "涨" if right > left else "跌"
    rows = []
    for vote in votes:
        call = vote.get("call") or {}
        hits = []
        for _key, name in PRODUCTS:
            side = call.get(name)
            if side not in ("涨", "跌"):
                continue
            hits.append(side == actual[name])
        if not hits:
            continue
        vote["hit"] = sum(hits)
        vote["asked"] = len(hits)
        rows.append(f"{vote['model']} {vote['hit']}/{vote['asked']}")
    payload["graded"] = {
        "to": nxt,
        "actual": actual,
        "models": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"新闻提问对答案 {day}→{nxt} " + "，".join(rows)


def before_train() -> None:
    """Grade old polls at 18:00. Do not ask here, and do not hand anything to the fit."""

    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    if today > WATCH_UNTIL:
        print(f"新闻提问观察期到 {WATCH_UNTIL} 已结束，不进训练", flush=True)
        return
    skip = "、".join(_rules().get("不问") or [])
    print(f"新闻提问只记账，不进训练。观察到 {WATCH_UNTIL}。不问 {skip}", flush=True)
    if not POLL.exists():
        return
    notes = []
    for path in sorted(POLL.glob("20*.json")):
        try:
            note = grade_file(path)
        except Exception as exc:
            print(f"新闻提问对答案失败 {path.name} {exc}", flush=True)
            continue
        if note:
            notes.append(note)
    for note in notes:
        print(note, flush=True)
    today_file = POLL / f"{today}.json"
    if today_file.exists() and json.loads(today_file.read_text()).get("news_votes"):
        print(f"今晚新闻提问已有 {today_file.name}", flush=True)
    else:
        print("今晚新闻提问还没有。训练不等它，也不用它。", flush=True)
