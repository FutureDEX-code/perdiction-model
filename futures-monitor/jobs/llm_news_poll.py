#!/usr/bin/env python3
"""News-only model votes. Asked at 14:40. Never a training feature.

Ask from headlines only. Do not read the board or feed prices into the question.
Score later, from that day's 14:55 to the next session's 14:55. Watch through 2026-11-26.
The 20:30 fit does not call this. After the votes are saved, one mail goes out so
the 14:50 order decision can see them: python3 jobs/llm_news_poll.py mail
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    """Grade old polls. Do not ask here, and do not hand anything to the fit."""

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
        print(f"今天新闻提问已有 {today_file.name}", flush=True)
    else:
        print("今天新闻提问还没有。训练不等它，也不用它。", flush=True)


def _tally(votes: list[dict]) -> dict[str, list[str]]:
    """Product -> the models calling it up, and the models calling it down."""

    out = {name: [0, 0, [], []] for _key, name in PRODUCTS}
    for vote in votes:
        call = vote.get("call") or {}
        for _key, name in PRODUCTS:
            side = call.get(name)
            if side == "涨":
                out[name][0] += 1
                out[name][2].append(vote["model"])
            elif side == "跌":
                out[name][1] += 1
                out[name][3].append(vote["model"])
    return out


def poll_mail_body(day: str) -> str:
    """The 14:40 mail: who says up or down, plus any polls that can be graded now."""

    path = POLL / f"{day}.json"
    if not path.exists():
        raise RuntimeError(f"没有 {day} 的投票记录")
    payload = json.loads(path.read_text())
    votes = payload.get("news_votes") or []
    if not votes:
        raise RuntimeError(f"{day} 还没有有效投票")
    tally = _tally(votes)
    lines = [
        f"{day} 14:40 问的结果（只从新闻，不看盘，不进训练，仅作 14:50 下单参考）",
        "",
    ]
    for _key, name in PRODUCTS:
        up, down, up_models, down_models = tally[name]
        lean = "偏涨" if up > down else ("偏跌" if down > up else "对半")
        lines.append(f"{name}  {up}涨{down}跌 {lean}")
        lines.append(f"  涨：{'、'.join(up_models) or '无'}")
        lines.append(f"  跌：{'、'.join(down_models) or '无'}")
    lines.append("")
    lines.append("各家明细：")
    for vote in votes:
        call = vote.get("call") or {}
        cells = " ".join(f"{name}{call.get(name, '？')}" for _key, name in PRODUCTS)
        lines.append(f"  {vote['model']}  {cells}")
    skipped = payload.get("not_asked") or []
    blocked = payload.get("blocked") or []
    abandoned = payload.get("abandoned") or []
    if skipped or blocked or abandoned:
        lines.append("")
        lines.append(f"没问：{'、'.join([*skipped, *blocked, *abandoned])}")
    graded_notes = []
    for old in sorted(POLL.glob("20*.json")):
        try:
            note = grade_file(old)
        except Exception:
            continue
        if note:
            graded_notes.append(note)
    lines.append("")
    if graded_notes:
        lines.extend(graded_notes)
    else:
        lines.append("没有对答案的新结果。最近一票要等今天 14:55 之后才能对。")
    lines.append(f"只记账，不进训练。观察到 {WATCH_UNTIL}。")
    return "\n".join(lines)


def send_poll_mail(day: str | None = None, dry: bool = False) -> str:
    from jobs.mail_outlook import _send

    today = day or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    body = poll_mail_body(today)
    subject = f"大模型新闻投票 {today}"
    if dry:
        print(f"主题：{subject}\n\n{body}", flush=True)
        return body
    _send(subject, body)
    print(f"已寄出 {subject}", flush=True)
    return body


if __name__ == "__main__":
    args = [item for item in sys.argv[1:] if not item.startswith("--")]
    if args and args[0] == "mail":
        send_poll_mail(args[1] if len(args) > 1 else None, dry="--dry" in sys.argv)
    else:
        before_train()
