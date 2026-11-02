#!/usr/bin/env python3
"""Pull the eight Sina daily closes (five traded + 棕榈油/菜籽油/花生). Nothing else.

From 18:00, retry until today's 15:00 bar is in. One failure mail per day
if it is still missing at 18:20. The subject is fixed. The 14:50 mail does
not wait for this bar; the 20:30 prediction reads it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import train
from jobs.mail_outlook import SENDER, TO, _forbid_company, _google_enabled, _locate, _outbox_count, _pull_from_bin, _send
TZ = ZoneInfo("Asia/Shanghai")
LOG = ROOT / "sina-daily-daemon.log"
PIDFILE = ROOT / "data" / "sina_daily_daemon.pid"
GOT = ROOT / "data" / "sina_daily_fetched.txt"
MAILED = ROOT / "data" / "sina_daily_fail_mailed.txt"
SUBJECT = "新浪日线获取失败"


def _today(now: datetime) -> str:
    return now.date().isoformat()


def _marked(path: Path, today: str) -> bool:
    return path.exists() and path.read_text().strip() == today


def pull() -> tuple[bool, str, dict]:
    """Write the domestic files. True only when every one ends on today."""

    today = datetime.now(TZ).date().isoformat()
    problems = []
    fresh: dict[str, list[dict]] = {}
    train.DATA.mkdir(parents=True, exist_ok=True)
    for key, symbol in (*train.DOMESTIC.items(), *train.SINA_EXTRA.items()):
        name = train.NAMES.get(key) or train.EXTRA_NAMES.get(key, key)
        path = train.series_path(key)
        try:
            rows = train.fetch_domestic(symbol)
            rows = [row for row in rows if str(row.get("日期") or "") >= train.DATA_FLOOR]
            if not rows:
                raise RuntimeError("空行情")
        except Exception as exc:
            problems.append(f"{name} 拉取失败 {exc}")
            continue
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        fresh[key] = rows
        last = str(rows[-1]["日期"])
        close = float(rows[-1]["收盘"])
        if last != today:
            problems.append(f"{name} 最后一根是 {last} {close:.0f}")
            continue
        print(f"{name} {last} {close:.0f}", flush=True)
    if problems:
        return False, "\n".join(problems), fresh
    return True, "八个品种今天的15:00收盘都到了", fresh


def _update_roll_calendar(fresh: dict) -> None:
    """Append today to the roll calendar; a confirmed roll for a traded product
    also reconciles the CTP subscription before the night session."""

    from ingest import roll_calendar
    from training_dataset.main_minute.main_minute_daemon import _ensure_subscribed, drop_foreign_day_bars

    def hook(product: str, _old: str, new: str) -> None:
        _ensure_subscribed(product, new)

    rolls = roll_calendar.update_today(fresh, traded_hook=hook)
    for product in roll_calendar.PRODUCTS:
        drop_foreign_day_bars(product)
    from ingest.roll_reference import apply

    for product in {item[0] for item in rolls}:
        apply(product)


def _mail(body: str) -> None:
    if not _google_enabled():
        raise RuntimeError("Mail 里的 Gmail 是停用的，信寄不出去")
    _forbid_company(SENDER)
    _forbid_company(TO)
    if _outbox_count(SUBJECT) == 0 and _locate(SUBJECT) == "no":
        _send(SUBJECT, body)
    for _ in range(8):
        place = _locate(SUBJECT)
        if place == "Bin":
            _pull_from_bin(SUBJECT)
            place = _locate(SUBJECT)
        if place in {"INBOX", "Sent Mail", "All Mail"}:
            return
        time.sleep(2)
    raise RuntimeError("失败信还没出现在收件箱")


def serve() -> None:
    print(f"sina daily daemon {os.getpid()} started", flush=True)
    while True:
        try:
            now = datetime.now(TZ)
            today = _today(now)
            if now.weekday() < 5 and (now.hour, now.minute) >= (18, 0) and not _marked(GOT, today):
                ok, detail, fresh = pull()
                if ok:
                    GOT.write_text(today + "\n")
                    print(detail, flush=True)
                    try:
                        _update_roll_calendar(fresh)
                    except Exception as exc:
                        print(f"换月日历续期失败 {exc}", flush=True)
                else:
                    print(detail, flush=True)
                    if (now.hour, now.minute) >= (18, 20) and not _marked(MAILED, today):
                        _mail(f"{today} 的新浪日线没有拉下来。\n\n{detail}")
                        MAILED.write_text(today + "\n")
                        print("失败信已发出", flush=True)
        except Exception as exc:
            print(f"sina daily error {exc}", flush=True)
        time.sleep(60)


def daemonize() -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)
    os.chdir(ROOT)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    log = os.open(str(LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open("/dev/null", os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(log, 1)
    os.dup2(log, 2)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        daemonize()
    serve()
