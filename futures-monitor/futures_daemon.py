#!/usr/bin/env python3
"""Keep the futures board up, and refit the model from the 14:50 price.

Start with: python3 futures_daemon.py --daemon
The process detaches from the terminal. It does not place orders.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PY = "/opt/homebrew/bin/python3"
TZ = ZoneInfo("Asia/Shanghai")
LOG = ROOT / "futures-daemon.log"
PIDFILE = ROOT / "data" / "futures_daemon.pid"
TRAINED = ROOT / "data" / "last_train_date.txt"


def _pids(mark: str) -> list[int]:
    result = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True)
    found = []
    for line in result.stdout.splitlines():
        if mark not in line or "futures_daemon.py" in line:
            continue
        found.append(int(line.split(None, 1)[0]))
    return found


def ensure_board() -> None:
    if _pids("monitor.py --web"):
        return
    log = open(ROOT / "monitor.log", "a")
    err = open(ROOT / "monitor.err.log", "a")
    proc = subprocess.Popen(
        [PY, "monitor.py", "--web", "--port", "8765"],
        cwd=ROOT,
        stdout=log,
        stderr=err,
        start_new_session=True,
    )
    print(f"started board pid {proc.pid}", flush=True)


def train_if_due() -> None:
    now = datetime.now(TZ)
    # 14:50 的价格当作收盘。四个合约都到了才训练，不等到 15:00。
    if (now.hour, now.minute) < (14, 50):
        return
    today = now.date().isoformat()
    if TRAINED.exists() and TRAINED.read_text().strip() == today:
        return
    stamp = ROOT / "data" / "train_retry_after.txt"
    retry_after = None
    if stamp.exists():
        retry_after = datetime.fromisoformat(stamp.read_text().strip())
        if now < retry_after:
            return
    print(f"running train.py {now:%Y-%m-%d %H:%M}", flush=True)
    result = subprocess.run([PY, "train.py"], cwd=ROOT)
    if result.returncode == 0:
        TRAINED.write_text(today)
        stamp.unlink(missing_ok=True)
        print("train.py finished", flush=True)
        tape = subprocess.run([PY, "session_tape.py", today], cwd=ROOT, capture_output=True, text=True)
        print((tape.stdout or tape.stderr).strip() or f"session_tape {tape.returncode}", flush=True)
        mail_outlook()
        return
    print(f"train.py failed {result.returncode}", flush=True)
    if (now.hour, now.minute) < (15, 0):
        return
    if retry_after is not None and now >= retry_after:
        TRAINED.write_text(today)
        return
    stamp.write_text((now + timedelta(hours=1)).isoformat())


def session_tape_if_due() -> None:
    now = datetime.now(TZ)
    if (now.hour, now.minute) < (15, 1):
        return
    today = now.date().isoformat()
    mark = ROOT / "data" / "session_tape_day.txt"
    if mark.exists() and mark.read_text().strip() == today:
        return
    result = subprocess.run([PY, "session_tape.py", today], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text or f"session_tape {result.returncode}", flush=True)
    if text.startswith("盘面记下") or text.startswith("盘面已记"):
        mark.write_text(today)


def mail_outlook() -> None:
    result = subprocess.run([PY, "mail_outlook.py"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        if text.startswith("already sent"):
            return
        print(f"mail {text}", flush=True)
        return
    print(f"mail failed {text}", flush=True)


def _market_open(now: datetime) -> bool:
    """Dalian day and night sessions. The breaks and the close are not mailed."""

    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    sessions = ((9 * 60, 10 * 60 + 15), (10 * 60 + 30, 11 * 60 + 30), (13 * 60 + 30, 15 * 60), (21 * 60, 23 * 60))
    return any(start <= minutes < end for start, end in sessions)


def tape_mail_if_due() -> None:
    """One Gmail tape note per clock hour while the market is open. Never the company mailbox."""

    now = datetime.now(TZ)
    if not _market_open(now):
        return
    key = now.strftime("%Y-%m-%d %H")
    mark = ROOT / "data" / "tape_emailed_hour.txt"
    if mark.exists() and mark.read_text().strip() == key:
        return
    result = subprocess.run([PY, "mail_tape.py"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text or f"tape mail {result.returncode}", flush=True)
    if result.returncode == 0 and text.startswith("sent"):
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(key + "\n")


def mail_if_due() -> None:
    """Retry Gmail after 14:50 until today's outlook has been sent. Never uses the company mailbox."""

    now = datetime.now(TZ)
    if (now.hour, now.minute) < (14, 50):
        return
    outlook = ROOT / "data" / "outlook.json"
    if not outlook.exists():
        return
    view = json.loads(outlook.read_text())
    if str(view.get("as_of") or "") != now.date().isoformat():
        return
    mail_outlook()


def ensure_tape() -> None:
    if _pids("tape.py"):
        return
    out = open(ROOT / "tape.log", "a")
    err = open(ROOT / "tape.err.log", "a")
    proc = subprocess.Popen(
        [PY, "-u", "tape.py"],
        cwd=ROOT,
        stdout=out,
        stderr=err,
        start_new_session=True,
    )
    print(f"started tape pid {proc.pid}", flush=True)


def ensure_quotes() -> None:
    """Keep the SimNow recorder up. The venv has CTP; the system Python does not."""

    if _pids("simnow_md.py"):
        return
    out = open(ROOT / "simnow_md.log", "a")
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python3"), "-u", "simnow_md.py"],
        cwd=ROOT,
        stdout=out,
        stderr=out,
        start_new_session=True,
    )
    print(f"started quotes pid {proc.pid}", flush=True)


def serve() -> None:
    print(f"futures daemon {os.getpid()} started", flush=True)
    while True:
        try:
            ensure_board()
            ensure_tape()
            ensure_quotes()
            train_if_due()
            session_tape_if_due()
            mail_if_due()
            tape_mail_if_due()
        except Exception as exc:
            print(f"daemon error {exc}", flush=True)
        time.sleep(30)


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
