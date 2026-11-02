#!/usr/bin/env python3
"""Keep the futures board up. Data refreshes in the afternoon. 20:30 predicts again without refitting.

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    if _pids("live.monitor --web"):
        return
    log = open(ROOT / "monitor.log", "a")
    err = open(ROOT / "monitor.err.log", "a")
    proc = subprocess.Popen(
        [PY, "-m", "live.monitor", "--web", "--port", "8765"],
        cwd=ROOT,
        stdout=log,
        stderr=err,
        start_new_session=True,
    )
    print(f"started board pid {proc.pid}", flush=True)


def train_if_due() -> None:
    now = datetime.now(TZ)
    # 下午只补数据并给出方向。20:30 用已有模型再预测，不重训。
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
    print(f"refreshing data {now:%Y-%m-%d %H:%M}", flush=True)
    result = subprocess.run([_venv(), "-u", "-m", "jobs.evening_compare", "--refresh"], cwd=ROOT)
    if result.returncode == 0:
        TRAINED.write_text(today)
        stamp.unlink(missing_ok=True)
        print("data refresh finished", flush=True)
        tape = subprocess.run([PY, "-m", "jobs.session_tape", today], cwd=ROOT, capture_output=True, text=True)
        print((tape.stdout or tape.stderr).strip() or f"session_tape {tape.returncode}", flush=True)
        snapshot_1450()
        mail_1450()
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
    result = subprocess.run([PY, "-m", "jobs.session_tape", today], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text or f"session_tape {result.returncode}", flush=True)
    if text.startswith("盘面记下") or text.startswith("盘面已记"):
        mark.write_text(today)


def mail_1450() -> None:
    """The one afternoon mail. 20:30 sends the second. Logistic and tree are not mailed apart."""

    result = subprocess.run([_venv(), "-u", "-m", "jobs.evening_compare", "--mail1450"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        if text.startswith("already sent"):
            return
        print(f"14:50 mail {text.splitlines()[-1] if text else 'sent'}", flush=True)
        return
    print(f"14:50 mail failed {text}", flush=True)


def _market_open(now: datetime) -> bool:
    """Dalian day and night sessions. The breaks and the close are not mailed."""

    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    sessions = ((9 * 60, 10 * 60 + 15), (10 * 60 + 30, 11 * 60 + 30), (13 * 60 + 30, 15 * 60), (21 * 60, 23 * 60))
    return any(start <= minutes < end for start, end in sessions)


def tape_mail_if_due() -> None:
    """One Gmail tape note per half hour while the market is open. Never the company mailbox."""

    now = datetime.now(TZ)
    if not _market_open(now):
        return
    minute = 0 if now.minute < 30 else 30
    key = f"{now:%Y-%m-%d %H}:{minute:02d}"
    mark = ROOT / "data" / "tape_emailed_hour.txt"
    if mark.exists() and mark.read_text().strip() == key:
        return
    result = subprocess.run([PY, "-m", "jobs.mail_tape"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text or f"tape mail {result.returncode}", flush=True)
    if result.returncode == 0 and text.startswith("sent"):
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(key + "\n")


def _venv() -> str:
    return str(ROOT / ".venv/bin/python3")


def snapshot_1450() -> None:
    """Keep the afternoon directions so 20:30 can compare against them."""

    result = subprocess.run([_venv(), "-u", "-m", "jobs.evening_compare", "--snapshot"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text or f"snapshot {result.returncode}", flush=True)


def snapshot_if_missing() -> None:
    now = datetime.now(TZ)
    if (now.hour, now.minute) < (14, 50):
        return
    today = now.date().isoformat()
    if not TRAINED.exists() or TRAINED.read_text().strip() != today:
        return
    if (ROOT / "data" / "compare" / f"{today}_1450.json").exists():
        return
    snapshot_1450()


def evening_if_due() -> None:
    """The only daily fit. Afternoon has already stored today's direction."""

    now = datetime.now(TZ)
    if (now.hour, now.minute) < (20, 30):
        return
    today = now.date().isoformat()
    mark = ROOT / "data" / "evening_compared.txt"
    if mark.exists() and mark.read_text().strip() == today:
        return
    if not (ROOT / "data" / "compare" / f"{today}_1450.json").exists():
        return
    print(f"running 20:30 compare {now:%Y-%m-%d %H:%M}", flush=True)
    result = subprocess.run([_venv(), "-u", "-m", "jobs.evening_compare"], cwd=ROOT)
    if result.returncode == 0:
        mark.write_text(today)
        print("20:30 compare finished", flush=True)
        review = subprocess.run([_venv(), "-u", "-m", "jobs.review"], cwd=ROOT, capture_output=True, text=True)
        print((review.stdout or review.stderr).strip() or f"review {review.returncode}", flush=True)
        return
    print(f"20:30 compare failed {result.returncode}", flush=True)


def halfmonth_if_due() -> None:
    """The 1st and 16th review mail, after the evening compare."""

    now = datetime.now(TZ)
    today = now.date().isoformat()
    if now.day not in (1, 16) or (now.hour, now.minute) < (20, 35):
        return
    mark = ROOT / "data" / "halfmonth_reviewed.txt"
    if mark.exists() and mark.read_text().strip() == today:
        return
    result = subprocess.run([_venv(), "-u", "-m", "jobs.review", "--halfmonth"], cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    print(text.splitlines()[-1] if text else f"halfmonth {result.returncode}", flush=True)
    if result.returncode == 0:
        mark.write_text(today)


def gainmoney_mail_if_due() -> None:
    """The 20:35 mail for the eight new models. Sent once a day."""

    now = datetime.now(TZ)
    if (now.hour, now.minute) < (20, 35):
        return
    today = now.date().isoformat()
    sent = ROOT / "data" / "gainmoney_new_models_emailed.txt"
    if sent.exists() and sent.read_text().strip() == today:
        return
    result = subprocess.run([_venv(), "-u", "-m", "gainmoney.mail_new_models"], cwd=ROOT, capture_output=True, text=True)
    print((result.stdout or result.stderr).strip(), flush=True)


def mail_if_due() -> None:
    """Retry the single 14:50 mail. The 20:30 mail is separate. Never the company mailbox."""

    now = datetime.now(TZ)
    if (now.hour, now.minute) < (14, 50):
        return
    today = now.date().isoformat()
    sent = ROOT / "data" / "futures_1450_emailed.txt"
    if sent.exists() and sent.read_text().strip() == today:
        return
    if not (ROOT / "data" / "compare" / f"{today}_1450.json").exists():
        return
    mail_1450()


def ensure_tape() -> None:
    if _pids("live.tape"):
        return
    out = open(ROOT / "tape.log", "a")
    err = open(ROOT / "tape.err.log", "a")
    proc = subprocess.Popen(
        [PY, "-u", "-m", "live.tape"],
        cwd=ROOT,
        stdout=out,
        stderr=err,
        start_new_session=True,
    )
    print(f"started tape pid {proc.pid}", flush=True)


def ensure_quotes() -> None:
    """Keep the SimNow recorder up. The venv has CTP; the system Python does not."""

    if _pids("live.simnow_md"):
        return
    out = open(ROOT / "simnow_md.log", "a")
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python3"), "-u", "-m", "live.simnow_md"],
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
            snapshot_if_missing()
            evening_if_due()
            halfmonth_if_due()
            session_tape_if_due()
            mail_if_due()
            gainmoney_mail_if_due()
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
