#!/usr/bin/env python3
"""Keep the Hong Kong / Paris / New York paper-trade watcher running.

Start with: python3 watch_daemon.py --daemon
At 09:00 HKT it appends the paper-ledger totals. It does not change rules,
place orders, or send mail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PY = "/opt/homebrew/bin/python3"
TZ = ZoneInfo("Asia/Hong_Kong")
DATA = ROOT / "hk_temp_model" / "data"
LOG = DATA / "watch-daemon.log"
PIDFILE = DATA / "watch_daemon.pid"
WATCH_PID = DATA / "watch.pid"
REVIEW = DATA / "daily_review.log"
LEDGER = DATA / "paper_trades.json"


def _watch_pids() -> list[int]:
    result = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True)
    found = []
    for line in result.stdout.splitlines():
        if "hk_temp_model" in line and "--watch" in line and "watch_daemon.py" not in line:
            found.append(int(line.split(None, 1)[0]))
    return found


def ensure_watch() -> None:
    running = _watch_pids()
    if running:
        WATCH_PID.write_text(str(running[0]))
        return
    out = open(DATA / "watch.log", "a")
    err = open(DATA / "watch.err.log", "a")
    proc = subprocess.Popen(
        [PY, "-u", "-m", "hk_temp_model", "--watch", "10", "--cities", "hk,paris,nyc"],
        cwd=ROOT,
        stdout=out,
        stderr=err,
        start_new_session=True,
    )
    WATCH_PID.write_text(str(proc.pid))
    print(f"started watch pid {proc.pid}", flush=True)


def review_if_due() -> None:
    now = datetime.now(TZ)
    if now.hour < 9:
        return
    today = now.date().isoformat()
    mark = DATA / "last_review_date.txt"
    if mark.exists() and mark.read_text().strip() == today:
        return
    rows = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    done = [row for row in rows if row.get("pnl") is not None]
    pnl = sum(row["pnl"] for row in done)
    line = (
        f"{now:%Y-%m-%d %H:%M} HKT  模拟盘  已结算 {len(done)} 笔  "
        f"盈亏 {pnl:+.2f} 美元  未结算 {len(rows) - len(done)} 笔"
    )
    with REVIEW.open("a") as handle:
        handle.write(line + "\n")
    mark.write_text(today)
    print(line, flush=True)


def serve() -> None:
    print(f"watch daemon {os.getpid()} started", flush=True)
    while True:
        try:
            ensure_watch()
            review_if_due()
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
    DATA.mkdir(parents=True, exist_ok=True)
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
