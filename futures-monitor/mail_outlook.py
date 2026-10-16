#!/usr/bin/env python3
"""Email the next-session outlook after training.

The Google account in Mail.app is disabled, and smtp.bixin.com does not
resolve, so Mail.app only queues a message and never delivers it. This script
therefore sends with Mail only when that account is enabled, and it does not
record the day as emailed if the message is still sitting in the Outbox.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTLOOK = ROOT / "data" / "outlook.json"
SENT = ROOT / "data" / "futures_outlook_emailed.txt"
# Gmail to the same Gmail. The company mailbox is never a sender or a recipient.
SENDER = "467588110zx@gmail.com"
TO = SENDER


def _forbid_company(address: str) -> str:
    if "bixin.com" in address.lower():
        raise RuntimeError("不使用公司邮箱")
    return address


def _body(view: dict) -> str:
    lines = [f"基于{view['as_of']} 14:50 的价格，下一交易日：", ""]
    for item in view["views"]:
        lines.append(f"{item['name']} 上涨概率 {item['prob_up'] * 100:.1f}%  {item['lean']}")
    lines.append("")
    lines.append("45%到55%之间按规则不下新单。")
    return "\n".join(lines)


def _google_enabled() -> bool:
    script = 'tell application "Mail" to get enabled of account "Google"'
    result = subprocess.run(["osascript", "-e", script], text=True, capture_output=True)
    return result.returncode == 0 and result.stdout.strip() == "true"


def _outbox_count(subject: str) -> int:
    script = """
on run argv
  set wanted to item 1 of argv
  tell application "Mail"
    set n to 0
    repeat with msg in outgoing messages
      if subject of msg is wanted then set n to n + 1
    end repeat
    return n
  end tell
end run
"""
    result = subprocess.run(["osascript", "-", subject], input=script, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return int(result.stdout.strip() or "0")


def _locate(subject: str) -> str:
    """Where this subject sits. Sent mail that Gmail accepted can still land in the Bin."""

    script = """
on run argv
  set wanted to item 1 of argv
  tell application "Mail"
    set acct to account "Google"
    repeat with boxName in {"INBOX", "Sent Mail", "All Mail", "Bin"}
      set wantedBox to boxName as text
      repeat with mbox in mailboxes of acct
        if (name of mbox as text) is wantedBox then
          set hits to messages of mbox whose subject is wanted
          if (count of hits) > 0 then return wantedBox
        end if
      end repeat
    end repeat
  end tell
  return "no"
end run
"""
    result = subprocess.run(["osascript", "-", subject], input=script, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip() or "no"


def _pull_from_bin(subject: str) -> None:
    script = """
on run argv
  set wanted to item 1 of argv
  tell application "Mail"
    set acct to account "Google"
    set dest to missing value
    set src to missing value
    repeat with mbox in mailboxes of acct
      if name of mbox is "INBOX" then set dest to mbox
      if name of mbox is "Bin" then set src to mbox
    end repeat
    if dest is missing value or src is missing value then error "找不到收件箱"
    set hits to messages of src whose subject is wanted
    if (count of hits) is 0 then return "none"
    move item 1 of hits to dest
    return "moved"
  end tell
end run
"""
    result = subprocess.run(["osascript", "-", subject], input=script, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _send(subject: str, body: str) -> None:
    script = """
on run argv
  set theSubject to item 1 of argv
  set theBody to item 2 of argv
  set theTo to item 3 of argv
  set theSender to "ZHENG XU <" & theTo & ">"
  tell application "Mail"
    set msg to make new outgoing message with properties {subject:theSubject, content:theBody, visible:false, sender:theSender}
    tell msg
      make new to recipient at end of to recipients with properties {address:theTo}
    end tell
    set who to sender of msg
    if who contains "bixin.com" then
      delete msg
      error "refusing company mailbox"
    end if
    send msg
  end tell
end run
"""
    result = subprocess.run(
        ["osascript", "-", subject, body, _forbid_company(SENDER)],
        input=script,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"osascript exited {result.returncode}")
    time.sleep(3)


def send_outlook() -> str:
    view = json.loads(OUTLOOK.read_text())
    as_of = str(view.get("as_of") or "")
    if not as_of or not view.get("views"):
        return "no outlook"
    sent = SENT.read_text().strip() if SENT.exists() else ""
    if sent == as_of:
        return f"already sent {as_of}"
    if not _google_enabled():
        raise RuntimeError("Mail 里的 Gmail 是停用的，信寄不出去")
    _forbid_company(SENDER)
    _forbid_company(TO)
    subject = f"下一交易日期货预测（基于{as_of}收盘）"
    if _outbox_count(subject) == 0 and _locate(subject) == "no":
        _send(subject, _body(view))
    for _ in range(8):
        place = _locate(subject)
        if place == "Bin":
            _pull_from_bin(subject)
            place = _locate(subject)
        if place in {"INBOX", "Sent Mail", "All Mail"}:
            SENT.write_text(as_of + "\n")
            return f"sent {as_of}"
        time.sleep(2)
    raise RuntimeError("信还没出现在收件箱")


def main() -> int:
    try:
        print(send_outlook(), flush=True)
    except Exception as exc:
        print(f"mail failed {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
