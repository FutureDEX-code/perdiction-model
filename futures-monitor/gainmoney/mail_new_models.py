#!/usr/bin/env python3
"""Mail the gainmoney picks once at 20:35.

Body follows the evening template: for each product the four chosen models
with menu, accuracy and holdout PnL, then the picked ensemble rule. Subject is
「新模型」.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.mail_outlook import _forbid_company, _google_enabled, _locate, _outbox_count, _pull_from_bin, _send, SENDER, TO  # noqa: E402

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
SUBJECT = "新模型"
STATUS = ROOT / "data" / "record_status"
SENT = STATUS / "gainmoney_new_models_emailed.txt"
ORDER = ["A", "B", "M", "RM", "Y", "P", "OI", "PK"]


def _body() -> str:
    lines = ["新模型，训练 2022-01-01 至 2025-12-31，验证 2026-01-01 至 2026-10-31。", "参数在训练段内自己拟合，利润按反手算，含手续费和滑点。", ""]
    total = 0.0
    for product in ORDER:
        payload = json.loads((HERE / product / "综合对比.json").read_text())
        lines.append(payload["品种"])
        for family, item in payload["四个模型"].items():
            menu = item["菜单"]
            tag = f"（{menu}）" if menu != "主菜" else ""
            lines.append(f"  {family}{tag}  {item['准确率']:.1f}%  {item['利润'] / 10000:+.2f}万")
        pick = payload["选定"]
        lines.append(f"  综合（{pick['综合']}）  {pick['准确率']:.1f}%  {pick['利润'] / 10000:+.2f}万")
        total += pick["利润"]
        lines.append("")
    lines.append(f"合计  {total / 10000:+.2f}万")
    return "\n".join(lines)


def send_new_models() -> None:
    if not _google_enabled():
        raise RuntimeError("Mail 里的 Gmail 是停用的，信寄不出去")
    _forbid_company(SENDER)
    _forbid_company(TO)
    if _outbox_count(SUBJECT) == 0 and _locate(SUBJECT) == "no":
        _send(SUBJECT, _body())
    for _ in range(8):
        place = _locate(SUBJECT)
        if place == "Bin":
            _pull_from_bin(SUBJECT)
            place = _locate(SUBJECT)
        if place in {"INBOX", "Sent Mail", "All Mail"}:
            SENT.write_text(datetime.now(TZ).date().isoformat() + "\n")
            print(f"新模型已发出 {datetime.now(TZ):%Y-%m-%d %H:%M}", flush=True)
            return
        import time

        time.sleep(2)
    raise RuntimeError("新模型的信还没出现在收件箱")


def main() -> int:
    now = datetime.now(TZ)
    if SENT.exists() and SENT.read_text().strip() == now.date().isoformat():
        print("already sent today", flush=True)
        return 0
    send_new_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
