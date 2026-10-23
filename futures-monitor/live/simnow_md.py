#!/usr/bin/env python3
"""Record SimNow depth snapshots for 豆一 / 豆二 / 豆粕 / 菜粕 / 豆油.

Market data only. This process never sends an order.
Credentials live in data/simnow.json, which is not committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import ctp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
TAPE = DATA / "ctp_tape.jsonl"
LATEST = DATA / "ctp_latest.json"
CFG = DATA / "simnow.json"
FLOW = DATA / "ctp_flow"
TZ = ZoneInfo("Asia/Shanghai")

# 豆一、豆二、豆粕、豆油在大连，代码小写。菜粕在郑州，CTP 用一位年份：RM701 即 2701。
# 豆二主连目前是 2611。2701 仍留给半小时行情。主连进程来问时按合约对上。
INSTRUMENTS = ("a2701", "b2701", "b2611", "m2701", "y2701", "RM701")
NAMES = {
    "a2701": "豆一2701",
    "b2701": "豆二2701",
    "b2611": "豆二2611",
    "m2701": "豆粕2701",
    "y2701": "豆油2701",
    "rm701": "菜粕2701",
}
EXTRA_PORT = 8771


def now_cn() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _px(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) > 1e100:
        return None
    return number


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    return str(value).strip()


def load_cfg() -> dict:
    if not CFG.exists():
        raise SystemExit(f"缺少 {CFG.name}")
    cfg = json.loads(CFG.read_text())
    for key in ("broker", "user", "password", "front"):
        if not str(cfg.get(key, "")).strip():
            raise SystemExit(f"{CFG.name} 缺少 {key}")
    return cfg


class Market(ctp.CThostFtdcMdSpi):
    def __init__(self, api, cfg: dict) -> None:
        super().__init__()
        self.api = api
        self.cfg = cfg
        self.login_error = ""
        self.logged_in = False
        self.ticks = 0
        self.latest: dict[str, dict] = {}
        self._req = 0
        self._lock = threading.Lock()
        self._minute: dict[str, dict] = {}
        self._finished: dict[str, deque] = {}
        self._cum: dict[str, float] = {}

    def OnFrontConnected(self) -> None:
        print(f"{now_cn()} 行情前置已连上", flush=True)
        field = ctp.CThostFtdcReqUserLoginField()
        field.BrokerID = str(self.cfg["broker"])
        field.UserID = str(self.cfg["user"])
        field.Password = str(self.cfg["password"])
        self._req += 1
        self.api.ReqUserLogin(field, self._req)

    def OnFrontDisconnected(self, reason: int) -> None:
        self.logged_in = False
        print(f"{now_cn()} 行情前置断开 {reason:#x}", flush=True)

    def OnRspUserLogin(self, login, info, request_id, is_last) -> None:
        error_id = 0 if info is None else int(info.ErrorID)
        if error_id:
            self.login_error = f"{error_id} {_text(info.ErrorMsg)}"
            print(f"{now_cn()} 登录失败 {self.login_error}", flush=True)
            return
        self.logged_in = True
        day = "" if login is None else _text(login.TradingDay)
        print(f"{now_cn()} 登录成功 交易日 {day}", flush=True)
        code = self.api.SubscribeMarketData(list(INSTRUMENTS))
        print(f"{now_cn()} 订阅 {','.join(INSTRUMENTS)} 返回 {code}", flush=True)

    def OnRspError(self, info, request_id, is_last) -> None:
        if info is None:
            return
        print(f"{now_cn()} 错误 {info.ErrorID} {_text(info.ErrorMsg)}", flush=True)

    def OnRspSubMarketData(self, instrument, info, request_id, is_last) -> None:
        name = "" if instrument is None else _text(instrument.InstrumentID)
        error_id = 0 if info is None else int(info.ErrorID)
        if error_id:
            print(f"{now_cn()} 订阅失败 {name} {error_id} {_text(info.ErrorMsg)}", flush=True)
            return
        print(f"{now_cn()} 已订阅 {name}", flush=True)

    def OnRtnDepthMarketData(self, data) -> None:
        if data is None:
            return
        code = _text(data.InstrumentID).lower()
        row = {
            "at": now_cn(),
            "instrument": code,
            "name": NAMES.get(code, code),
            "last": _px(data.LastPrice),
            "bid": _px(data.BidPrice1),
            "ask": _px(data.AskPrice1),
            "bid_vol": _px(data.BidVolume1),
            "ask_vol": _px(data.AskVolume1),
            "volume": _px(data.Volume),
            "oi": _px(data.OpenInterest),
            "open": _px(data.OpenPrice),
            "high": _px(data.HighestPrice),
            "low": _px(data.LowestPrice),
            "pre_close": _px(data.PreClosePrice),
            "pre_settle": _px(data.PreSettlementPrice),
            "update": f"{_text(data.UpdateTime)}.{int(data.UpdateMillisec):03d}",
            "trading_day": _text(data.TradingDay),
        }
        self.ticks += 1
        self.latest[code] = row
        self._fold_minute(code, row)
        DATA.mkdir(parents=True, exist_ok=True)
        with TAPE.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        LATEST.write_text(
            json.dumps({"at": row["at"], "quotes": self.latest}, ensure_ascii=False, indent=2)
        )
        if self.ticks <= 3 or self.ticks % 120 == 0:
            print(
                f"{row['at']} {row['name']} {row['last']} 买{row['bid']} 卖{row['ask']} {row['update']}",
                flush=True,
            )

    def _fold_minute(self, code: str, row: dict) -> None:
        """Remember the minute that just finished. The minute daemon asks for it."""

        at = str(row.get("at") or "")
        update = str(row.get("update") or "")
        price = row.get("last")
        if len(at) < 16 or len(update) < 8 or price is None or price <= 0:
            return
        if update >= "21:00:00" and at[11:19] < "21:00:00":
            return
        stamp = f"{at[:10]} {update[:5]}:00"
        cum = row.get("volume")
        with self._lock:
            previous = self._cum.get(code)
            delta = 0.0
            if isinstance(cum, (int, float)) and isinstance(previous, (int, float)) and cum >= previous:
                delta = cum - previous
            if isinstance(cum, (int, float)):
                self._cum[code] = cum
            acc = self._minute.get(code)
            if acc is None or acc["trade_time"] != stamp:
                if acc is not None:
                    self._finished.setdefault(code, deque(maxlen=30)).append(acc)
                acc = {
                    "trade_time": stamp,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "vol": delta,
                    "oi": row.get("oi"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "bid_vol": row.get("bid_vol"),
                    "ask_vol": row.get("ask_vol"),
                    "pre_close": row.get("pre_close"),
                    "pre_settle": row.get("pre_settle"),
                }
                self._minute[code] = acc
                return
            acc["high"] = max(acc["high"], price)
            acc["low"] = min(acc["low"], price)
            acc["close"] = price
            acc["vol"] = (acc.get("vol") or 0.0) + delta
            acc["oi"] = row.get("oi")
            acc["bid"] = row.get("bid")
            acc["ask"] = row.get("ask")
            acc["bid_vol"] = row.get("bid_vol")
            acc["ask_vol"] = row.get("ask_vol")
            acc["pre_close"] = row.get("pre_close")
            acc["pre_settle"] = row.get("pre_settle")

    def snapshot(self) -> dict:
        """One answer for the minute daemon: latest bid/ask, and minutes already closed."""

        with self._lock:
            quotes = {}
            for code, row in self.latest.items():
                quotes[code] = {
                    "name": row.get("name"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "bid_vol": row.get("bid_vol"),
                    "ask_vol": row.get("ask_vol"),
                    "pre_close": row.get("pre_close"),
                    "pre_settle": row.get("pre_settle"),
                    "at": row.get("at"),
                    "minutes": list(self._finished.get(code, ())),
                }
        return {"at": now_cn(), "quotes": quotes}


class _LatestHandler(BaseHTTPRequestHandler):
    spi: Market | None = None

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/latest":
            self.send_error(404)
            return
        spi = _LatestHandler.spi
        body = json.dumps(spi.snapshot() if spi is not None else {"quotes": {}}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def _serve_latest(spi: Market) -> None:
    _LatestHandler.spi = spi
    server = ThreadingHTTPServer(("127.0.0.1", EXTRA_PORT), _LatestHandler)
    print(f"{now_cn()} 主连进程可来取独有行情 127.0.0.1:{EXTRA_PORT}/latest", flush=True)
    server.serve_forever()


def run(probe_seconds: int) -> int:
    cfg = load_cfg()
    FLOW.mkdir(parents=True, exist_ok=True)
    print(f"{now_cn()} CTP {ctp.CThostFtdcMdApi.GetApiVersion()} 前置 {cfg['front']}", flush=True)
    api = ctp.CThostFtdcMdApi.CreateFtdcMdApi(str(FLOW) + "/")
    spi = Market(api, cfg)
    api.RegisterSpi(spi)
    api.RegisterFront(str(cfg["front"]))
    api.Init()
    threading.Thread(target=_serve_latest, args=(spi,), daemon=True).start()
    started = time.time()
    next_beat = started + 30
    try:
        while True:
            time.sleep(0.5)
            now = time.time()
            if probe_seconds and now - started >= probe_seconds:
                if spi.login_error:
                    return 2
                if not spi.logged_in:
                    print(f"{now_cn()} 探测结束，还没登录上", flush=True)
                    return 3
                print(f"{now_cn()} 探测结束，已登录，快照 {spi.ticks} 笔", flush=True)
                return 0
            if now >= next_beat:
                next_beat = now + 30
                state = "已登录" if spi.logged_in else "未登录"
                print(f"{now_cn()} {state} 快照 {spi.ticks} 笔", flush=True)
    finally:
        try:
            api.Release()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SimNow 行情快照，不下单")
    parser.add_argument("--probe", type=int, default=0, help="探测秒数，到点退出")
    args = parser.parse_args()
    sys.exit(run(args.probe))


if __name__ == "__main__":
    main()
