#!/usr/bin/env python3
"""Local Polymarket order bridge for HK max-temp strategy.
Listens on 127.0.0.1:16888. Cloud agent decides; this host signs+posts from HK IP.
"""
from __future__ import annotations

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            # .env is source of truth (override stale LaunchAgent/env)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


load_env()

AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
PK = os.environ.get("POLYMARKET_PK", "")
FUNDER = os.environ.get("FUNDER", "0x5a64359724b0faa963ee3d500af7d14972bf4c7a")
SIG_TYPE = int(os.environ.get("SIGNATURE_TYPE", "3"))
HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID = int(os.environ.get("CHAIN_ID", "137"))
MAX_ORDER_USD = float(os.environ.get("MAX_ORDER_USD", "10"))
MAX_PARIS_USD = float(os.environ.get("MAX_PARIS_USD", "5"))
PORT = int(os.environ.get("PORT", "16888"))

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    if not PK:
        raise RuntimeError("POLYMARKET_PK missing in .env")
    from py_clob_client_v2 import ClobClient

    pk = PK if PK.startswith("0x") else "0x" + PK
    c = ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=pk,
        signature_type=SIG_TYPE,
        funder=FUNDER,
    )
    c.set_api_creds(c.create_or_derive_api_key())
    _client = c
    return c


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # avoid logging secrets; keep short access log
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _auth_ok(self) -> bool:
        if not AUTH_TOKEN:
            return False
        got = self.headers.get("X-Auth-Token") or self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return got == AUTH_TOKEN

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return json_response(self, 200, {"ok": True, "service": "polymarket-temperature", "port": PORT})
        if not self._auth_ok():
            return json_response(self, 401, {"ok": False, "error": "unauthorized"})
        try:
            if path == "/balance":
                from py_clob_client_v2 import AssetType, BalanceAllowanceParams

                c = get_client()
                try:
                    bal = c.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
                except TypeError:
                    bal = c.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
                usd = int(bal.get("balance", 0)) / 1e6
                return json_response(self, 200, {"ok": True, "balance_usd": usd, "raw": bal})
            if path == "/asks":
                return self._asks()
            if path == "/open-orders":
                c = get_client()
                # best-effort
                orders = []
                if hasattr(c, "get_orders"):
                    orders = c.get_orders()
                elif hasattr(c, "get_open_orders"):
                    orders = c.get_open_orders()
                return json_response(self, 200, {"ok": True, "orders": orders})
            return json_response(self, 404, {"ok": False, "error": "not_found"})
        except Exception as e:
            return json_response(self, 500, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-800:]})

    def do_POST(self):
        path = urlparse(self.path).path
        if not self._auth_ok():
            return json_response(self, 401, {"ok": False, "error": "unauthorized"})
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return json_response(self, 400, {"ok": False, "error": "invalid_json"})

        try:
            if path == "/order":
                return self._place_order(data)
            if path == "/cancel":
                return self._cancel(data)
            return json_response(self, 404, {"ok": False, "error": "not_found"})
        except Exception as e:
            return json_response(self, 500, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-800:]})

    def _asks(self):
        """Yes best ask for each daily-high bracket. Read-only."""

        query = parse_qs(urlparse(self.path).query)
        city = (query.get("city") or ["hk"])[0].lower()
        day = (query.get("day") or [""])[0]
        places = {"hk": "hong-kong", "hongkong": "hong-kong", "paris": "paris", "nyc": "nyc"}
        months = (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        )
        if city not in places or len(day) != 10:
            return json_response(self, 400, {"ok": False, "error": "city and day=YYYY-MM-DD required"})
        year, month, dom = (int(part) for part in day.split("-"))
        slug = f"highest-temperature-in-{places[city]}-on-{months[month - 1]}-{dom}-{year}"
        request = Request(
            f"https://gamma-api.polymarket.com/events?slug={slug}",
            headers={"User-Agent": "polymarket-temperature/1.0"},
        )
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        event = payload[0] if isinstance(payload, list) and payload else {}
        from py_clob_client_v2 import BookParams

        labels: list[tuple[str, str]] = []
        for market in event.get("markets") or []:
            label = market.get("groupItemTitle")
            raw_ids = market.get("clobTokenIds")
            if isinstance(raw_ids, str):
                raw_ids = json.loads(raw_ids)
            if not label or not raw_ids:
                continue
            labels.append((str(label), str(raw_ids[0])))
        books = get_client().get_order_books([BookParams(token_id=token) for _, token in labels]) if labels else []
        by_token = {}
        for book in books or []:
            token = str(book.get("asset_id") or "")
            levels = book.get("asks") or []
            prices = [float(level["price"]) for level in levels if level.get("price") is not None]
            bid_levels = book.get("bids") or []
            bid_prices = [float(level["price"]) for level in bid_levels if level.get("price") is not None]
            by_token[token] = {
                "ask": min(prices) if prices else None,
                "bid": max(bid_prices) if bid_prices else None,
            }
        asks = {label: (by_token.get(token) or {}).get("ask") for label, token in labels}
        bids = {label: (by_token.get(token) or {}).get("bid") for label, token in labels}
        return json_response(self, 200, {"ok": True, "city": city, "day": day, "asks": asks, "bids": bids})

    def _place_order(self, data: dict):
        from py_clob_client_v2 import OrderArgs, OrderType, Side

        token_id = str(data.get("token_id") or "")
        side = str(data.get("side") or "BUY").upper()
        price = float(data.get("price"))
        size = float(data.get("size"))
        market = str(data.get("market") or "hk").lower()  # hk | paris
        post_only = bool(data.get("post_only", False))

        if not token_id:
            return json_response(self, 400, {"ok": False, "error": "token_id required"})
        if price <= 0 or price >= 1:
            return json_response(self, 400, {"ok": False, "error": "price must be in (0,1)"})
        if size <= 0:
            return json_response(self, 400, {"ok": False, "error": "size must be > 0"})

        # BUY: collateral spent ≈ price*size; SELL exit: skip notional cap when reduce_only/exit
        notional = price * size
        cap = MAX_PARIS_USD if market.startswith("par") else MAX_ORDER_USD
        exit_order = bool(data.get("exit") or data.get("reduce_only") or side == "SELL")
        if (not exit_order) and notional > cap + 1e-9:
            return json_response(
                self,
                400,
                {"ok": False, "error": f"notional {notional:.4f} exceeds cap {cap}", "cap": cap},
            )

        side_enum = Side.BUY if side == "BUY" else Side.SELL
        c = get_client()
        args = OrderArgs(token_id=token_id, price=price, size=size, side=side_enum)
        signed = c.create_order(args)
        resp = c.post_order(signed, OrderType.GTC, post_only=post_only)
        return json_response(
            self,
            200,
            {
                "ok": True,
                "notional": notional,
                "cap": cap,
                "market": market,
                "response": resp,
            },
        )

    def _cancel(self, data: dict):
        oid = data.get("order_id") or data.get("orderID")
        if not oid:
            return json_response(self, 400, {"ok": False, "error": "order_id required"})
        from py_clob_client_v2 import OrderPayload
        c = get_client()
        resp = c.cancel_order(OrderPayload(orderID=str(oid)))
        return json_response(self, 200, {"ok": True, "response": resp})


def main():
    if not AUTH_TOKEN:
        raise SystemExit("AUTH_TOKEN missing")
    if not PK:
        raise SystemExit("POLYMARKET_PK missing")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"polymarket-temperature listening on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
