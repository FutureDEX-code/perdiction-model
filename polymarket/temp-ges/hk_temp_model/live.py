"""Send the model's new tickets to Polymarket, then record the same ticket on paper.

Five dollars, same city/day/bracket/side once. A failed order is not written
to the paper ledger, so the next cycle can try again.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import date

from hk_temp_model.market import _BRIDGE, _token, outcome_tokens
from hk_temp_model.paper import _load, record_exit, record_fill

_USD = 5.0


def _size(price: float) -> float:
    size = math.floor(_USD / price * 10000) / 10000
    if size <= 0 or price * size > _USD + 1e-9:
        raise RuntimeError("size would spend more than 5 dollars")
    return size


def _post_buy(city: str, token_id: str, price: float) -> None:
    body = json.dumps(
        {
            "token_id": token_id,
            "side": "BUY",
            "price": price,
            "size": _size(price),
            "market": city,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{_BRIDGE}/order",
        data=body,
        headers={
            "X-Auth-Token": _token(),
            "Content-Type": "application/json",
            "User-Agent": "hk-temp-model/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "order failed")


def _sell_price(side: str, item: dict) -> float | None:
    if side == "YES":
        price = item.get("yes_bid")
    else:
        ask = item.get("yes_ask")
        price = None if ask is None else math.floor((1.0 - float(ask)) * 100) / 100
    if price is None:
        return None
    price = float(price)
    if price <= 0 or price >= 1:
        return None
    return price


def _post_sell(city: str, token_id: str, price: float, size: float) -> None:
    body = json.dumps(
        {
            "token_id": token_id,
            "side": "SELL",
            "price": price,
            "size": size,
            "market": city,
            "exit": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{_BRIDGE}/order",
        data=body,
        headers={
            "X-Auth-Token": _token(),
            "Content-Type": "application/json",
            "User-Agent": "hk-temp-model/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "order failed")


def exit_hopeless(city: str, day: date, board: list[dict]) -> list[str]:
    """Sell a held ticket once the print has made it unable to pay."""

    by_label = {item["label"]: item for item in board}
    notes: list[str] = []
    try:
        tokens = outcome_tokens(city, day)
    except Exception as exc:
        return [f"实盘卖出失败  {city}  {exc}"]
    for row in _load():
        if row["city"] != city or row["day"] != day.isoformat() or row.get("pnl") is not None:
            continue
        item = by_label.get(row["label"])
        if item is None:
            continue
        dead = (row["side"] == "YES" and item.get("yes_dead")) or (row["side"] == "NO" and item.get("no_dead"))
        if not dead:
            continue
        price = _sell_price(row["side"], item)
        if price is None:
            notes.append(f"实盘卖出失败  {row['side']}  {city} {row['label']}  没有买价")
            continue
        pair = tokens.get(row["label"])
        if pair is None:
            notes.append(f"实盘卖出失败  {city} {row['label']}  没有这个档的代币")
            continue
        token_id = pair[0] if row["side"] == "YES" else pair[1]
        size = math.floor(float(row["shares"]) * 10000) / 10000
        try:
            _post_sell(city, token_id, price, size)
        except Exception as exc:
            notes.append(f"实盘卖出失败  {row['side']}  {city} {row['label']}  {price:.3f}  {exc}")
            continue
        paper = record_exit(city, day, row["label"], row["side"], price)
        notes.append(
            f"实盘卖出 {row['side']}  {city} {row['label']}  {price:.3f}  ×{size:.2f}份"
        )
        if paper:
            notes.append(paper)
    return notes


def execute(city: str, day: date, observed_at: str, board: list[dict]) -> list[str]:
    """Place each new model buy, then mirror it in the paper ledger."""

    taken = {(row["city"], row["day"], row["label"], row["side"]) for row in _load()}
    wanted: list[tuple[str, str, str, float]] = []
    for item in board:
        if item.get("buy_no"):
            wanted.append((item["label"], "NO", item["reason"], item.get("no_ask")))
        if item.get("buy_yes"):
            wanted.append((item["label"], "YES", "模型高于卖价", item.get("yes_ask")))
    if not wanted:
        return []
    tokens = outcome_tokens(city, day)
    notes: list[str] = []
    for label, side, reason, price in wanted:
        key = (city, day.isoformat(), label, side)
        if key in taken or price is None or price <= 0 or price >= 1:
            continue
        pair = tokens.get(label)
        if pair is None:
            notes.append(f"实盘失败  {city} {label}  没有这个档的代币")
            continue
        token_id = pair[0] if side == "YES" else pair[1]
        try:
            _post_buy(city, token_id, float(price))
        except Exception as exc:
            notes.append(f"实盘失败  {side}  {city} {label}  {price:.3f}  {exc}")
            continue
        paper = record_fill(city, day, observed_at, label, side, reason, float(price))
        shares = _size(float(price))
        notes.append(f"实盘买入 {side}  {city} {label}  {price:.3f}  ×{shares:.2f}份  共 5 美元")
        if paper:
            notes.append(paper)
        taken.add(key)
    return notes
