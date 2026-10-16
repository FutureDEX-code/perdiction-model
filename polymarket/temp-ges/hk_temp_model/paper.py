"""Paper ledger. Five dollars when the model trades. Live orders are sent separately."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from hk_temp_model.decide import contains

_PATH = Path(__file__).resolve().parent / "data" / "paper_trades.json"
_USD = 5.0


def _load(path: Path | None = None) -> list[dict]:
    file = path or _PATH
    if not file.exists():
        return []
    return json.loads(file.read_text())


def _save(rows: list[dict], path: Path | None = None) -> None:
    file = path or _PATH
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def _append(rows, taken, notes, city, day, observed_at, label, side, reason, price) -> None:
    key = (city, day.isoformat(), label, side)
    if key in taken or price is None or price <= 0 or price >= 1:
        return
    shares = round(_USD / float(price), 4)
    rows.append(
        {
            "city": city,
            "day": day.isoformat(),
            "observed_at": observed_at,
            "label": label,
            "side": side,
            "reason": reason,
            "price": round(float(price), 4),
            "usd": _USD,
            "shares": shares,
            "pnl": None,
        }
    )
    taken.add(key)
    notes.append(
        f"模拟买入 {side.title()}  {city} {label}  {price:.3f}  ×{shares:.2f}份  共 {_USD:.0f} 美元  {reason}"
    )


def record_fill(
    city: str,
    day: date,
    observed_at: str,
    label: str,
    side: str,
    reason: str,
    price: float | None,
    path: Path | None = None,
) -> str:
    """Record one new ticket. Empty string if this city, day, bracket, and side is already there."""

    rows = _load(path)
    taken = {(row["city"], row["day"], row["label"], row["side"]) for row in rows}
    notes: list[str] = []
    _append(rows, taken, notes, city, day, observed_at, label, side, reason, price)
    if notes:
        _save(rows, path)
    return notes[0] if notes else ""


def note_fills(city: str, day: date, observed_at: str, board: list[dict], path: Path | None = None) -> list[str]:
    """Five dollars on each new city, day, bracket, and side. The same ticket is recorded once."""

    rows = _load(path)
    taken = {(row["city"], row["day"], row["label"], row["side"]) for row in rows}
    notes: list[str] = []
    for item in board:
        if item.get("buy_no"):
            _append(rows, taken, notes, city, day, observed_at, item["label"], "NO", item["reason"], item.get("no_ask"))
        if item.get("buy_yes"):
            _append(rows, taken, notes, city, day, observed_at, item["label"], "YES", "模型高于卖价", item.get("yes_ask"))
    if notes:
        _save(rows, path)
    return notes


def settle(city: str, day: str, settled: float, path: Path | None = None) -> list[str]:
    """No wins when the settlement is outside that bracket."""

    rows = _load(path)
    notes = []
    changed = False
    for row in rows:
        if row["city"] != city or row["day"] != day or row["pnl"] is not None:
            continue
        hit = contains(row["label"], settled)
        won = hit if row["side"] == "YES" else not hit
        row["pnl"] = round(row["shares"] - row["usd"], 4) if won else round(-row["usd"], 4)
        row["won"] = won
        changed = True
        verb = "赢" if won else "输"
        notes.append(
            f"模拟结算 {verb}  {city} {day} {row['label']} {row['side'].title()}  {row['pnl']:+.2f} 美元  结算 {settled:g}"
        )
    if changed:
        _save(rows, path)
    return notes


def record_exit(
    city: str,
    day: date,
    label: str,
    side: str,
    sell_price: float,
    path: Path | None = None,
) -> str:
    """Close one open ticket at the sale price. Empty if it is already closed."""

    rows = _load(path)
    for row in rows:
        if row["city"] != city or row["day"] != day.isoformat() or row["label"] != label or row["side"] != side:
            continue
        if row.get("pnl") is not None:
            return ""
        pnl = round(float(row["shares"]) * float(sell_price) - float(row["usd"]), 4)
        row["pnl"] = pnl
        row["won"] = pnl > 0
        row["exit_price"] = round(float(sell_price), 4)
        row["exit_reason"] = "印出后卖出"
        _save(rows, path)
        return (
            f"模拟卖出 {side.title()}  {city} {label}  {sell_price:.3f}  "
            f"盈亏 {pnl:+.2f} 美元  印出后卖出"
        )
    return ""


def settle_scored(city: str, scored: dict | None, path: Path | None = None) -> list[str]:
    notes = []
    for day, row in (scored or {}).items():
        value = row.get("settled_c", row.get("settled"))
        if value is None:
            continue
        notes.extend(settle(city, day, float(value), path))
    return notes


def status_line(path: Path | None = None) -> str:
    rows = _load(path)
    done = [row for row in rows if row.get("pnl") is not None]
    pnl = sum(row["pnl"] for row in done)
    return f"模拟盘  已结算 {len(done)} 笔  盈亏 {pnl:+.2f} 美元  未结算 {len(rows) - len(done)} 笔"
