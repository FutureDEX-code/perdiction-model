"""Each cycle: which bracket is already the high, and which Yes books can be No."""

from __future__ import annotations

import re

from hk_temp_model.strategies import allow_yes

_EDGE = 0.04
_OVERSHOOT = 1.5
_MIN_YES_ASK = 0.35
_MIN_YES_P = 0.40
_FORECAST_NO_LEAD = 4.0


def _span(label: str) -> tuple[float, float] | None:
    text = label.replace("°C", "").replace("°F", "").strip()
    if text.endswith("or below"):
        return float("-inf"), float(text.split()[0]) + 1
    if text.endswith("or higher") or text.endswith("or above"):
        return float(text.split()[0]), float("inf")
    pair = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if pair:
        return float(pair.group(1)), float(pair.group(2)) + 1
    single = re.fullmatch(r"\d+(?:\.\d+)?", text)
    if single:
        lo = float(text)
        return lo, lo + 1
    return None


def contains(label: str, value: float) -> bool:
    span = _span(label)
    if span is None:
        return False
    lo, hi = span
    return value >= lo and value < hi


def _prob(fair, lo: float, hi: float) -> float:
    total = 0.0
    for row in fair:
        if row.lo_c >= lo and row.lo_c < hi:
            total += row.probability
    return total


def no_board(
    *,
    running: float,
    current: float,
    hour: float,
    peak: float,
    forecast: float | None,
    fair,
    asks: dict,
    bids: dict,
    overshoot: float = _OVERSHOOT,
    city: str = "hk",
    mu: float | None = None,
    fallen_dip: float = 0.4,
    fallen_room: float = 0.3,
) -> list[dict]:
    """One row per market bracket.

    Shared Yes gates: the most likely bracket, at least 40 percent, ask from
    0.35 up to but not including 0.97, and at least 4 cents cheaper than the
    model. Asks under 0.35 lost money both before and after live orders:
    50 settled tickets, 10 wins, about -32 dollars. The last hour does not
    waive that floor. City files may refuse that bracket before these gates run.
    """

    guide = running if forecast is None else max(running, forecast)
    fallen = hour >= peak and running - current >= fallen_dip
    ceiling = running + fallen_room if fallen else guide + overshoot
    rows = []
    for label in list(asks) or list(bids):
        span = _span(label)
        if span is None:
            continue
        lo, hi = span
        passed = running >= hi
        unreachable = lo > ceiling
        too_low = forecast is not None and hour < peak and hi <= forecast - overshoot
        model_p = _prob(fair, lo, hi)
        yes_bid = bids.get(label)
        no_ask = None if yes_bid is None else round(1.0 - float(yes_bid), 4)
        edge = None if no_ask is None else round((1.0 - model_p) - no_ask, 4)
        # "已过" is the printed maximum. "到不了" is only a forecast, so it waits
        # until four hours before the peak, when most of the rise is already in.
        forecast_dead = (unreachable or too_low) and hour >= peak - _FORECAST_NO_LEAD
        buy_no = bool((passed or forecast_dead) and edge is not None and edge >= _EDGE and no_ask <= 0.97)
        if passed:
            reason = "已过"
        elif unreachable:
            reason = "到不了"
        elif too_low:
            reason = "低于预报"
        elif model_p <= 0.08:
            reason = "模型很低"
        else:
            reason = "还可能"
        rows.append(
            {
                "label": label,
                "reason": reason,
                "model_p": round(model_p, 4),
                "no_ask": no_ask,
                "edge": edge,
                "yes_ask": None if asks.get(label) is None else round(float(asks[label]), 4),
                "yes_bid": None if yes_bid is None else round(float(yes_bid), 4),
                "yes_edge": None
                if asks.get(label) is None
                else round(model_p - float(asks[label]), 4),
                "buy_no": buy_no,
                "buy_yes": False,
                "yes_block": "",
                "yes_dead": passed,
                "no_dead": (not passed) and running >= lo and fallen and hi > ceiling,
                "alive": not passed and not unreachable and not too_low,
            }
        )
    # Buy the likely bracket only. The widest edge is usually the cheap
    # degree just below the mode, and that ticket lost almost every time.
    best = _yes_candidate(
        rows,
        city=city,
        mu=mu,
        hour=hour,
        peak=peak,
        forecast=forecast,
    )
    if best is None:
        return rows
    ask = best["yes_ask"]
    edge = best["yes_edge"]
    # One to four hours before the local peak the mode walks up and those
    # added Yes tickets lost. Earlier tickets and the last hour stay at $5.
    lead = peak - hour
    if 1 <= lead < 4:
        best["yes_block"] = "高峰前1到4小时不加买"
    elif ask is not None and ask < _MIN_YES_ASK:
        best["yes_block"] = "卖价低于0.35"
    elif (
        best["alive"]
        and ask is not None
        and ask < 0.97
        and best["model_p"] >= _MIN_YES_P
        and edge is not None
        and edge >= _EDGE
    ):
        best["buy_yes"] = True
    return rows


def _yes_candidate(
    rows: list[dict],
    *,
    city: str,
    mu: float | None,
    hour: float,
    peak: float,
    forecast: float | None,
) -> dict | None:
    alive = [row for row in rows if row["alive"]]
    if not alive:
        return None
    best = max(alive, key=lambda row: row["model_p"])
    block = allow_yes(
        city,
        best,
        hour=hour,
        peak=peak,
        forecast=forecast,
        span=_span(best["label"]),
        mu=mu,
    )
    if block:
        best["yes_block"] = block
        return None
    return best


def format_no_board(rows: list[dict]) -> str:
    alive = [row["label"] for row in rows if row["alive"]]
    if len(alive) == 1:
        head = f"最高温锁定  {alive[0]}"
    elif not alive:
        head = "最高温锁定  没有还开着的档"
    else:
        head = "还可能  " + "  ".join(alive)
    buys = [
        f"{row['label']}  No卖价 {row['no_ask']:.3f}  空间 {row['edge']:+.3f}  {row['reason']}"
        for row in rows
        if row["buy_no"]
    ]
    yeses = [
        f"{row['label']}  Yes卖价 {row['yes_ask']:.3f}  空间 {row['yes_edge']:+.3f}"
        for row in rows
        if row.get("buy_yes")
    ]
    lines = [head]
    if yeses:
        lines.append("可下Yes  " + "；".join(yeses))
    else:
        blocked = next((row["yes_block"] for row in rows if row.get("yes_block")), "")
        if blocked:
            lines.append(f"可下Yes  没有。{blocked}")
        else:
            lines.append("可下Yes  没有。卖价不低于模型")
    if buys:
        lines.append("可下No  " + "；".join(buys))
    else:
        lines.append("可下No  没有。已排除的档没有人还在买 Yes")
    return "\n".join(lines)
