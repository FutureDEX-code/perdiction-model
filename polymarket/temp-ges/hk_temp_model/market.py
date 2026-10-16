"""Yes asks from the local Polymarket bridge on 127.0.0.1:16888."""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from pathlib import Path

_BRIDGE = os.environ.get("POLY_BRIDGE_URL", "http://127.0.0.1:16888")
_ENV = Path("/Users/jasperZ@bixin.com/Documents/Prediction-model-ensemble/polymarket/polymarket-temperature/.env")


def _token() -> str:
    token = os.environ.get("POLY_BRIDGE_TOKEN", "")
    if token or not _ENV.exists():
        return token
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("AUTH_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def fetch_yes_book(city_key: str, day: date) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Best Yes ask and bid on each bracket, via the local bridge."""

    url = f"{_BRIDGE}/asks?city={city_key}&day={day.isoformat()}"
    request = urllib.request.Request(url, headers={"X-Auth-Token": _token(), "User-Agent": "hk-temp-model/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "asks failed")

    def _prices(raw: dict) -> dict[str, float | None]:
        return {str(label): (None if price is None else float(price)) for label, price in (raw or {}).items()}

    return _prices(payload.get("asks")), _prices(payload.get("bids"))


def fetch_yes_asks(city_key: str, day: date) -> dict[str, float | None]:
    asks, _bids = fetch_yes_book(city_key, day)
    return asks


def outcome_tokens(city_key: str, day: date) -> dict[str, tuple[str, str]]:
    """Yes and No token ids for each bracket. Outcomes are ordered Yes, then No."""

    places = {"hk": "hong-kong", "paris": "paris", "nyc": "nyc"}
    months = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    )
    place = places.get(city_key)
    if place is None:
        raise RuntimeError(f"unknown city {city_key}")
    slug = f"highest-temperature-in-{place}-on-{months[day.month - 1]}-{day.day}-{day.year}"
    request = urllib.request.Request(
        f"https://gamma-api.polymarket.com/events?slug={slug}",
        headers={"User-Agent": "hk-temp-model/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    event = payload[0] if isinstance(payload, list) and payload else {}
    found: dict[str, tuple[str, str]] = {}
    for market in event.get("markets") or []:
        label = market.get("groupItemTitle")
        raw_ids = market.get("clobTokenIds")
        if isinstance(raw_ids, str):
            raw_ids = json.loads(raw_ids)
        if not label or not raw_ids or len(raw_ids) < 2:
            continue
        found[str(label)] = (str(raw_ids[0]), str(raw_ids[1]))
    if not found:
        raise RuntimeError(f"no tokens for {city_key} {day.isoformat()}")
    return found
