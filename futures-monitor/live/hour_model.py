#!/usr/bin/env python3
"""Half-hour forecast from the half hour that just traded.

This layer is separate from the 14:50 next-session model in train.py.
It is refit from the tape each time. Weights are not written into data/models.
"""

from __future__ import annotations

import sys

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TAPE = ROOT / "data" / "ctp_tape.jsonl"
NAMES = ("豆一2701", "豆二2701", "豆粕2701", "菜粕2701", "豆油2701")
SESSIONS = ((9 * 60, 10 * 60 + 15), (10 * 60 + 30, 11 * 60 + 30), (13 * 60 + 30, 15 * 60), (21 * 60, 23 * 60))
# Five finished half hours are enough to start. The fit stays shrunk while the tape is short.
MIN_SAMPLES = 5
SLOT = 30
MIN_TICKS = 15


def _minutes(at: str) -> int:
    return int(at[11:13]) * 60 + int(at[14:16])


def _session(at: str) -> tuple[int, int] | None:
    minutes = _minutes(at)
    for start, end in SESSIONS:
        if start <= minutes < end:
            return start, end
    return None


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    table = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(table[row][col]))
        table[col], table[pivot] = table[pivot], table[col]
        scale = table[col][col]
        if abs(scale) < 1e-12:
            raise ValueError("half-hour model is singular")
        for col2 in range(col, size + 1):
            table[col][col2] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = table[row][col]
            for col2 in range(col, size + 1):
                table[row][col2] -= factor * table[col][col2]
    return [table[row][size] for row in range(size)]


def _bar() -> dict:
    return {"o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "n": 0}


def _add(bar: dict, price: float) -> None:
    if bar["n"] == 0:
        bar["o"] = bar["h"] = bar["l"] = price
    else:
        bar["h"] = max(bar["h"], price)
        bar["l"] = min(bar["l"], price)
    bar["c"] = price
    bar["n"] += 1


def _features(bar: dict) -> dict[str, float] | None:
    if bar["n"] < MIN_TICKS:
        return None
    span = bar["h"] - bar["l"]
    loc = 0.5 if span <= 0 else (bar["c"] - bar["l"]) / span
    return {"ret": bar["c"] - bar["o"], "rng": span, "loc": loc}


def _clock(day: str, minutes: int) -> str:
    return f"{day} {minutes // 60:02d}:{minutes % 60:02d}"


def _segments() -> list[tuple[int, int]]:
    """Half-hour bars inside every session, plus the 10:00-10:15 stub. Sessions chain."""

    out = []
    for start, end in SESSIONS:
        mark = start
        while mark + SLOT <= end:
            out.append((mark, mark + SLOT))
            mark += SLOT
        if mark < end:
            out.append((mark, end))
    return out


SEGMENTS = _segments()


def _segment_of(at: str) -> tuple[int, int] | None:
    minutes = _minutes(at)
    for start, end in SEGMENTS:
        if start <= minutes < end:
            return start, end
    return None


def _load(now: datetime) -> tuple[dict[str, list[dict]], dict[str, dict | None], dict[tuple[str, str], dict]]:
    """Bars chain across breaks: every sample is one bar, its label the next bar's close."""

    stamps: dict[tuple[str, str], dict] = {}
    cutoff = now.strftime("%Y-%m-%d %H:%M:%S")
    empty_end: dict[tuple[str, str], dict] = {}
    if not TAPE.exists():
        return {name: [] for name in NAMES}, {name: None for name in NAMES}, empty_end
    with TAPE.open() as handle:
        for line in handle:
            row = json.loads(line)
            name = row.get("name")
            at = str(row.get("at") or "")
            if name not in NAMES or len(at) < 16 or at > cutoff or _session(at) is None:
                continue
            try:
                price = float(row["last"])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue
            segment = _segment_of(at)
            if segment is None:
                continue
            start_min, end_min = segment
            # An unfinished bar is not a sample and not the bar we forecast from.
            end_text = f"{at[:10]} {end_min // 60:02d}:{end_min % 60:02d}:00"
            if end_text > cutoff:
                continue
            slot = _clock(at[:10], start_min)
            key = (slot, name)
            bar = stamps.get(key)
            if bar is None:
                bar = _bar()
                bar["end"] = _clock(at[:10], end_min)
                stamps[key] = bar
            _add(bar, price)
    by_name: dict[str, list[tuple[str, dict]]] = {name: [] for name in NAMES}
    for (stamp, name), bar in stamps.items():
        feat = _features(bar)
        if feat is not None:
            by_name[name].append((stamp, {**feat, "open": bar["o"], "close": bar["c"], "end": bar["end"]}))
    samples: dict[str, list[dict]] = {name: [] for name in NAMES}
    live_feat: dict[str, dict | None] = {name: None for name in NAMES}
    by_end: dict[tuple[str, str], dict] = {}
    for name in NAMES:
        rows = sorted(by_name[name])
        if not rows:
            continue
        live_feat[name] = rows[-1][1]
        for (_left, prev), (_right, nxt) in zip(rows, rows[1:]):
            samples[name].append({**prev, "y": nxt["close"] - prev["close"]})
        for _start, feat in rows:
            by_end[(name, feat["end"])] = feat
    return samples, live_feat, by_end


def _fit(samples: list[dict], live: dict) -> dict | None:
    if len(samples) < MIN_SAMPLES:
        return None
    cols = ("ret", "rng", "loc")
    mean = {col: sum(row[col] for row in samples) / len(samples) for col in cols}
    scale = {}
    for col in cols:
        var = sum((row[col] - mean[col]) ** 2 for row in samples) / len(samples)
        scale[col] = math.sqrt(var) or 1.0

    def row_z(row: dict) -> list[float]:
        return [1.0, *[(row[col] - mean[col]) / scale[col] for col in cols]]

    size = 4
    xtx = [[0.0] * size for _ in range(size)]
    xty = [0.0] * size
    for sample in samples:
        z = row_z(sample)
        for i in range(size):
            xty[i] += z[i] * sample["y"]
            for j in range(size):
                xtx[i][j] += z[i] * z[j]
    # Light shrink. The intercept is the baseline; the other three are this half hour's tape.
    for i in range(1, size):
        xtx[i][i] += 1.0
    beta = _solve(xtx, xty)
    errors = []
    for sample in samples:
        yhat = sum(weight * value for weight, value in zip(beta, row_z(sample)))
        errors.append(sample["y"] - yhat)
    resid = math.sqrt(sum(err * err for err in errors) / len(errors)) or 1.0
    z_live = row_z(live)
    points = sum(weight * value for weight, value in zip(beta, z_live))
    baseline = beta[0]
    # Short tapes cannot support a sharp probability. Pull the log-odds back toward a coin flip.
    shrink = len(samples) / (len(samples) + 8)
    prob_up = 1.0 / (1.0 + math.exp(-shrink * points / resid))
    if points > 0.5:
        side, shown = "涨", prob_up
    elif points < -0.5:
        side, shown = "跌", 1.0 - prob_up
    else:
        side, shown = "平", 0.5
    return {
        "points": points,
        "baseline": baseline,
        "tape": points - baseline,
        "side": side,
        "prob": shown,
        "n": len(samples),
        "resid": resid,
        "ret": live["ret"],
        "rng": live["rng"],
        "loc": live["loc"],
        "end": live.get("end"),
    }


def prepare(now: datetime) -> tuple[dict[str, dict | None], dict[tuple[str, str], dict]]:
    """Forecasts, plus every finished bar keyed by (name, end 'YYYY-MM-DD HH:MM')."""

    samples, live, by_end = _load(now)
    out: dict[str, dict | None] = {}
    for name in NAMES:
        if live[name] is None:
            out[name] = None
            continue
        fitted = _fit(samples[name], live[name])
        out[name] = fitted if fitted is not None else {"n": len(samples[name]), **live[name]}
    return out, by_end


def forecast(now: datetime) -> dict[str, dict | None]:
    views, _bars = prepare(now)
    return views
