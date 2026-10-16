"""Remember forecasts and nudge the model when the Daily Extract settles.

Each watch cycle stores the headquarters forecast. A day is scored only after
its Absolute Daily Max appears in the Daily Extract. The error moves a bias
and a spread multiplier a little; one day cannot rewrite the model.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from hk_temp_model.cases import append_case
from hk_temp_model.model import Adjustment, Prediction
from hk_temp_model.observations import Observation
from hk_temp_model.settlement import DAILY_EXTRACT_URL, absolute_daily_maxes, bracket_for

_STATE_PATH = Path(__file__).resolve().parent / "data" / "online_state.json"
_BIAS_STEP = 0.2
_SIGMA_STEP = 0.1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class OnlineState:
    bias_c: float = 0.0
    sigma_mult: float = 1.0
    forecasts: dict[str, list[dict]] | None = None
    scored: dict[str, dict] | None = None
    forecast_max_by_day: dict[str, float] | None = None
    forecast_trust: float = 0.62

    def __post_init__(self) -> None:
        if self.forecasts is None:
            self.forecasts = {}
        if self.scored is None:
            self.scored = {}
        if self.forecast_max_by_day is None:
            self.forecast_max_by_day = {}

    def hold_forecast(self, day: date, fresh: float | None) -> float | None:
        """Keep the last nine-day maximum issued for this settlement date.

        After late morning the feed drops today. That absence must not replace
        the morning forecast with the climate mean.
        """

        key = day.isoformat()
        if fresh is not None:
            self.forecast_max_by_day[key] = fresh
            return fresh
        held = self.forecast_max_by_day.get(key)
        return None if held is None else float(held)

    @property
    def adjustment(self) -> Adjustment:
        return Adjustment(
            bias_c=self.bias_c, sigma_mult=self.sigma_mult, forecast_trust=self.forecast_trust
        )

    def record(
        self,
        result: Prediction,
        obs: Observation | None = None,
        *,
        forecast_carried: bool = False,
        climate_std_c: float | None = None,
        asks: dict[str, float | None] | None = None,
        bids: dict[str, float | None] | None = None,
        nos: list[dict] | None = None,
    ) -> None:
        day = result.observed_at[:10]
        book = {label: (None if price is None else round(float(price), 4)) for label, price in (asks or {}).items()}
        bid_book = {label: (None if price is None else round(float(price), 4)) for label, price in (bids or {}).items()}
        snap = {
            "observed_at": result.observed_at,
            "mu_c": round(result.mu_c, 3),
            "sigma_c": round(result.sigma_c, 3),
            "hko_max_c": result.hko_max_c,
            "mode": result.mode.label,
            "asks": book,
            "bids": bid_book,
        }
        snaps = self.forecasts.setdefault(day, [])
        if snaps and snaps[-1]["observed_at"] == snap["observed_at"]:
            snaps[-1] = snap
        else:
            snaps.append(snap)
        del snaps[:-288]
        append_case(
            {
                "kind": "forecast",
                "city": "hk",
                "observed_at": result.observed_at,
                "inputs": None
                if obs is None
                else {
                    "hko_temp_c": obs.hko_temp_c,
                    "hko_max_c": obs.hko_max_c,
                    "hka_temp_c": obs.hka_temp_c,
                    "hka_max_c": obs.hka_max_c,
                    "global_wm2": obs.global_wm2,
                    "direct_wm2": obs.direct_wm2,
                    "diffuse_wm2": obs.diffuse_wm2,
                    "uv_index": obs.uv_index,
                    "wind_kmh": obs.wind_kmh,
                    "wind_from_deg": obs.wind_from_deg,
                    "icon": obs.icon,
                    "forecast_max_c": obs.forecast_max_c,
                    "forecast_carried": forecast_carried,
                    "climate_mean_c": result.climate_mean_c,
                    "climate_std_c": climate_std_c,
                },
                "adjustment": {
                    "bias_c": round(self.bias_c, 4),
                    "sigma_mult": round(self.sigma_mult, 4),
                    "forecast_trust": round(self.forecast_trust, 4),
                },
                "outputs": {
                    "anchor_c": round(result.anchor_c, 3),
                    "heating": round(result.heating, 3),
                    "wind": round(result.wind, 3),
                    "remaining_clear_rise_c": round(result.remaining_clear_rise_c, 3),
                    "path_max_c": round(result.path_max_c, 3),
                    "mu_c": round(result.mu_c, 3),
                    "sigma_c": round(result.sigma_c, 3),
                    "mode": result.mode.label,
                    "fair": {row.label: round(row.probability, 4) for row in result.fair},
                },
                "asks": book,
                "bids": bid_book,
                "nos": nos or [],
            }
        )

    def apply_settlement(self, day: date, settled_c: float) -> str | None:
        key = day.isoformat()
        snaps = (self.forecasts or {}).get(key) or []
        if key in (self.scored or {}) or not snaps:
            return None
        chosen = min(snaps, key=lambda row: abs(_minutes(row["observed_at"]) - 16 * 60))
        error = settled_c - float(chosen["mu_c"])
        self.bias_c = _clamp((1.0 - _BIAS_STEP) * self.bias_c + _BIAS_STEP * error, -1.0, 1.0)
        ratio = abs(error) / max(float(chosen["sigma_c"]), 0.05)
        target = _clamp(ratio, 0.7, 1.8)
        self.sigma_mult = _clamp(
            (1.0 - _SIGMA_STEP) * self.sigma_mult + _SIGMA_STEP * target,
            0.6,
            1.8,
        )
        morning = _nearest(snaps, 8 * 60 + 30)
        morning_error = None
        if morning is not None:
            morning_error = settled_c - float(morning["mu_c"])
            self.forecast_trust = _clamp(self.forecast_trust + 0.08 * _clamp(morning_error, -1.5, 1.5), 0.2, 0.9)
        self.scored[key] = {
            "settled_c": settled_c,
            "predicted_mu_c": chosen["mu_c"],
            "predicted_at": chosen["observed_at"],
            "error_c": round(error, 3),
            "morning_at": None if morning is None else morning["observed_at"],
            "morning_error_c": None if morning_error is None else round(morning_error, 3),
            "bracket": bracket_for(settled_c).label,
            "bias_c": round(self.bias_c, 3),
            "sigma_mult": round(self.sigma_mult, 3),
            "forecast_trust": round(self.forecast_trust, 3),
        }
        return (
            f"验证 {key}  结算 {settled_c:.1f}°C（{bracket_for(settled_c).label}）  "
            f"对照 {chosen['observed_at']} 的预测 {chosen['mu_c']:.2f}°C  "
            f"误差 {error:+.2f}°C  偏差改为 {self.bias_c:+.2f}°C  波动 ×{self.sigma_mult:.2f}"
            + (
                ""
                if morning_error is None
                else f"  早间误差 {morning_error:+.2f}°C  预报信任 {self.forecast_trust:.2f}"
            )
        )

    def status_line(self) -> str:
        return (
            f"在线修正  偏差 {self.bias_c:+.2f}°C   波动 ×{self.sigma_mult:.2f}   "
            f"已验证 {len(self.scored or {})} 日   早间预报信任 {self.forecast_trust:.2f}"
        )


def _nearest(snaps: list[dict], target_minute: int) -> dict | None:
    chosen = min(snaps, key=lambda row: abs(_minutes(row["observed_at"]) - target_minute))
    if abs(_minutes(chosen["observed_at"]) - target_minute) > 90:
        return None
    return chosen


def _minutes(stamp: str) -> int:
    clock = stamp.split()[1]
    hour, minute = clock.split(":")[:2]
    return int(hour) * 60 + int(minute)


def load_state(path: Path | None = None) -> OnlineState:
    file = path or _STATE_PATH
    if not file.exists():
        return OnlineState()
    payload = json.loads(file.read_text())
    return OnlineState(
        bias_c=float(payload.get("bias_c", 0.0)),
        sigma_mult=float(payload.get("sigma_mult", 1.0)),
        forecasts=payload.get("forecasts") or {},
        scored=payload.get("scored") or {},
        forecast_max_by_day=payload.get("forecast_max_by_day") or {},
        forecast_trust=float(payload.get("forecast_trust", 0.62)),
    )


def save_state(state: OnlineState, path: Path | None = None) -> None:
    file = path or _STATE_PATH
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(
            {
                "bias_c": round(state.bias_c, 4),
                "sigma_mult": round(state.sigma_mult, 4),
                "forecasts": state.forecasts,
                "scored": state.scored,
                "forecast_max_by_day": state.forecast_max_by_day,
                "forecast_trust": round(state.forecast_trust, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def fetch_settled_maxima(today: date) -> dict[date, float]:
    """Read Absolute Daily Max for this month, and last month early in a new month."""

    months = {today.replace(day=1)}
    if today.day <= 3:
        months.add((today.replace(day=1) - timedelta(days=1)).replace(day=1))
    found: dict[date, float] = {}
    for month_start in months:
        url = DAILY_EXTRACT_URL.format(yyyymm=f"{month_start.year}{month_start.month:02d}")
        request = urllib.request.Request(url, headers={"User-Agent": "hk-temp-model/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        found.update(absolute_daily_maxes(payload, month_start.year))
    return found


def verify_settled(state: OnlineState, today: date | None = None) -> list[str]:
    maxima = fetch_settled_maxima(today or datetime.now().date())
    notes = []
    for day, settled in sorted(maxima.items()):
        note = state.apply_settlement(day, settled)
        if note:
            notes.append(note)
            scored = (state.scored or {}).get(day.isoformat()) or {}
            append_case(
                {
                    "kind": "settlement",
                    "city": "hk",
                    "day": day.isoformat(),
                    "settled_c": settled,
                    "error_c": scored.get("error_c"),
                    "morning_error_c": scored.get("morning_error_c"),
                    "forecast_trust": scored.get("forecast_trust"),
                }
            )
    return notes
