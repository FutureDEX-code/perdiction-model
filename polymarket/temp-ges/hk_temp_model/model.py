"""Intraday distribution of the HKO Absolute Daily Max.

Climatology is estimated from the settlement station. The heating term is an
operational prior: global radiation carries cloud thickness, the direct-beam
fraction separates bright overcast from sun, UV is a small lagging check, and
easterly wind caps the waterfront station. King's Park is only a sensor proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hk_temp_model.climatology import Climatology, DayClimate
from hk_temp_model.observations import Observation
from hk_temp_model.settlement import SETTLEMENT_NAME, SETTLEMENT_STATION, Bracket, default_brackets

# September clear-sky noon is near 12.4 HKT. The daily max usually lags it.
_SOLAR_NOON_HKT = 12.4
_HALF_DAY_HOURS = 6.2
_CLEAR_GLOBAL_NOON = 980.0
_CLEAR_UV_NOON = 10.5
_TEMP_PEAK_HKT = 14.75
_HEATING_WINDOW_HOURS = 5.0
_REMAINING_RISE_PER_HOUR = 0.48

# Icon cloud fraction. Missing icons leave the radiation term on its own.
_ICON_CLOUD = {
    50: 0.10,
    51: 0.30,
    52: 0.45,
    53: 0.40,
    54: 0.55,
    60: 0.75,
    61: 0.90,
    62: 0.80,
    63: 0.85,
    64: 0.92,
    65: 0.90,
    76: 0.80,
    77: 0.85,
    90: 0.20,
    91: 0.15,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _phi(z: float) -> float:
    if math.isinf(z):
        return 1.0 if z > 0 else 0.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clear_sky(hour: float) -> tuple[float, float]:
    scale = max(0.0, math.cos(math.pi / 2 * abs(hour - _SOLAR_NOON_HKT) / _HALF_DAY_HOURS))
    return _CLEAR_GLOBAL_NOON * scale**1.2, _CLEAR_UV_NOON * scale**1.15


def _icon_heating(icon: int | None) -> float | None:
    if icon not in _ICON_CLOUD:
        return None
    return _clamp(1.0 - 0.85 * _ICON_CLOUD[icon], 0.15, 1.0)


def heating_factor(obs: Observation) -> float:
    """Combine radiation, UV, and the weather icon into a 0–1 heating factor."""

    icon_factor = _icon_heating(obs.icon)
    if obs.global_wm2 is None or obs.global_wm2 <= 0 or obs.direct_wm2 is None:
        return 0.55 if icon_factor is None else icon_factor

    global_clear, uv_clear = _clear_sky(obs.hour_hkt)
    transmittance = _clamp(obs.global_wm2 / global_clear, 0.0, 1.0) if global_clear > 1 else 0.0
    direct_fraction = _clamp((obs.direct_wm2 / obs.global_wm2) / 0.75, 0.0, 1.0)
    uv_factor = 0.0
    if obs.uv_index is not None and uv_clear > 0.2:
        uv_factor = _clamp(obs.uv_index / uv_clear, 0.0, 1.0)
    radiation = _clamp(0.15 + 0.55 * transmittance + 0.25 * direct_fraction + 0.05 * uv_factor, 0.0, 1.0)
    if icon_factor is None:
        return radiation
    return 0.65 * radiation + 0.35 * icon_factor


def wind_factor(obs: Observation) -> float:
    """Easterly flow off the harbour suppresses the headquarters maximum."""

    if obs.wind_kmh is None or obs.wind_from_deg is None:
        return 1.0
    onshore = max(0.0, math.cos(math.radians(obs.wind_from_deg - 90.0)))
    return _clamp(1.05 - 0.012 * max(0.0, obs.wind_kmh - 10.0) * onshore, 0.65, 1.10)


def remaining_clear_rise(hour: float) -> float:
    return max(0.0, _REMAINING_RISE_PER_HOUR * (_TEMP_PEAK_HKT - hour))


@dataclass(frozen=True)
class BracketFair:
    label: str
    lo_c: float
    hi_c: float | None
    probability: float
    dist_c: float | None
    already_printing: bool


@dataclass(frozen=True)
class Prediction:
    settlement_station: str
    settlement_name: str
    observed_at: str
    hko_max_c: float
    hka_max_c: float | None
    airport_minus_hko_c: float | None
    historical_airport_gap_c: float
    climate_mean_c: float
    forecast_max_c: float | None
    anchor_c: float
    heating: float
    wind: float
    remaining_clear_rise_c: float
    path_max_c: float
    mu_c: float
    sigma_c: float
    fair: tuple[BracketFair, ...]

    @property
    def mode(self) -> BracketFair:
        return max(self.fair, key=lambda row: row.probability)


def _anchor(climate: DayClimate, forecast_max_c: float | None) -> float:
    if forecast_max_c is None:
        return climate.mean_c
    # The issued daily max beat the September climate mean by about 2°C today.
    return 0.85 * forecast_max_c + 0.15 * climate.mean_c


def _respect_issued_forecast(
    path: float, forecast_max_c: float | None, hour: float, trust: float
) -> float:
    """A dim morning sensor must not erase the forecast while hours remain.

    On 24 Sep the nine-day maximum was 32°C from midnight, and the day printed
    32.2°C. The radiation path sat near 30°C until the headquarters max itself
    had already climbed.
    """

    if forecast_max_c is None or path >= forecast_max_c or hour >= _TEMP_PEAK_HKT:
        return path
    time_left = _clamp((_TEMP_PEAK_HKT - hour) / _HEATING_WINDOW_HOURS, 0.0, 1.0)
    return path + (forecast_max_c - path) * (_clamp(trust, 0.2, 0.9) * time_left)


def _sigma(
    hour: float,
    heating: float,
    climate: DayClimate,
    radiation_missing: bool,
    shortfall_c: float,
    upside_c: float,
) -> float:
    # After the peak, a fallen temperature means the printed max is likely final.
    # A wide sigma around 31.9 would dump the truncated tail into the next degree.
    if hour >= _TEMP_PEAK_HKT and upside_c < 0.05:
        rebound = math.exp(-shortfall_c / 0.35)
        sigma = 0.04 + 0.18 * heating * rebound
        if hour >= 16.0:
            sigma *= 0.6
        return max(sigma, 0.04)

    progress = _clamp((_TEMP_PEAK_HKT - hour) / _HEATING_WINDOW_HOURS, 0.0, 1.0)
    uncertainty = 1.0 + 0.20 * (1.0 - abs(2.0 * heating - 1.0))
    sigma = (0.40 + 0.50 * progress) * uncertainty
    if progress > 0.8:
        sigma = max(sigma, min(climate.std_c, 1.6) * 0.55)
    if radiation_missing:
        sigma *= 1.15
    if hour >= 18.0:
        return 0.15
    return max(sigma, 0.35)


def _distance(bracket: Bracket, running_max_c: float) -> float | None:
    if bracket.hi is not None and running_max_c >= bracket.hi:
        return None
    if running_max_c >= bracket.lo:
        return 0.0
    return bracket.lo - running_max_c


def _fair_values(
    mu_c: float,
    sigma_c: float,
    running_max_c: float,
    brackets: tuple[Bracket, ...],
) -> tuple[BracketFair, ...]:
    floor_z = (running_max_c - mu_c) / sigma_c
    surviving = max(1.0 - _phi(floor_z), 1e-12)
    raw: list[float] = []
    rows: list[tuple[Bracket, float | None, bool]] = []
    for bracket in brackets:
        distance = _distance(bracket, running_max_c)
        printing = bracket.contains(running_max_c)
        rows.append((bracket, distance, printing))
        if bracket.hi is not None and bracket.hi <= running_max_c:
            raw.append(0.0)
            continue
        lower = max(bracket.lo, running_max_c)
        upper_phi = 1.0 if bracket.hi is None else _phi((bracket.hi - mu_c) / sigma_c)
        raw.append(max(0.0, upper_phi - _phi((lower - mu_c) / sigma_c)) / surviving)
    total = sum(raw) or 1.0
    return tuple(
        BracketFair(
            label=bracket.label,
            lo_c=bracket.lo,
            hi_c=bracket.hi,
            probability=probability / total,
            dist_c=distance,
            already_printing=printing,
        )
        for (bracket, distance, printing), probability in zip(rows, raw)
    )


@dataclass(frozen=True)
class Adjustment:
    """Slow correction learned from settled Absolute Daily Max errors."""

    bias_c: float = 0.0
    sigma_mult: float = 1.0
    forecast_trust: float = 0.62


def predict(
    obs: Observation,
    climatology: Climatology,
    brackets: tuple[Bracket, ...] | None = None,
    adjustment: Adjustment | None = None,
) -> Prediction:
    """Predict the settlement maximum. The HKO running max is the floor."""

    climate = climatology.for_date(obs.observed_at.date())
    heating = heating_factor(obs)
    wind = wind_factor(obs)
    rise = remaining_clear_rise(obs.hour_hkt)
    current = obs.hko_max_c if obs.hko_temp_c is None else obs.hko_temp_c
    shortfall = max(0.0, obs.hko_max_c - current)
    potential = rise * heating * wind
    upside = max(0.0, potential - shortfall)
    trust = 0.62 if adjustment is None else adjustment.forecast_trust
    path = _respect_issued_forecast(obs.hko_max_c + upside, obs.forecast_max_c, obs.hour_hkt, trust)
    anchor = _anchor(climate, obs.forecast_max_c)
    progress = _clamp((_TEMP_PEAK_HKT - obs.hour_hkt) / _HEATING_WINDOW_HOURS, 0.0, 1.0)
    anchor_weight = 0.55 * progress + 0.15
    mu = anchor_weight * anchor + (1.0 - anchor_weight) * path
    bias = 0.0 if adjustment is None else adjustment.bias_c
    sigma_mult = 1.0 if adjustment is None else adjustment.sigma_mult
    mu += bias
    # The published maximum cannot finish below the value already printed.
    mu = max(mu, obs.hko_max_c)
    radiation_missing = obs.global_wm2 is None or obs.direct_wm2 is None
    sigma = _sigma(obs.hour_hkt, heating, climate, radiation_missing, shortfall, upside)
    sigma = max(0.04, sigma * sigma_mult)
    gap = None if obs.hka_max_c is None else obs.hka_max_c - obs.hko_max_c
    return Prediction(
        settlement_station=SETTLEMENT_STATION,
        settlement_name=SETTLEMENT_NAME,
        observed_at=obs.observed_at.strftime("%Y-%m-%d %H:%M HKT"),
        hko_max_c=obs.hko_max_c,
        hka_max_c=obs.hka_max_c,
        airport_minus_hko_c=gap,
        historical_airport_gap_c=climatology.airport_gap.mean_c,
        climate_mean_c=climate.mean_c,
        forecast_max_c=obs.forecast_max_c,
        anchor_c=anchor,
        heating=heating,
        wind=wind,
        remaining_clear_rise_c=rise,
        path_max_c=path,
        mu_c=mu,
        sigma_c=sigma,
        fair=_fair_values(mu, sigma, obs.hko_max_c, brackets or default_brackets()),
    )
