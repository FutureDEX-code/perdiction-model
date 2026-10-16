"""Paris (LFPB) and New York (KLGA) daily-high markets.

Polymarket settles Paris on Le Bourget whole °C and NYC on LaGuardia whole °F.
Live floor comes from today's METARs; the daily-max forecast is Open-Meteo at the
station coordinates (guide only, not the settlement feed).
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from hk_temp_model.cases import append_case
from hk_temp_model.model import Adjustment, BracketFair, _clamp, _phi
from hk_temp_model.settlement import Bracket

_DATA = Path(__file__).resolve().parent / "data"
_BIAS_STEP = 0.2
_SIGMA_STEP = 0.1
_TRUST_STEP = 0.08
_MORNING_MINUTE = 9 * 60
_METAR_URL = "https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours={hours}"
_METAR_CLOCK = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")
_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,cloud_cover,wind_speed_10m,wind_direction_10m"
    "&daily=temperature_2m_max,temperature_2m_min"
    "&timezone={tz}&forecast_days=3"
)


@dataclass(frozen=True)
class CitySpec:
    key: str
    icao: str
    name: str
    tz: str
    tz_label: str
    unit: str  # "C" or "F"
    lat: float
    lon: float
    low: int
    high: int
    solar_noon: float
    temp_peak: float
    score_hour: float  # local hour used when scoring a settled day


PARIS = CitySpec(
    key="paris",
    icao="LFPB",
    name="Paris-Le Bourget",
    tz="Europe/Paris",
    tz_label="CEST",
    unit="C",
    lat=48.9694,
    lon=2.4414,
    low=14,
    high=28,
    solar_noon=12.5,
    temp_peak=14.5,
    score_hour=16.0,
)
NYC = CitySpec(
    key="nyc",
    icao="KLGA",
    name="New York LaGuardia",
    tz="America/New_York",
    tz_label="ET",
    unit="F",
    lat=40.7769,
    lon=-73.8740,
    low=55,
    high=85,
    solar_noon=12.5,
    temp_peak=14.5,
    score_hour=16.0,
)

INTL_CITIES = (PARIS, NYC)


def unit_symbol(unit: str) -> str:
    return "°C" if unit == "C" else "°F"


def c_to_settlement(temp_c: float, unit: str) -> float:
    if unit == "C":
        return float(round(temp_c))
    return float(round(temp_c * 9.0 / 5.0 + 32.0))


def whole_brackets(low: int, high: int, unit: str) -> tuple[Bracket, ...]:
    """Integer ladder: ``{low} or below``, each integer, ``{high} or higher``."""

    if high <= low:
        raise ValueError("high bracket must be above the low tail")
    sym = unit_symbol(unit)
    brackets = [Bracket(f"{low}{sym} or below", float("-inf"), float(low + 1))]
    for degree in range(low + 1, high):
        brackets.append(Bracket(f"{degree}{sym}", float(degree), float(degree + 1)))
    brackets.append(Bracket(f"{high}{sym} or higher", float(high), None))
    return tuple(brackets)


def bracket_for(temp: float, brackets: tuple[Bracket, ...]) -> Bracket:
    for bracket in brackets:
        if bracket.contains(temp):
            return bracket
    raise ValueError(f"no bracket contains {temp}")


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "hk-temp-model-intl/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def fetch_metars(icao: str, hours: int = 30) -> list[dict]:
    payload = _fetch_json(_METAR_URL.format(icao=icao, hours=hours))
    if not isinstance(payload, list):
        raise ValueError(f"unexpected METAR payload for {icao}")
    return payload


@dataclass(frozen=True)
class IntlObservation:
    city: CitySpec
    observed_at: datetime
    temp: float
    running_max: float
    forecast_max: float | None
    climate_mean: float
    climate_std: float
    cloud_cover: float | None
    wind_kmh: float | None
    wind_from_deg: float | None

    @property
    def hour_local(self) -> float:
        local = self.observed_at
        return local.hour + local.minute / 60 + local.second / 3600


def _observation_utc(row: dict) -> datetime | None:
    """UTC time the temperature was observed.

    Aviation Weather rounds ``reportTime`` up to the hour. A 23:51 reading
    then falls on the next local midnight and is counted as the next day's high.
    The day and clock in the METAR text are the observation.
    """

    stamp = row.get("reportTime") or row.get("receiptTime")
    if not stamp:
        return None
    reported = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(timezone.utc)
    match = _METAR_CLOCK.search(str(row.get("rawOb") or ""))
    if match is None:
        return reported
    day, hour, minute = (int(match.group(index)) for index in (1, 2, 3))
    if hour > 23 or minute > 59 or not 1 <= day <= 31:
        return reported
    year, month = reported.year, reported.month
    for _ in range(2):
        if 1 <= day <= monthrange(year, month)[1]:
            observed = reported.replace(
                year=year, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0
            )
            if observed <= reported + timedelta(minutes=2):
                if reported - observed <= timedelta(hours=2):
                    return observed
                return reported
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return reported


def _metar_local_day(row: dict, tz: ZoneInfo) -> date | None:
    observed = _observation_utc(row)
    if observed is None:
        return None
    return observed.astimezone(tz).date()


def running_max_from_metars(city: CitySpec, metars: list[dict], day: date) -> tuple[float, float, datetime]:
    """Return (running_max, latest_temp, latest_local_time) for the local calendar day."""

    tz = ZoneInfo(city.tz)
    temps: list[tuple[datetime, float]] = []
    for row in metars:
        if row.get("temp") is None:
            continue
        observed = _observation_utc(row)
        if observed is None:
            continue
        local = observed.astimezone(tz).replace(tzinfo=None)
        if local.date() != day:
            continue
        temps.append((local, c_to_settlement(float(row["temp"]), city.unit)))
    if not temps:
        raise ValueError(f"no {city.icao} METARs for {day.isoformat()}")
    temps.sort(key=lambda item: item[0])
    latest_at, latest_temp = temps[-1]
    return max(t for _, t in temps), latest_temp, latest_at


def fetch_open_meteo(city: CitySpec) -> dict:
    return _fetch_json(
        _FORECAST_URL.format(lat=city.lat, lon=city.lon, tz=city.tz)
    )  # type: ignore[return-value]


def _daily_max_c(payload: dict, day: date) -> float | None:
    days = (payload.get("daily") or {}).get("time") or []
    maxima = (payload.get("daily") or {}).get("temperature_2m_max") or []
    key = day.isoformat()
    for stamp, value in zip(days, maxima):
        if stamp == key and value is not None:
            return float(value)
    return None


def fetch_observation(city: CitySpec, metars: list[dict] | None = None) -> IntlObservation:
    tz = ZoneInfo(city.tz)
    now_local = datetime.now(tz).replace(tzinfo=None)
    rows = metars if metars is not None else fetch_metars(city.icao)
    running_max, latest_temp, observed_at = running_max_from_metars(city, rows, now_local.date())
    forecast_payload = fetch_open_meteo(city)
    forecast_c = _daily_max_c(forecast_payload, now_local.date())
    # soft climate: blend today/tomorrow Open-Meteo maxima when available
    climate_vals = []
    for offset in (0, 1, -1):
        day = now_local.date() + timedelta(days=offset)
        value = _daily_max_c(forecast_payload, day)
        if value is not None:
            climate_vals.append(value)
    climate_mean_c = sum(climate_vals) / len(climate_vals) if climate_vals else (forecast_c or latest_temp)
    climate_std = 1.6 if city.unit == "C" else 2.8
    current = forecast_payload.get("current") or {}
    cloud = current.get("cloud_cover")
    wind_kmh = current.get("wind_speed_10m")
    wind_dir = current.get("wind_direction_10m")
    forecast = None if forecast_c is None else c_to_settlement(forecast_c, city.unit)
    climate_mean = c_to_settlement(climate_mean_c, city.unit)
    return IntlObservation(
        city=city,
        observed_at=observed_at,
        temp=latest_temp,
        running_max=running_max,
        forecast_max=forecast,
        climate_mean=float(climate_mean),
        climate_std=climate_std,
        cloud_cover=None if cloud is None else float(cloud),
        wind_kmh=None if wind_kmh is None else float(wind_kmh),
        wind_from_deg=None if wind_dir is None else float(wind_dir),
    )


def _heating(obs: IntlObservation) -> float:
    if obs.cloud_cover is None:
        return 0.55
    clear = _clamp(1.0 - 0.85 * (obs.cloud_cover / 100.0), 0.15, 1.0)
    # weaken heating far from solar noon
    hour = obs.hour_local
    span = abs(hour - obs.city.solar_noon)
    day_scale = _clamp(1.0 - span / 7.0, 0.25, 1.0)
    return clear * day_scale


def _wind(obs: IntlObservation) -> float:
    if obs.wind_kmh is None:
        return 1.0
    return _clamp(1.05 - 0.008 * max(0.0, obs.wind_kmh - 8.0), 0.70, 1.10)


def _remaining_rise(obs: IntlObservation) -> float:
    per_hour = 0.55 if obs.city.unit == "C" else 1.0
    return max(0.0, per_hour * (obs.city.temp_peak - obs.hour_local))


def _sigma(
    hour: float,
    heating: float,
    climate_std: float,
    shortfall: float,
    upside: float,
    peak: float,
) -> float:
    if hour >= peak and upside < 0.05:
        rebound = math.exp(-shortfall / 0.6)
        sigma = 0.15 + 0.35 * heating * rebound
        if hour >= 16.0:
            sigma *= 0.7
        return max(sigma, 0.15)
    progress = _clamp((peak - hour) / 5.0, 0.0, 1.0)
    uncertainty = 1.0 + 0.20 * (1.0 - abs(2.0 * heating - 1.0))
    sigma = (0.55 + 0.70 * progress) * uncertainty
    if progress > 0.8:
        sigma = max(sigma, min(climate_std, 3.0) * 0.55)
    if hour >= 18.0:
        return 0.25
    return max(sigma, 0.45)


def _distance(bracket: Bracket, running_max: float) -> float | None:
    if bracket.hi is not None and running_max >= bracket.hi:
        return None
    if running_max >= bracket.lo:
        return 0.0
    return bracket.lo - running_max


def _fair_values(
    mu: float,
    sigma: float,
    running_max: float,
    brackets: tuple[Bracket, ...],
) -> tuple[BracketFair, ...]:
    floor_z = (running_max - mu) / sigma
    surviving = max(1.0 - _phi(floor_z), 1e-12)
    raw: list[float] = []
    rows: list[tuple[Bracket, float | None, bool]] = []
    for bracket in brackets:
        distance = _distance(bracket, running_max)
        printing = bracket.contains(running_max)
        rows.append((bracket, distance, printing))
        if bracket.hi is not None and bracket.hi <= running_max:
            raw.append(0.0)
            continue
        lower = max(bracket.lo, running_max)
        upper_phi = 1.0 if bracket.hi is None else _phi((bracket.hi - mu) / sigma)
        raw.append(max(0.0, upper_phi - _phi((lower - mu) / sigma)) / surviving)
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
class IntlPrediction:
    city_key: str
    settlement_station: str
    settlement_name: str
    unit: str
    observed_at: str
    running_max: float
    current: float
    climate_mean: float
    forecast_max: float | None
    heating: float
    wind: float
    remaining_clear_rise: float
    path_max: float
    mu: float
    sigma: float
    fair: tuple[BracketFair, ...]

    @property
    def mode(self) -> BracketFair:
        return max(self.fair, key=lambda row: row.probability)


def predict(
    obs: IntlObservation,
    brackets: tuple[Bracket, ...] | None = None,
    adjustment: Adjustment | None = None,
) -> IntlPrediction:
    city = obs.city
    heating = _heating(obs)
    wind = _wind(obs)
    rise = _remaining_rise(obs)
    shortfall = max(0.0, obs.running_max - obs.temp)
    potential = rise * heating * wind
    upside = max(0.0, potential - shortfall)
    path = obs.running_max + upside
    trust = 0.62 if adjustment is None else adjustment.forecast_trust
    if obs.forecast_max is not None and path < obs.forecast_max and obs.hour_local < city.temp_peak:
        time_left = _clamp((city.temp_peak - obs.hour_local) / 5.0, 0.0, 1.0)
        path = path + (obs.forecast_max - path) * (_clamp(trust, 0.2, 0.9) * time_left)
    anchor = obs.climate_mean if obs.forecast_max is None else 0.85 * obs.forecast_max + 0.15 * obs.climate_mean
    progress = _clamp((city.temp_peak - obs.hour_local) / 5.0, 0.0, 1.0)
    anchor_weight = 0.55 * progress + 0.15
    mu = anchor_weight * anchor + (1.0 - anchor_weight) * path
    bias = 0.0 if adjustment is None else adjustment.bias_c
    sigma_mult = 1.0 if adjustment is None else adjustment.sigma_mult
    mu = max(mu + bias, obs.running_max)
    sigma = _sigma(obs.hour_local, heating, obs.climate_std, shortfall, upside, city.temp_peak)
    sigma = max(0.15, sigma * sigma_mult)
    ladder = brackets or whole_brackets(city.low, city.high, city.unit)
    local_stamp = obs.observed_at.strftime(f"%Y-%m-%d %H:%M {city.tz_label}")
    return IntlPrediction(
        city_key=city.key,
        settlement_station=city.icao,
        settlement_name=city.name,
        unit=city.unit,
        observed_at=local_stamp,
        running_max=obs.running_max,
        current=obs.temp,
        climate_mean=obs.climate_mean,
        forecast_max=obs.forecast_max,
        heating=heating,
        wind=wind,
        remaining_clear_rise=rise,
        path_max=path,
        mu=mu,
        sigma=sigma,
        fair=_fair_values(mu, sigma, obs.running_max, ladder),
    )


def _snap_minutes(stamp: str) -> int:
    clock = stamp.split()[1]
    hour, minute = clock.split(":")[:2]
    return int(hour) * 60 + int(minute)


def _nearest_snap(snaps: list[dict], target_minute: int) -> dict | None:
    chosen = min(snaps, key=lambda row: abs(_snap_minutes(row["observed_at"]) - target_minute))
    if abs(_snap_minutes(chosen["observed_at"]) - target_minute) > 90:
        return None
    return chosen


@dataclass
class IntlOnlineState:
    bias: float = 0.0
    sigma_mult: float = 1.0
    forecasts: dict[str, list[dict]] | None = None
    scored: dict[str, dict] | None = None
    forecast_trust: float = 0.62

    def __post_init__(self) -> None:
        if self.forecasts is None:
            self.forecasts = {}
        if self.scored is None:
            self.scored = {}

    @property
    def adjustment(self) -> Adjustment:
        return Adjustment(
            bias_c=self.bias, sigma_mult=self.sigma_mult, forecast_trust=self.forecast_trust
        )

    def record(
        self,
        result: IntlPrediction,
        obs: IntlObservation | None = None,
        asks: dict | None = None,
        bids: dict | None = None,
        nos: list | None = None,
    ) -> None:
        day = result.observed_at[:10]
        book = {label: (None if price is None else round(float(price), 4)) for label, price in (asks or {}).items()}
        bid_book = {label: (None if price is None else round(float(price), 4)) for label, price in (bids or {}).items()}
        snap = {
            "observed_at": result.observed_at,
            "mu": round(result.mu, 3),
            "sigma": round(result.sigma, 3),
            "running_max": result.running_max,
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
                "city": result.city_key,
                "observed_at": result.observed_at,
                "inputs": None
                if obs is None
                else {
                    "temp": obs.temp,
                    "running_max": obs.running_max,
                    "forecast_max": obs.forecast_max,
                    "climate_mean": obs.climate_mean,
                    "climate_std": obs.climate_std,
                    "cloud_cover": obs.cloud_cover,
                    "wind_kmh": obs.wind_kmh,
                    "wind_from_deg": obs.wind_from_deg,
                },
                "adjustment": {
                    "bias": round(self.bias, 4),
                    "sigma_mult": round(self.sigma_mult, 4),
                    "forecast_trust": round(self.forecast_trust, 4),
                },
                "outputs": {
                    "heating": round(result.heating, 3),
                    "wind": round(result.wind, 3),
                    "remaining_clear_rise": round(result.remaining_clear_rise, 3),
                    "path_max": round(result.path_max, 3),
                    "mu": round(result.mu, 3),
                    "sigma": round(result.sigma, 3),
                    "mode": result.mode.label,
                    "fair": {row.label: round(row.probability, 4) for row in result.fair},
                },
                "asks": book,
                "bids": bid_book,
                "nos": nos or [],
            }
        )

    def apply_settlement(self, city: CitySpec, day: date, settled: float) -> str | None:
        key = day.isoformat()
        snaps = (self.forecasts or {}).get(key) or []
        if key in (self.scored or {}) or not snaps:
            return None
        target_min = int(city.score_hour * 60)

        def minutes(stamp: str) -> int:
            return _snap_minutes(stamp)

        chosen = min(snaps, key=lambda row: abs(minutes(row["observed_at"]) - target_min))
        error = settled - float(chosen["mu"])
        self.bias = _clamp((1.0 - _BIAS_STEP) * self.bias + _BIAS_STEP * error, -2.0, 2.0)
        ratio = abs(error) / max(float(chosen["sigma"]), 0.1)
        target = _clamp(ratio, 0.7, 1.8)
        self.sigma_mult = _clamp(
            (1.0 - _SIGMA_STEP) * self.sigma_mult + _SIGMA_STEP * target,
            0.6,
            1.8,
        )
        morning = _nearest_snap(snaps, _MORNING_MINUTE)
        morning_error = None
        if morning is not None:
            morning_error = settled - float(morning["mu"])
            # Paris mornings sit under the issued forecast. Trust up to 0.9
            # closed part of that gap and added no Yes tickets. The same pull
            # on New York bought the next bracket; those tickets all lost,
            # so New York stays on the original 0.62.
            ceiling = 0.90 if city.key == "paris" else 0.62
            self.forecast_trust = _clamp(
                self.forecast_trust + _TRUST_STEP * _clamp(morning_error, -1.5, 1.5),
                0.2,
                ceiling,
            )
        ladder = whole_brackets(city.low, city.high, city.unit)
        self.scored[key] = {
            "settled": settled,
            "predicted_mu": chosen["mu"],
            "predicted_at": chosen["observed_at"],
            "error": round(error, 3),
            "morning_at": None if morning is None else morning["observed_at"],
            "morning_error": None if morning_error is None else round(morning_error, 3),
            "bracket": bracket_for(settled, ladder).label,
            "bias": round(self.bias, 3),
            "sigma_mult": round(self.sigma_mult, 3),
            "forecast_trust": round(self.forecast_trust, 3),
        }
        return (
            f"验证 {city.key} {key}  结算 {settled:.0f}{unit_symbol(city.unit)}"
            f"（{bracket_for(settled, ladder).label}）  "
            f"对照 {chosen['observed_at']} 的预测 {chosen['mu']:.2f}  "
            f"误差 {error:+.2f}  偏差改为 {self.bias:+.2f}  波动 ×{self.sigma_mult:.2f}"
            + (
                ""
                if morning_error is None
                else f"  早间误差 {morning_error:+.2f}  预报信任 {self.forecast_trust:.2f}"
            )
        )

    def status_line(self) -> str:
        return (
            f"在线修正  偏差 {self.bias:+.2f}   波动 ×{self.sigma_mult:.2f}   "
            f"已验证 {len(self.scored or {})} 日   早间预报信任 {self.forecast_trust:.2f}"
        )


def state_path(city: CitySpec) -> Path:
    return _DATA / f"online_state_{city.key}.json"


def load_state(city: CitySpec) -> IntlOnlineState:
    path = state_path(city)
    if not path.exists():
        return IntlOnlineState()
    payload = json.loads(path.read_text())
    return IntlOnlineState(
        bias=float(payload.get("bias", payload.get("bias_c", 0.0))),
        sigma_mult=float(payload.get("sigma_mult", 1.0)),
        forecasts=payload.get("forecasts") or {},
        scored=payload.get("scored") or {},
        forecast_trust=float(payload.get("forecast_trust", 0.62)),
    )


def save_state(city: CitySpec, state: IntlOnlineState) -> None:
    path = state_path(city)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "bias": round(state.bias, 4),
                "sigma_mult": round(state.sigma_mult, 4),
                "forecasts": state.forecasts,
                "scored": state.scored,
                "forecast_trust": round(state.forecast_trust, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def verify_settled(city: CitySpec, state: IntlOnlineState) -> list[str]:
    """Score yesterday (and earlier unscored days) from METAR daily maxima."""

    tz = ZoneInfo(city.tz)
    today = datetime.now(tz).date()
    metars = fetch_metars(city.icao, hours=60)
    notes: list[str] = []
    for key in sorted((state.forecasts or {}).keys()):
        day = date.fromisoformat(key)
        if day >= today:
            continue
        if key in (state.scored or {}):
            continue
        try:
            settled, _, _ = running_max_from_metars(city, metars, day)
        except ValueError:
            continue
        note = state.apply_settlement(city, day, settled)
        if note:
            notes.append(note)
            scored = (state.scored or {}).get(key) or {}
            append_case(
                {
                    "kind": "settlement",
                    "city": city.key,
                    "day": key,
                    "settled": settled,
                    "error": scored.get("error"),
                }
            )
    return notes


def format_report(result: IntlPrediction, state_line: str) -> str:
    sym = unit_symbol(result.unit)
    ladder = whole_brackets(
        PARIS.low if result.city_key == "paris" else NYC.low,
        PARIS.high if result.city_key == "paris" else NYC.high,
        result.unit,
    )
    printing = bracket_for(result.running_max, ladder).label
    forecast = "无今日预报" if result.forecast_max is None else f"{result.forecast_max:.0f}{sym}"
    lines = [
        f"结算站  {result.settlement_name} ({result.settlement_station})",
        f"观测    {result.observed_at}",
        f"已印最高  {result.running_max:.0f}{sym}（当前 {result.current:.0f}{sym}）  →  若现在截停落 {printing}",
        f"气候均值  {result.climate_mean:.0f}{sym}    预报最高  {forecast}",
        f"加热 {result.heating:.2f}   风 {result.wind:.2f}   "
        f"晴空剩余升幅 {result.remaining_clear_rise:.2f}{sym}   路径 {result.path_max:.2f}{sym}",
        f"公允模型预测  {result.mu:.2f}{sym}    标准差 {result.sigma:.2f}{sym}    最高概率  {result.mode.label}",
        state_line,
        "",
        f"{'档位':<22}{'公允':>8}{'距下沿':>10}",
    ]
    for row in result.fair:
        if row.probability < 0.005 and row.dist_c != 0.0:
            continue
        dist = "已过" if row.dist_c is None else ("已印" if row.dist_c == 0.0 else f"{row.dist_c:.0f}{sym}")
        lines.append(f"{row.label:<22}{row.probability * 100:7.1f}%{dist:>10}")
    return "\n".join(lines)
