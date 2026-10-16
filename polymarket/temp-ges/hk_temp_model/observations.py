"""Live HKO observations for the settlement station.

Temperature comes from the ``HK Observatory`` row. King's Park supplies
radiation, UV, and wind because the headquarters row has no radiometer.
Chek Lap Kok is parsed only so the report can show the gap.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

REGIONAL_URL = "https://www.weather.gov.hk/textonly/v2/forecast/text_readings_e.htm"
FORECAST_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en"
)
CURRENT_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
)

_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
_WIND_DIRS = {
    "Northeast": 45.0,
    "Northwest": 315.0,
    "Southeast": 135.0,
    "Southwest": 225.0,
    "North": 0.0,
    "East": 90.0,
    "South": 180.0,
    "West": 270.0,
}
_DIR_PATTERN = "|".join(sorted(_WIND_DIRS, key=len, reverse=True))
_TEMP_LINE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z' ]+?)\s+"
    r"(?P<temp>N/A|-?\d+\.\d)\s+"
    r"(?P<rh>N/A|\d+)\s+"
    r"(?P<tmax>-?\d+\.\d)\s*/\s*(?P<tmin>-?\d+\.\d)"
)
_WIND_LINE = re.compile(
    rf"^(?P<name>.+?)\s+(?P<dir>{_DIR_PATTERN})\s+(?P<spd>N/A|\d+)\s+(?P<gust>N/A|\d+)\s*$"
)
_SOLAR_LINE = re.compile(
    r"^(?P<name>King's Park|Kau Sai Chau)\s+(?P<global>\d+(?:\.\d+)?)\s+"
    r"(?P<direct>\d+(?:\.\d+)?)\s+(?P<diffuse>\d+(?:\.\d+)?)"
)
_STAMP = re.compile(
    r"Latest readings recorded at (\d{2}):(\d{2}) Hong Kong Time (\d{1,2}) ([A-Za-z]+) (\d{4})"
)


@dataclass(frozen=True)
class Observation:
    observed_at: datetime
    hko_temp_c: float | None
    hko_max_c: float
    hka_temp_c: float | None
    hka_max_c: float | None
    global_wm2: float | None
    direct_wm2: float | None
    diffuse_wm2: float | None
    uv_index: float | None
    wind_kmh: float | None
    wind_from_deg: float | None
    icon: int | None
    forecast_max_c: float | None

    @property
    def hour_hkt(self) -> float:
        return self.observed_at.hour + self.observed_at.minute / 60 + self.observed_at.second / 3600


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "hk-temp-model/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_json(url: str) -> dict:
    return json.loads(_fetch_text(url))


def _number(token: str) -> float | None:
    if token == "N/A":
        return None
    return float(token)


def parse_regional_readings(text: str) -> dict:
    stamp = _STAMP.search(text)
    if stamp is None:
        raise ValueError("regional readings have no Hong Kong timestamp")
    hour, minute, day, month_name, year = stamp.groups()
    observed_at = datetime(
        int(year),
        _MONTHS[month_name],
        int(day),
        int(hour),
        int(minute),
    )
    stations: dict[str, dict[str, float | None]] = {}
    winds: dict[str, tuple[float, float]] = {}
    solar: dict[str, tuple[float, float, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        temp = _TEMP_LINE.match(line)
        if temp:
            stations[temp.group("name")] = {
                "temp_c": _number(temp.group("temp")),
                "max_c": float(temp.group("tmax")),
                "min_c": float(temp.group("tmin")),
            }
            continue
        wind = _WIND_LINE.match(line)
        if wind and wind.group("spd") != "N/A":
            winds[wind.group("name").strip()] = (
                float(wind.group("spd")),
                _WIND_DIRS[wind.group("dir")],
            )
            continue
        sun = _SOLAR_LINE.match(line)
        if sun:
            solar[sun.group("name")] = (
                float(sun.group("global")),
                float(sun.group("direct")),
                float(sun.group("diffuse")),
            )
    if "HK Observatory" not in stations:
        raise ValueError("regional readings have no HK Observatory row")
    return {
        "observed_at": observed_at,
        "stations": stations,
        "winds": winds,
        "solar": solar,
    }


def forecast_max_for(payload: dict, day: date) -> float | None:
    """Use the nine-day forecast only when its date is the settlement date.

    After late morning the feed drops today and starts at tomorrow. Tomorrow's
    maximum must not be reused as today's anchor.
    """

    key = day.strftime("%Y%m%d")
    for row in payload.get("weatherForecast", []):
        if row.get("forecastDate") == key:
            return float(row["forecastMaxtemp"]["value"])
    return None


def _uv_and_icon(payload: dict) -> tuple[float | None, int | None]:
    icons = payload.get("icon") or []
    icon = int(icons[0]) if icons else None
    uv = None
    for row in (payload.get("uvindex") or {}).get("data") or []:
        if row.get("place") == "King's Park":
            uv = float(row["value"])
            break
    return uv, icon


def _wind(parsed: dict) -> tuple[float | None, float | None]:
    winds = parsed["winds"]
    for name in ("King's Park", "Star Ferry", "Kai Tak"):
        if name in winds:
            return winds[name]
    return None, None


def observation_from_sources(
    regional_text: str,
    forecast_payload: dict | None = None,
    current_payload: dict | None = None,
) -> Observation:
    parsed = parse_regional_readings(regional_text)
    hko = parsed["stations"]["HK Observatory"]
    if hko["max_c"] is None:
        raise ValueError("HK Observatory running maximum is missing")
    hka = parsed["stations"].get("Chek Lap Kok")
    sun = parsed["solar"].get("King's Park", (None, None, None))
    wind_kmh, wind_deg = _wind(parsed)
    uv = icon = None
    if current_payload is not None:
        uv, icon = _uv_and_icon(current_payload)
    forecast = None
    if forecast_payload is not None:
        forecast = forecast_max_for(forecast_payload, parsed["observed_at"].date())
    return Observation(
        observed_at=parsed["observed_at"],
        hko_temp_c=hko["temp_c"],
        hko_max_c=float(hko["max_c"]),
        hka_temp_c=None if hka is None else hka["temp_c"],
        hka_max_c=None if hka is None else hka["max_c"],
        global_wm2=sun[0],
        direct_wm2=sun[1],
        diffuse_wm2=sun[2],
        uv_index=uv,
        wind_kmh=wind_kmh,
        wind_from_deg=wind_deg,
        icon=icon,
        forecast_max_c=forecast,
    )


def fetch_observation() -> Observation:
    regional = _fetch_text(REGIONAL_URL)
    forecast = _fetch_json(FORECAST_URL)
    current = _fetch_json(CURRENT_URL)
    return observation_from_sources(regional, forecast, current)
