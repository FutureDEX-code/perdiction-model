"""Calendar-day prior for the HKO headquarters daily maximum."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "hko_climatology.json"


@dataclass(frozen=True)
class DayClimate:
    month_day: str
    mean_c: float
    std_c: float
    same_day_n: int
    window_mean_c: float
    window_std_c: float
    p10_c: float
    p90_c: float


@dataclass(frozen=True)
class AirportGap:
    """September mean of Chek Lap Kok daily max minus HKO daily max."""

    years: str
    n: int
    mean_c: float
    std_c: float
    min_c: float
    max_c: float


@dataclass(frozen=True)
class Climatology:
    source_years: str
    airport_gap: AirportGap
    days: dict[str, DayClimate]

    def for_date(self, day: date) -> DayClimate:
        key = f"{day.month:02d}-{day.day:02d}"
        try:
            return self.days[key]
        except KeyError as exc:
            raise KeyError(f"no HKO climatology for {key}") from exc


def load_climatology(path: Path | None = None) -> Climatology:
    payload = json.loads((path or _DATA_PATH).read_text())
    gap = payload["september_hka_minus_hko"]
    days: dict[str, DayClimate] = {}
    for key, row in payload["by_month_day"].items():
        use_same_day = int(row["same_day_n"]) >= 10
        days[key] = DayClimate(
            month_day=key,
            mean_c=float(row["same_day_mean"] if use_same_day else row["mean"]),
            std_c=float(row["same_day_std"] if use_same_day else row["std"]),
            same_day_n=int(row["same_day_n"]),
            window_mean_c=float(row["mean"]),
            window_std_c=float(row["std"]),
            p10_c=float(row["p10"]),
            p90_c=float(row["p90"]),
        )
    return Climatology(
        source_years=str(payload["source_years"]),
        airport_gap=AirportGap(
            years=str(gap["years"]),
            n=int(gap["n"]),
            mean_c=float(gap["mean_c"]),
            std_c=float(gap["std_c"]),
            min_c=float(gap["min_c"]),
            max_c=float(gap["max_c"]),
        ),
        days=days,
    )
