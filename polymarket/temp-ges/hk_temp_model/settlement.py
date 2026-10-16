"""Polymarket Hong Kong daily-high settlement.

The contract resolves to the bracket that contains the Hong Kong Observatory
Daily Extract value ``Absolute Daily Max (deg. C)``, to one decimal place.
That column is the headquarters station in Tsim Sha Tsui. It is the same
series as open-data ``CLMMAXT`` with ``station=HKO``.

August 2026 was checked day by day: extract column 2 and CLMMAXT HKO matched
on all 31 days. Chek Lap Kok (``HKA``) is a different station. In September
2015–2025 it ran about +1.0°C warmer than HKO, and on some days more than +3°C.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

SETTLEMENT_STATION = "HKO"
SETTLEMENT_NAME = "Hong Kong Observatory headquarters"
AIRPORT_STATION = "HKA"
AIRPORT_NAME = "Chek Lap Kok"

# Daily Extract row: day, mean pressure, absolute daily max, mean temp, ...
ABSOLUTE_DAILY_MAX_COLUMN = 2

DAILY_EXTRACT_URL = "https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{yyyymm}.xml"
CLMMAXT_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    "?dataType=CLMMAXT&station=HKO&year={year}&rformat=json"
)


@dataclass(frozen=True)
class Bracket:
    """Half-open temperature interval [lo, hi). ``hi is None`` means no upper bound."""

    label: str
    lo: float
    hi: float | None

    def contains(self, temp_c: float) -> bool:
        if temp_c < self.lo:
            return False
        if self.hi is None:
            return True
        return temp_c < self.hi


def default_brackets(low: int = 24, high: int = 34) -> tuple[Bracket, ...]:
    """Polymarket ladder: ``{low}°C or below``, each integer, ``{high}°C or higher``.

    An interior label ``n°C`` contains ``n.0`` up to but not including ``(n+1).0``.
    30.2°C therefore settles in 30°C, and 31.0°C settles in 31°C.
    """

    if high <= low:
        raise ValueError("high bracket must be above the low tail")
    brackets = [Bracket(f"{low}°C or below", float("-inf"), float(low + 1))]
    for degree in range(low + 1, high):
        brackets.append(Bracket(f"{degree}°C", float(degree), float(degree + 1)))
    brackets.append(Bracket(f"{high}°C or higher", float(high), None))
    return tuple(brackets)


def bracket_for(temp_c: float, brackets: tuple[Bracket, ...] | None = None) -> Bracket:
    ladder = brackets or default_brackets()
    for bracket in ladder:
        if bracket.contains(temp_c):
            return bracket
    raise ValueError(f"no bracket contains {temp_c}")


def absolute_daily_maxes(payload: dict, year: int) -> dict[date, float]:
    """Read Absolute Daily Max from a Daily Extract JSON payload."""

    found: dict[date, float] = {}
    for block in payload["stn"]["data"]:
        month = int(block["month"])
        for row in block["dayData"]:
            day_text = str(row[0]).strip()
            if not day_text.isdigit():
                continue
            if len(row) <= ABSOLUTE_DAILY_MAX_COLUMN:
                continue
            value = str(row[ABSOLUTE_DAILY_MAX_COLUMN]).strip()
            if not value or value in {"***", "N/A"}:
                continue
            found[date(year, month, int(day_text))] = float(value)
    return found
