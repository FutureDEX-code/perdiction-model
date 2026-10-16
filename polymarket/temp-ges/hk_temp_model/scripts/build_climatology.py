"""Rebuild data/hko_climatology.json from HKO open data.

The settlement series is CLMMAXT station=HKO, plus finalized Daily Extract
days that are not in CLMMAXT yet. HKA is downloaded only to measure the
September gap and is not written into the temperature prior.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "data" / "hko_climatology.json"


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "hk-temp-model/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _clmmaxt(station: str, year: int) -> list[tuple[int, int, int, float]]:
    url = (
        "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
        f"?dataType=CLMMAXT&station={station}&year={year}&rformat=json"
    )
    rows = []
    for year_text, month_text, day_text, value, completeness in _get(url)["data"]:
        if str(completeness).strip() != "C":
            continue
        rows.append((int(year_text), int(month_text), int(day_text), float(value)))
    return rows


def _percentile(sorted_vals: list[float], probability: float) -> float:
    index = (len(sorted_vals) - 1) * probability
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return sorted_vals[low]
    weight = index - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def main() -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        hko_years = list(pool.map(lambda year: _clmmaxt("HKO", year), range(1997, 2026)))
        hka_years = list(pool.map(lambda year: _clmmaxt("HKA", year), range(2015, 2026)))
    hko = [row for year in hko_years for row in year]
    hka = [row for year in hka_years for row in year]
    extract = _get("https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_202609.xml")
    for block in extract["stn"]["data"]:
        if int(block["month"]) != 9:
            continue
        for row in block["dayData"]:
            day_text = str(row[0]).strip()
            if day_text.isdigit():
                hko.append((2026, 9, int(day_text), float(str(row[2]).strip())))

    by_month_day: dict[str, list[float]] = defaultdict(list)
    for year, month, day, temp in hko:
        by_month_day[f"{month:02d}-{day:02d}"].append(temp)

    def base_date(month: int, day: int) -> date:
        return date(2020 if (month, day) == (2, 29) else 2021, month, day)

    summary: dict[str, dict] = {}
    for key in sorted(by_month_day):
        month, day = (int(part) for part in key.split("-"))
        base = base_date(month, day)
        pooled: list[float] = []
        for delta in range(-7, 8):
            neighbour = date.fromordinal(base.toordinal() + delta)
            pooled.extend(by_month_day.get(f"{neighbour.month:02d}-{neighbour.day:02d}", []))
        pooled.sort()
        same = sorted(by_month_day[key])
        summary[key] = {
            "n": len(pooled),
            "mean": round(statistics.fmean(pooled), 3),
            "std": round(statistics.pstdev(pooled), 3),
            "p10": round(_percentile(pooled, 0.10), 2),
            "p50": round(_percentile(pooled, 0.50), 2),
            "p90": round(_percentile(pooled, 0.90), 2),
            "same_day_n": len(same),
            "same_day_mean": round(statistics.fmean(same), 3),
            "same_day_std": round(statistics.pstdev(same), 3),
        }

    hko_index = {(year, month, day): temp for year, month, day, temp in hko}
    gaps = [
        temp - hko_index[(year, month, day)]
        for year, month, day, temp in hka
        if month == 9 and (year, month, day) in hko_index
    ]
    payload = {
        "settlement_station": "HKO",
        "series": "Daily Extract Absolute Daily Max at Hong Kong Observatory headquarters. Identical to CLMMAXT station=HKO.",
        "source_years": "1997-2025 CLMMAXT plus finalized 2026 Daily Extract days",
        "window_days": 7,
        "september_hka_minus_hko": {
            "years": "2015-2025",
            "n": len(gaps),
            "mean_c": round(statistics.fmean(gaps), 3),
            "std_c": round(statistics.pstdev(gaps), 3),
            "min_c": round(min(gaps), 2),
            "max_c": round(max(gaps), 2),
        },
        "by_month_day": summary,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload))
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
