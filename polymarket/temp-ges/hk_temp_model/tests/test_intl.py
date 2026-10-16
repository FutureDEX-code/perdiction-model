import unittest
from datetime import date

from hk_temp_model.intl import (
    NYC,
    PARIS,
    IntlOnlineState,
    bracket_for,
    c_to_settlement,
    running_max_from_metars,
    whole_brackets,
)


class IntlSettlementTests(unittest.TestCase):
    def test_paris_whole_celsius_brackets(self):
        brackets = whole_brackets(14, 28, "C")
        self.assertEqual(bracket_for(22.0, brackets).label, "22°C")
        self.assertEqual(bracket_for(14.0, brackets).label, "14°C or below")
        self.assertEqual(bracket_for(28.0, brackets).label, "28°C or higher")

    def test_nyc_converts_celsius_metar_to_fahrenheit(self):
        self.assertEqual(c_to_settlement(20.6, "F"), 69.0)
        self.assertEqual(c_to_settlement(13.9, "F"), 57.0)
        brackets = whole_brackets(NYC.low, NYC.high, "F")
        self.assertEqual(bracket_for(69.0, brackets).label, "69°F")

    def test_city_specs(self):
        self.assertEqual(PARIS.icao, "LFPB")
        self.assertEqual(NYC.icao, "KLGA")

    def test_cold_morning_raises_forecast_trust(self):
        state = IntlOnlineState()
        state.forecasts["2026-10-15"] = [
            {"observed_at": "2026-10-15 09:00 CEST", "mu": 21.1, "sigma": 1.3, "running_max": 13.0, "mode": "21°C"},
            {"observed_at": "2026-10-15 16:00 CEST", "mu": 24.0, "sigma": 0.2, "running_max": 24.0, "mode": "24°C"},
        ]
        note = state.apply_settlement(PARIS, date(2026, 10, 15), 24.0)
        self.assertIn("预报信任", note)
        self.assertGreater(state.forecast_trust, 0.62)
        self.assertLessEqual(state.forecast_trust, 0.9)
        self.assertIsNone(state.apply_settlement(PARIS, date(2026, 10, 15), 24.0))

    def test_new_york_morning_does_not_raise_trust(self):
        state = IntlOnlineState()
        state.forecasts["2026-10-14"] = [
            {"observed_at": "2026-10-14 09:00 ET", "mu": 76.0, "sigma": 1.5, "running_max": 68.0, "mode": "76°F"},
            {"observed_at": "2026-10-14 16:00 ET", "mu": 79.0, "sigma": 0.2, "running_max": 79.0, "mode": "79°F"},
        ]
        state.apply_settlement(NYC, date(2026, 10, 14), 79.0)
        self.assertAlmostEqual(state.forecast_trust, 0.62)

    def test_hour_rounded_report_time_does_not_cross_midnight(self):
        metars = [
            {
                "temp": 20.0,
                "reportTime": "2026-10-15T04:00:00.000Z",
                "rawOb": "METAR KLGA 150351Z 31014G19KT 10SM 20/13",
            },
            {
                "temp": 18.9,
                "reportTime": "2026-10-15T05:00:00.000Z",
                "rawOb": "METAR KLGA 150451Z 33010KT 10SM 19/12",
            },
        ]
        running, latest, _at = running_max_from_metars(NYC, metars, date(2026, 10, 15))
        self.assertEqual(running, 66.0)
        self.assertEqual(latest, 66.0)


if __name__ == "__main__":
    unittest.main()
