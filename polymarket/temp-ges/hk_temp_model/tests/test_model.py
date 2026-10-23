import json
import unittest
from datetime import date, datetime
from pathlib import Path

from hk_temp_model.climatology import load_climatology
from hk_temp_model.learn import OnlineState
from hk_temp_model.cli import _trading_floor
from hk_temp_model.decide import no_board
from hk_temp_model.model import Adjustment, predict
from hk_temp_model.observations import Observation, forecast_max_for, observation_from_sources
from hk_temp_model.settlement import absolute_daily_maxes, bracket_for, default_brackets

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SettlementTests(unittest.TestCase):
    def test_one_decimal_lands_in_the_printed_degree(self):
        brackets = default_brackets()
        self.assertEqual(bracket_for(30.2, brackets).label, "30°C")
        self.assertEqual(bracket_for(31.0, brackets).label, "31°C")
        self.assertEqual(bracket_for(31.9, brackets).label, "31°C")
        self.assertEqual(bracket_for(32.0, brackets).label, "32°C")
        self.assertEqual(bracket_for(24.9, brackets).label, "24°C or below")
        self.assertEqual(bracket_for(25.0, brackets).label, "25°C")
        self.assertEqual(bracket_for(34.0, brackets).label, "34°C or higher")

    def test_daily_extract_column_is_absolute_daily_max(self):
        payload = json.loads((_FIXTURES / "extract_202609_day22.json").read_text())
        maxima = absolute_daily_maxes(payload, 2026)
        self.assertEqual(maxima[datetime(2026, 9, 22).date()], 30.2)
        self.assertEqual(bracket_for(maxima[datetime(2026, 9, 22).date()]).label, "30°C")

    def test_climatology_is_the_headquarters_series(self):
        climate = load_climatology()
        day = climate.for_date(datetime(2026, 9, 23).date())
        self.assertGreaterEqual(day.same_day_n, 20)
        self.assertGreater(day.mean_c, 29.0)
        self.assertLess(day.mean_c, 32.0)
        gap = climate.airport_gap
        self.assertGreater(gap.n, 300)
        self.assertAlmostEqual(gap.mean_c, 1.04, places=2)


class ObservationTests(unittest.TestCase):
    def test_parser_keeps_headquarters_and_airport_apart(self):
        text = (_FIXTURES / "readings_20260923_1200.txt").read_text()
        current = {"icon": [51], "uvindex": {"data": [{"place": "King's Park", "value": 7}]}}
        forecast = {
            "weatherForecast": [
                {"forecastDate": "20260924", "forecastMaxtemp": {"value": 32}},
                {"forecastDate": "20260923", "forecastMaxtemp": {"value": 31}},
            ]
        }
        obs = observation_from_sources(text, forecast, current)
        self.assertEqual(obs.hko_max_c, 30.1)
        self.assertEqual(obs.hka_max_c, 32.2)
        self.assertEqual(obs.direct_wm2, 28.0)
        self.assertEqual(obs.wind_from_deg, 135.0)
        self.assertEqual(obs.forecast_max_c, 31.0)
        self.assertEqual(obs.uv_index, 7.0)

    def test_dropped_today_keeps_the_morning_forecast(self):
        state = OnlineState()
        day = datetime(2026, 9, 24).date()
        self.assertEqual(state.hold_forecast(day, 32.0), 32.0)
        self.assertEqual(state.hold_forecast(day, None), 32.0)
        self.assertIsNone(state.hold_forecast(datetime(2026, 9, 25).date(), None))

    def test_tomorrow_forecast_is_not_todays_anchor(self):
        payload = {"weatherForecast": [{"forecastDate": "20260924", "forecastMaxtemp": {"value": 32}}]}
        self.assertIsNone(forecast_max_for(payload, datetime(2026, 9, 23).date()))


class PredictTests(unittest.TestCase):
    def test_airport_print_does_not_move_the_settlement_floor(self):
        text = (_FIXTURES / "readings_20260923_1200.txt").read_text()
        current = {"icon": [51], "uvindex": {"data": [{"place": "King's Park", "value": 7}]}}
        forecast = {"weatherForecast": [{"forecastDate": "20260923", "forecastMaxtemp": {"value": 31}}]}
        result = predict(observation_from_sources(text, forecast, current), load_climatology())
        self.assertEqual(result.settlement_station, "HKO")
        self.assertAlmostEqual(result.hko_max_c, 30.1)
        self.assertAlmostEqual(result.hka_max_c, 32.2)
        probabilities = {row.label: row.probability for row in result.fair}
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(probabilities["24°C or below"], 0.0)
        self.assertEqual(probabilities["29°C"], 0.0)
        self.assertGreater(probabilities["30°C"], probabilities["32°C"])
        self.assertGreater(probabilities["31°C"], probabilities["33°C"])
        self.assertLess(probabilities["32°C"], 0.25)
        printing = next(row for row in result.fair if row.already_printing)
        self.assertEqual(printing.label, "30°C")
        self.assertEqual(printing.dist_c, 0.0)

    def test_morning_forecast_of_32_is_not_buried_by_a_cold_path(self):
        obs = Observation(
            observed_at=datetime(2026, 9, 24, 8, 20),
            hko_temp_c=28.0,
            hko_max_c=28.2,
            hka_temp_c=30.0,
            hka_max_c=30.4,
            global_wm2=420.0,
            direct_wm2=80.0,
            diffuse_wm2=340.0,
            uv_index=4.0,
            wind_kmh=14.0,
            wind_from_deg=90.0,
            icon=51,
            forecast_max_c=32.0,
        )
        result = predict(obs, load_climatology())
        probabilities = {row.label: row.probability for row in result.fair}
        self.assertGreater(probabilities["32°C"], 0.25)
        self.assertGreater(probabilities["32°C"], probabilities["30°C"])

    def test_fallen_temperature_keeps_the_printed_degree(self):
        obs = Observation(
            observed_at=datetime(2026, 9, 23, 14, 50),
            hko_temp_c=31.2,
            hko_max_c=31.9,
            hka_temp_c=31.4,
            hka_max_c=33.0,
            global_wm2=458.0,
            direct_wm2=77.0,
            diffuse_wm2=391.0,
            uv_index=5.0,
            wind_kmh=12.0,
            wind_from_deg=90.0,
            icon=51,
            forecast_max_c=None,
        )
        result = predict(obs, load_climatology())
        probabilities = {row.label: row.probability for row in result.fair}
        self.assertEqual(result.settlement_station, "HKO")
        self.assertAlmostEqual(result.hko_max_c, 31.9)
        self.assertEqual(probabilities["30°C"], 0.0)
        self.assertGreater(probabilities["31°C"], 0.8)
        self.assertLess(probabilities["32°C"], 0.2)
        self.assertGreater(probabilities["31°C"], probabilities["32°C"])


class LearningTests(unittest.TestCase):
    def test_settlement_nudges_bias_once(self):
        state = OnlineState()
        state.forecasts["2026-09-22"] = [
            {
                "observed_at": "2026-09-22 16:00 HKT",
                "mu_c": 30.0,
                "sigma_c": 0.5,
                "hko_max_c": 29.8,
                "mode": "30°C",
            }
        ]
        note = state.apply_settlement(date(2026, 9, 22), 31.0)
        self.assertIsNotNone(note)
        self.assertAlmostEqual(state.bias_c, 0.2)
        self.assertAlmostEqual(state.sigma_mult, 1.08)
        self.assertIsNone(state.apply_settlement(date(2026, 9, 22), 31.0))
        self.assertIsNone(state.apply_settlement(date(2026, 9, 21), 30.2))

    def test_cold_morning_raises_forecast_trust(self):
        state = OnlineState()
        state.forecasts["2026-09-24"] = [
            {
                "observed_at": "2026-09-24 08:20 HKT",
                "mu_c": 30.7,
                "sigma_c": 1.0,
                "hko_max_c": 28.2,
                "mode": "30°C",
            },
            {
                "observed_at": "2026-09-24 16:00 HKT",
                "mu_c": 32.2,
                "sigma_c": 0.1,
                "hko_max_c": 32.2,
                "mode": "32°C",
            },
        ]
        state.apply_settlement(date(2026, 9, 24), 32.2)
        self.assertGreater(state.forecast_trust, 0.62)
        self.assertLess(state.forecast_trust, 0.9)

    def test_adjustment_cannot_pull_below_the_printed_max(self):
        text = (_FIXTURES / "readings_20260923_1200.txt").read_text()
        current = {"icon": [51], "uvindex": {"data": [{"place": "King's Park", "value": 7}]}}
        forecast = {"weatherForecast": [{"forecastDate": "20260923", "forecastMaxtemp": {"value": 31}}]}
        obs = observation_from_sources(text, forecast, current)
        result = predict(obs, load_climatology(), adjustment=Adjustment(bias_c=-5.0, sigma_mult=1.0))
        self.assertGreaterEqual(result.mu_c, obs.hko_max_c)


class _Fair:
    def __init__(self, lo, probability):
        self.lo_c = lo
        self.probability = probability


class PaperTests(unittest.TestCase):
    def test_five_dollar_no_wins_when_the_bracket_misses(self):
        from pathlib import Path
        import tempfile

        from hk_temp_model.paper import note_fills, settle, status_line

        board = [{"buy_no": True, "label": "25°C", "no_ask": 0.8, "reason": "到不了"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper.json"
            notes = note_fills("paris", date(2026, 9, 24), "2026-09-24 11:30 CEST", board, path)
            self.assertEqual(len(notes), 1)
            self.assertEqual(note_fills("paris", date(2026, 9, 24), "later", board, path), [])
            settled = settle("paris", "2026-09-24", 23.0, path)
            self.assertIn("+1.25", settled[0])
            self.assertIn("+1.25", status_line(path))
            self.assertEqual(settle("paris", "2026-09-24", 23.0, path), [])

    def test_no_loses_the_stake_when_the_bracket_hits(self):
        from pathlib import Path
        import tempfile

        from hk_temp_model.paper import note_fills, settle

        board = [{"buy_no": True, "label": "32°C", "no_ask": 0.5, "reason": "已过"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper.json"
            note_fills("hk", date(2026, 9, 24), "2026-09-24 10:00 HKT", board, path)
            settled = settle("hk", "2026-09-24", 32.2, path)
            self.assertIn("-5.00", settled[0])

    def test_yes_wins_when_the_bracket_hits(self):
        from pathlib import Path
        import tempfile

        from hk_temp_model.paper import note_fills, settle

        board = [{"buy_yes": True, "label": "32°C", "yes_ask": 0.4, "reason": "还可能"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper.json"
            notes = note_fills("hk", date(2026, 9, 24), "2026-09-24 08:20 HKT", board, path)
            self.assertIn("Yes", notes[0])
            self.assertEqual(note_fills("hk", date(2026, 9, 24), "later", board, path), [])
            second = note_fills("hk", date(2026, 9, 24), "later", [{"buy_yes": True, "label": "33°C", "yes_ask": 0.2, "reason": "还可能"}], path)
            self.assertIn("33°C", second[0])
            settled = settle("hk", "2026-09-24", 32.2, path)
            self.assertIn("+7.50", settled[0])
            self.assertIn("-5.00", settled[1])


class NoTests(unittest.TestCase):
    def test_printed_32_locks_and_fades_a_stale_31_bid(self):
        fair = [_Fair(31, 0.0), _Fair(32, 1.0), _Fair(33, 0.0)]
        rows = no_board(
            running=32.2,
            current=29.7,
            hour=17.5,
            peak=14.75,
            forecast=32.0,
            fair=fair,
            asks={"31°C": 0.20, "32°C": None, "33°C": 0.001},
            bids={"31°C": 0.15, "32°C": 0.998, "33°C": None},
        )
        by_label = {row["label"]: row for row in rows}
        alive = [row["label"] for row in rows if row["alive"]]
        self.assertEqual(alive, ["32°C"])
        self.assertTrue(by_label["31°C"]["buy_no"])
        self.assertFalse(by_label["32°C"]["buy_no"])
        self.assertFalse(by_label["33°C"]["buy_no"])
        self.assertTrue(by_label["31°C"]["yes_dead"])
        self.assertFalse(by_label["31°C"]["no_dead"])
        self.assertFalse(by_label["32°C"]["yes_dead"])
        self.assertTrue(by_label["32°C"]["no_dead"])

    def test_a_print_that_can_still_rise_does_not_kill_the_no(self):
        fair = [_Fair(30, 0.8), _Fair(31, 0.2)]
        rows = no_board(
            running=30.2,
            current=30.1,
            hour=12.0,
            peak=14.75,
            forecast=31.0,
            fair=fair,
            asks={"30°C": 0.6, "31°C": 0.4},
            bids={"30°C": 0.55, "31°C": 0.35},
        )
        by_label = {row["label"]: row for row in rows}
        self.assertFalse(by_label["30°C"]["yes_dead"])
        self.assertFalse(by_label["30°C"]["no_dead"])

    def test_selling_a_dead_yes_keeps_part_of_the_stake(self):
        from pathlib import Path
        import tempfile

        from hk_temp_model.paper import note_fills, record_exit

        board = [{"buy_yes": True, "label": "30°C", "yes_ask": 0.4, "reason": "还可能"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper.json"
            note_fills("hk", date(2026, 10, 15), "2026-10-15 10:00 HKT", board, path)
            note = record_exit("hk", date(2026, 10, 15), "30°C", "YES", 0.12, path)
            self.assertIn("-3.50", note)
            self.assertEqual(record_exit("hk", date(2026, 10, 15), "30°C", "YES", 0.12, path), "")

    def test_morning_forecast_still_leaves_the_next_degree_open(self):
        fair = [_Fair(31, 0.2), _Fair(32, 0.7), _Fair(33, 0.1)]
        rows = no_board(
            running=28.2,
            current=28.0,
            hour=8.5,
            peak=14.75,
            forecast=32.0,
            fair=fair,
            asks={"31°C": 0.4, "32°C": 0.4, "33°C": 0.2, "35°C": 0.05},
            bids={"31°C": 0.3, "32°C": 0.3, "33°C": 0.1, "35°C": 0.04},
        )
        alive = {row["label"] for row in rows if row["alive"]}
        self.assertIn("32°C", alive)
        self.assertIn("33°C", alive)
        self.assertNotIn("35°C", alive)

    def test_penny_yes_is_not_bought(self):
        fair = [_Fair(21, 0.34), _Fair(23, 0.2)]
        rows = no_board(
            running=17,
            current=17,
            hour=12,
            peak=15,
            forecast=23,
            fair=fair,
            asks={"21°C": 0.002, "23°C": 0.35},
            bids={"21°C": 0.001, "23°C": 0.3},
        )
        self.assertFalse(any(row["buy_yes"] for row in rows))

    def test_fahrenheit_overshoot_leaves_three_degrees_open(self):
        fair = [_Fair(65, 0.4), _Fair(68, 0.05)]
        rows = no_board(
            running=59,
            current=54,
            hour=6,
            peak=15,
            forecast=65,
            fair=fair,
            asks={"64-65°F": 0.18, "68-69°F": 0.23},
            bids={"64-65°F": 0.15, "68-69°F": 0.19},
            overshoot=3,
        )
        by_label = {row["label"]: row for row in rows}
        self.assertTrue(by_label["68-69°F"]["alive"])
        self.assertFalse(by_label["68-69°F"]["buy_no"])

    def test_yes_does_not_buy_the_cheap_degree_under_the_mode(self):
        fair = [_Fair(31, 0.24), _Fair(32, 0.46), _Fair(33, 0.24)]
        rows = no_board(
            running=28.0,
            current=28.0,
            hour=0.2,
            peak=14.75,
            forecast=32.0,
            fair=fair,
            asks={"31°C": 0.07, "32°C": 0.48, "33°C": 0.45},
            bids={"31°C": 0.06, "32°C": 0.45, "33°C": 0.40},
        )
        by_label = {row["label"]: row for row in rows}
        self.assertFalse(by_label["31°C"]["buy_yes"])
        self.assertFalse(by_label["32°C"]["buy_yes"])

    def test_yes_below_20_cents_is_not_bought(self):
        fair = [_Fair(68, 0.26), _Fair(69, 0.20), _Fair(70, 0.10)]
        rows = no_board(
            running=59,
            current=59,
            hour=3,
            peak=14.5,
            forecast=70,
            fair=fair,
            asks={"68-69°F": 0.12, "70-71°F": 0.41},
            bids={"68-69°F": 0.10, "70-71°F": 0.38},
            overshoot=3,
        )
        by_label = {row["label"]: row for row in rows}
        self.assertFalse(by_label["68-69°F"]["buy_yes"])
        self.assertEqual(by_label["68-69°F"]["yes_block"], "卖价低于0.35")
        self.assertFalse(by_label["70-71°F"]["buy_yes"])

    def test_mispriced_likely_bracket_is_still_bought(self):
        fair = [_Fair(68, 0.26), _Fair(69, 0.20), _Fair(70, 0.10)]
        rows = no_board(
            running=59,
            current=59,
            hour=3,
            peak=14.5,
            forecast=70,
            fair=fair,
            asks={"68-69°F": 0.40, "70-71°F": 0.41},
            bids={"68-69°F": 0.36, "70-71°F": 0.38},
            overshoot=3,
        )
        by_label = {row["label"]: row for row in rows}
        self.assertTrue(by_label["68-69°F"]["buy_yes"])
        self.assertFalse(by_label["70-71°F"]["buy_yes"])

    def test_unreachable_before_late_morning_is_not_bought(self):
        fair = [_Fair(26, 0.40), _Fair(28, 0.01)]
        common = dict(
            running=17,
            current=17,
            peak=14.5,
            forecast=26,
            fair=fair,
            asks={"28°C": 0.34},
            bids={"28°C": 0.31},
        )
        early = {row["label"]: row for row in no_board(hour=0, **common)}
        late = {row["label"]: row for row in no_board(hour=11.5, **common)}
        self.assertFalse(early["28°C"]["buy_no"])
        self.assertTrue(late["28°C"]["buy_no"])

    def test_hong_kong_still_buys_one_degree_under_the_forecast(self):
        fair = [_Fair(31, 0.55), _Fair(32, 0.30)]
        rows = no_board(
            running=29.4,
            current=29.0,
            hour=9,
            peak=14.75,
            forecast=32.0,
            fair=fair,
            asks={"31°C": 0.35, "32°C": 0.40},
            bids={"31°C": 0.32, "32°C": 0.36},
            city="hk",
            mu=31.2,
        )
        by_label = {row["label"]: row for row in rows}
        self.assertTrue(by_label["31°C"]["buy_yes"])

    def test_paris_does_not_buy_below_the_forecast_before_the_peak(self):
        fair = [_Fair(21, 0.46), _Fair(22, 0.30)]
        rows = no_board(
            running=20,
            current=20,
            hour=12.5,
            peak=14.5,
            forecast=22,
            fair=fair,
            asks={"21°C": 0.055, "22°C": 0.27},
            bids={"21°C": 0.04, "22°C": 0.24},
            city="paris",
            mu=21.6,
        )
        self.assertFalse(any(row["buy_yes"] for row in rows))
        self.assertEqual(rows[0]["yes_block"], "高峰前低于预报")

    def test_new_york_can_buy_a_later_yes_the_same_day(self):
        fair = [_Fair(72, 0.33), _Fair(73, 0.15)]
        rows = no_board(
            running=71,
            current=68,
            hour=2,
            peak=14.5,
            forecast=72,
            fair=fair,
            asks={"72-73°F": 0.40},
            bids={"72-73°F": 0.36},
            overshoot=3,
            city="nyc",
        )
        self.assertTrue(rows[0]["buy_yes"])

    def test_yes_is_not_added_in_the_four_hours_before_the_peak(self):
        fair = [_Fair(30, 0.55), _Fair(31, 0.30)]
        common = dict(
            running=29.0,
            current=29.0,
            peak=14.75,
            forecast=31.0,
            fair=fair,
            asks={"30°C": 0.40, "31°C": 0.45},
            bids={"30°C": 0.36, "31°C": 0.40},
            city="hk",
        )
        climbing = no_board(hour=12.5, **common)
        late = no_board(hour=14.2, **common)
        early = no_board(hour=6.0, **common)
        by_climb = {row["label"]: row for row in climbing}
        by_late = {row["label"]: row for row in late}
        by_early = {row["label"]: row for row in early}
        self.assertFalse(by_climb["30°C"]["buy_yes"])
        self.assertEqual(by_climb["30°C"]["yes_block"], "高峰前1到4小时不加买")
        self.assertTrue(by_late["30°C"]["buy_yes"])
        self.assertTrue(by_early["30°C"]["buy_yes"])

    def test_yes_under_35_cents_is_not_bought_even_in_the_last_hour(self):
        fair = [_Fair(30, 0.55), _Fair(31, 0.30)]
        common = dict(
            running=29.0,
            current=29.0,
            peak=14.75,
            forecast=31.0,
            fair=fair,
            bids={"30°C": 0.28, "31°C": 0.36},
            city="hk",
        )
        early = no_board(hour=6.0, asks={"30°C": 0.30, "31°C": 0.40}, **common)
        near = no_board(hour=14.2, asks={"30°C": 0.30, "31°C": 0.40}, **common)
        pricey = no_board(hour=6.0, asks={"30°C": 0.40, "31°C": 0.45}, **common)
        by_early = {row["label"]: row for row in early}
        by_near = {row["label"]: row for row in near}
        by_pricey = {row["label"]: row for row in pricey}
        self.assertFalse(by_early["30°C"]["buy_yes"])
        self.assertEqual(by_early["30°C"]["yes_block"], "卖价低于0.35")
        self.assertFalse(by_near["30°C"]["buy_yes"])
        self.assertEqual(by_near["30°C"]["yes_block"], "卖价低于0.35")
        self.assertTrue(by_pricey["30°C"]["buy_yes"])

    def test_first_hour_does_not_trade_yesterdays_maximum(self):
        running, trade = _trading_floor(32.2, 28.0, 0.0)
        self.assertEqual(running, 28.0)
        self.assertFalse(trade)
        running, trade = _trading_floor(32.4, 32.0, 15.0)
        self.assertEqual(running, 32.4)
        self.assertTrue(trade)


if __name__ == "__main__":
    unittest.main()
