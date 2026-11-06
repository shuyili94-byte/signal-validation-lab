"""Behavioral contracts for strategy-agnostic signal validation."""

import csv
from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from signal_validation import research
except ImportError:
    research = None


def market(prices, signals, dates=None, eligible=None):
    dates = dates or [date(2020, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    if not isinstance(signals, dict):
        signals = {"signal": signals}
    return research.SignalData(tuple(dates), tuple(sorted(signals)), {
        "A": {"close": np.array(prices, dtype=float),
              "eligible": np.array(eligible if eligible is not None else [1] * len(prices)),
              "signals": {name: np.array(values, dtype=float) for name, values in signals.items()}}})


class SignalContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(research, "Generic signal validation is not implemented")

    # Catches using next day / next row instead of the third market session.
    def test_three_day_endpoint_and_strict_up_label(self):
        data = market([10, 11, 12, 10, 13, 15], [1, 0, 0, 0, 0, 0])
        result = research.evaluate(data, "signal", bootstrap_samples=0)
        first = result["signal_rows"][0]
        self.assertEqual((first["date"], first["label_end_date"]),
                         ("2020-01-01", "2020-01-04"))
        self.assertIs(first["win"], False)

    # A change to <2 or <=3 would accept t+2 or suppress t+3 incorrectly.
    def test_cooldown_allows_t_plus_three(self):
        data = market([10] * 9, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        result = research.evaluate(data, "signal", bootstrap_samples=0)
        self.assertEqual([r["date"] for r in result["signal_rows"]],
                         ["2020-01-01", "2020-01-04", "2020-01-07"])
        self.assertEqual(result["cooldown_suppressed_count"], 6)
        self.assertEqual(result["baseline_count"], 0)

    # Future outcomes must never decide whether an input signal is retained.
    def test_cooldown_does_not_depend_on_future_missing_prices(self):
        signals = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        original = research.evaluate(market([10] * 9, signals), "signal", bootstrap_samples=0)
        missing = research.evaluate(market([10, 10, 10, np.nan, 20, 30, 40, 50, 60], signals),
                                    "signal", bootstrap_samples=0)
        self.assertEqual([r["date"] for r in original["signal_rows"]],
                         [r["date"] for r in missing["signal_rows"]])
        self.assertEqual(missing["missing_outcome_count"], 2)

    # Missing input close or exact endpoint is not substituted with another day.
    def test_missing_close_is_counted_and_tail_does_not_cross_window(self):
        data = market([np.nan, 10, 10, 11, 12, 13, 14, 15], [1, 0, 0, 1, 0, 0, 0, 0])
        result = research.evaluate(data, "signal", end="2020-01-06", bootstrap_samples=0)
        self.assertEqual(result["signal_count"], 2)
        self.assertEqual(result["missing_outcome_count"], 1)
        self.assertEqual(result["unmatured_count"], 1)
        self.assertEqual(result["valid_count"], 0)

    # Unknown and cooldown-suppressed signals must not become baseline controls.
    def test_baseline_uses_only_known_zero_and_eligible_dates(self):
        data = market([10, 10, 10, 10, 11, 9, 12, 20],
                      [1, 0, np.nan, 0, 0, 0, 0, 0], eligible=[1, 1, 1, 0, 1, 1, 1, 1])
        result = research.evaluate(data, "signal", bootstrap_samples=0)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["baseline_count"], 2)
        self.assertEqual(result["baseline_rate"], 1.0)
        self.assertEqual(result["lift"], -1.0)


class CsvContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(research, "Generic signal validation is not implemented")

    def csv(self, header, rows):
        workspace = Path(__file__).resolve().parents[1] / ".test-work"
        workspace.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=workspace)
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    # An external calendar retains days absent from every symbol.
    def test_calendar_and_unknown_signal_are_preserved(self):
        path = self.csv(["date", "symbol", "close", "signal"], [
            ["2020-01-03", "A", 10, 1], ["2020-01-06", "A", 11, ""],
            ["2020-01-08", "A", 14, 0], ["2020-01-09", "A", 15, 0]])
        calendar = self.csv(["date"], [["2020-01-03"], ["2020-01-06"],
                                      ["2020-01-07"], ["2020-01-08"], ["2020-01-09"]])
        data = research.load_csv(path, calendar)
        self.assertTrue(np.isnan(data.series["A"]["signals"]["signal"][1]))
        self.assertTrue(np.isnan(data.series["A"]["close"][2]))
        first = research.evaluate(data, "signal", bootstrap_samples=0)["signal_rows"][0]
        self.assertEqual(first["label_end_date"], "2020-01-08")
        self.assertIs(first["win"], True)

    # Detects interpreting candidate names as embedded trading logic.
    def test_multiple_candidates_are_plain_external_markers(self):
        path = self.csv(["date", "symbol", "close", "signal_B", "signal_A"], [
            ["2020-01-01", "A", 10, 1, 0], ["2020-01-02", "A", "", "", 1]])
        data = research.load_csv(path)
        self.assertEqual(data.candidates, ("A", "B"))
        self.assertTrue(np.isnan(data.series["A"]["close"][1]))

    # Invalid markers and prices must not silently become zeros or missing rows.
    def test_bad_rows_and_outside_calendar_fail(self):
        header = ["date", "symbol", "close", "signal"]
        bad_rows = [
            [["2020-01-01", "A", 0, 1]],
            [["2020-01-01", "A", "inf", 1]],
            [["2020-01-01", "A", 10, 2]],
            [["2020-01-01", "A", 10, "yes"]],
            [["bad-date", "A", 10, 1]],
            [["2020-01-01", "A", 10, 1], ["2020-01-01", "A", 11, 0]],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                research.load_csv(self.csv(header, rows))
        with self.assertRaises(ValueError):
            research.load_csv(self.csv(header, [["2020-01-02", "A", 10, 1]]),
                              self.csv(["date"], [["2020-01-01"]]))


class ResearchContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(research, "Generic signal validation is not implemented")

    # Selection must pool outcomes, honor minimum evidence, and ignore future rows.
    def test_phase1_selects_observed_best_with_minimum_count(self):
        self.assertTrue(callable(getattr(research, "phase1", None)), "phase1 is not implemented")
        data = market([10, 10, 10, 12, 8, 10, 14, 10, 10, 15, 10, 20], {
            "A": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "B": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]})
        first = research.phase1(data, min_signals=2, bootstrap_samples=0)
        self.assertEqual(first["selected_candidate"], "A")
        self.assertEqual(first["selected"]["wins"], 2)
        self.assertEqual(first["selected"]["valid_count"], 2)
        self.assertEqual(research.phase1(data, min_signals=3, bootstrap_samples=0)["selected_candidate"], "B")
        self.assertIsNone(research.phase1(data, min_signals=4, bootstrap_samples=0)["selected_candidate"])
        frozen = research.phase1(data, end="2020-01-07", min_signals=1, bootstrap_samples=0)
        data.series["A"]["close"][6:] = [1, 800, 1, 800, 1, 800]
        changed = research.phase1(data, end="2020-01-07", min_signals=1, bootstrap_samples=0)
        self.assertEqual(frozen["selected_candidate"], changed["selected_candidate"])
        self.assertEqual(frozen["selected"]["wins"], changed["selected"]["wins"])

    # Sampling individual symbols instead of whole dates breaks this exact zero.
    def test_date_block_bootstrap_preserves_same_day_cross_section(self):
        function = getattr(research, "bootstrap_lift", None)
        self.assertTrue(callable(function), "date-block bootstrap is not implemented")
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
        signals, baselines = [], []
        for i, day in enumerate(days):
            row = {"date": day.isoformat(), "status": "valid", "win": i % 2 == 0}
            signals.extend([dict(row, symbol="A"), dict(row, symbol="B")])
            baselines.append(dict(row, symbol="C"))
        ci = function(signals, baselines, days, samples=200, block_days=3, seed=7)
        self.assertEqual(ci["status"], "computed")
        self.assertEqual((ci["ci_low"], ci["ci_high"]), (0.0, 0.0))
        absent = function(signals, [], days, samples=200, block_days=3, seed=7)
        self.assertIsNone(absent["ci_low"])
        self.assertEqual(absent["status"], "insufficient")

    # CSCV labels crossing a raw time-block boundary leak outcome information.
    def test_pbo_uses_only_labels_mature_inside_each_block(self):
        self.assertTrue(callable(getattr(research, "phase1", None)), "phase1 is not implemented")
        data = market(list(range(1, 33)), {"A": [1] * 32, "B": [1] * 32})
        result = research.phase1(data, min_signals=1, cooldown_sessions=0, bootstrap_samples=0)
        self.assertEqual(result["pbo"]["block_counts"], [[1, 1]] * 8)
        self.assertEqual(result["selected"]["valid_count"], 29)

    def walk_data(self):
        days, day = [], date(2020, 1, 1)
        while day < date(2022, 1, 1):
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        n = len(days)
        return market(list(range(100, 100 + n)), {
            "A": [int(i % 5 == 0) for i in range(n)],
            "B": [int(i % 5 == 1) for i in range(n)]}, dates=days)

    # Catches calendar-day gaps, label leakage, incomplete quarters and overlap.
    def test_walk_forward_purges_labels_and_uses_five_market_day_gap(self):
        self.assertTrue(callable(getattr(research, "walk_forward", None)), "WFA is not implemented")
        result = research.walk_forward(self.walk_data(), train_months=12, min_signals=10,
                                       bootstrap_samples=0)
        first = result["folds"][0]
        self.assertEqual(first["train_start"], "2020-01-01")
        self.assertEqual(first["train_end"], "2021-01-01")
        self.assertEqual(first["gap_dates"], ["2021-01-01", "2021-01-04", "2021-01-05",
                                              "2021-01-06", "2021-01-07"])
        self.assertEqual((first["oos_start"], first["oos_end"]), ("2021-01-08", "2021-04-08"))
        self.assertEqual(first["selected_candidate"], "A")
        for fold in result["folds"]:
            if fold["status"] == "evaluated":
                self.assertTrue(all(r["label_end_date"] < fold["train_end"]
                                    for r in fold["train"]["selected"]["signal_rows"] if r["status"] == "valid"))
                self.assertTrue(all(r["label_end_date"] < fold["oos_end"]
                                    for r in fold["oos"]["signal_rows"] if r["status"] == "valid"))
        for previous, following in zip(result["folds"], result["folds"][1:]):
            self.assertGreaterEqual(following["oos_start"], previous["oos_end"])
        self.assertEqual(result["aggregate"]["wins"], result["aggregate"]["valid_count"])
        self.assertGreater(result["aggregate"]["fold_count"], 0)

    # A lack of eligible training signals must not invent a selected model / OOS zero.
    def test_walk_forward_skips_unselectable_folds_and_rejects_overlap(self):
        self.assertTrue(callable(getattr(research, "walk_forward", None)), "WFA is not implemented")
        result = research.walk_forward(self.walk_data(), train_months=12,
                                       min_signals=10000, bootstrap_samples=0)
        self.assertTrue(all(f["status"] == "skipped" and f["oos"] is None for f in result["folds"]))
        self.assertIsNone(result["aggregate"]["win_rate"])
        self.assertEqual(result["aggregate"]["fold_count"], 0)
        with self.assertRaises(ValueError):
            research.walk_forward(self.walk_data(), oos_months=3, step_months=1, bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
