"""Hand-checked CSCV cases: selection reversals, ties and missing observations."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from signal_validation.pbo import cscv_pbo
except ModuleNotFoundError as error:
    if error.name not in {"signal_validation", "signal_validation.pbo"}:
        raise
    cscv_pbo = None


class CSCVPBOTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(cscv_pbo), "CSCV PBO computation has not been implemented")

    def test_reversing_winner_has_pbo_one(self):
        result = cscv_pbo([[4, 0], [0, 4]], [[4, 4], [4, 4]], partitions=2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pbo"], 1.0)
        self.assertEqual(result["valid_splits"], 2)
        self.assertEqual(result["selection_counts"], [1, 1])

    def test_same_winner_in_both_halves_has_pbo_zero(self):
        result = cscv_pbo([[4, 0], [4, 0]], [[4, 4], [4, 4]], partitions=2)
        self.assertEqual(result["pbo"], 0.0)
        self.assertEqual(result["selection_counts"], [2, 0])

    def test_zero_counts_are_insufficient_instead_of_guessed_probability(self):
        result = cscv_pbo([[0, 0], [0, 0]], [[0, 0], [0, 0]], partitions=2)
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["pbo"])
        self.assertEqual(result["valid_splits"], 0)
        self.assertEqual(result["skipped_splits"], 2)
        self.assertEqual(result["selection_counts"], [0, 0])
        self.assertTrue(result["skip_reasons"])

    def test_is_tie_selects_first_and_os_tie_uses_average_rank(self):
        # Both IS candidates tie; selected candidate 0 ties all OS candidates.
        # The OS average rank is the median, which is included in the bad half.
        result = cscv_pbo([[2, 2], [2, 2]], [[4, 4], [4, 4]], partitions=2)
        self.assertEqual(result["selection_counts"], [2, 0])
        self.assertEqual(result["pbo"], 1.0)

    def test_zero_count_candidate_does_not_disappear_from_os_comparison(self):
        result = cscv_pbo([[4, 0], [0, 4]], [[4, 0], [4, 4]], partitions=2)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["valid_splits"], 0)
        self.assertEqual(result["skipped_splits"], 2)

    def test_rank_uses_all_four_candidates_with_top_ties(self):
        result = cscv_pbo([[3, 3, 1, 0]] * 2, [[4, 4, 4, 4]] * 2, partitions=2)
        self.assertEqual(result["selection_counts"], [2, 0, 0, 0])
        self.assertEqual(result["pbo"], 0.0)

    def test_counts_weight_is_selection_instead_of_mean_of_block_rates(self):
        result = cscv_pbo(
            [[1, 0], [0, 4], [4, 0], [4, 0]],
            [[1, 4], [100, 4], [4, 4], [4, 4]],
            partitions=4,
        )
        self.assertEqual(result["selection_counts"], [3, 3])
        self.assertEqual(result["pbo"], 1.0)

    def test_default_eight_partitions_enumerate_all_seventy_splits(self):
        wins = np.array([[3, 1]] * 8)
        counts = np.array([[4, 4]] * 8)
        original = wins.copy()
        result = cscv_pbo(wins, counts)
        self.assertEqual(result["total_splits"], 70)
        self.assertEqual(result["valid_splits"], 70)
        self.assertEqual(result["skipped_splits"], 0)
        self.assertEqual(result["selection_counts"], [70, 0])
        self.assertEqual(result["pbo"], 0.0)
        np.testing.assert_array_equal(wins, original)

    def test_invalid_shapes_and_counts_are_rejected(self):
        cases = [
            ([[1, 0]], [[1, 1]], 2),
            ([[1, 0], [1, 0]], [[1], [1]], 2),
            ([[1], [1]], [[1], [1]], 2),
            ([[2, 0], [1, 0]], [[1, 1], [1, 1]], 2),
            ([[-1, 0], [1, 0]], [[1, 1], [1, 1]], 2),
            ([[float("nan"), 0], [1, 0]], [[1, 1], [1, 1]], 2),
            ([[0.5, 0], [1, 0]], [[1, 1], [1, 1]], 2),
            ([[True, 0], [1, 0]], [[1, 1], [1, 1]], 2),
            ([[1, 0]] * 3, [[1, 1]] * 3, 3),
            ([[1, 0]] * 14, [[1, 1]] * 14, 14),
        ]
        for wins, counts, partitions in cases:
            with self.subTest(wins=wins, partitions=partitions):
                with self.assertRaises(ValueError):
                    cscv_pbo(wins, counts, partitions=partitions)


if __name__ == "__main__":
    unittest.main()
