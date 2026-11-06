"""CSCV of candidate win/count matrices, using pooled win rate as the metric.

Rows must already be ordered temporal partitions.  This module does not create
partitions, purge labels, enforce a minimum sample size, or establish independence.
CSCV recombines historical blocks and is a retrospective selection-risk diagnostic;
it is not chronological WFA and is not evidence of live predictive performance.
"""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb, isfinite
from numbers import Integral, Real

import numpy as np


def _count_matrix(values, name):
    """Normalize integer counts without silently truncating fractional inputs."""
    try:
        matrix = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular two-dimensional matrix") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a rectangular two-dimensional matrix")
    normalized = []
    for row in matrix.tolist():
        target = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must contain nonnegative integer counts")
            if not isinstance(value, Integral) and (not isfinite(value) or value != int(value)):
                raise ValueError(f"{name} must contain nonnegative integer counts")
            number = int(value)
            if number < 0:
                raise ValueError(f"{name} must contain nonnegative integer counts")
            target.append(number)
        normalized.append(target)
    return normalized, matrix.shape


def cscv_pbo(wins, counts, *, partitions=8) -> dict:
    """Enumerate every half/half CSCV split of ``partitions`` temporal blocks.

    ``wins`` and ``counts`` have shape (partitions, candidates).  For each split,
    IS selection maximizes sum(wins)/sum(counts), with first-column tie breaking.
    The same candidate's OS rate is ranked against *all* candidate OS rates.
    Rank 1 is worst; ties receive average ranks.  Ranks at or below (N+1)/2
    contribute to PBO.  Thus an all-tied OS comparison counts as a bad split.

    A split is skipped if any candidate has zero total observations in either
    half; the comparison universe is never silently reduced.  Invalid matrices
    raise ValueError.  With no valid splits, PBO is None with an explicit reason.
    Raw candidate counts are pooled, not averages of per-partition win rates.
    """
    if (isinstance(partitions, bool) or not isinstance(partitions, Integral)
            or partitions < 2 or partitions > 12 or partitions % 2):
        raise ValueError("partitions must be an even integer from 2 through 12")
    partitions = int(partitions)
    win_rows, shape = _count_matrix(wins, "wins")
    count_rows, count_shape = _count_matrix(counts, "counts")
    if shape != count_shape or shape[0] != partitions or shape[1] < 2:
        raise ValueError("wins/counts must have matching shape (partitions, at least two candidates)")
    candidates = shape[1]
    if any(w > n for wr, nr in zip(win_rows, count_rows) for w, n in zip(wr, nr)):
        raise ValueError("wins cannot exceed counts")

    selections = [0] * candidates
    skip_reasons = Counter()
    valid = skipped = bad = 0
    total_wins = [sum(row[j] for row in win_rows) for j in range(candidates)]
    total_counts = [sum(row[j] for row in count_rows) for j in range(candidates)]
    for train in combinations(range(partitions), partitions // 2):
        is_wins = [sum(win_rows[i][j] for i in train) for j in range(candidates)]
        is_counts = [sum(count_rows[i][j] for i in train) for j in range(candidates)]
        os_wins = [total_wins[j] - is_wins[j] for j in range(candidates)]
        os_counts = [total_counts[j] - is_counts[j] for j in range(candidates)]
        zero_is, zero_os = 0 in is_counts, 0 in os_counts
        if zero_is or zero_os:
            skipped += 1
            if zero_is:
                skip_reasons["zero_is_count"] += 1
            if zero_os:
                skip_reasons["zero_os_count"] += 1
            continue
        is_rates = [Fraction(w, n) for w, n in zip(is_wins, is_counts)]
        selected = max(range(candidates), key=is_rates.__getitem__)
        os_rates = [Fraction(w, n) for w, n in zip(os_wins, os_counts)]
        selected_rate = os_rates[selected]
        lower = sum(rate < selected_rate for rate in os_rates)
        tied = sum(rate == selected_rate for rate in os_rates)
        # Average rank is lower + (tied + 1)/2. Compare exactly to (N+1)/2.
        bad += int(2 * lower + tied <= candidates)
        selections[selected] += 1
        valid += 1

    return {
        "status": "ok" if valid else "insufficient",
        "reason": None if valid else "no_valid_cscv_splits",
        "pbo": bad / valid if valid else None,
        "bad_splits": bad,
        "valid_splits": valid,
        "skipped_splits": skipped,
        "total_splits": comb(partitions, partitions // 2),
        "partitions": partitions,
        "candidates": candidates,
        "selection_counts": selections,
        "skip_reasons": dict(skip_reasons),
        "metric": "pooled_win_rate",
        "os_rank_rule": "average ascending rank <= (candidates + 1) / 2",
        "is_tie_rule": "first candidate column",
        "statement": (
            "Retrospective CSCV selection-risk diagnostic on the supplied fixed candidate family; "
            "not chronological WFA, independent-sample proof, a future win probability, "
            "or evidence of live predictive performance."
        ),
    }
