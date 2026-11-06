"""Validate externally supplied binary signals without embedding a strategy."""

from bisect import bisect_left
from calendar import monthrange
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np


@dataclass
class SignalData:
    calendar: tuple
    candidates: tuple
    series: dict


def _date(value):
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _price(value):
    if value is None or not str(value).strip():
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("close must be a positive finite number or blank") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError("close must be a positive finite number or blank")
    return number


def _flag(value, column, allow_missing):
    value = (value or "").strip()
    if value == "" and allow_missing:
        return float("nan")
    if value not in {"0", "1"}:
        raise ValueError(f"{column} must be 0 or 1" + (" or blank" if allow_missing else ""))
    return int(value)


def load_csv(path, calendar_path=None):
    """Align to an explicit calendar or the union of all input dates, never fill.

    In union mode, days absent for every symbol are unknowable. Optional
    columns not used by the engine (such as benchmark_close) are not inferred
    as filters. Candidate identifiers carry no strategy semantics.
    """
    rows, seen = [], set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("CSV has duplicate column names")
        if not {"symbol", "date", "close"}.issubset(fields):
            raise ValueError("CSV needs date,symbol,close and signal columns")
        named = [field for field in fields if field.startswith("signal_")]
        if "signal" in fields and named:
            raise ValueError("Use signal or signal_NAME columns, not both")
        columns = {"signal": "signal"} if "signal" in fields else {
            field[7:]: field for field in named}
        if not columns or "" in columns:
            raise ValueError("CSV needs signal or nonempty signal_NAME columns")
        candidates = tuple(sorted(columns))
        for line, row in enumerate(reader, 2):
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                raise ValueError(f"Missing symbol at CSV line {line}")
            day = _date((row.get("date") or "").strip())
            if (symbol, day) in seen:
                raise ValueError(f"Duplicate symbol/date at CSV line {line}")
            seen.add((symbol, day))
            rows.append((symbol, day, _price(row.get("close")),
                         _flag(row.get("eligible"), "eligible", False) if "eligible" in fields else 1,
                         {name: _flag(row.get(column), column, True) for name, column in columns.items()}))
    if not rows:
        raise ValueError("CSV contains no signal rows")
    if calendar_path is None:
        days = tuple(sorted({row[1] for row in rows}))
    else:
        with Path(calendar_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "date" not in (reader.fieldnames or []):
                raise ValueError("Calendar CSV needs date column")
            observed = [_date((row.get("date") or "").strip()) for row in reader]
        if not observed or len(observed) != len(set(observed)):
            raise ValueError("Calendar must be nonempty with unique dates")
        days = tuple(sorted(observed))
        if any(row[1] not in set(days) for row in rows):
            raise ValueError("Signal date is outside the supplied calendar")
    positions = {day: index for index, day in enumerate(days)}
    series = {symbol: {"close": np.full(len(days), np.nan),
                       "eligible": np.full(len(days), np.nan),
                       "signals": {name: np.full(len(days), np.nan) for name in candidates}}
              for symbol in sorted({row[0] for row in rows})}
    for symbol, day, close, eligible, signals in rows:
        index, values = positions[day], series[symbol]
        values["close"][index], values["eligible"][index] = close, eligible
        for candidate, signal in signals.items():
            values["signals"][candidate][index] = signal
    return SignalData(days, candidates, series)


def _bounds(data, start, end):
    if not data.calendar:
        raise ValueError("Market calendar is empty")
    start = _date(start) if start is not None else data.calendar[0]
    end = _date(end) if end is not None else data.calendar[-1] + timedelta(days=1)
    if start >= end:
        raise ValueError("start must be before exclusive end")
    return start, end, bisect_left(data.calendar, start), bisect_left(data.calendar, end)


def _event(data, symbol, candidate, index, end_index):
    endpoint = index + 3
    close = data.series[symbol]["close"][index]
    row = {"symbol": symbol, "candidate": candidate, "date": data.calendar[index].isoformat(),
           "label_end_date": data.calendar[endpoint].isoformat() if endpoint < len(data.calendar) else None,
           "close": float(close) if np.isfinite(close) else None, "end_close": None, "win": None}
    if endpoint >= end_index:
        row["status"] = "unmatured"
    elif not np.isfinite(close) or not np.isfinite(data.series[symbol]["close"][endpoint]):
        row["status"] = "missing_outcome"
    else:
        row["status"] = "valid"
        row["end_close"] = float(data.series[symbol]["close"][endpoint])
        row["win"] = bool(row["end_close"] > close)
    return row


def _summary(signal_rows, baseline_rows, calendar, bootstrap_samples, block_days, seed):
    valid = [row for row in signal_rows if row["status"] == "valid"]
    baseline = [row for row in baseline_rows if row["status"] == "valid"]
    wins, baseline_wins = sum(row["win"] for row in valid), sum(row["win"] for row in baseline)
    rate = wins / len(valid) if valid else None
    baseline_rate = baseline_wins / len(baseline) if baseline else None
    return {"signal_count": len(signal_rows), "valid_count": len(valid), "wins": wins,
            "win_rate": rate, "missing_outcome_count": sum(
                row["status"] == "missing_outcome" for row in signal_rows),
            "unmatured_count": sum(row["status"] == "unmatured" for row in signal_rows),
            "baseline_count": len(baseline), "baseline_wins": baseline_wins,
            "baseline_rate": baseline_rate,
            "lift": rate - baseline_rate if rate is not None and baseline_rate is not None else None,
            "bootstrap": bootstrap_lift(signal_rows, baseline_rows, calendar,
                                        bootstrap_samples, block_days, seed),
            "signal_rows": signal_rows, "baseline_rows": baseline_rows}


def evaluate(data, candidate, start=None, end=None, cooldown_sessions=3,
             bootstrap_samples=500, block_days=5, seed=42):
    """Count external markers and strictly mature outcomes inside [start,end).

    Cooldown resets for each symbol and evaluation window. Labels do not
    influence event acceptance. Baselines include only original known zeros.
    The engine cannot verify that externally supplied signals were causal.
    """
    if candidate not in data.candidates:
        raise ValueError(f"Unknown candidate: {candidate}")
    if isinstance(cooldown_sessions, bool) or not isinstance(cooldown_sessions, int) or cooldown_sessions < 0:
        raise ValueError("cooldown_sessions must be a nonnegative integer")
    start, end, left, right = _bounds(data, start, end)
    signals, baselines = [], []
    counts = {"cooldown_suppressed_count": 0, "unknown_signal_count": 0,
              "ineligible_count": 0, "raw_signal_count": 0}
    for symbol, values in data.series.items():
        last_signal = left - cooldown_sessions
        for index in range(left, right):
            if values["eligible"][index] != 1:
                counts["ineligible_count"] += 1
                continue
            marker = values["signals"][candidate][index]
            if not np.isfinite(marker):
                counts["unknown_signal_count"] += 1
            elif marker == 1:
                counts["raw_signal_count"] += 1
                if index - last_signal < cooldown_sessions:
                    counts["cooldown_suppressed_count"] += 1
                    continue
                last_signal = index
                signals.append(_event(data, symbol, candidate, index, right))
            elif marker == 0:
                baselines.append(_event(data, symbol, candidate, index, right))
            else:
                raise ValueError("Signals must be 0, 1 or missing")
    result = _summary(signals, baselines, data.calendar[left:right], bootstrap_samples, block_days, seed)
    result.update(counts)
    result.update(candidate=candidate, start=start.isoformat(), end=end.isoformat(),
                  cooldown_sessions=cooldown_sessions)
    return result


def bootstrap_lift(signal_rows, baseline_rows, calendar, samples=500, block_days=5, seed=42):
    """Circular date-block resampling keeps every same-date symbol together.

    A descriptive percentile interval of pooled signal minus baseline win rate;
    it neither corrects strategy selection nor proves causal signal quality.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("bootstrap samples must be a nonnegative integer")
    if isinstance(block_days, bool) or not isinstance(block_days, int) or block_days < 1:
        raise ValueError("block_days must be a positive integer")
    result = {"status": "disabled" if samples == 0 else "insufficient",
              "ci_low": None, "ci_high": None, "samples": samples,
              "valid_replicates": 0, "block_days": block_days, "seed": seed,
              "method": "circular_date_block_percentile_95",
              "statement": "Descriptive only; no correction for repeated strategy selection."}
    if samples == 0:
        return result
    days = [_date(day).isoformat() for day in calendar]
    n = len(days)
    positions = {day: i for i, day in enumerate(days)}
    if n < 2 * block_days or samples < 20:
        result["reason"] = "need_at_least_two_blocks_and_twenty_replicates"
        return result
    daily = np.zeros((n, 4), dtype=float)
    for rows, win_column, count_column in ((signal_rows, 0, 1), (baseline_rows, 2, 3)):
        for row in rows:
            if row["status"] == "valid":
                index = positions[row["date"]]
                daily[index, win_column] += int(row["win"])
                daily[index, count_column] += 1
    if daily[:, 1].sum() < 2 or daily[:, 3].sum() < 2:
        result["reason"] = "need_two_observations_in_each_comparison_group"
        return result
    rng, lifts = np.random.default_rng(seed), []
    offsets = np.arange(block_days)
    blocks = (n + block_days - 1) // block_days
    for _ in range(samples):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        signal_wins, signal_count, baseline_wins, baseline_count = daily[indices].sum(axis=0)
        if signal_count and baseline_count:
            lifts.append(signal_wins / signal_count - baseline_wins / baseline_count)
    result["valid_replicates"] = len(lifts)
    if len(lifts) < max(20, samples // 2):
        result["reason"] = "too_few_replicates_contain_both_groups"
        return result
    low, high = np.quantile(lifts, [0.025, 0.975])
    result.update(status="computed", ci_low=float(low), ci_high=float(high), reason=None)
    return result


def _pbo(data, results, start, end):
    from .pbo import cscv_pbo

    _, _, left, right = _bounds(data, start, end)
    if len(results) < 2 or right - left < 32:
        return {"status": "insufficient", "pbo": None,
                "reason": "need_two_candidates_and_eight_blocks_longer_than_three_sessions"}
    blocks = np.array_split(np.arange(left, right), 8)
    counts, wins, boundaries = [], [], []
    for indices in blocks:
        first, last = int(indices[0]), int(indices[-1])
        block_start = data.calendar[first].isoformat()
        block_end = (data.calendar[last + 1].isoformat() if last + 1 < right else _date(end).isoformat())
        row_counts, row_wins = [], []
        for result in results:
            mature = [row for row in result["signal_rows"] if row["status"] == "valid"
                      and block_start <= row["date"] < block_end and row["label_end_date"] < block_end]
            row_counts.append(len(mature))
            row_wins.append(sum(row["win"] for row in mature))
        counts.append(row_counts)
        wins.append(row_wins)
        boundaries.append({"start": block_start, "end": block_end})
    diagnostic = cscv_pbo(wins, counts, partitions=8)
    diagnostic.update(block_counts=counts, block_wins=wins, block_boundaries=boundaries,
                      candidate_names=[result["candidate"] for result in results])
    return diagnostic


def phase1(data, start=None, end=None, cooldown_sessions=3, bootstrap_samples=500,
           block_days=5, seed=42, min_signals=10, candidates=None):
    """Select highest pooled training win rate, breaking ties by candidate name.

    This is exploratory selection, never an out-of-sample acceptance verdict.
    """
    if isinstance(min_signals, bool) or not isinstance(min_signals, int) or min_signals < 1:
        raise ValueError("min_signals must be a positive integer")
    candidate_names = sorted(data.candidates if candidates is None else candidates)
    if not candidate_names or len(candidate_names) != len(set(candidate_names)):
        raise ValueError("Need nonempty unique candidates")
    start, end, _, _ = _bounds(data, start, end)
    results = [evaluate(data, candidate, start, end, cooldown_sessions, bootstrap_samples,
                        block_days, seed) for candidate in candidate_names]
    eligible = [result for result in results if result["valid_count"] >= min_signals]
    selected = min(eligible, key=lambda result: (-result["win_rate"], result["candidate"])) if eligible else None
    return {"status": "selected" if selected else "insufficient",
            "start": start.isoformat(), "end": end.isoformat(), "min_signals": min_signals,
            "selection_rule": "highest_pooled_win_rate_then_candidate_name",
            "candidates": results, "selected_candidate": selected["candidate"] if selected else None,
            "selected": selected, "pbo": _pbo(data, results, start, end),
            "statement": "Exploratory in-sample selection, not proof of predictive performance."}


def _add_months(day, months):
    serial = day.year * 12 + day.month - 1 + months
    year, month = divmod(serial, 12)
    month += 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def walk_forward(data, candidates=None, train_months=18, oos_months=3, step_months=3,
                 gap_days=5, start=None, end=None, min_signals=10,
                 cooldown_sessions=3, bootstrap_samples=500, block_days=5, seed=42):
    """Chronological rolling selection with mature IS and OOS labels.

    Calendar months schedule the windows; gap_days counts the supplied market
    sessions. Only complete OOS windows are included. Training failures produce
    explicit skipped folds, not invented zero-return periods. No OOS score is
    used for candidate selection or dropping a completed fold.
    """
    for name, value in (("train_months", train_months), ("oos_months", oos_months),
                        ("step_months", step_months)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(gap_days, bool) or not isinstance(gap_days, int) or gap_days < 0:
        raise ValueError("gap_days must be a nonnegative integer")
    if step_months < oos_months:
        raise ValueError("Overlapping OOS windows are not allowed: step_months < oos_months")
    start, end, _, _ = _bounds(data, start, end)
    if start < data.calendar[0]:
        raise ValueError("Training start precedes supplied market history")
    candidate_names = tuple(sorted(data.candidates if candidates is None else candidates))
    if not candidate_names or len(candidate_names) != len(set(candidate_names)) or not set(candidate_names).issubset(data.candidates):
        raise ValueError("Need nonempty unique known candidates")
    available_end = min(end, data.calendar[-1] + timedelta(days=1))
    first_index = bisect_left(data.calendar, _add_months(start, train_months)) + gap_days
    folds, signal_rows, baseline_rows, evaluated_calendar = [], [], [], []
    if first_index < len(data.calendar):
        anchor = data.calendar[first_index]
        number = 0
        while True:
            oos_start = _add_months(anchor, number * step_months)
            oos_end = _add_months(anchor, number * step_months + oos_months)
            if oos_end > available_end:
                break
            oos_index = bisect_left(data.calendar, oos_start)
            train_end = data.calendar[oos_index - gap_days] if gap_days else oos_start
            train_start = _add_months(train_end, -train_months)
            gap = [day.isoformat() for day in data.calendar[oos_index - gap_days:oos_index]]
            fold = {"fold": number + 1, "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(), "oos_start": oos_start.isoformat(),
                    "oos_end": oos_end.isoformat(), "gap_dates": gap,
                    "selected_candidate": None, "train": None, "oos": None}
            if train_start < start:
                fold.update(status="skipped", reason="insufficient_training_history")
            else:
                train = phase1(data, train_start, train_end, cooldown_sessions, 0,
                               block_days, seed, min_signals, candidate_names)
                fold.update(train=train, selected_candidate=train["selected_candidate"])
                if train["selected_candidate"] is None:
                    fold.update(status="skipped", reason="insufficient_training_signals")
                else:
                    oos = evaluate(data, train["selected_candidate"], oos_start, oos_end,
                                   cooldown_sessions, 0, block_days, seed)
                    fold.update(status="evaluated", reason=None, oos=oos)
                    signal_rows.extend(oos["signal_rows"])
                    baseline_rows.extend(oos["baseline_rows"])
                    left, right = bisect_left(data.calendar, oos_start), bisect_left(data.calendar, oos_end)
                    evaluated_calendar.extend(data.calendar[left:right])
            folds.append(fold)
            number += 1
    aggregate = _summary(signal_rows, baseline_rows, evaluated_calendar,
                         bootstrap_samples, block_days, seed)
    aggregate.update(fold_count=sum(fold["status"] == "evaluated" for fold in folds),
                     skipped_fold_count=sum(fold["status"] == "skipped" for fold in folds))
    return {"status": "evaluated" if aggregate["fold_count"] else "insufficient",
            "start": start.isoformat(), "end": end.isoformat(), "candidates": list(candidate_names),
            "train_months": train_months, "oos_months": oos_months, "step_months": step_months,
            "gap_days": gap_days, "folds": folds, "aggregate": aggregate,
            "statement": "Pooled nonoverlapping OOS outcomes; descriptive validation, not trading P&L."}
