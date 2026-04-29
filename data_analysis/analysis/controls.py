from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_FORECAST_PCT = 0.05
MAX_FORECAST_PCT = 0.5
MIN_CONTAMINATION = 0.001
MAX_CONTAMINATION = 0.2


@dataclass(frozen=True, slots=True)
class DataRangeSelection:
    """Resolved user data-range control values."""

    requested: float
    ratio: float
    rows: int


def _to_finite_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float(default)
    except Exception:
        return float(default)


def clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    """Parse and clamp a numeric user-control value."""
    parsed = _to_finite_float(value, default)
    return float(max(float(min_value), min(float(max_value), parsed)))


def parse_forecast_pct(value: Any, default: float = DEFAULT_FORECAST_PCT) -> float:
    """Parse forecast share as a clamped [0, 0.5] float."""
    return clamp_float(value, default, 0.0, MAX_FORECAST_PCT)


def parse_contamination(value: Any, default: float = 0.02) -> float:
    """Parse IsolationForest contamination as a clamped safe float."""
    return clamp_float(value, default, MIN_CONTAMINATION, MAX_CONTAMINATION)


def forecast_steps_for_history(history_rows: int, forecast_pct: float) -> int:
    """Map a visual forecast share to a concrete forecast horizon."""
    try:
        rows = int(history_rows)
    except Exception:
        rows = 0
    pct = parse_forecast_pct(forecast_pct)
    if pct <= 0 or rows <= 0:
        return 0
    pct_den = max(1e-9, 1.0 - float(pct))
    return max(1, int(math.floor(float(rows) * float(pct) / pct_den)))


def resolve_data_range_selection(raw_value: Any, total_rows: int) -> DataRangeSelection:
    """Resolve range control into cache key value, display ratio, and tail row count."""
    try:
        row_count = max(0, int(total_rows))
    except Exception:
        row_count = 0

    requested = _to_finite_float(raw_value, 1.0)
    if requested <= 0:
        return DataRangeSelection(requested=1.0, ratio=1.0, rows=0)

    if requested <= 1.0:
        if row_count <= 0:
            return DataRangeSelection(requested=requested, ratio=1.0, rows=0)
        rows = max(1, min(int(math.ceil(row_count * requested)), row_count))
        if rows >= row_count:
            return DataRangeSelection(requested=requested, ratio=1.0, rows=0)
        return DataRangeSelection(requested=requested, ratio=float(requested), rows=rows)

    if row_count <= 0:
        return DataRangeSelection(requested=requested, ratio=1.0, rows=0)

    rows = min(int(requested), row_count)
    if rows <= 0 or rows >= row_count:
        return DataRangeSelection(requested=requested, ratio=1.0, rows=0)
    return DataRangeSelection(requested=requested, ratio=float(rows) / float(row_count), rows=rows)


__all__ = [
    "DataRangeSelection",
    "DEFAULT_FORECAST_PCT",
    "MAX_FORECAST_PCT",
    "MIN_CONTAMINATION",
    "MAX_CONTAMINATION",
    "clamp_float",
    "forecast_steps_for_history",
    "parse_contamination",
    "parse_forecast_pct",
    "resolve_data_range_selection",
]
