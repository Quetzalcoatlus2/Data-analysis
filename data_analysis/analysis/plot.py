from __future__ import annotations

# ruff: noqa: F821
import base64
import contextlib
import io
import math
import textwrap
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, Literal, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_numeric_trailing_zeros(raw: str) -> str:
    """Remove insignificant trailing zeros from a decimal string."""
    if "." not in raw:
        return raw
    return raw.rstrip("0").rstrip(".")


def _format_precise_axis_value(value: float, decimals: int | None = None) -> str:
    """Format a numeric value without compact suffixes."""
    try:
        numeric = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(numeric):
        return str(value)

    if decimals is None:
        decimals = 0 if abs(numeric) >= 100 else 1 if abs(numeric) >= 10 else 2 if abs(numeric) >= 1 else 3

    return _strip_numeric_trailing_zeros(f"{numeric:.{max(0, decimals)}f}")


def _format_stat_value(v: float) -> str:
    """Return a compact string for a statistic value.

    Uses ``M/B/T`` suffixes only when there is meaningful space savings and
    falls back to precise numeric labels below one million.
    """
    try:
        value = float(v)
        mag = abs(value)
        if mag >= 1e15:                       # > 999.999 T
            return f"{value:.3e}"
        if mag >= 1e12:
            raw = f"{value / 1e12:.3f}"
            return _strip_numeric_trailing_zeros(raw) + "T"
        if mag >= 1e9:
            raw = f"{value / 1e9:.3f}"
            return _strip_numeric_trailing_zeros(raw) + "B"
        if mag >= 1e6:
            raw = f"{value / 1e6:.3f}"
            return _strip_numeric_trailing_zeros(raw) + "M"
        if mag >= 1e3:
            raw = f"{value / 1e3:.2f}"
            return _strip_numeric_trailing_zeros(raw) + "k"
        return _format_precise_axis_value(value)
    except Exception:
        return str(v)


def _apply_sci_formatter(
    ax,
    *,
    y_threshold: float = 1e6,
    x_threshold: float = 1e6,
) -> None:
    """Apply compact k/M/B/T axis formatter when values exceed thresholds.

    Inspects both x-axis and y-axis limits and applies the formatter to
    each axis independently when needed.
    """
    _fmt = mticker.FuncFormatter(lambda val, _pos: _format_stat_value(float(val)))

    try:
        y_limit = float(y_threshold)
        if not math.isfinite(y_limit) or y_limit <= 0:
            raise ValueError
    except Exception:
        y_limit = 1e6

    try:
        x_limit = float(x_threshold)
        if not math.isfinite(x_limit) or x_limit <= 0:
            raise ValueError
    except Exception:
        x_limit = 1e6

    def _has_explicit_tick_labels(axis) -> bool:
        try:
            labels = [str(tick.get_text() or '').strip() for tick in axis.get_ticklabels()]
            has_text = any(labels)
            return has_text and isinstance(axis.get_major_locator(), mticker.FixedLocator)
        except Exception:
            return False

    try:
        y_formatter = ax.yaxis.get_major_formatter()
        if isinstance(y_formatter, mticker.FixedFormatter) or _has_explicit_tick_labels(ax.yaxis):
            raise RuntimeError("skip-fixed-y-formatter")
        ymin, ymax = ax.get_ylim()
        if max(abs(ymin), abs(ymax)) >= y_limit:
            ax.yaxis.set_major_formatter(_fmt)
    except Exception:
        pass
    try:
        x_formatter = ax.xaxis.get_major_formatter()
        if isinstance(x_formatter, mticker.FixedFormatter) or _has_explicit_tick_labels(ax.xaxis):
            raise RuntimeError("skip-fixed-x-formatter")
        xmin, xmax = ax.get_xlim()
        if max(abs(xmin), abs(xmax)) >= x_limit:
            ax.xaxis.set_major_formatter(_fmt)
    except Exception:
        pass


def _format_axis_tick_value(value: float) -> str:
    """Format numeric axis ticks using the shared compact suffix rules."""
    try:
        numeric = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(numeric):
        return str(value)

    return _format_stat_value(numeric)


def _sample_numeric_axis_ticks(
    values: Sequence[Any] | np.ndarray | pd.Series,
    max_tick_labels: int,
    min_spacing_ratio: float = 0.0,
) -> tuple[list[float], list[str]]:
    """Sample readable numeric ticks using nice-step anchors and full-range coverage."""
    finite_values: list[float] = []
    for value in list(values):
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            finite_values.append(numeric)

    if not finite_values:
        return [], []

    min_value = float(min(finite_values))
    max_value = float(max(finite_values))
    safe_max = max(2, int(max_tick_labels))
    value_range = float(max_value - min_value)

    if value_range <= 0:
        return [min_value], [_format_precise_axis_value(min_value)]

    try:
        ratio = max(0.0, float(min_spacing_ratio))
    except Exception:
        ratio = 0.0

    # Keep ticks dense while allowing spacing ratio to soften overly crowded labels.
    density_scale = max(0.60, 1.0 - min(0.42, ratio * 0.55))
    target_tick_count = max(2, min(safe_max, int(round(safe_max * density_scale))))
    if safe_max >= 4:
        target_tick_count = max(4, target_tick_count)

    def _is_integer_like(v: float) -> bool:
        tol = 1e-9 * max(1.0, abs(v))
        return abs(v - round(v)) <= tol

    all_integer_like = bool(finite_values) and all(_is_integer_like(v) for v in finite_values)
    min_int = int(round(min_value))
    max_int = int(round(max_value))
    integer_span = max_int - min_int + 1

    # If the integer span is reasonably small, show every value (avoid skipped years/categories).
    if all_integer_like and integer_span >= 2 and integer_span <= safe_max:
        tick_values = [float(v) for v in range(min_int, max_int + 1)]
        return tick_values, [str(int(v)) for v in tick_values]

    anchors = (1.0, 2.0, 2.5, 5.0, 10.0)

    def _candidate_nice_steps(raw_step: float) -> list[float]:
        if raw_step <= 0 or not math.isfinite(raw_step):
            return [1.0]
        exponent = int(math.floor(math.log10(raw_step)))
        candidates: list[float] = []
        for exp in range(exponent - 1, exponent + 12):
            base = 10.0 ** exp
            for anchor in anchors:
                step = anchor * base
                if step > 0:
                    candidates.append(step)
        floor_step = raw_step * 0.999999
        return [step for step in sorted(set(candidates)) if step >= floor_step]

    def _build_ticks(step: float) -> list[float]:
        start = math.floor(min_value / step) * step
        end = math.ceil(max_value / step) * step
        tick_count = max(2, int(round((end - start) / step)) + 1)
        raw_ticks = [start + idx * step for idx in range(tick_count)]

        if raw_ticks and raw_ticks[0] > min_value:
            raw_ticks.insert(0, raw_ticks[0] - step)
        if raw_ticks and raw_ticks[-1] < max_value:
            raw_ticks.append(raw_ticks[-1] + step)

        if step >= 1:
            round_decimals = 6
        else:
            round_decimals = min(12, max(4, int(math.ceil(-math.log10(step))) + 2))

        ticks: list[float] = []
        for raw in raw_ticks:
            value = float(round(raw, round_decimals))
            if abs(value) < 1e-12:
                value = 0.0
            if not ticks or not math.isclose(value, ticks[-1], rel_tol=0.0, abs_tol=1e-12):
                ticks.append(value)
        return ticks

    rough_step = value_range / max(1, target_tick_count - 1)
    chosen_step = rough_step
    sampled_ticks: list[float] = []

    for step in _candidate_nice_steps(rough_step):
        candidate_ticks = _build_ticks(step)
        chosen_step = step
        sampled_ticks = candidate_ticks
        if len(candidate_ticks) <= safe_max:
            break

    if not sampled_ticks:
        sampled_ticks = [min_value, max_value]

    if len(sampled_ticks) > safe_max:
        sampled_idx = sorted({
            int(round(i))
            for i in np.linspace(0, len(sampled_ticks) - 1, num=safe_max)
        })
        sampled_ticks = [sampled_ticks[i] for i in sampled_idx]

    max_abs_bound = max(abs(min_value), abs(max_value))
    small_fractional_window = bool(value_range <= 4.0 and max_abs_bound <= 4.0)
    prefer_plain_integer_labels = bool(
        chosen_step >= 1.0
        and not small_fractional_window
        and max_abs_bound < 1e3
    )

    if prefer_plain_integer_labels:
        integer_labels = [str(int(round(v))) for v in sampled_ticks]
        if len(set(integer_labels)) == len(integer_labels):
            return sampled_ticks, integer_labels

    if small_fractional_window:
        base_decimals = min(
            10,
            max(2, int(math.ceil(-math.log10(max(chosen_step, 1e-12)))) + 1),
        )
        for extra in range(0, 4):
            decimals = min(10, base_decimals + extra)
            fractional = [_format_precise_axis_value(v, decimals=decimals) for v in sampled_ticks]
            if len(set(fractional)) == len(fractional):
                return sampled_ticks, fractional

    compact_labels = [_format_axis_tick_value(v) for v in sampled_ticks]
    if len(set(compact_labels)) == len(compact_labels):
        return sampled_ticks, compact_labels

    spacing = value_range / max(1, len(sampled_ticks) - 1)
    base_decimals = 2 if spacing >= 1 else min(
        10,
        max(3, int(math.ceil(-math.log10(max(spacing, 1e-12)))) + 1),
    )
    for extra in range(0, 5):
        decimals = min(10, base_decimals + extra)
        precise_labels = [_format_precise_axis_value(v, decimals=decimals) for v in sampled_ticks]
        if len(set(precise_labels)) == len(precise_labels):
            return sampled_ticks, precise_labels

    return sampled_ticks, [f"{float(v):.12g}" for v in sampled_ticks]


def _resolve_distribution_histogram_bins(
    values: Sequence[Any] | np.ndarray | pd.Series,
    *,
    min_bins: int = 10,
    max_bins: int = 50,
    integer_span_threshold: int = 240,
) -> np.ndarray | int:
    """Resolve histogram bins for static distribution charts.

    For integer-like ranges (e.g., years), prefer unit bins centered on each
    integer value so x-ticks can map exactly to bar centers.
    """
    finite_values: list[float] = []
    for value in list(values):
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            finite_values.append(numeric)

    if not finite_values:
        return max(2, int(min_bins))

    safe_min_bins = max(2, int(min_bins))
    safe_max_bins = max(safe_min_bins, int(max_bins))

    def _is_integer_like(v: float) -> bool:
        tol = 1e-9 * max(1.0, abs(v))
        return abs(v - round(v)) <= tol

    all_integer_like = bool(finite_values) and all(_is_integer_like(v) for v in finite_values)
    if all_integer_like:
        min_int = int(round(min(finite_values)))
        max_int = int(round(max(finite_values)))
        integer_span = max_int - min_int + 1
        if integer_span >= 2 and integer_span <= int(integer_span_threshold):
            start = float(min_int) - 0.5
            stop = float(max_int) + 1.5
            return np.arange(start, stop, 1.0, dtype=float)

    adaptive_bins = max(safe_min_bins, min(safe_max_bins, int(len(finite_values) // 10)))
    return int(adaptive_bins)


def _sample_histogram_bin_ticks(
    bin_edges: Sequence[Any] | np.ndarray,
    max_tick_labels: int,
    min_spacing_ratio: float = 0.0,
) -> tuple[list[float], list[str]]:
    """Return x-ticks sampled directly from histogram bar centers.

    Sampling from actual bin centers guarantees that each rendered tick maps to
    a real bar position, avoiding center/label drift.
    """
    if isinstance(bin_edges, np.ndarray):
        edges_arr = bin_edges.astype(float, copy=False)
    else:
        with contextlib.suppress(Exception):
            edges_arr = np.asarray(list(bin_edges), dtype=float)
        if 'edges_arr' not in locals():
            edges_arr = np.asarray([], dtype=float)

    edges = edges_arr[np.isfinite(edges_arr)]
    if edges.size < 2:
        return [], []

    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        return [], []

    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(float)
    if centers.size == 0:
        return [], []

    safe_max = max(2, int(max_tick_labels))

    def _is_integer_like(v: float) -> bool:
        tol = 1e-8 * max(1.0, abs(v))
        return abs(v - round(v)) <= tol

    all_centers_integer_like = bool(centers.size) and all(
        _is_integer_like(float(v)) for v in centers.tolist()
    )

    if all_centers_integer_like and centers.size <= safe_max:
        sampled = centers
    else:
        try:
            ratio = max(0.0, float(min_spacing_ratio))
        except Exception:
            ratio = 0.0
        density_scale = max(0.62, 1.0 - min(0.45, ratio * 0.70))
        target_tick_count = max(2, min(safe_max, int(round(safe_max * density_scale))))

        if centers.size <= target_tick_count:
            sampled = centers
        else:
            sampled_idx = sorted({
                int(round(i))
                for i in np.linspace(0, centers.size - 1, num=target_tick_count)
            })
            sampled = centers[sampled_idx]

    tick_values = [float(v) for v in sampled.tolist()]
    if not tick_values:
        return [], []

    compact_labels = [_format_axis_tick_value(v) for v in tick_values]
    if len(set(compact_labels)) == len(compact_labels):
        return tick_values, compact_labels

    if all(_is_integer_like(v) for v in tick_values):
        integer_labels = [str(int(round(v))) for v in tick_values]
        if len(set(integer_labels)) == len(integer_labels):
            return tick_values, integer_labels

    step_candidates = np.diff(edges)
    finite_steps = step_candidates[np.isfinite(step_candidates) & (step_candidates > 0)]
    step = float(np.median(finite_steps)) if finite_steps.size else 0.0
    base_decimals = 1 if step >= 1 else min(
        10,
        max(3, int(math.ceil(-math.log10(max(step, 1e-12)))) + 1),
    )
    for extra in range(0, 5):
        decimals = min(10, base_decimals + extra)
        precise_labels = [_format_precise_axis_value(v, decimals=decimals) for v in tick_values]
        if len(set(precise_labels)) == len(precise_labels):
            return tick_values, precise_labels

    return tick_values, [f"{float(v):.12g}" for v in tick_values]


def _tick_fontsize_for_labels(
    labels: Sequence[object] | pd.Index,
    *,
    dense_cutoff: int = 10,
) -> float:
    """Choose a smaller x-tick font when many labels are long or densely packed."""
    clean_labels = [_stringify_axis_label(label) for label in list(labels) if str(label).strip()]
    if not clean_labels:
        return 8.0

    max_label_length = max((len(label) for label in clean_labels), default=0)
    label_count = len(clean_labels)

    if max_label_length > 42 or label_count > max(dense_cutoff + 8, 18):
        return 5.0
    if max_label_length > 32 or label_count > max(dense_cutoff + 5, 14):
        return 5.5
    if max_label_length > 24 or label_count > max(dense_cutoff + 3, 12):
        return 6.0
    if max_label_length > 18 or label_count > dense_cutoff:
        return 6.5
    if max_label_length > 12 or label_count > max(6, dense_cutoff - 2):
        return 7.0
    return 8.0


def _apply_dense_non_overlapping_y_ticks(
    ax: Any,
    *,
    integer: bool = False,
    label_fontsize: float = 8.0,
    min_ticks: int = 6,
    max_ticks: int = 22,
    min_spacing_factor: float = 1.9,
) -> None:
    """Maximize y-axis tick density while avoiding label overlap."""
    try:
        fig = getattr(ax, "figure", None)
        fig_height_inches = float(fig.get_size_inches()[1]) if fig is not None else 6.0
        axes_height_fraction = float(getattr(ax.get_position(), "height", 0.75))
        axes_height_inches = max(1.0, fig_height_inches * max(0.2, axes_height_fraction))

        safe_fontsize = max(6.0, float(label_fontsize))
        safe_spacing_factor = max(1.6, float(min_spacing_factor))
        min_label_spacing_inches = max(0.11, (safe_fontsize * safe_spacing_factor) / 72.0)
        candidate_ticks = int(math.floor(axes_height_inches / min_label_spacing_inches))

        safe_min = max(3, int(min_ticks))
        safe_max = max(safe_min, int(max_ticks))
        nbins = max(safe_min, min(safe_max, candidate_ticks))
        min_target = max(safe_min, min(12, nbins))

        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(
                nbins=nbins,
                min_n_ticks=min_target,
                integer=bool(integer),
                steps=[1, 2, 2.5, 5, 10],
            )
        )
        ax.tick_params(axis='y', labelsize=safe_fontsize)
    except Exception:
        with contextlib.suppress(Exception):
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(
                    nbins=max(6, int(min_ticks)),
                    min_n_ticks=max(4, int(min_ticks) - 1),
                    integer=bool(integer),
                )
            )
            ax.tick_params(axis='y', labelsize=max(6.0, float(label_fontsize)))


def _resolve_static_tick_policy(
    labels: Sequence[object] | pd.Index,
    *,
    chart_type: Literal["trend", "forecast", "distribution"] = "forecast",
    spacing_profile: Literal["default", "detailed"] = "default",
) -> dict[str, float | int | str]:
    """Return shared static x-tick policy for Detailed Analysis, ZIP and PDF outputs."""
    clean_labels = [_stringify_axis_label(label) for label in list(labels) if str(label or "").strip()]
    label_count = len(clean_labels)
    max_label_length = max((len(label) for label in clean_labels), default=0)
    if chart_type == "distribution":
        # Keep distribution labels non-inclined and rely on evenly-spaced sampling
        # to increase readability while avoiding overlaps.
        tick_angle = 0
        tick_ha = 'center'
        target_tick_labels = min(24, max(6, label_count))
        if label_count <= 26:
            max_tick_labels = max(4, label_count)
        elif label_count <= 40:
            max_tick_labels = 24
        elif label_count <= 80:
            max_tick_labels = 20
        else:
            max_tick_labels = 18
        if max_label_length > 18 and label_count > 26:
            max_tick_labels = max(10, max_tick_labels - 4)
        elif max_label_length > 14 and label_count > 26:
            max_tick_labels = max(12, max_tick_labels - 2)
        min_spacing_ratio = 0.14 if label_count <= 26 else 0.20 if (label_count > 30 or max_label_length > 10) else 0.16

        # Detailed Analysis: show more x ticks while preserving readability.
        if spacing_profile == "detailed" and label_count > 20:
            max_tick_labels = min(28, max_tick_labels + 4)
            if max_label_length >= 10:
                max_tick_labels = min(max_tick_labels, 14)
                min_spacing_ratio = max(min_spacing_ratio, 0.24)
            elif max_label_length >= 8:
                max_tick_labels = min(max_tick_labels, 16)
                min_spacing_ratio = max(min_spacing_ratio, 0.21)
            else:
                max_tick_labels = min(max_tick_labels, 18)
                min_spacing_ratio = max(min_spacing_ratio, 0.18)
    else:
        target_tick_labels = min(25, label_count)
        can_fit_horizontal = target_tick_labels * (max_label_length + 1) <= 75
        if can_fit_horizontal:
            tick_angle = 0
            tick_ha = 'center'
            max_tick_labels = max(4, int(150 / max(1, max_label_length)))
        else:
            tick_angle = -20
            tick_ha = 'left'
            max_tick_labels = min(40, label_count)
        min_spacing_ratio = 0.0

    dense_cutoff = 8 if chart_type == "distribution" else 10
    tick_fontsize = _tick_fontsize_for_labels(clean_labels, dense_cutoff=dense_cutoff)
    if chart_type == "distribution":
        tick_fontsize = min(10.0, tick_fontsize * 1.28)
    
    if max_label_length > 46:
        tick_fontsize = min(tick_fontsize, 4.8)
    elif max_label_length > 36:
        tick_fontsize = min(tick_fontsize, 5.2)
    elif max_label_length > 28:
        tick_fontsize = min(tick_fontsize, 5.8)
    elif max_label_length > 22:
        tick_fontsize = min(tick_fontsize, 6.2)

    return {
        "max_tick_labels": int(max_tick_labels),
        "tick_fontsize": float(tick_fontsize),
        "tick_angle": int(tick_angle),
        "tick_ha": tick_ha,
        "min_spacing_ratio": float(min_spacing_ratio),
    }


def build_distribution_axis_spec(
    values: Sequence[Any] | np.ndarray | pd.Series,
    *,
    min_bins: int = 8,
    max_bins: int = 52,
    integer_span_threshold: int = 260,
    spacing_profile: Literal["default", "detailed"] = "default",
) -> dict[str, Any]:
    """Return a canonical distribution axis/bin spec shared across views/exports."""
    finite = pd.to_numeric(pd.Series(values), errors='coerce').dropna().astype(float)
    if finite.empty:
        return {}

    values_arr = np.asarray(finite.to_numpy(dtype=float), dtype=float)
    hist_bins = _resolve_distribution_histogram_bins(
        values_arr.tolist(),
        min_bins=min_bins,
        max_bins=max_bins,
        integer_span_threshold=integer_span_threshold,
    )
    if isinstance(hist_bins, np.ndarray):
        hist_edges = hist_bins.astype(float, copy=False)
    else:
        target_bins = max(1, int(hist_bins))
        min_val, max_val = float(np.min(values_arr)), float(np.max(values_arr))
        if min_val >= max_val:
            hist_edges = np.array([min_val - 0.5, min_val + 0.5])
        else:
            rough_step = (max_val - min_val) / target_bins
            exponent = int(math.floor(math.log10(rough_step))) if rough_step > 0 else 0
            base = 10.0 ** exponent
            best_step = base * 1.0
            min_diff = float('inf')
            for cand in [a * base for a in (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)]:
                if abs(cand - rough_step) < min_diff:
                    min_diff = abs(cand - rough_step)
                    best_step = cand
            start = math.floor(min_val / best_step) * best_step
            end = math.ceil(max_val / best_step) * best_step
            n_bins = max(1, int(round((end - start) / best_step)))
            hist_edges = np.array([start + i * best_step for i in range(n_bins + 1)])

    hist_edges = hist_edges[np.isfinite(hist_edges)]
    hist_edges = np.unique(np.sort(hist_edges))
    if hist_edges.size < 2:
        return {}

    finite_unique_values = np.unique(values_arr[np.isfinite(values_arr)])
    tick_policy = _resolve_static_tick_policy(
        finite_unique_values.tolist(),
        chart_type='distribution',
        spacing_profile=spacing_profile,
    )
    tick_values, tick_labels = _sample_histogram_bin_ticks(
        hist_edges,
        max_tick_labels=int(tick_policy['max_tick_labels']),
        min_spacing_ratio=float(tick_policy['min_spacing_ratio']),
    )

    bin_size: float | None = None
    widths = np.diff(hist_edges)
    finite_widths = widths[np.isfinite(widths) & (widths > 0)]
    if finite_widths.size and np.allclose(finite_widths, finite_widths[0], rtol=1e-9, atol=1e-12):
        bin_size = float(finite_widths[0])

    return {
        'hist_bins': [float(edge) for edge in hist_edges.tolist()],
        'bin_edges': [float(edge) for edge in hist_edges.tolist()],
        'bin_start': float(hist_edges[0]),
        'bin_end': float(hist_edges[-1]),
        'bin_size': bin_size,
        'tickvals': [float(v) for v in tick_values],
        'ticktext': [str(v) for v in tick_labels],
        'tick_angle': int(tick_policy['tick_angle']),
        'tick_ha': str(tick_policy['tick_ha']),
        'tick_fontsize': float(tick_policy['tick_fontsize']),
        'max_tick_labels': int(tick_policy['max_tick_labels']),
        'min_spacing_ratio': float(tick_policy['min_spacing_ratio']),
    }


def apply_distribution_axis_spec(ax: Any, axis_spec: dict[str, Any] | None) -> None:
    """Apply a distribution axis spec produced by ``build_distribution_axis_spec``."""
    if not isinstance(axis_spec, dict) or not axis_spec:
        return

    tick_values = axis_spec.get('tickvals')
    tick_labels = axis_spec.get('ticktext')
    if isinstance(tick_values, list) and tick_values:
        ax.set_xticks([float(v) for v in tick_values])
        if isinstance(tick_labels, list) and tick_labels:
            ax.set_xticklabels(
                [str(label) for label in tick_labels],
                rotation=int(axis_spec.get('tick_angle', 0)),
                ha=str(axis_spec.get('tick_ha', 'center')),
                fontsize=float(axis_spec.get('tick_fontsize', 8.0)),
            )

    try:
        raw_start = axis_spec.get('bin_start')
        raw_end = axis_spec.get('bin_end')
        if raw_start is None or raw_end is None:
            return
        start = float(raw_start)
        end = float(raw_end)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            ax.set_xlim(start, end)
    except Exception:
        pass


def apply_static_distribution_compact_layout(
    fig: Any,
    ax: Any,
    *,
    right: float = 0.947,
    top: float = 0.898,
) -> None:
    """Compact distribution footer spacing while preserving no-overlap guardrails."""
    with contextlib.suppress(Exception):
        ax.xaxis.set_label_coords(0.5, -0.072)

    max_rotation = 0.0
    max_label_len = 0
    with contextlib.suppress(Exception):
        tick_labels = [str(tick.get_text() or '').strip() for tick in ax.get_xticklabels()]
        max_label_len = max((len(label) for label in tick_labels), default=0)
        rotations = [abs(float(tick.get_rotation() or 0.0)) for tick in ax.get_xticklabels()]
        max_rotation = max(rotations, default=0.0)

    if max_rotation >= 28 or max_label_len > 16:
        bottom = 0.255
    elif max_rotation >= 16 or max_label_len > 11:
        bottom = 0.235
    else:
        bottom = 0.215

    fig.subplots_adjust(bottom=bottom, right=right, top=top)


def _stringify_axis_label(label: object) -> str:
    """Normalize axis labels to a compact single-line string."""
    raw = str(label or "")
    return " ".join(raw.split())


def _sample_axis_tick_labels(
    labels: Sequence[object] | pd.Index,
    max_tick_labels: int,
) -> tuple[list[int], list[str]]:
    """Sample axis labels while always preserving the last visible point."""
    clean_labels = [_stringify_axis_label(label) for label in list(labels)]
    total = len(clean_labels)
    if total <= 0:
        return [], []

    safe_max = max(1, int(max_tick_labels))
    if total > safe_max:
        positions = sorted({
            int(round(pos))
            for pos in np.linspace(0, total - 1, num=safe_max)
        })
        if positions[0] != 0:
            positions.insert(0, 0)
        if positions[-1] != total - 1:
            positions.append(total - 1)
    else:
        positions = list(range(total))

    return positions, [clean_labels[pos] for pos in positions]


def _build_non_timeseries_tick_labels(
    history_labels: Sequence[object] | pd.Index,
    forecast_labels: Sequence[object] | pd.Index | None = None,
    *,
    max_tick_labels: int,
) -> tuple[list[int], list[str]]:
    """Return tick positions/labels for non-timeseries forecast charts.

    When the original history index is descriptive text (for example countries)
    but the forecast helper has to synthesize numeric future indices, keep the
    visible ticks anchored to the real history labels instead of surfacing the
    synthetic numeric forecast index.
    """
    history_index = pd.Index(list(history_labels))
    forecast_index = pd.Index(list(forecast_labels)) if forecast_labels is not None else pd.Index([])

    try:
        history_is_numeric = bool(pd.api.types.is_numeric_dtype(history_index))
    except Exception:
        history_is_numeric = False

    try:
        forecast_is_numeric = bool(pd.api.types.is_numeric_dtype(forecast_index))
    except Exception:
        forecast_is_numeric = False

    if len(forecast_index) == 0 or (forecast_is_numeric and not history_is_numeric):
        return _sample_axis_tick_labels(history_index, max_tick_labels=max_tick_labels)

    full_labels = list(history_index) + list(forecast_index)
    return _sample_axis_tick_labels(full_labels, max_tick_labels=max_tick_labels)


def _is_numeric_index_like(index: pd.Index) -> bool:
    """Return True when an index behaves like a numeric position axis."""
    try:
        return bool(pd.api.types.is_numeric_dtype(index))
    except Exception:
        return False


def _resolve_plot_display_axis(
    series: pd.Series,
    *,
    source_df: pd.DataFrame | None = None,
    fallback_label: str | None = None,
    prefer_first_column: bool = False,
) -> tuple[str, pd.Index]:
    """Resolve the descriptive display axis used by interactive charts.

    Static charts sometimes receive numeric series whose index is a cached
    positional axis even though the resolved analysis x-axis is descriptive
    text (for example, a first-column country name). This helper mirrors the
    interactive view's preference for that descriptive axis without changing the
    underlying numeric plotting positions used for forecasts/anomalies.
    """
    base_index = pd.Index(series.index)
    default_label = (
        str(base_index.name)
        if getattr(base_index, "name", None)
        else (fallback_label or ("Timestamp" if _is_reliable_timeseries_index(base_index) else "Index"))
    )

    if prefer_first_column and source_df is not None and not source_df.empty and len(base_index) > 0:
        if len(base_index) == len(source_df):
            first_col = source_df.columns[0]
            first_values_full = source_df.iloc[:, 0]
            try:
                first_is_numeric_full = bool(pd.api.types.is_numeric_dtype(first_values_full))
            except Exception:
                first_is_numeric_full = False

            if not first_is_numeric_full:
                try:
                    source_index = pd.Index(source_df.index)
                    if base_index.equals(source_index):
                        resolved_index = pd.Index(first_values_full.tolist(), name=first_col)
                        return str(first_col), resolved_index
                except Exception:
                    pass

        try:
            raw_positions = [int(v) for v in base_index.tolist()]
        except Exception:
            raw_positions = []

        if len(raw_positions) == len(base_index) and all(0 <= pos < len(source_df) for pos in raw_positions):
            first_col = source_df.columns[0]
            first_values = source_df.iloc[raw_positions, 0]
            try:
                first_is_numeric = bool(pd.api.types.is_numeric_dtype(first_values))
            except Exception:
                first_is_numeric = False

            if not first_is_numeric:
                resolved_index = pd.Index(first_values.tolist(), name=first_col)
                if len(resolved_index) == len(base_index):
                    return str(first_col), resolved_index

    if len(base_index) == 0 or _is_reliable_timeseries_index(base_index) or not _is_numeric_index_like(base_index):
        return default_label, base_index

    if source_df is None or source_df.empty:
        return default_label, base_index

    try:
        raw_positions = [int(v) for v in base_index.tolist()]
    except Exception:
        raw_positions = []

    if len(raw_positions) != len(base_index) or any(pos < 0 or pos >= len(source_df) for pos in raw_positions):
        return default_label, base_index

    source_index = pd.Index(source_df.index)
    if len(source_index) == len(source_df) and not _is_numeric_index_like(source_index):
        resolved_index = pd.Index([source_index[pos] for pos in raw_positions], name=source_index.name)
        resolved_label = str(resolved_index.name) if resolved_index.name else default_label
        return resolved_label, resolved_index

    if len(source_df.columns) > 0:
        first_col = source_df.columns[0]
        first_values = source_df.iloc[raw_positions, 0]
        resolved_index = pd.Index(first_values.tolist(), name=first_col)
        return str(first_col), resolved_index

    return default_label, base_index


def _wrap_category_legend_label(
    label: object,
    *,
    width: int = 44,
    max_lines: int = 3,
    html_breaks: bool = False,
) -> str:
    """Wrap category legend items without truncating the underlying label text."""
    raw = _stringify_axis_label(label)
    if not raw:
        return raw
    wrapped = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False) or [raw]
    if len(wrapped) > max_lines:
        remaining = " ".join(wrapped[max_lines - 1 :])
        wrapped = wrapped[: max_lines - 1] + [remaining]
    joiner = "<br>" if html_breaks else "\n"
    return joiner.join(wrapped)


def get_export_chart_figsize(
    chart_kind: Literal["trend", "forecast", "distribution"],
    *,
    context: Literal["zip", "pdf"] = "pdf",
) -> tuple[float, float]:
    """Return shared export figure sizes for ZIP/PDF chart rendering.

    Sizes are intentionally a bit taller than previous defaults to improve
    legibility of axis labels, legends, and annotations in static exports.
    """
    size_map: dict[tuple[str, str], tuple[float, float]] = {
        ("zip", "trend"): (10.0, 7.4),
        ("zip", "forecast"): (10.0, 7.4),
        ("zip", "distribution"): (8.0, 8.2),
        ("pdf", "trend"): (10.0, 5.2),
        ("pdf", "forecast"): (10.0, 5.2),
        ("pdf", "distribution"): (10.0, 8.2),
    }
    return size_map.get((str(context), str(chart_kind)), (10.0, 7.2))


def generate_static_distribution_plot(
    values: Sequence[Any] | np.ndarray | pd.Series,
    column: object,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.2),
    dpi: int = 150,
    pad_inches: float = 0.02,
    spacing_profile: Literal["default", "detailed"] = "default",
    value_formatter: Callable[[float], str] | None = None,
    label: str | None = None,
) -> str | None:
    """Render the shared static distribution chart used by pages and exports."""
    fig = None
    try:
        series = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
        if series.empty:
            return None
        arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
        if arr.size == 0:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        axis_spec = build_distribution_axis_spec(
            arr.tolist(),
            min_bins=max(8, min(12, len(arr) // 5)) if len(arr) >= 20 else 8,
            max_bins=52,
            integer_span_threshold=260,
            spacing_profile=spacing_profile,
        )
        hist_bins = axis_spec.get('hist_bins') if isinstance(axis_spec, dict) else None
        if not hist_bins:
            hist_bins = max(8, min(52, int(len(arr) // 10) if len(arr) >= 20 else 8))

        _hist_counts, hist_edges, _hist_patches = ax.hist(
            arr,
            bins=hist_bins,
            color='#5d84d8',
            alpha=0.74,
            edgecolor='#1f2937',
            linewidth=0.45,
            label=str(label if label is not None else column),
        )
        ax.set_title(title or f"Distribution: {column}", fontsize=10, pad=16)
        ax.set_xlabel(str(column), fontsize=9, labelpad=0)
        ax.set_ylabel("Frequency", fontsize=9, labelpad=8)
        ax.grid(True, alpha=0.3)
        _apply_dense_non_overlapping_y_ticks(
            ax,
            integer=True,
            label_fontsize=8.0,
            min_ticks=6,
            max_ticks=18,
        )
        if isinstance(axis_spec, dict) and axis_spec:
            apply_distribution_axis_spec(ax, axis_spec)
        elif len(hist_edges) >= 2:
            ax.set_xlim(float(hist_edges[0]), float(hist_edges[-1]))

        _add_static_distribution_overlays(
            ax,
            arr,
            value_formatter=value_formatter or _format_stat_value,
            legend_fontsize=6,
            legend_columns=6,
            legend_y=-0.12,
            expand_xlim=False,
            right_pad_ratio=0.015,
            top_lane=1.005 if spacing_profile == "default" else 1.006,
            line_tag_offset_ratio=0.006,
        )
        _apply_sci_formatter(ax, y_threshold=1e3, x_threshold=1e6)
        apply_static_distribution_compact_layout(fig, ax)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as exc:
        with contextlib.suppress(Exception):
            app.logger.debug("Static distribution plot failed for %s: %s", column, exc)
        return None
    finally:
        if fig is not None:
            with contextlib.suppress(Exception):
                plt.close(fig)


def _build_static_category_chart(all_counts: pd.Series, col: str) -> tuple[Any, Any] | None:
    """Build a shared Matplotlib category chart used by PDF and ZIP exports."""
    if all_counts.empty or len(all_counts) < 2:
        return None

    category_count = len(all_counts)
    total_unique = len(all_counts)
    max_count = int(all_counts.max())
    min_count = int(all_counts.min())
    avg_count = float(all_counts.mean())
    med_count = float(all_counts.median())
    most_freq = str(all_counts.index[0])
    least_freq = str(all_counts.index[-1]) if len(all_counts) > 0 else "N/A"
    most_label_value = textwrap.shorten(
        _stringify_axis_label(most_freq),
        width=20,
        placeholder='…',
    )
    least_label_value = textwrap.shorten(
        _stringify_axis_label(least_freq),
        width=20,
        placeholder='…',
    )
    most_label = f"Most: '{most_label_value}' ({max_count})"
    least_label = f"Least: '{least_label_value}' ({min_count})"
    legend_labels = [
        "Count",
        f"Avg: {_format_stat_value(avg_count)}",
        f"Med: {_format_stat_value(med_count)}",
        most_label,
        least_label,
    ]

    category_labels = [_stringify_axis_label(label) for label in all_counts.index.tolist()]
    tick_positions: list[float] = [float(i) for i in range(category_count)]

    # Export requirement: always show every category value on the x-axis,
    # regardless of category count (PDF report + images ZIP).
    tick_labels = category_labels
    visible_tick_labels = [label for label in tick_labels if label]
    visible_tick_count = max(1, len(visible_tick_labels))
    max_label_length = max((len(label) for label in visible_tick_labels), default=0)
    long_label_scale = min(1.0, max_label_length / 42.0)
    label_density_scale = min(1.0, visible_tick_count / 180.0)
    max_legend_label_length = max((len(label) for label in legend_labels), default=0)
    legend_fontsize = 15 if max_legend_label_length > 26 else 16
    legend_columns = len(legend_labels)
    legend_rows = 1
    can_fit_horizontal = visible_tick_count * (max_label_length + 1) <= 120
    tick_ha: Literal['left', 'center', 'right']
    if can_fit_horizontal:
        tick_angle = 0
        tick_ha = 'center'
    elif category_count > 120:
        tick_angle = -54
        tick_ha = 'left'
    elif category_count > 70 or max_label_length > 18:
        tick_angle = -46
        tick_ha = 'left'
    else:
        tick_angle = -32
        tick_ha = 'left'
    # Increase typography while preserving dense x-axis readability around ~200 categories.
    if visible_tick_count >= 180:
        tick_fontsize = 6.0
    elif visible_tick_count > 120:
        tick_fontsize = 8.8
    elif visible_tick_count > 60:
        tick_fontsize = 10.8
    else:
        tick_fontsize = 12.2
    # Keep export width in a narrow band so PDF/ZIP renders do not visually shrink
    # bar heights or footer/title fonts when many categories are present.
    fig_width = min(18.8, max(16.0, 16.0 + (category_count / 70.0)))
    plot_area_height = 12.2
    top_padding_inches = 1.0
    tick_to_xlabel_gap_inches = 0.04 if tick_angle == 0 else 0.06
    tick_footer_inches = (
        0.26
        if tick_angle == 0
        else 1.80 + long_label_scale * 0.30 + label_density_scale * 0.25
    )
    xlabel_lane_inches = 0.34
    xlabel_to_legend_gap_inches = 0.06
    legend_lane_inches = 0.50 + max(0, legend_rows - 1) * 0.22
    bottom_inches = (
        tick_footer_inches
        + tick_to_xlabel_gap_inches
        + xlabel_lane_inches
        + xlabel_to_legend_gap_inches
        + legend_lane_inches
    )
    fig_height = min(30.0, max(16.0, plot_area_height + bottom_inches + top_padding_inches))
    # Use actual content height -- no inflated minimum that wastes vertical space
    bottom_fraction = min(
        0.86,
        max(0.10 if tick_angle == 0 else 0.16, bottom_inches / fig_height),
    )
    top_fraction = min(0.14, top_padding_inches / fig_height)
    axes_top_fraction = max(0.72, 1.0 - top_fraction)
    axes_height_fraction = max(1e-6, axes_top_fraction - bottom_fraction)
    legend_bottom_axes = (0.0 - bottom_fraction) / axes_height_fraction
    xlabel_center_fraction = (
        legend_lane_inches
        + xlabel_to_legend_gap_inches
        + (xlabel_lane_inches * 0.5)
    ) / fig_height
    xlabel_y_axes = (xlabel_center_fraction - bottom_fraction) / axes_height_fraction

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_positions = np.arange(category_count, dtype=np.int64)
    y_values = [int(float(v)) for v in all_counts.to_numpy(dtype=float).tolist()]
    bar_color = '#2e7d32'
    # Keep dense-category bar spacing visually consistent in rasterized exports.
    # We target a minimum gap in *pixels* (not data units) so very dense charts
    # avoid occasional conjoined-looking neighbors from subpixel rounding.
    axes_right_fraction = 0.92
    axes_width_fraction = max(1e-6, axes_right_fraction - 0.125)
    estimated_axes_px = max(320.0, fig_width * 120.0 * axes_width_fraction)
    slot_px = estimated_axes_px / max(1.0, float(category_count))
    target_gap_px = 2.0 if category_count >= 100 else 1.5
    bar_width = 1.0 - (target_gap_px / max(1.0, slot_px))
    bar_width = float(min(0.90, max(0.68, bar_width)))
    bar_container = ax.bar(
        x_positions,
        y_values,
        width=bar_width,
        color=bar_color,
        alpha=1.0,
        edgecolor='none',
        linewidth=0.0,
        antialiased=False,
        snap=True,
        label='Count',
    )

    if isinstance(bar_container, BarContainer):
        with contextlib.suppress(Exception):
            tick_positions = [
                float(bar.get_x() + (bar.get_width() / 2.0))
                for bar in bar_container.patches
            ]

    if category_count <= 72:
        with contextlib.suppress(Exception):
            if isinstance(bar_container, BarContainer):
                ax.bar_label(
                    bar_container,
                    labels=[str(v) for v in y_values],
                    padding=2,
                    fontsize=14,
                )

    ax.set_title(f"Categories: {col} ({total_unique} unique values)", fontsize=20, pad=17)
    ax.set_xlabel(col, fontsize=19, labelpad=0)
    ax.set_ylabel("Count", fontsize=17, labelpad=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.margins(x=0.01)
    ax.set_ylim(0, max(max_count * 1.06, 1.0))
    right_empty_space_slots = 0.16
    ax.set_xlim(-0.5, category_count - 0.5 + right_empty_space_slots)

    _apply_dense_non_overlapping_y_ticks(
        ax,
        integer=True,
        label_fontsize=15.0,
        min_ticks=6,
        max_ticks=22,
    )

    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
    x_tick_pad = 2.0 if tick_angle == 0 else 0.0
    ax.tick_params(axis='x', pad=x_tick_pad, labelsize=tick_fontsize, direction='out')
    with contextlib.suppress(Exception):
        # Keep axis baseline anchored to bars for all category density profiles.
        ax.spines['bottom'].set_position(('outward', 0.0))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(tick_angle)
        tick_label.set_horizontalalignment(tick_ha)
        if tick_angle != 0:
            tick_label.set_rotation_mode('anchor')
            tick_label.set_verticalalignment('top')

    avg_color = '#a16207'
    med_color = '#6b21a8'
    ax.axhline(y=avg_count, color=avg_color, linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {_format_stat_value(avg_count)}')
    ax.axhline(y=med_count, color=med_color, linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Med: {_format_stat_value(med_count)}')

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    base_text_offset = max((y_range * 0.006) if y_range else 0.0, 0.08)
    min_vertical_gap = max(
        base_text_offset * 2.2,
        (y_range * 0.012) if y_range else 0.0,
        0.16,
    )
    if avg_count >= med_count:
        avg_y = avg_count + base_text_offset
        avg_va = 'bottom'
        med_y = med_count - base_text_offset
        med_va = 'top'
        pair_gap = avg_y - med_y
        if pair_gap < min_vertical_gap:
            push = (min_vertical_gap - pair_gap) / 2.0
            avg_y += push
            med_y -= push
    else:
        avg_y = avg_count - base_text_offset
        avg_va = 'top'
        med_y = med_count + base_text_offset
        med_va = 'bottom'
        pair_gap = med_y - avg_y
        if pair_gap < min_vertical_gap:
            push = (min_vertical_gap - pair_gap) / 2.0
            avg_y -= push
            med_y += push

    ax.text(
        1.008,
        avg_y,
        f'Avg: {_format_stat_value(avg_count)}',
        transform=ax.get_yaxis_transform(),
        va=avg_va,
        ha='left',
        fontsize=14,
        color=avg_color,
        fontweight='semibold',
        clip_on=False,
    )
    ax.text(
        1.008,
        med_y,
        f'Med: {_format_stat_value(med_count)}',
        transform=ax.get_yaxis_transform(),
        va=med_va,
        ha='left',
        fontsize=14,
        color=med_color,
        fontweight='semibold',
        clip_on=False,
    )

    ax.plot(
        [],
        [],
        color='#27ae60',
        marker='s',
        linestyle='',
        markersize=8,
        label=most_label,
    )
    ax.plot(
        [],
        [],
        color='#f1c40f',
        marker='s',
        linestyle='',
        markersize=8,
        label=least_label,
    )

    ax.legend(
        fontsize=legend_fontsize,
        loc='lower center',
        bbox_to_anchor=(0.5, legend_bottom_axes),
        ncol=legend_columns,
        frameon=False,
        columnspacing=0.26,
        handletextpad=0.16,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=bottom_fraction, right=axes_right_fraction, top=axes_top_fraction)
    ax.xaxis.set_label_coords(0.5, xlabel_y_axes)
    return fig, ax


def _add_static_distribution_overlays(
    ax: Any,
    values: pd.Series | np.ndarray | list[float] | tuple[float, ...],
    *,
    value_formatter: Callable[[float], str] | None = None,
    legend_fontsize: float = 6,
    legend_columns: int = 6,
    legend_y: float = -0.12,
    expand_xlim: bool = True,
    right_pad_ratio: float = 0.0,
    left_pad_ratio: float = 0.0,
    top_lane: float = 1.008,
    line_tag_offset_ratio: float = 0.006,
) -> dict[str, float]:
    """Add consistent stat overlays to a static histogram axis.

    The overlays place Min/Max markers on the x-axis lane while keeping their
    value tags in a raised annotation lane so labels do not collide with
    x-axis ticks.
    """
    formatter = value_formatter or _format_stat_value
    finite = pd.to_numeric(pd.Series(values), errors='coerce').dropna().astype(float)
    if finite.empty:
        raise ValueError("No finite values available for distribution overlays")

    stats_min = float(finite.min())
    stats_max = float(finite.max())
    stats_mean = float(finite.mean())
    stats_median = float(finite.median())
    stats_std = float(finite.std())

    avg_color = '#a16207'
    med_color = '#6b21a8'

    ax.axvline(
        x=stats_mean,
        color=avg_color,
        linestyle=':',
        linewidth=2,
        alpha=0.8,
        label=f'Avg: {formatter(stats_mean)}',
    )
    ax.axvline(
        x=stats_median,
        color=med_color,
        linestyle='-.',
        linewidth=1.5,
        alpha=0.7,
        label=f'Med: {formatter(stats_median)}',
    )

    xlim = ax.get_xlim()
    x_range = max(float(xlim[1] - xlim[0]), 1e-9)
    if bool(expand_xlim):
        ax.set_xlim(xlim[0] - x_range * 0.03, xlim[1] + x_range * 0.03)
        xlim = ax.get_xlim()
        x_range = max(float(xlim[1] - xlim[0]), 1e-9)
    else:
        safe_left_ratio = max(0.0, float(left_pad_ratio))
        safe_right_ratio = max(0.0, float(right_pad_ratio))
        if safe_left_ratio > 0 or safe_right_ratio > 0:
            ax.set_xlim(
                xlim[0] - (x_range * safe_left_ratio),
                xlim[1] + (x_range * safe_right_ratio),
            )
        xlim = ax.get_xlim()
        x_range = max(float(xlim[1] - xlim[0]), 1e-9)

    xaxis_transform = blended_transform_factory(ax.transData, ax.transAxes)
    top_lane = max(1.001, float(top_lane))
    safe_line_offset_ratio = max(0.0, float(line_tag_offset_ratio))
    x_offset = max(1e-9, x_range * safe_line_offset_ratio)

    if stats_mean <= stats_median:
        ax.text(
            stats_mean - x_offset,
            top_lane,
            f'Avg: {formatter(stats_mean)}',
            transform=xaxis_transform,
            va='bottom',
            ha='right',
            fontsize=7.2,
            color=avg_color,
            fontweight='semibold',
            clip_on=False,
        )
        ax.text(
            stats_median + x_offset,
            top_lane,
            f'Med: {formatter(stats_median)}',
            transform=xaxis_transform,
            va='bottom',
            ha='left',
            fontsize=7.2,
            color=med_color,
            fontweight='semibold',
            clip_on=False,
        )
    else:
        ax.text(
            stats_median - x_offset,
            top_lane,
            f'Med: {formatter(stats_median)}',
            transform=xaxis_transform,
            va='bottom',
            ha='right',
            fontsize=7.2,
            color=med_color,
            fontweight='semibold',
            clip_on=False,
        )
        ax.text(
            stats_mean + x_offset,
            top_lane,
            f'Avg: {formatter(stats_mean)}',
            transform=xaxis_transform,
            va='bottom',
            ha='left',
            fontsize=7.2,
            color=avg_color,
            fontweight='semibold',
            clip_on=False,
        )

    # Min/Max: sit on x-axis (marker_lane_y=0.008), text at 0.04 above
    marker_lane_y = 0.008
    # Colors legible on both blue histogram bars and white background
    min_color = '#d97706'  # vivid amber - high contrast on blue and white
    max_color = '#16a34a'  # vivid green - high contrast on blue and white
    edge_color = '#0b1220'
    min_xytext = (2, 6)
    max_xytext = (-2, 6)
    if abs(stats_max - stats_min) <= x_range * 0.04:
        min_xytext = (2, 6)
        max_xytext = (-4, 6)

    ax.scatter(
        [stats_min],
        [marker_lane_y],
        transform=xaxis_transform,
        color=min_color,
        s=28,
        zorder=10,
        marker='v',
        edgecolors=edge_color,
        linewidths=1.4,
        clip_on=False,
        label=f'Min: {formatter(stats_min)}',
    )
    ax.scatter(
        [stats_max],
        [marker_lane_y],
        transform=xaxis_transform,
        color=max_color,
        s=28,
        zorder=10,
        marker='^',
        edgecolors=edge_color,
        linewidths=1.4,
        clip_on=False,
        label=f'Max: {formatter(stats_max)}',
    )
    ax.annotate(
        formatter(stats_min),
        (stats_min, marker_lane_y),
        xycoords=xaxis_transform,
        textcoords='offset points',
        xytext=min_xytext,
        ha='left',
        va='bottom',
        fontsize=6.4,
        color='#7c2d12',
        fontweight='semibold',
        bbox={
            'boxstyle': 'round,pad=0.18',
            'facecolor': '#fff7ed',
            'edgecolor': min_color,
            'linewidth': 0.6,
            'alpha': 0.96,
        },
        annotation_clip=False,
    )
    ax.annotate(
        formatter(stats_max),
        (stats_max, marker_lane_y),
        xycoords=xaxis_transform,
        textcoords='offset points',
        xytext=max_xytext,
        ha='right',
        va='bottom',
        fontsize=6.4,
        color='#14532d',
        fontweight='semibold',
        bbox={
            'boxstyle': 'round,pad=0.18',
            'facecolor': '#f0fdf4',
            'edgecolor': max_color,
            'linewidth': 0.6,
            'alpha': 0.96,
        },
        annotation_clip=False,
    )

    ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {formatter(stats_std)}')
    legend_anchor = max(-0.16, min(float(legend_y), -0.09))
    ax.legend(
        fontsize=legend_fontsize,
        loc='upper center',
        bbox_to_anchor=(0.5, legend_anchor),
        ncol=legend_columns,
        frameon=False,
        columnspacing=0.80,
        handletextpad=0.20,
        borderaxespad=0.0,
    )

    return {
        'min': stats_min,
        'max': stats_max,
        'mean': stats_mean,
        'median': stats_median,
        'std': stats_std,
    }

# Bound at runtime via _bind_runtime_globals().
app: Any = cast(Any, None)

_LOCAL_SYMBOLS = {
    '_LOCAL_SYMBOLS',
    '_bind_runtime_globals',
    '_is_reliable_timeseries_index',
    '_is_numeric_index_like',
    '_resolve_plot_display_axis',
    '_add_static_distribution_overlays',
    '_legacy',
    '_cap_anomalies_for_display',
    '_anomaly_positions_for_index',
    'generate_plot',
    'generate_forecast_plot',
    '_build_category_plotly_chart',
    'generate_correlation_heatmap',
    'get_cached_heatmap',
    'generate_stl_plot',
    'get_cached_stl_plot',
    '__all__',
}



def _bind_runtime_globals():
    import data_analysis.runtime_app as rt

    sync = getattr(rt, "_sync_ai_engine_state", None)
    if callable(sync):
        sync()

    global app
    app = getattr(rt, "app", None)

    g = globals()
    for key, value in rt.__dict__.items():
        if key.startswith("__") or key in _LOCAL_SYMBOLS:
            continue
        g[key] = value
    return rt


def _is_reliable_timeseries_index(idx) -> bool:
    """Resolve time-series index checker lazily from runtime_app."""
    rt = _bind_runtime_globals()
    checker = getattr(rt, "_is_reliable_timeseries_index", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(idx))
    except Exception:
        return False


def _cap_anomalies_for_display(
    anomalies_idx: pd.Index | None,
    anomalies_score: pd.Series | None = None,
    max_points: int | None = None,
) -> pd.Index:
    """Return a display-only subset of anomalies (detection results remain unchanged)."""
    if anomalies_idx is None:
        return pd.Index([])
    if max_points is None:
        max_points = 20
    if max_points <= 0:
        return pd.Index([])
    if len(anomalies_idx) <= max_points:
        return pd.Index(anomalies_idx)

    if anomalies_score is None or anomalies_score.empty:
        return pd.Index(anomalies_idx[:max_points])

    score_buckets: dict[Any, list[float]] = {}
    try:
        for idx_val, score_val in anomalies_score.items():
            if pd.notna(score_val):
                score_buckets.setdefault(idx_val, []).append(float(score_val))
    except Exception:
        score_buckets = {}

    ranked_rows: list[tuple[int, float]] = []
    for rel_pos, idx_val in enumerate(list(anomalies_idx)):
        scores = score_buckets.get(idx_val, [])
        score_val = scores.pop(0) if scores else float("-inf")
        score_buckets[idx_val] = scores
        ranked_rows.append((int(rel_pos), float(score_val)))

    ranked_rows.sort(key=lambda row: row[1], reverse=True)
    top_rel_positions = sorted(pos for pos, _ in ranked_rows[:max_points])
    return pd.Index([anomalies_idx[pos] for pos in top_rel_positions])


def _anomaly_positions_for_index(data_index: pd.Index, anomalies_idx: pd.Index | None) -> list[int]:
    """Map anomaly labels to concrete plotted positions using occurrence counts."""
    if anomalies_idx is None or len(anomalies_idx) == 0:
        return []

    # Explicit positional anomaly indices:
    # - preferred format: index name "__pos__"
    # - backward-compat: integer index for non-unique labels.
    try:
        a = pd.Index(anomalies_idx)
        idx_name = str(getattr(a, "name", "") or "")
        is_positional = idx_name.lower() == "__pos__"
        if not is_positional and (not data_index.is_unique) and pd.api.types.is_integer_dtype(a):
            is_positional = True
        if is_positional:
            max_pos = len(data_index) - 1
            pos = [int(v) for v in a if isinstance(v, (int, np.integer)) and 0 <= int(v) <= max_pos]
            if pos:
                return list(dict.fromkeys(pos))
    except Exception:
        pass

    remaining = Counter(list(anomalies_idx))
    positions: list[int] = []
    for i, idx_val in enumerate(list(data_index)):
        if remaining.get(idx_val, 0) > 0:
            positions.append(i)
            remaining[idx_val] -= 1
    return positions


def generate_plot(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    anomalies_idx: Any = None,
    use_webp: bool = False,
    display_index: pd.Index | list[object] | tuple[object, ...] | None = None,
) -> str:
    _bind_runtime_globals()
    try:
        max_anomaly_markers = int(app.config.get('ANOMALY_MARKER_CAP', 20))
    except Exception:
        max_anomaly_markers = 20
    # HIGH QUALITY: Larger figure for better image quality
    fig, ax = plt.subplots(figsize=(10, 4.8))
    
    # Use numeric x-positions for non-datetime indexes to ensure proper alignment
    is_datetime = _is_reliable_timeseries_index(data.index)
    y_values = pd.to_numeric(data, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    
    if is_datetime:
        # For datetime index, use the index directly
        ax.plot(data.index, y_values, label='History', color='tab:blue', lw=1.0)
        if anomalies_idx is not None and len(anomalies_idx):
            try:
                an_display = _cap_anomalies_for_display(pd.Index(anomalies_idx), max_points=max_anomaly_markers)
                an_positions = _anomaly_positions_for_index(data.index, an_display)
                if an_positions:
                    an_pos_arr = np.asarray(an_positions, dtype=np.int64)
                    aligned = data.iloc[an_pos_arr]
                    aligned_y = pd.to_numeric(
                        aligned,
                        errors='coerce',
                    ).to_numpy(dtype=float, na_value=np.nan)
                    ax.scatter(aligned.index, aligned_y, color='red', s=2, zorder=5, label='Anomaly')
            except Exception:
                pass
    else:
        # For non-datetime index (e.g., country names), use numeric positions
        x_positions = np.arange(len(data), dtype=np.int64)
        ax.plot(x_positions, y_values, label='History', color='tab:blue', lw=1.0)
        
        # Map index labels to positions for tick labels
        display_tick_index = data.index
        if display_index is not None:
            try:
                candidate_display_index = pd.Index(list(display_index))
                if len(candidate_display_index) == len(data):
                    display_tick_index = candidate_display_index
            except Exception:
                pass
        tick_positions, tick_labels = _sample_axis_tick_labels(display_tick_index, max_tick_labels=8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=25, ha='right', fontsize=8)
        
        # Plot anomalies at correct numeric positions
        if anomalies_idx is not None and len(anomalies_idx):
            try:
                an_display = _cap_anomalies_for_display(pd.Index(anomalies_idx), max_points=max_anomaly_markers)
                an_positions = _anomaly_positions_for_index(data.index, an_display)
                if an_positions:
                    an_pos_arr = np.asarray(an_positions, dtype=np.int64)
                    an_values = pd.to_numeric(
                        data.iloc[an_pos_arr],
                        errors='coerce',
                    ).to_numpy(dtype=float, na_value=np.nan)
                    ax.scatter(an_pos_arr, an_values, color='red', s=2, zorder=5, label='Anomaly')
            except Exception:
                pass
    
    ax.set_title(title, fontsize=10, pad=17)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)
    _apply_dense_non_overlapping_y_ticks(
        ax,
        integer=False,
        label_fontsize=8.0,
        min_ticks=6,
        max_ticks=20,
    )
    
    # Add visual statistics markers on the chart
    try:
        stats_min = float(data.min())
        stats_max = float(data.max())
        stats_mean = float(data.mean())
        stats_median = float(data.median())
        stats_std = float(data.std())
        
        # Draw horizontal lines for Avg and Median
        avg_color = '#a16207'
        med_color = '#6b21a8'
        ax.axhline(y=stats_mean, color=avg_color, linestyle=':', linewidth=1.5, alpha=0.8, label=f'Avg: {_format_stat_value(stats_mean)}')
        ax.axhline(y=stats_median, color=med_color, linestyle='-.', linewidth=1.2, alpha=0.7, label=f'Median: {_format_stat_value(stats_median)}')
        
        # Add value tags - position based on which line is higher
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_offset = (ylim[1] - ylim[0]) * 0.004  # 0.4% offset for tighter Avg/Med tags
        # Position tags so they don't overlap: higher line's tag above it, lower line's tag below it
        if stats_mean >= stats_median:
            # Avg is above Med - Avg tag above its line, Med tag below its line
            ax.text(xlim[1], stats_mean + y_offset, f' Avg: {_format_stat_value(stats_mean)}', va='bottom', ha='left', fontsize=7, color=avg_color, fontweight='bold')
            ax.text(xlim[1], stats_median - y_offset, f' Med: {_format_stat_value(stats_median)}', va='top', ha='left', fontsize=7, color=med_color, fontweight='bold')
        else:
            # Med is above Avg - Med tag above its line, Avg tag below its line
            ax.text(xlim[1], stats_median + y_offset, f' Med: {_format_stat_value(stats_median)}', va='bottom', ha='left', fontsize=7, color=med_color, fontweight='bold')
            ax.text(xlim[1], stats_mean - y_offset, f' Avg: {_format_stat_value(stats_mean)}', va='top', ha='left', fontsize=7, color=avg_color, fontweight='bold')
        
        # Mark the actual Min and Max points on the data with value annotations
        min_color = '#d97706'  # vivid amber - high contrast on blue and white
        max_color = '#16a34a'  # vivid green - high contrast on blue and white
        edge_color = '#0b1220'

        def _annotate_extreme(x_val, y_val, label_text, color, side):
            # Keep labels horizontally next to symbol at the same y-value.
            if side == 'left':
                x_offset_pts = -5
                horizontal_align = 'right'
                # Keep Min label just below the x-axis lane without drifting too low.
                y_offset_pts = -1
                vertical_align = 'top'
            else:
                x_offset_pts = 5
                horizontal_align = 'left'
                y_offset_pts = 0
                vertical_align = 'center'

            ax.annotate(
                label_text,
                (x_val, y_val),
                textcoords='offset points',
                xytext=(x_offset_pts, y_offset_pts),
                ha=horizontal_align,
                va=vertical_align,
                fontsize=7,
                color=color,
                fontweight='semibold',
                annotation_clip=False,
                clip_on=False,
                zorder=12
            )

        if is_datetime:
            min_idx = data.idxmin()
            max_idx = data.idxmax()
            ax.scatter([min_idx], [stats_min], color=min_color, s=30, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {_format_stat_value(stats_min)}', clip_on=False)
            ax.scatter([max_idx], [stats_max], color=max_color, s=30, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {_format_stat_value(stats_max)}', clip_on=False)
            _annotate_extreme(min_idx, stats_min, f'{_format_stat_value(stats_min)}', min_color, 'left')
            _annotate_extreme(max_idx, stats_max, f'{_format_stat_value(stats_max)}', max_color, 'right')
        else:
            finite_mask = np.isfinite(y_values)
            if not bool(finite_mask.any()):
                raise ValueError("No finite data available for min/max annotation")
            min_pos = int(np.nanargmin(y_values))
            max_pos = int(np.nanargmax(y_values))
            min_pos_arr = np.asarray([min_pos], dtype=np.int64)
            max_pos_arr = np.asarray([max_pos], dtype=np.int64)
            ax.scatter(min_pos_arr, np.asarray([stats_min], dtype=float), color=min_color, s=30, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {_format_stat_value(stats_min)}', clip_on=False)
            ax.scatter(max_pos_arr, np.asarray([stats_max], dtype=float), color=max_color, s=30, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {_format_stat_value(stats_max)}', clip_on=False)
            _annotate_extreme(min_pos, stats_min, f'{_format_stat_value(stats_min)}', min_color, 'left')
            _annotate_extreme(max_pos, stats_max, f'{_format_stat_value(stats_max)}', max_color, 'right')
        
        # Std legend entry
        ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {_format_stat_value(stats_std)}')

        # Legend on single line - at the lowest position below x-axis label
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=6, frameon=False, columnspacing=0.6, handletextpad=0.3)

        with contextlib.suppress(Exception):
            fig.subplots_adjust(bottom=0.25, right=0.972, top=0.895)

        # Std appears in legend only
    except Exception as e:
        app.logger.debug("generate_plot stats overlay skipped for '%s': %s", title, e)
    
    buf = io.BytesIO()
    # PERFORMANCE: Use WebP if available (smaller), fallback to PNG
    fmt = 'webp' if use_webp else 'png'
    try:
        _apply_dense_non_overlapping_y_ticks(
            ax,
            integer=False,
            label_fontsize=8.0,
            min_ticks=6,
            max_ticks=20,
        )
        _apply_sci_formatter(ax, y_threshold=1e3, x_threshold=1e6)
        fig.savefig(buf, format=fmt, bbox_inches='tight', pad_inches=0.02)
    except Exception as e:
        app.logger.debug("generate_plot save as %s failed; falling back to png: %s", fmt, e)
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def generate_forecast_plot(
    history: pd.Series,
    forecast_series: pd.Series | None,
    title: str,
    xlabel: str,
    ylabel: str,
    conf_int: pd.DataFrame | None = None,
    history_tail: int | None = None,
    anomalies_idx: pd.Index | None = None,
    anomalies_score: pd.Series | None = None,
    stats: dict[str, float] | None = None,
    legend_y: float | None = None,
    xlabel_labelpad: float | None = None,
    x_tick_angle_override: int | None = None,
    figsize: tuple[float, float] | None = None,
    display_index: pd.Index | list[object] | tuple[object, ...] | None = None,
):
    """Generate a plot showing historical data and forecast with confidence intervals and anomaly markers.
       If forecast_series is None or empty, only history is shown (0% forecast mode).
    """
    _bind_runtime_globals()
    resolved_figsize = (10.0, 5.2)
    if isinstance(figsize, tuple) and len(figsize) == 2:
        try:
            w = float(figsize[0])
            h = float(figsize[1])
            if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
                resolved_figsize = (w, h)
        except Exception:
            resolved_figsize = (10.0, 5.2)

    fig, ax = plt.subplots(figsize=resolved_figsize)

    history_tail_series = history if not history_tail or history_tail <= 0 else history.tail(history_tail)
    display_history_index = history_tail_series.index
    if display_index is not None:
        try:
            candidate_display_index = pd.Index(list(display_index))
            if len(candidate_display_index) == len(history_tail_series):
                display_history_index = candidate_display_index
            elif len(candidate_display_index) == len(history):
                if len(history_tail_series) < len(history):
                    display_history_index = candidate_display_index[-len(history_tail_series):]
                else:
                    display_history_index = candidate_display_index
        except Exception:
            display_history_index = history_tail_series.index
    
    forecast: pd.Series | None = None
    if isinstance(forecast_series, pd.Series) and not forecast_series.empty:
        forecast = forecast_series

    forecast_index = forecast.index if forecast is not None else pd.Index([])
    anomalies_index = pd.Index(anomalies_idx) if anomalies_idx is not None else pd.Index([])

    # Check if we have a valid forecast
    has_forecast = forecast is not None
    
    # For non-datetime indices, use numeric positions to ensure proper alignment
    has_reliable_index = _is_reliable_timeseries_index(history_tail_series.index)
    use_numeric_x = not has_reliable_index
    
    if use_numeric_x:
        # Use numeric x positions (0, 1, 2...) for plotting
        n_hist = len(history_tail_series)
        hist_x = list(range(n_hist))
        hist_y = history_tail_series.to_numpy(dtype=float, na_value=np.nan)
        
        # Plot history
        ax.plot(hist_x, hist_y, linestyle='-', color='tab:blue', linewidth=1.2, label='History', zorder=2)
        
        # Only plot forecast elements if we have a forecast
        if forecast is not None:
            n_fc = len(forecast)
            fc_x = list(range(n_hist, n_hist + n_fc))  # Continue from where history ends
            fc_y = forecast.to_numpy(dtype=float, na_value=np.nan)
            
            # Plot forecast - prepend last history point for continuity
            ax.plot([n_hist - 1] + fc_x, [float(hist_y[-1])] + list(fc_y),
                    linestyle='-', color='orangered', linewidth=1.2, alpha=0.9, label='Forecast', zorder=3)
            
            # Confidence interval
            if conf_int is not None:
                try:
                    lower = conf_int.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
                    upper = conf_int.iloc[:, 1].to_numpy(dtype=float, na_value=np.nan)
                    ax.fill_between(fc_x, lower, upper, color='orangered', alpha=0.22, label='95% CI', zorder=2)
                except Exception as e:
                    app.logger.debug("generate_forecast_plot numeric CI skipped for '%s': %s", title, e)
            
            # Forecast start line
            ax.axvline(n_hist - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
            ax.axvspan(n_hist - 0.5, n_hist + n_fc - 0.5, color='orange', alpha=0.08, zorder=0)
        
        # Anomaly markers (shown regardless of forecast)
        if not anomalies_index.empty:
            try:
                # Anomalies are already capped by the caller — use as-is
                an_display = anomalies_index
                an_positions = _anomaly_positions_for_index(history_tail_series.index, an_display)
                if an_positions:
                    an_pos_arr = np.asarray(an_positions, dtype=np.int64)
                    an_values = history_tail_series.iloc[an_pos_arr].to_numpy(dtype=float, na_value=np.nan)
                    ax.scatter(an_pos_arr, an_values, color='red', s=4, zorder=5,
                              label='Anomaly', marker='o', edgecolors='darkred', linewidths=0.6)
            except Exception as e:
                app.logger.warning(f"Could not plot anomalies: {e}")

        # Set Y limits based on data
        try:
            all_y = list(hist_y)
            if has_forecast:
                all_y += list(fc_y)
            y_min, y_max = min(all_y), max(all_y)

            # Keep full-series extrema visible when stats are provided.
            if isinstance(stats, dict):
                try:
                    smin = float(stats.get('min', np.nan))
                    smax = float(stats.get('max', np.nan))
                    if np.isfinite(smin):
                        y_min = min(y_min, smin)
                    if np.isfinite(smax):
                        y_max = max(y_max, smax)
                except Exception:
                    pass

            if np.isfinite(y_min) and np.isfinite(y_max):
                if y_max > y_min:
                    pad_top = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min, y_max + pad_top)
                else:
                    flat_pad = max(1e-6, abs(y_min) * 0.05, 1.0)
                    ax.set_ylim(y_min - (flat_pad * 0.02), y_max + flat_pad)
        except Exception as e:
            app.logger.debug("generate_forecast_plot numeric y-limits skipped for '%s': %s", title, e)
            
        # Set x-ticks to match the original index labels (history + forecast)
        forecast_tick_policy = _resolve_static_tick_policy(
            list(display_history_index) + list(forecast_index),
            chart_type='forecast',
        )

        try:
            # Combine history and forecast indices
            tick_positions, tick_labels = _build_non_timeseries_tick_labels(
                display_history_index,
                forecast_index,
                max_tick_labels=int(forecast_tick_policy['max_tick_labels']),
            )

            tick_fontsize = float(forecast_tick_policy['tick_fontsize'])
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                tick_labels,
                rotation=int(forecast_tick_policy['tick_angle']),
                ha=str(forecast_tick_policy['tick_ha']),
                fontsize=tick_fontsize,
            )
        except Exception as e:
            app.logger.debug("generate_forecast_plot numeric x-ticks skipped for '%s': %s", title, e)
        
    else:
        # Original datetime-based plotting
        ax.plot(history_tail_series.index, history_tail_series.to_numpy(dtype=float, na_value=np.nan),
                linestyle='-', color='tab:blue', linewidth=1.2, label='History', zorder=2)
        
        # Only plot forecast elements if we have a forecast
        if forecast is not None:
            # Forecast with continuity
            try:
                last_x = history_tail_series.index[-1]
                last_y = float(history_tail_series.iloc[-1])
                x_plot = [last_x] + list(forecast_index)
                y_plot = [last_y] + list(forecast.to_numpy(dtype=float, na_value=np.nan))
            except Exception:
                x_plot = list(forecast_index)
                y_plot = list(forecast.to_numpy(dtype=float, na_value=np.nan))
            
            ax.plot(x_plot, y_plot, linestyle='-', color='orangered', linewidth=1.2, alpha=0.9, label='Forecast', zorder=3)
            
            if conf_int is not None:
                try:
                    lower = conf_int.iloc[:, 0]
                    upper = conf_int.iloc[:, 1]
                    lower.index = forecast_index
                    upper.index = forecast_index
                    ax.fill_between(forecast_index, lower, upper, color='orangered', alpha=0.22, label='95% CI', zorder=2)
                except Exception as e:
                    app.logger.debug("generate_forecast_plot datetime CI skipped for '%s': %s", title, e)
            
            try:
                split_x = history.index[-1]
                ax.axvline(split_x, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
                ax.axvspan(split_x, forecast_index[-1], color='orange', alpha=0.08, zorder=0)
            except Exception as e:
                app.logger.debug("generate_forecast_plot split marker skipped for '%s': %s", title, e)
        
        # Add anomaly markers if provided (shown regardless of forecast)
        if not anomalies_index.empty:
            try:
                # Anomalies are already capped by the caller — use as-is
                an_display = anomalies_index
                an_positions = _anomaly_positions_for_index(history_tail_series.index, an_display)
                if an_positions:
                    an_pos_arr = np.asarray(an_positions, dtype=np.int64)
                    aligned_anomalies = history_tail_series.iloc[an_pos_arr]
                    ax.scatter(
                        aligned_anomalies.index,
                        aligned_anomalies.to_numpy(dtype=float, na_value=np.nan),
                        color='red',
                        s=4,
                        zorder=5,
                        label='Anomaly', marker='o', edgecolors='darkred', linewidths=0.6)
            except Exception as e:
                app.logger.warning(f"Could not plot anomalies: {e}")
        
        try:
            if has_forecast and forecast is not None:
                y_stack = pd.concat([history_tail_series, forecast]).astype(float)
            else:
                y_stack = history_tail_series.astype(float)
            y_stack_values = y_stack.to_numpy(dtype=float, na_value=np.nan)
            y_min = float(np.nanmin(y_stack_values))
            y_max = float(np.nanmax(y_stack_values))

            # Keep full-series extrema visible when stats are provided.
            if isinstance(stats, dict):
                try:
                    smin = float(stats.get('min', np.nan))
                    smax = float(stats.get('max', np.nan))
                    if np.isfinite(smin):
                        y_min = min(y_min, smin)
                    if np.isfinite(smax):
                        y_max = max(y_max, smax)
                except Exception:
                    pass

            if np.isfinite(y_min) and np.isfinite(y_max):
                if y_max > y_min:
                    pad_top = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min, y_max + pad_top)
                else:
                    flat_pad = max(1e-6, abs(y_min) * 0.05, 1.0)
                    ax.set_ylim(y_min - (flat_pad * 0.02), y_max + flat_pad)
        except Exception as e:
            app.logger.debug("generate_forecast_plot datetime y-limits skipped for '%s': %s", title, e)

    ax.set_title(title, pad=17)
    # Use a sensible x-axis label depending on index type
    try:
        label_pad = 2 if xlabel_labelpad is None else xlabel_labelpad
        display_axis_name = ''
        if display_index is not None:
            with contextlib.suppress(Exception):
                display_idx = display_index if isinstance(display_index, pd.Index) else pd.Index(list(display_index))
                display_axis_name = str(display_idx.name or '').strip()
                if display_axis_name.lower() in {'index', 'unnamed: 0', 'unnamed'}:
                    display_axis_name = ''

        raw_xlabel = str(xlabel or '').strip()
        raw_xlabel_lower = raw_xlabel.lower()
        generic_labels = {'', 'timestamp', 'time', 'date', 'index'}

        if has_reliable_index:
            if raw_xlabel_lower not in generic_labels:
                resolved_xlabel = raw_xlabel
            elif display_axis_name:
                resolved_xlabel = display_axis_name
            else:
                resolved_xlabel = 'Timestamp'
        else:
            if raw_xlabel_lower not in generic_labels:
                resolved_xlabel = raw_xlabel
            elif display_axis_name:
                resolved_xlabel = display_axis_name
            else:
                resolved_xlabel = 'Index'
        ax.set_xlabel(resolved_xlabel, labelpad=label_pad)
    except Exception:
        label_pad = 2 if xlabel_labelpad is None else xlabel_labelpad
        ax.set_xlabel(xlabel, labelpad=label_pad)
    ax.set_ylabel(ylabel, labelpad=9)
    _apply_dense_non_overlapping_y_ticks(
        ax,
        integer=False,
        label_fontsize=8.0,
        min_ticks=6,
        max_ticks=20,
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Keep headroom/side room so value tags can stay fully visible.
    with contextlib.suppress(Exception):
        ax.margins(x=0.06)
    
    # Improve X-axis readability
    try:
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]
        final_tick_policy = _resolve_static_tick_policy(tick_labels, chart_type='forecast')
        tick_fontsize = float(final_tick_policy['tick_fontsize'])
        tick_angle = int(final_tick_policy['tick_angle'])
        tick_ha = str(final_tick_policy['tick_ha'])
        if x_tick_angle_override is not None:
            tick_angle = int(x_tick_angle_override)
            tick_ha = 'left' if tick_angle != 0 else 'center'
        # Always rotate visible labels for better readability
        # Use small font to fit more labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=tick_angle,
            ha=tick_ha,
            fontsize=tick_fontsize,
        )
        
        # Apply tight layout to prevent cutoff
        fig.tight_layout()
    except Exception:
        pass
    
    # Add visual statistics markers on the chart (use FULL history stats for consistency with distribution)
    try:
        if stats:
            hist_min = float(stats.get('min', np.nan))
            hist_max = float(stats.get('max', np.nan))
            hist_mean = float(stats.get('mean', np.nan))
            hist_median = float(stats.get('median', np.nan))
            hist_std = float(stats.get('std', np.nan))
        else:
            hist_vals = history.astype(float)  # Use full history, not just tail
            hist_min = float(hist_vals.min())
            hist_max = float(hist_vals.max())
            hist_mean = float(hist_vals.mean())
            hist_median = float(hist_vals.median())
            hist_std = float(hist_vals.std())

        if not all(np.isfinite(v) for v in [hist_min, hist_max, hist_mean, hist_median, hist_std]):
            raise ValueError("Non-finite stats")
        
        # Draw horizontal lines for Avg and Median
        avg_color = '#a16207'
        med_color = '#6b21a8'
        ax.axhline(y=hist_mean, color=avg_color, linestyle=':', linewidth=1.5, alpha=0.7, label=f'Avg: {_format_stat_value(hist_mean)}')
        ax.axhline(y=hist_median, color=med_color, linestyle='-.', linewidth=1.2, alpha=0.6, label=f'Median: {_format_stat_value(hist_median)}')
        
        # Put Avg/Med labels next to their horizontal lines (line-relative placement).
        ylim = ax.get_ylim()
        y_offset = (ylim[1] - ylim[0]) * 0.004
        yaxis_transform = blended_transform_factory(ax.transAxes, ax.transData)
        stats_lane_x = 1.004
        if hist_mean >= hist_median:
            ax.text(stats_lane_x, hist_mean + y_offset, f'Avg: {_format_stat_value(hist_mean)}', transform=yaxis_transform,
                va='bottom', ha='left', fontsize=7, color=avg_color, fontweight='bold', clip_on=False)
            ax.text(stats_lane_x, hist_median - y_offset, f'Med: {_format_stat_value(hist_median)}', transform=yaxis_transform,
                va='top', ha='left', fontsize=7, color=med_color, fontweight='bold', clip_on=False)
        else:
            ax.text(stats_lane_x, hist_mean - y_offset, f'Avg: {_format_stat_value(hist_mean)}', transform=yaxis_transform,
                va='top', ha='left', fontsize=7, color=avg_color, fontweight='bold', clip_on=False)
            ax.text(stats_lane_x, hist_median + y_offset, f'Med: {_format_stat_value(hist_median)}', transform=yaxis_transform,
                va='bottom', ha='left', fontsize=7, color=med_color, fontweight='bold', clip_on=False)
        
        # Add Min/Max markers - find positions in visible data closest to the global min/max
        # Use FULL HISTORY stats (hist_min, hist_max) for consistency with distribution
        tail_vals = history_tail_series.astype(float)
        tail_values = tail_vals.to_numpy(dtype=float, na_value=np.nan)
        
        # Find positions in visible data where values are closest to global min/max
        if use_numeric_x:
            tail_min_pos = int(np.nanargmin(tail_values))
            tail_max_pos = int(np.nanargmax(tail_values))
        else:
            tail_min_pos = tail_vals.idxmin()
            tail_max_pos = tail_vals.idxmax()

        def _annotate_extreme(x_val, y_val, label_text, color, side):
            # Keep labels horizontally next to symbol at the same y-value.
            if side == 'left':
                x_offset_pts = -5
                horizontal_align = 'right'
                y_offset_pts = -1
                vertical_align = 'top'
            else:
                x_offset_pts = 5
                horizontal_align = 'left'
                y_offset_pts = 0
                vertical_align = 'center'

            ax.annotate(
                label_text,
                (x_val, y_val),
                textcoords='offset points',
                xytext=(x_offset_pts, y_offset_pts),
                ha=horizontal_align,
                va=vertical_align,
                fontsize=7,
                color=color,
                fontweight='semibold',
                annotation_clip=False,
                clip_on=False,
                zorder=12
            )

        # Use global min/max values (from full history) for markers and annotations
        min_color = '#d97706'  # vivid amber - high contrast on blue and white
        max_color = '#16a34a'  # vivid green - high contrast on blue and white
        edge_color = '#0b1220'
        # Plot Min marker
        ax.scatter(
            [tail_min_pos],
            [hist_min],
            color=min_color,
            s=30,
            zorder=10,
            marker='v',
            edgecolors=edge_color,
            linewidths=1.5,
            label=f'Min: {_format_stat_value(hist_min)}',
            clip_on=False,
        )
        _annotate_extreme(tail_min_pos, hist_min, f'{_format_stat_value(hist_min)}', min_color, 'left')
        
        # Plot Max marker
        ax.scatter(
            [tail_max_pos],
            [hist_max],
            color=max_color,
            s=30,
            zorder=10,
            marker='^',
            edgecolors=edge_color,
            linewidths=1.5,
            label=f'Max: {_format_stat_value(hist_max)}',
            clip_on=False,
        )
        _annotate_extreme(tail_max_pos, hist_max, f'{_format_stat_value(hist_max)}', max_color, 'right')
        
        # Std legend entry
        ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {_format_stat_value(hist_std)}')

        # Reserve space for the right-side Avg/Med label lane.
        with contextlib.suppress(Exception):
            fig.subplots_adjust(right=0.93)

        # Keep x-axis title close to the legend lane while preserving readability.
        axis_title_y = -0.10
        with contextlib.suppress(Exception):
            ax.xaxis.set_label_coords(0.5, axis_title_y)

        # Keep legend close to axis title with a hard no-overlap minimum lane gap.
        default_legend_y = -0.16
        legend_anchor_requested = default_legend_y if legend_y is None else float(legend_y)
        legend_anchor = max(-0.20, min(legend_anchor_requested, axis_title_y - 0.03))
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, legend_anchor), ncol=12, frameon=False, columnspacing=0.45, handletextpad=0.25)

        with contextlib.suppress(Exception):
            tick_rotations = [abs(float(tick.get_rotation() or 0.0)) for tick in ax.get_xticklabels()]
            max_rotation = max(tick_rotations, default=0.0)
            if max_rotation >= 30:
                bottom_margin = 0.31
            elif max_rotation >= 18:
                bottom_margin = 0.28
            else:
                bottom_margin = 0.245
            fig.subplots_adjust(bottom=bottom_margin, right=0.93, top=0.915)
        
        # Std appears in legend only
    except Exception as e:
        app.logger.debug("generate_forecast_plot stats overlay skipped for '%s': %s", title, e)

    buf = io.BytesIO()
    _apply_dense_non_overlapping_y_ticks(
        ax,
        integer=False,
        label_fontsize=8.0,
        min_ticks=6,
        max_ticks=20,
    )
    _apply_sci_formatter(ax, y_threshold=1e3, x_threshold=1e6)
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _build_category_plotly_chart(s_cat: pd.Series, col: str) -> dict[str, object] | None:
    """Build Plotly traces/layout for a categorical bar chart with Avg/Med annotations.

    Args:
        s_cat: Series of categorical values.
        col: Column name for labels.

    Returns:
        Plotly chart data dict or None when insufficient categories.
    """
    _bind_runtime_globals()
    # Drop missing values before string conversion so NaN/NaT do not become literal labels.
    s_cat = s_cat.dropna().astype(str)
    if len(s_cat) < 3:
        return None

    all_counts = s_cat.value_counts()
    if len(all_counts) < 2:
        return None

    total_unique = len(all_counts)
    max_count = int(all_counts.max())
    min_count = int(all_counts.min())
    avg_count = float(all_counts.mean())
    med_count = float(all_counts.median())
    most_freq = str(all_counts.index[0])
    least_freq = str(all_counts.index[-1]) if len(all_counts) > 0 else "N/A"
    most_label = f"Most: '{_wrap_category_legend_label(most_freq, width=40, max_lines=3, html_breaks=True)}' ({max_count})"
    least_label = f"Least: '{_wrap_category_legend_label(least_freq, width=40, max_lines=3, html_breaks=True)}' ({min_count})"
    annotation_offset = max_count * 0.015 if abs(avg_count - med_count) < max_count * 0.05 else 0.0
    avg_annotation_above = avg_count >= med_count

    chart_title = f"Categories: {col} ({total_unique} unique values)"

    x_values = [str(x) for x in all_counts.index.tolist()]
    y_values = [int(float(y)) for y in all_counts.to_numpy(dtype=float).tolist()]

    show_all_tick_labels = len(x_values) <= 220
    if show_all_tick_labels:
        tick_indices = list(range(len(x_values)))
    else:
        max_label_length_overall = max((len(x) for x in x_values), default=1)
        max_tick_labels = max(5, int(150 / max_label_length_overall))
        tick_step = max(1, int(math.ceil(len(x_values) / max_tick_labels)))
        tick_indices = list(range(0, len(x_values), tick_step))
        if tick_indices and tick_indices[-1] != len(x_values) - 1:
            tick_indices.append(len(x_values) - 1)
        elif not tick_indices:
            tick_indices = [0]

    tickvals = [x_values[idx] for idx in tick_indices]
    ticktext = [x_values[idx] for idx in tick_indices]
    max_tick_line_length = max((len(label) for label in ticktext), default=0)
    long_label_scale = min(1.0, max_tick_line_length / 48.0)
    label_density_scale = min(1.0, len(ticktext) / 180.0)
    legend_names = [
        "Count",
        f"Avg: {_format_stat_value(avg_count)}",
        f"Med: {_format_stat_value(med_count)}",
        most_label.replace("<br>", " "),
        least_label.replace("<br>", " "),
    ]
    legend_font_size = 12
    legend_columns = 5
    legend_rows = int(math.ceil(max(1, len(legend_names)) / max(1, legend_columns)))
    tick_font_size = 6 if len(ticktext) > 180 else 6.5 if len(ticktext) > 120 else 7.5 if len(ticktext) > 70 else 10
    bottom_margin = int(round(min(166, 96 + max_tick_line_length * 0.46 + label_density_scale * 14 + max(0, legend_rows - 1) * 14)))
    legend_y = round(max(-0.54, -0.43 - long_label_scale * 0.012 - label_density_scale * 0.012 - max(0, legend_rows - 1) * 0.024), 3)
    title_standoff = 0
    chart_height = int(round(1325 + long_label_scale * 14 + label_density_scale * 18 + max(0, legend_rows - 1) * 8))
    plot_area_px = max(340, chart_height - bottom_margin - 120)
    y_tick_count = max(9, min(26, int(math.floor(plot_area_px / 38.0))))
    show_bar_labels = len(x_values) <= 120

    bar_trace = {
        "type": "bar",
        "name": "Count",
        "x": x_values,
        "y": y_values,
        "cliponaxis": False,
        "marker": {"color": "rgb(46, 204, 113)", "opacity": 0.7, "line": {"color": "black", "width": 0.5}},
        "hovertemplate": "%{x}<br>Count: %{y}<extra></extra>"
    }
    if show_bar_labels:
        bar_trace.update({
            "text": y_values,
            "textposition": "outside",
            "textfont": {"size": 9},
        })

    avg_trace = {
        "type": "scatter",
        "mode": "lines",
        "name": f"Avg: {_format_stat_value(avg_count)}",
        "x": [None],
        "y": [None],
        "line": {"color": "#f39c12", "width": 2, "dash": "dot"},
        "showlegend": True,
        "meta": "avg-control",
        "hoverinfo": "skip"
    }

    med_trace = {
        "type": "scatter",
        "mode": "lines",
        "name": f"Med: {_format_stat_value(med_count)}",
        "x": [None],
        "y": [None],
        "line": {"color": "#9b59b6", "width": 2, "dash": "dashdot"},
        "showlegend": True,
        "meta": "med-control",
        "hoverinfo": "skip"
    }

    traces = [bar_trace, avg_trace, med_trace]

    layout = {
        "title": {
            "text": chart_title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.99,
            "yanchor": "top",
            "pad": {"t": 6},
            "font": {"color": "#e0e0e0"},
        },
        "xaxis": {
            "title": {"text": col, "standoff": title_standoff, "font": {"color": "#c0c0c0", "size": 17}},
            "type": "category",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [-0.5, max(-0.5, len(x_values) - 0.5)],
            "autorange": False,
            "tickangle": 0,
            "tickfont": {"size": tick_font_size, "color": "#b0b0b0"},
            "titlefont": {"color": "#c0c0c0"},
            "automargin": True,
        },
        "yaxis": {
            "title": {"text": "Count", "standoff": 12},
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.3)",
            "tickfont": {"color": "#b0b0b0"},
            "titlefont": {"color": "#c0c0c0"},
            "nticks": y_tick_count,
            "tickmode": "auto",
            "automargin": True,
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": legend_y,
            "yanchor": "top",
            "font": {"color": "#d0d0d0", "size": legend_font_size},
            "tracegroupgap": 6,
            "bgcolor": "rgba(0,0,0,0)"
        },
        "margin": {"l": 62, "r": 30, "t": 42, "b": bottom_margin},
        "height": chart_height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#d0d0d0"},
        "hoverlabel": {"bgcolor": "#1e1e1e", "font": {"color": "#e0e0e0"}, "bordercolor": "#4a4a4a", "namelength": -1},
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": avg_count,
                "y1": avg_count,
                "line": {"color": "#f39c12", "width": 2, "dash": "dot"},
                "name": "avg-shape"
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": med_count,
                "y1": med_count,
                "line": {"color": "#9b59b6", "width": 2, "dash": "dashdot"},
                "name": "med-shape"
            }
        ],
        "annotations": [
            {
                "x": 0.995,
                "y": avg_count + (annotation_offset if avg_annotation_above else -annotation_offset),
                "xref": "paper",
                "yref": "y",
                "text": f"Avg: {_format_stat_value(avg_count)}",
                "showarrow": False,
                "font": {"size": 10, "color": "#f39c12"},
                "xanchor": "right",
                "yanchor": "bottom" if avg_annotation_above else "top",
                "align": "right",
                "name": "avg-annot"
            },
            {
                "x": 0.995,
                "y": med_count + (annotation_offset if not avg_annotation_above else -annotation_offset),
                "xref": "paper",
                "yref": "y",
                "text": f"Med: {_format_stat_value(med_count)}",
                "showarrow": False,
                "font": {"size": 10, "color": "#9b59b6"},
                "xanchor": "right",
                "yanchor": "bottom" if not avg_annotation_above else "top",
                "align": "right",
                "name": "med-annot"
            }
        ]
    }

    traces.append({
        "type": "scatter",
        "mode": "markers",
        "name": most_label,
        "x": [None],
        "y": [None],
        "marker": {"color": "#27ae60", "symbol": "triangle-up", "size": 8},
        "showlegend": True
    })
    traces.append({
        "type": "scatter",
        "mode": "markers",
        "name": least_label,
        "x": [None],
        "y": [None],
        "marker": {"color": "#e74c3c", "symbol": "triangle-down", "size": 8},
        "showlegend": True
    })

    return {"traces": traces, "layout": layout}


def generate_correlation_heatmap(
    df,
    method='spearman',
    title='Correlation Heatmap',
    *,
    layout_preset: Literal['default', 'export'] = 'default',
):
    """Generate a correlation heatmap as base64 image."""
    rt = _bind_runtime_globals()
    fig = None
    try:
        import base64
        import io

        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
        except Exception:
            sns = None
        
        # Get numeric columns
        df_num = rt.coerce_numeric_df(df)
        sel = df_num.select_dtypes(include='number')
        if sel.empty:
            return None
        
        # Remove constant columns
        nunique = sel.nunique(dropna=True)
        sel = sel.loc[:, nunique > 1]
        
        if sel.shape[1] < 2:
            return None
        
        # Compute correlation
        corr_method: Literal['pearson', 'kendall', 'spearman']
        if method in ('pearson', 'kendall', 'spearman'):
            corr_method = cast(Literal['pearson', 'kendall', 'spearman'], method)
        else:
            corr_method = 'spearman'
        corr = sel.corr(method=corr_method)
        n_cols = len(corr.columns)
        
        # Dynamic sizing
        preset = str(layout_preset).strip().lower()
        export_layout = preset == 'export'
        figsize_dim = max(10.8, n_cols * 0.62) if export_layout else max(10, n_cols * 0.6)
        fontsize = max(6, min(10, 150 / n_cols))
        corr_labels = [_stringify_axis_label(label) for label in corr.columns.tolist()]
        max_corr_label_len = max((len(label) for label in corr_labels), default=0)
        if export_layout:
            if n_cols <= 6 and max_corr_label_len <= 16:
                corr_x_tick_angle = 0
            elif n_cols <= 10 and max_corr_label_len <= 24:
                corr_x_tick_angle = -12
            elif n_cols <= 16 and max_corr_label_len <= 30:
                corr_x_tick_angle = -18
            else:
                corr_x_tick_angle = -24
        else:
            corr_x_tick_angle = -32
        corr_x_tick_ha = 'center' if corr_x_tick_angle == 0 else 'left'
        
        # Create heatmap
        fig_height = max(7.6, figsize_dim * 0.75) if export_layout else max(6.0, figsize_dim * 0.62)
        fig, ax = plt.subplots(figsize=(figsize_dim, fig_height))
        if sns is not None:
            sns.heatmap(
                corr,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=False,
                linewidths=0.5,
                cbar_kws={"shrink": 0.86 if export_layout else 0.8},
                vmin=-1,
                vmax=1,
                ax=ax,
                annot_kws={"size": fontsize},
            )
            with contextlib.suppress(Exception):
                ax.set_xticklabels(
                    [str(c) for c in corr.columns],
                    rotation=corr_x_tick_angle,
                    ha=corr_x_tick_ha,
                    rotation_mode='anchor',
                    fontsize=max(6, fontsize - 1),
                )
            with contextlib.suppress(Exception):
                ax.set_yticklabels(
                    [str(c) for c in corr.index],
                    fontsize=max(6, fontsize - 1),
                )
            with contextlib.suppress(Exception):
                ax.tick_params(axis='x', pad=1.5)
        else:
            data = corr.to_numpy(dtype=float)
            im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_cols))
            ax.set_xticklabels(
                [str(c) for c in corr.columns],
                rotation=corr_x_tick_angle,
                ha=corr_x_tick_ha,
                rotation_mode='anchor',
                fontsize=max(6, fontsize - 1),
            )
            ax.set_yticklabels([str(c) for c in corr.index], fontsize=max(6, fontsize - 1))
            annotate_cells = n_cols <= 28
            if annotate_cells:
                for r in range(n_cols):
                    for c in range(n_cols):
                        val = data[r, c]
                        color = 'white' if abs(val) > 0.55 else 'black'
                        ax.text(c, r, f"{val:.2f}", ha='center', va='center', fontsize=max(5, fontsize - 2), color=color)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.86 if export_layout else 0.8)
            ax.grid(False)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout(pad=0.55 if export_layout else 0.4)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        with contextlib.suppress(Exception):
            rt.app.logger.debug("generate_correlation_heatmap failed (%s): %s", method, e)
        return None


def get_cached_heatmap(
    filename: str,
    df: pd.DataFrame,
    method: str = 'spearman',
    *,
    layout_preset: Literal['default', 'export'] = 'default',
):
    """Get correlation heatmap from cache or generate and cache it.
    
    Avoids regenerating identical heatmaps for PDF when already generated for web view.
    """
    rt = _bind_runtime_globals()
    logger = rt.app.logger
    cache = rt.HEATMAP_CACHE
    cache_key = (filename, method, str(layout_preset))
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Heatmap cache HIT: %s/%s/%s", filename[:8], method, layout_preset)
        return cached
    logger.debug("Heatmap cache MISS: %s/%s/%s - generating", filename[:8], method, layout_preset)
    img = generate_correlation_heatmap(
        df,
        method=method,
        title=f'{method.capitalize()} Correlation',
        layout_preset=layout_preset,
    )
    if img:
        cache.set(cache_key, img)
    return img


def generate_stl_plot(series: pd.Series, title: str, seasonal_period: int):
    rt = _bind_runtime_globals()
    fig = None
    try:
        import base64
        import io

        import matplotlib.pyplot as plt
        from statsmodels.tsa.seasonal import STL

        s = rt.normalize_timeseries(series)
        if s is None or len(s) < max(28, seasonal_period * 2):
            return None
        res = STL(s.astype(float), period=int(seasonal_period), robust=True).fit()  # robust=True for quality

        # HIGH QUALITY: Larger figure for better image quality
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(s.index, s.values, color='tab:blue', lw=1.2)
        axes[0].set_ylabel("Observed")
        axes[0].grid(True, alpha=0.3)
        _apply_dense_non_overlapping_y_ticks(axes[0], integer=False, label_fontsize=7.0, min_ticks=4, max_ticks=10)
        axes[1].plot(res.trend.index, res.trend.values, color='tab:orange', lw=1.6)
        axes[1].set_ylabel("Trend")
        axes[1].grid(True, alpha=0.3)
        _apply_dense_non_overlapping_y_ticks(axes[1], integer=False, label_fontsize=7.0, min_ticks=4, max_ticks=10)
        axes[2].plot(res.seasonal.index, res.seasonal.values, color='tab:green', lw=1.6)
        axes[2].set_ylabel("Seasonal")
        axes[2].grid(True, alpha=0.3)
        _apply_dense_non_overlapping_y_ticks(axes[2], integer=False, label_fontsize=7.0, min_ticks=4, max_ticks=10)
        axes[3].plot(res.resid.index, res.resid.values, color='tab:red', lw=1.6)
        axes[3].axhline(0, color='gray', ls=':', lw=1)
        axes[3].set_ylabel("Residual")
        axes[3].grid(True, alpha=0.3)
        _apply_dense_non_overlapping_y_ticks(axes[3], integer=False, label_fontsize=7.0, min_ticks=4, max_ticks=10)
        axes[0].set_title(title, pad=12)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img
    except Exception:
        if fig is not None:
            plt.close(fig)
        return None


def get_cached_stl_plot(filename: str, column: str, series: pd.Series, seasonal_period: int):
    """Get STL decomposition plot from cache or generate and cache it.
    
    STL decomposition is computationally expensive. This ensures each unique
    (filename, column, seasonal_period) is only computed once.
    """
    rt = _bind_runtime_globals()
    logger = rt.app.logger
    cache = rt.STL_CACHE
    if seasonal_period is None or seasonal_period < 2:
        return None
    cache_key = (filename, str(column), int(seasonal_period))
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("STL cache HIT: %s/%s", filename[:8], column)
        return cached
    logger.debug("STL cache MISS: %s/%s - generating", filename[:8], column)
    s_norm = rt.normalize_timeseries(series)
    if s_norm is None or len(s_norm) < max(28, seasonal_period * 2):
        return None
    stl_img = generate_stl_plot(s_norm, f"STL decomposition for {column}", seasonal_period=seasonal_period)
    if stl_img:
        cache.set(cache_key, stl_img)
    return stl_img


__all__ = [
    "_cap_anomalies_for_display",
    "_anomaly_positions_for_index",
    "build_distribution_axis_spec",
    "apply_distribution_axis_spec",
    "apply_static_distribution_compact_layout",
    "generate_static_distribution_plot",
    "generate_plot",
    "generate_forecast_plot",
    "_build_category_plotly_chart",
    "generate_correlation_heatmap",
    "get_cached_heatmap",
    "generate_stl_plot",
    "get_cached_stl_plot",
]
