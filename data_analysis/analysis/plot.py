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
            return _strip_numeric_trailing_zeros(raw) + "K"
        return _format_precise_axis_value(value)
    except Exception:
        return str(v)


def _apply_sci_formatter(ax) -> None:
    """Apply compact M/B/T axis formatter when values exceed one million.

    Inspects both x-axis and y-axis limits and applies the formatter to
    each axis independently when needed.
    """
    _fmt = mticker.FuncFormatter(lambda val, _pos: _format_stat_value(float(val)))

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
        if max(abs(ymin), abs(ymax)) >= 1e6:
            ax.yaxis.set_major_formatter(_fmt)
    except Exception:
        pass
    try:
        x_formatter = ax.xaxis.get_major_formatter()
        if isinstance(x_formatter, mticker.FixedFormatter) or _has_explicit_tick_labels(ax.xaxis):
            raise RuntimeError("skip-fixed-x-formatter")
        xmin, xmax = ax.get_xlim()
        if max(abs(xmin), abs(xmax)) >= 1e6:
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
    """Sample numeric axis ticks while preserving as many unique values as feasible."""
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

    unique_sorted = sorted(set(finite_values))
    safe_max = max(2, int(max_tick_labels))

    if len(unique_sorted) > safe_max:
        sampled_positions = sorted({
            int(round(pos))
            for pos in np.linspace(0, len(unique_sorted) - 1, num=safe_max)
        })
        if sampled_positions[0] != 0:
            sampled_positions.insert(0, 0)
        if sampled_positions[-1] != len(unique_sorted) - 1:
            sampled_positions.append(len(unique_sorted) - 1)
        tick_values = [unique_sorted[pos] for pos in sampled_positions]
    else:
        tick_values = unique_sorted

    if len(tick_values) > 2:
        try:
            ratio = max(0.0, float(min_spacing_ratio))
        except Exception:
            ratio = 0.0
        if ratio > 0:
            data_range = float(unique_sorted[-1] - unique_sorted[0])
            if data_range > 0:
                base_spacing = data_range / max(1, safe_max - 1)
                min_spacing = base_spacing * ratio
                if min_spacing > 0:
                    spaced_values: list[float] = [float(tick_values[0])]
                    for value in tick_values[1:-1]:
                        numeric = float(value)
                        if (numeric - spaced_values[-1]) >= min_spacing:
                            spaced_values.append(numeric)
                    if tick_values[-1] != spaced_values[-1]:
                        spaced_values.append(float(tick_values[-1]))
                    tick_values = spaced_values

    if len(tick_values) > safe_max:
        step = max(1, int(math.ceil(len(tick_values) / safe_max)))
        reduced_tick_values = tick_values[::step]
        if reduced_tick_values[-1] != tick_values[-1]:
            reduced_tick_values.append(tick_values[-1])
        tick_values = reduced_tick_values

    compact_labels = [_format_axis_tick_value(value) for value in tick_values]
    if len(set(compact_labels)) != len(compact_labels):
        return tick_values, [_format_precise_axis_value(value) for value in tick_values]
    return tick_values, compact_labels


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


def _resolve_static_tick_policy(
    labels: Sequence[object] | pd.Index,
    *,
    chart_type: Literal["trend", "forecast", "distribution"] = "forecast",
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
        target_tick_labels = min(16, label_count)
        if label_count <= 8:
            max_tick_labels = max(4, label_count)
        elif label_count <= 16:
            max_tick_labels = min(14, label_count)
        else:
            max_tick_labels = min(12, label_count)
        if max_label_length > 12:
            max_tick_labels = max(6, min(max_tick_labels, 10))
        min_spacing_ratio = 0.34 if (label_count > 16 or max_label_length > 10) else 0.28
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
        ("pdf", "trend"): (10.0, 7.4),
        ("pdf", "forecast"): (10.0, 7.4),
        ("pdf", "distribution"): (10.0, 8.2),
    }
    return size_map.get((str(context), str(chart_kind)), (10.0, 7.2))


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
    most_label = f"Most: '{_wrap_category_legend_label(most_freq, width=44, max_lines=3)}' ({max_count})"
    least_label = f"Least: '{_wrap_category_legend_label(least_freq, width=44, max_lines=3)}' ({min_count})"
    legend_labels = [
        "Count",
        f"Avg: {_format_stat_value(avg_count)}",
        f"Med: {_format_stat_value(med_count)}",
        most_label,
        least_label,
    ]

    category_labels = [_stringify_axis_label(label) for label in all_counts.index.tolist()]
    tick_positions: list[float] = [float(i) for i in range(category_count)]
    tick_labels = category_labels
    max_label_length = max((len(label) for label in tick_labels), default=0)
    long_label_scale = min(1.0, max_label_length / 42.0)
    label_density_scale = min(1.0, category_count / 180.0)
    legend_fontsize = 12
    legend_columns = 5
    legend_rows = int(math.ceil(max(1, len(legend_labels)) / max(1, legend_columns)))
    can_fit_horizontal = category_count * (max_label_length + 1) <= 120
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
    tick_fontsize = 5.0 if category_count > 180 else 5.6 if category_count > 140 else 6.2 if category_count > 100 else 7.0 if category_count > 60 else 8.0
    fig_width = min(48.0, max(16.0, category_count * 0.16))
    fig_height = min(28.0, 18.5 + long_label_scale * 0.55 + label_density_scale * 0.85 + max(0, legend_rows - 1) * 0.24)
    legend_y = (
        -0.092
        - long_label_scale * 0.006
        - label_density_scale * 0.008
        - max(0, legend_rows - 1) * 0.014
        - (0.11 if tick_angle != 0 else 0.0)
    )
    bottom_padding = (
        0.25
        + long_label_scale * 0.045
        + label_density_scale * 0.055
        + max(0, legend_rows - 1) * 0.02
        + (0.34 if tick_angle != 0 else 0.0)
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_positions = np.arange(category_count, dtype=np.int64)
    y_values = [int(float(v)) for v in all_counts.to_numpy(dtype=float).tolist()]
    bar_container = ax.bar(x_positions, y_values, color='tab:green', alpha=0.7, edgecolor='black', label='Count')

    if isinstance(bar_container, BarContainer):
        with contextlib.suppress(Exception):
            tick_positions = [
                float(bar.get_x() + (bar.get_width() / 2.0))
                for bar in bar_container.patches
            ]

    if category_count <= 120:
        with contextlib.suppress(Exception):
            if isinstance(bar_container, BarContainer):
                ax.bar_label(
                    bar_container,
                    labels=[str(v) for v in y_values],
                    padding=2,
                    fontsize=8,
                )

    ax.set_title(f"Categories: {col} ({total_unique} unique values)", fontsize=13)
    ax.set_xlabel(col, fontsize=13, labelpad=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.margins(x=0.01)
    ax.set_ylim(0, max(max_count * 1.06, 1.0))
    ax.set_xlim(-0.5, category_count - 0.5)

    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
    x_tick_pad = 1.0 if tick_angle == 0 else 0.0
    ax.tick_params(axis='x', pad=x_tick_pad, labelsize=tick_fontsize, direction='out')
    if tick_angle != 0:
        with contextlib.suppress(Exception):
            # Keep only a minimal outward offset for rotated labels: enough to
            # preserve anchor readability while avoiding the larger historical
            # chart-to-label gap.
            ax.spines['bottom'].set_position(('outward', 6.0))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(tick_angle)
        tick_label.set_horizontalalignment(tick_ha)
        if tick_angle != 0:
            tick_label.set_rotation_mode('anchor')
            tick_label.set_verticalalignment('top')

    ax.axhline(y=avg_count, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {_format_stat_value(avg_count)}')
    ax.axhline(y=med_count, color='#9b59b6', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Med: {_format_stat_value(med_count)}')

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    text_offset = max((y_range * 0.02) if y_range else 0.0, 0.25)
    if avg_count >= med_count:
        avg_y = avg_count + text_offset
        avg_va = 'bottom'
        med_y = med_count - text_offset
        med_va = 'top'
    else:
        avg_y = avg_count - text_offset
        avg_va = 'top'
        med_y = med_count + text_offset
        med_va = 'bottom'

    ax.text(
        0.995,
        avg_y,
        f'Avg: {_format_stat_value(avg_count)}',
        transform=ax.get_yaxis_transform(),
        va=avg_va,
        ha='right',
        fontsize=8,
        color='#f39c12',
        fontweight='bold',
        clip_on=False,
    )
    ax.text(
        0.995,
        med_y,
        f'Med: {_format_stat_value(med_count)}',
        transform=ax.get_yaxis_transform(),
        va=med_va,
        ha='right',
        fontsize=8,
        color='#9b59b6',
        fontweight='bold',
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
        loc='upper center',
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_columns,
        frameon=False,
        columnspacing=0.35,
        handletextpad=0.20,
    )
    min_bottom = 0.24 if tick_angle == 0 else 0.60
    max_bottom = 0.42 if tick_angle == 0 else 0.82
    fig.subplots_adjust(bottom=min(max_bottom, max(min_bottom, bottom_padding)), right=0.94, top=0.93)
    return fig, ax


def _add_static_distribution_overlays(
    ax: Any,
    values: pd.Series | np.ndarray | list[float] | tuple[float, ...],
    *,
    value_formatter: Callable[[float], str] | None = None,
    legend_fontsize: float = 6,
    legend_columns: int = 6,
    legend_y: float = -0.20,
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

    ax.axvline(
        x=stats_mean,
        color='#f39c12',
        linestyle=':',
        linewidth=2,
        alpha=0.8,
        label=f'Avg: {formatter(stats_mean)}',
    )
    ax.axvline(
        x=stats_median,
        color='#9b59b6',
        linestyle='-.',
        linewidth=1.5,
        alpha=0.7,
        label=f'Med: {formatter(stats_median)}',
    )

    xlim = ax.get_xlim()
    x_range = max(float(xlim[1] - xlim[0]), 1e-9)
    ax.set_xlim(xlim[0] - x_range * 0.03, xlim[1] + x_range * 0.03)
    xlim = ax.get_xlim()
    x_range = max(float(xlim[1] - xlim[0]), 1e-9)

    xaxis_transform = blended_transform_factory(ax.transData, ax.transAxes)
    top_lane = 0.972
    x_offset = x_range * 0.012

    if stats_mean <= stats_median:
        ax.text(
            stats_mean - x_offset,
            top_lane,
            f'Avg: {formatter(stats_mean)}',
            transform=xaxis_transform,
            va='top',
            ha='right',
            fontsize=7,
            color='#f39c12',
            fontweight='bold',
            clip_on=False,
        )
        ax.text(
            stats_median + x_offset,
            top_lane,
            f'Med: {formatter(stats_median)}',
            transform=xaxis_transform,
            va='top',
            ha='left',
            fontsize=7,
            color='#9b59b6',
            fontweight='bold',
            clip_on=False,
        )
    else:
        ax.text(
            stats_median - x_offset,
            top_lane,
            f'Med: {formatter(stats_median)}',
            transform=xaxis_transform,
            va='top',
            ha='right',
            fontsize=7,
            color='#9b59b6',
            fontweight='bold',
            clip_on=False,
        )
        ax.text(
            stats_mean + x_offset,
            top_lane,
            f'Avg: {formatter(stats_mean)}',
            transform=xaxis_transform,
            va='top',
            ha='left',
            fontsize=7,
            color='#f39c12',
            fontweight='bold',
            clip_on=False,
        )

    marker_lane_y = 0.033
    min_color = '#f1c40f'  # Bright yellow
    max_color = '#008fa3'
    edge_color = '#0b1220'
    min_xytext = (-2, 4)
    max_xytext = (2, 4)
    if abs(stats_max - stats_min) <= x_range * 0.04:
        min_xytext = (-2, 6)
        max_xytext = (2, 10)

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
        ha='right',
        va='bottom',
        fontsize=6,
        color=min_color,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.16',
            'facecolor': (1.0, 1.0, 1.0, 0.82),
            'edgecolor': edge_color,
            'linewidth': 0.6,
        },
        annotation_clip=False,
    )
    ax.annotate(
        formatter(stats_max),
        (stats_max, marker_lane_y),
        xycoords=xaxis_transform,
        textcoords='offset points',
        xytext=max_xytext,
        ha='left',
        va='bottom',
        fontsize=6,
        color=max_color,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.16',
            'facecolor': (1.0, 1.0, 1.0, 0.82),
            'edgecolor': edge_color,
            'linewidth': 0.6,
        },
        annotation_clip=False,
    )

    ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {formatter(stats_std)}')
    legend_anchor = max(-0.24, min(float(legend_y), -0.14))
    ax.legend(
        fontsize=legend_fontsize,
        loc='upper center',
        bbox_to_anchor=(0.5, legend_anchor),
        ncol=legend_columns,
        frameon=False,
        columnspacing=0.40,
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
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # Add visual statistics markers on the chart
    try:
        stats_min = float(data.min())
        stats_max = float(data.max())
        stats_mean = float(data.mean())
        stats_median = float(data.median())
        stats_std = float(data.std())
        
        # Draw horizontal lines for Avg and Median
        ax.axhline(y=stats_mean, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Avg: {_format_stat_value(stats_mean)}')
        ax.axhline(y=stats_median, color='#9b59b6', linestyle='-.', linewidth=1.2, alpha=0.7, label=f'Median: {_format_stat_value(stats_median)}')
        
        # Add value tags - position based on which line is higher
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_offset = (ylim[1] - ylim[0]) * 0.004  # 0.4% offset for tighter Avg/Med tags
        # Position tags so they don't overlap: higher line's tag above it, lower line's tag below it
        if stats_mean >= stats_median:
            # Avg is above Med - Avg tag above its line, Med tag below its line
            ax.text(xlim[1], stats_mean + y_offset, f' Avg: {_format_stat_value(stats_mean)}', va='bottom', ha='left', fontsize=7, color='#f39c12', fontweight='bold')
            ax.text(xlim[1], stats_median - y_offset, f' Med: {_format_stat_value(stats_median)}', va='top', ha='left', fontsize=7, color='#9b59b6', fontweight='bold')
        else:
            # Med is above Avg - Med tag above its line, Avg tag below its line
            ax.text(xlim[1], stats_median + y_offset, f' Med: {_format_stat_value(stats_median)}', va='bottom', ha='left', fontsize=7, color='#9b59b6', fontweight='bold')
            ax.text(xlim[1], stats_mean - y_offset, f' Avg: {_format_stat_value(stats_mean)}', va='top', ha='left', fontsize=7, color='#f39c12', fontweight='bold')
        
        # Mark the actual Min and Max points on the data with value annotations
        min_color = '#ff3b30'
        max_color = '#00e5ff'
        edge_color = '#0b1220'

        def _annotate_extreme(x_val, y_val, label_text, color, side):
            # Keep labels horizontally next to symbol at the same y-value.
            if side == 'left':
                x_offset_pts = -5
                horizontal_align = 'right'
            else:
                x_offset_pts = 5
                horizontal_align = 'left'

            ax.annotate(
                label_text,
                (x_val, y_val),
                textcoords='offset points',
                xytext=(x_offset_pts, 0),
                ha=horizontal_align,
                va='center',
                fontsize=7,
                color=color,
                fontweight='bold',
                annotation_clip=False,
                clip_on=False,
                zorder=12
            )

        if is_datetime:
            min_idx = data.idxmin()
            max_idx = data.idxmax()
            ax.scatter([min_idx], [stats_min], color=min_color, s=30, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {_format_stat_value(stats_min)}')
            ax.scatter([max_idx], [stats_max], color=max_color, s=30, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {_format_stat_value(stats_max)}')
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
            ax.scatter(min_pos_arr, np.asarray([stats_min], dtype=float), color=min_color, s=30, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {_format_stat_value(stats_min)}')
            ax.scatter(max_pos_arr, np.asarray([stats_max], dtype=float), color=max_color, s=30, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {_format_stat_value(stats_max)}')
            _annotate_extreme(min_pos, stats_min, f'{_format_stat_value(stats_min)}', min_color, 'left')
            _annotate_extreme(max_pos, stats_max, f'{_format_stat_value(stats_max)}', max_color, 'right')
        
        # Std legend entry
        ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {_format_stat_value(stats_std)}')

        # Legend on single line - at the lowest position below x-axis label
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=6, frameon=False, columnspacing=0.6, handletextpad=0.3)

        with contextlib.suppress(Exception):
            fig.subplots_adjust(bottom=0.36, right=0.98, top=0.90)

        # Std appears in legend only
    except Exception as e:
        app.logger.debug("generate_plot stats overlay skipped for '%s': %s", title, e)
    
    buf = io.BytesIO()
    # PERFORMANCE: Use WebP if available (smaller), fallback to PNG
    fmt = 'webp' if use_webp else 'png'
    try:
        _apply_sci_formatter(ax)
        fig.savefig(buf, format=fmt, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        app.logger.debug("generate_plot save as %s failed; falling back to png: %s", fmt, e)
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def generate_forecast_plot(
    history,
    forecast_series,
    title,
    xlabel,
    ylabel,
    conf_int=None,
    history_tail=None,
    anomalies_idx=None,
    anomalies_score=None,
    stats=None,
    legend_y=None,
    xlabel_labelpad=None,
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
    
    # Check if we have a valid forecast
    has_forecast = forecast_series is not None and len(forecast_series) > 0
    
    # For non-datetime indices, use numeric positions to ensure proper alignment
    has_reliable_index = _is_reliable_timeseries_index(history_tail_series.index)
    use_numeric_x = not has_reliable_index
    
    if use_numeric_x:
        # Use numeric x positions (0, 1, 2...) for plotting
        n_hist = len(history_tail_series)
        hist_x = list(range(n_hist))
        hist_y = history_tail_series.values.astype(float)
        
        # Plot history
        ax.plot(hist_x, hist_y, linestyle='-', color='tab:blue', linewidth=1.2, label='History', zorder=2)
        
        # Only plot forecast elements if we have a forecast
        if has_forecast:
            n_fc = len(forecast_series)
            fc_x = list(range(n_hist, n_hist + n_fc))  # Continue from where history ends
            fc_y = forecast_series.values.astype(float)
            
            # Plot forecast - prepend last history point for continuity
            ax.plot([n_hist - 1] + fc_x, [float(hist_y[-1])] + list(fc_y),
                    linestyle='-', color='orangered', linewidth=1.2, alpha=0.9, label='Forecast', zorder=3)
            
            # Confidence interval
            if conf_int is not None:
                try:
                    lower = conf_int.iloc[:, 0].values.astype(float)
                    upper = conf_int.iloc[:, 1].values.astype(float)
                    ax.fill_between(fc_x, lower, upper, color='orangered', alpha=0.22, label='95% CI', zorder=2)
                except Exception as e:
                    app.logger.debug("generate_forecast_plot numeric CI skipped for '%s': %s", title, e)
            
            # Forecast start line
            ax.axvline(n_hist - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
            ax.axvspan(n_hist - 0.5, n_hist + n_fc - 0.5, color='orange', alpha=0.08, zorder=0)
        
        # Anomaly markers (shown regardless of forecast)
        if anomalies_idx is not None and len(anomalies_idx):
            try:
                # Anomalies are already capped by the caller — use as-is
                an_display = pd.Index(anomalies_idx)
                an_positions = _anomaly_positions_for_index(history_tail_series.index, an_display)
                if an_positions:
                    an_values = history_tail_series.iloc[an_positions].astype(float).values
                    ax.scatter(an_positions, an_values, color='red', s=4, zorder=5,
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

            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
                ax.set_ylim(y_min - pad, y_max + pad)
        except Exception as e:
            app.logger.debug("generate_forecast_plot numeric y-limits skipped for '%s': %s", title, e)
            
        # Set x-ticks to match the original index labels (history + forecast)
        forecast_tick_policy = _resolve_static_tick_policy(
            list(display_history_index) + (list(forecast_series.index) if has_forecast else []),
            chart_type='forecast',
        )

        try:
            # Combine history and forecast indices
            tick_positions, tick_labels = _build_non_timeseries_tick_labels(
                display_history_index,
                forecast_series.index if has_forecast else None,
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
        ax.plot(history_tail_series.index, history_tail_series.values,
                linestyle='-', color='tab:blue', linewidth=1.2, label='History', zorder=2)
        
        # Only plot forecast elements if we have a forecast
        if has_forecast:
            # Forecast with continuity
            try:
                last_x = history_tail_series.index[-1]
                last_y = float(history_tail_series.iloc[-1])
                x_plot = [last_x] + list(forecast_series.index)
                y_plot = [last_y] + list(forecast_series.values.astype(float))
            except Exception:
                x_plot = list(forecast_series.index)
                y_plot = list(forecast_series.values)
            
            ax.plot(x_plot, y_plot, linestyle='-', color='orangered', linewidth=1.2, alpha=0.9, label='Forecast', zorder=3)
            
            if conf_int is not None:
                try:
                    lower = conf_int.iloc[:, 0]
                    upper = conf_int.iloc[:, 1]
                    lower.index = forecast_series.index
                    upper.index = forecast_series.index
                    ax.fill_between(forecast_series.index, lower, upper, color='orangered', alpha=0.22, label='95% CI', zorder=2)
                except Exception as e:
                    app.logger.debug("generate_forecast_plot datetime CI skipped for '%s': %s", title, e)
            
            try:
                split_x = history.index[-1]
                ax.axvline(split_x, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
                ax.axvspan(split_x, forecast_series.index[-1], color='orange', alpha=0.08, zorder=0)
            except Exception as e:
                app.logger.debug("generate_forecast_plot split marker skipped for '%s': %s", title, e)
        
        # Add anomaly markers if provided (shown regardless of forecast)
        if anomalies_idx is not None and len(anomalies_idx):
            try:
                # Anomalies are already capped by the caller — use as-is
                an_display = pd.Index(anomalies_idx)
                an_positions = _anomaly_positions_for_index(history_tail_series.index, an_display)
                if an_positions:
                    aligned_anomalies = history_tail_series.iloc[an_positions]
                    ax.scatter(aligned_anomalies.index, aligned_anomalies.values, color='red', s=4, zorder=5,
                              label='Anomaly', marker='o', edgecolors='darkred', linewidths=0.6)
            except Exception as e:
                app.logger.warning(f"Could not plot anomalies: {e}")
        
        try:
            if has_forecast:
                y_stack = pd.concat([history_tail_series, forecast_series]).astype(float)
            else:
                y_stack = history_tail_series.astype(float)
            y_min = float(np.nanmin(y_stack.values))
            y_max = float(np.nanmax(y_stack.values))

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

            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
                ax.set_ylim(y_min - pad, y_max + pad)
        except Exception as e:
            app.logger.debug("generate_forecast_plot datetime y-limits skipped for '%s': %s", title, e)

    ax.set_title(title)
    # Use a sensible x-axis label depending on index type
    try:
        label_pad = 6 if xlabel_labelpad is None else xlabel_labelpad
        if has_reliable_index:
            resolved_xlabel = xlabel or 'Timestamp'
        else:
            raw_xlabel = str(xlabel or '').strip().lower()
            resolved_xlabel = 'Index' if raw_xlabel in {'', 'timestamp', 'time', 'date'} else str(xlabel)
        ax.set_xlabel(resolved_xlabel, labelpad=label_pad)
    except Exception:
        label_pad = 6 if xlabel_labelpad is None else xlabel_labelpad
        ax.set_xlabel(xlabel, labelpad=label_pad)
    ax.set_ylabel(ylabel)

    with contextlib.suppress(Exception):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, min_n_ticks=6))

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
        # Always rotate visible labels for better readability
        # Use small font to fit more labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=int(final_tick_policy['tick_angle']),
            ha=str(final_tick_policy['tick_ha']),
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
        ax.axhline(y=hist_mean, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Avg: {_format_stat_value(hist_mean)}')
        ax.axhline(y=hist_median, color='#9b59b6', linestyle='-.', linewidth=1.2, alpha=0.6, label=f'Median: {_format_stat_value(hist_median)}')
        
        # Put Avg/Med labels next to their horizontal lines (line-relative placement).
        ylim = ax.get_ylim()
        y_offset = (ylim[1] - ylim[0]) * 0.004
        yaxis_transform = blended_transform_factory(ax.transAxes, ax.transData)
        if hist_mean >= hist_median:
            ax.text(1.01, hist_mean + y_offset, f'Avg: {_format_stat_value(hist_mean)}', transform=yaxis_transform,
                va='bottom', ha='left', fontsize=7, color='#f39c12', fontweight='bold', clip_on=False)
            ax.text(1.01, hist_median - y_offset, f'Med: {_format_stat_value(hist_median)}', transform=yaxis_transform,
                va='top', ha='left', fontsize=7, color='#9b59b6', fontweight='bold', clip_on=False)
        else:
            ax.text(1.01, hist_mean - y_offset, f'Avg: {_format_stat_value(hist_mean)}', transform=yaxis_transform,
                va='top', ha='left', fontsize=7, color='#f39c12', fontweight='bold', clip_on=False)
            ax.text(1.01, hist_median + y_offset, f'Med: {_format_stat_value(hist_median)}', transform=yaxis_transform,
                va='bottom', ha='left', fontsize=7, color='#9b59b6', fontweight='bold', clip_on=False)
        
        # Add Min/Max markers - find positions in visible data closest to the global min/max
        # Use FULL HISTORY stats (hist_min, hist_max) for consistency with distribution
        tail_vals = history_tail_series.astype(float)
        
        # Find positions in visible data where values are closest to global min/max
        if use_numeric_x:
            tail_min_pos = int(tail_vals.values.argmin())
            tail_max_pos = int(tail_vals.values.argmax())
        else:
            tail_min_pos = tail_vals.idxmin()
            tail_max_pos = tail_vals.idxmax()

        def _annotate_extreme(x_val, y_val, label_text, color, side):
            # Keep labels horizontally next to symbol at the same y-value.
            if side == 'left':
                x_offset_pts = -5
                horizontal_align = 'right'
            else:
                x_offset_pts = 5
                horizontal_align = 'left'

            ax.annotate(
                label_text,
                (x_val, y_val),
                textcoords='offset points',
                xytext=(x_offset_pts, 0),
                ha=horizontal_align,
                va='center',
                fontsize=7,
                color=color,
                fontweight='bold',
                annotation_clip=False,
                clip_on=False,
                zorder=12
            )

        # Use global min/max values (from full history) for markers and annotations
        min_color = '#ff3b30'
        max_color = '#00BCD4'  # Cyan - works on both light and dark backgrounds
        edge_color = '#0b1220'
        # Plot Min marker
        ax.scatter([tail_min_pos], [hist_min], color=min_color, s=30, zorder=10, marker='v', 
               edgecolors=edge_color, linewidths=1.5, label=f'Min: {_format_stat_value(hist_min)}')
        _annotate_extreme(tail_min_pos, hist_min, f'{_format_stat_value(hist_min)}', min_color, 'left')
        
        # Plot Max marker
        ax.scatter([tail_max_pos], [hist_max], color=max_color, s=30, zorder=10, marker='^', 
               edgecolors=edge_color, linewidths=1.5, label=f'Max: {_format_stat_value(hist_max)}')
        _annotate_extreme(tail_max_pos, hist_max, f'{_format_stat_value(hist_max)}', max_color, 'right')
        
        # Std legend entry
        ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {_format_stat_value(hist_std)}')

        # Reserve space for the right-side Avg/Med label lane.
        with contextlib.suppress(Exception):
            fig.subplots_adjust(right=0.82)

        # Keep x-axis title above the legend lane for static outputs.
        with contextlib.suppress(Exception):
            ax.xaxis.set_label_coords(0.5, -0.18)

        # Legend on single line - below x-axis title (Index)
        legend_floor = -0.34
        legend_anchor = legend_floor if legend_y is None else max(legend_floor, min(float(legend_y), -0.22))
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, legend_anchor), ncol=12, frameon=False, columnspacing=0.45, handletextpad=0.25)

        with contextlib.suppress(Exception):
            fig.subplots_adjust(bottom=0.42, right=0.82, top=0.92)
        
        # Std appears in legend only
    except Exception as e:
        app.logger.debug("generate_forecast_plot stats overlay skipped for '%s': %s", title, e)

    buf = io.BytesIO()
    _apply_sci_formatter(ax)
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
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
        "title": {"text": chart_title, "x": 0.5, "xanchor": "center", "font": {"color": "#e0e0e0"}},
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
        "yaxis": {"title": "Count", "showgrid": True, "gridcolor": "rgba(128,128,128,0.3)", "tickfont": {"color": "#b0b0b0"}, "titlefont": {"color": "#c0c0c0"}},
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
        "margin": {"l": 60, "r": 28, "t": 40, "b": bottom_margin},
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


def generate_correlation_heatmap(df, method='spearman', title='Correlation Heatmap'):
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
        figsize_dim = max(10, n_cols * 0.6)
        fontsize = max(6, min(10, 150 / n_cols))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(figsize_dim, figsize_dim * 0.8))
        if sns is not None:
            sns.heatmap(
                corr,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                vmin=-1,
                vmax=1,
                ax=ax,
                annot_kws={"size": fontsize},
            )
        else:
            data = corr.to_numpy(dtype=float)
            im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_cols))
            ax.set_xticklabels([str(c) for c in corr.columns], rotation=45, ha='right', fontsize=max(6, fontsize - 1))
            ax.set_yticklabels([str(c) for c in corr.index], fontsize=max(6, fontsize - 1))
            annotate_cells = n_cols <= 28
            if annotate_cells:
                for r in range(n_cols):
                    for c in range(n_cols):
                        val = data[r, c]
                        color = 'white' if abs(val) > 0.55 else 'black'
                        ax.text(c, r, f"{val:.2f}", ha='center', va='center', fontsize=max(5, fontsize - 2), color=color)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
            ax.grid(False)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
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


def get_cached_heatmap(filename: str, df: pd.DataFrame, method: str = 'spearman'):
    """Get correlation heatmap from cache or generate and cache it.
    
    Avoids regenerating identical heatmaps for PDF when already generated for web view.
    """
    rt = _bind_runtime_globals()
    logger = rt.app.logger
    cache = rt.HEATMAP_CACHE
    cache_key = (filename, method)
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Heatmap cache HIT: %s/%s", filename[:8], method)
        return cached
    logger.debug("Heatmap cache MISS: %s/%s - generating", filename[:8], method)
    img = generate_correlation_heatmap(df, method=method, title=f'{method.capitalize()} Correlation')
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
        axes[1].plot(res.trend.index, res.trend.values, color='tab:orange', lw=1.6)
        axes[1].set_ylabel("Trend")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(res.seasonal.index, res.seasonal.values, color='tab:green', lw=1.6)
        axes[2].set_ylabel("Seasonal")
        axes[2].grid(True, alpha=0.3)
        axes[3].plot(res.resid.index, res.resid.values, color='tab:red', lw=1.6)
        axes[3].axhline(0, color='gray', ls=':', lw=1)
        axes[3].set_ylabel("Residual")
        axes[3].grid(True, alpha=0.3)
        axes[0].set_title(title)
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
    "generate_plot",
    "generate_forecast_plot",
    "_build_category_plotly_chart",
    "generate_correlation_heatmap",
    "get_cached_heatmap",
    "generate_stl_plot",
    "get_cached_stl_plot",
]

