from __future__ import annotations

# ruff: noqa: F821
import base64
import io
from collections import Counter
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_stat_value(v: float) -> str:
    """Return a compact string for a statistic value.

    Values >= 1e9 (1 billion) switch to B/T compact notation so they fit
    inside chart labels without overflowing.
    """
    try:
        value = float(v)
        mag = abs(value)
        if mag >= 1e12:
            return f"{value / 1e12:.3f}T"
        if mag >= 1e9:
            return f"{value / 1e9:.3f}B"
        return f"{value:.2f}"
    except Exception:
        return str(v)


def _apply_sci_formatter(ax) -> None:
    """Apply compact B/T y-axis formatter when values are very large.

    Inspects the current y-axis limits and applies B/T labels when needed.
    """
    try:
        ymin, ymax = ax.get_ylim()
        if max(abs(ymin), abs(ymax)) >= 1e9:
            formatter = mticker.FuncFormatter(lambda val, _pos: _format_stat_value(float(val)))
            ax.yaxis.set_major_formatter(formatter)
    except Exception:
        pass

# Bound at runtime via _bind_runtime_globals().
app: Any = cast(Any, None)

_LOCAL_SYMBOLS = {
    '_LOCAL_SYMBOLS',
    '_bind_runtime_globals',
    '_is_reliable_timeseries_index',
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
) -> str:
    _bind_runtime_globals()
    try:
        max_anomaly_markers = int(app.config.get('ANOMALY_MARKER_CAP', 20))
    except Exception:
        max_anomaly_markers = 20
    # HIGH QUALITY: Larger figure for better image quality
    fig, ax = plt.subplots(figsize=(10, 4))
    
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
        if len(data) > 20:
            # Show fewer tick labels if many points
            step = max(1, len(data) // 8)
            tick_positions = list(range(0, len(data), step))
            tick_labels = [str(data.index[i])[:15] for i in tick_positions]  # Truncate long labels
        else:
            tick_positions = list(x_positions)
            tick_labels = [str(idx)[:15] for idx in data.index]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=7)
        
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
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
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
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6, frameon=False, columnspacing=0.6, handletextpad=0.3)

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
):
    """Generate a plot showing historical data and forecast with confidence intervals and anomaly markers.
       If forecast_series is None or empty, only history is shown (0% forecast mode).
    """
    _bind_runtime_globals()
    fig, ax = plt.subplots(figsize=(10, 4))

    history_tail_series = history if not history_tail or history_tail <= 0 else history.tail(history_tail)
    
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
        non_reliable_dt = isinstance(history_tail_series.index, pd.DatetimeIndex) and not has_reliable_index
        
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
        try:
            # Combine history and forecast indices
            full_index = list(history_tail_series.index)
            if has_forecast:
                full_index.extend(forecast_series.index)
            
            total_points = len(full_index)
            
            # Decide on tick spacing
            if total_points > 20:
                step = max(1, total_points // 10)
                tick_positions = list(range(0, total_points, step))
                tick_labels = [str(i) for i in tick_positions] if non_reliable_dt else [str(full_index[i])[:15] for i in tick_positions]
            else:
                tick_positions = list(range(total_points))
                tick_labels = [str(i) for i in tick_positions] if non_reliable_dt else [str(idx)[:15] for idx in full_index]
                
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
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
        label_pad = 2 if xlabel_labelpad is None else xlabel_labelpad
        if has_reliable_index:
            ax.set_xlabel('Timestamp', labelpad=label_pad)
        else:
            ax.set_xlabel('Index', labelpad=label_pad)
    except Exception:
        label_pad = 2 if xlabel_labelpad is None else xlabel_labelpad
        ax.set_xlabel(xlabel, labelpad=label_pad)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Keep headroom/side room so value tags can stay fully visible.
    try:
        ax.margins(x=0.06)
    except Exception:
        pass
    
    # Improve X-axis readability
    try:
        # Always rotate visible labels for better readability
        # Use small font to fit more labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
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
        try:
            fig.subplots_adjust(right=0.84)
        except Exception:
            pass

        # Legend on single line - below x-axis title (Index)
        legend_anchor = -0.30 if legend_y is None else legend_y
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, legend_anchor), ncol=12, frameon=False, columnspacing=0.5, handletextpad=0.3)
        
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
    s_cat = s_cat.astype(str).dropna()
    if len(s_cat) < 3:
        return None

    all_counts = s_cat.value_counts()
    top_counts = all_counts.head(50)
    if len(top_counts) < 2:
        return None

    total_unique = len(all_counts)
    max_count = int(all_counts.max())
    min_count = int(all_counts.min())
    avg_count = float(all_counts.mean())
    med_count = float(all_counts.median())
    most_freq = str(all_counts.index[0])[:20]
    least_freq = str(all_counts.index[-1])[:20] if len(all_counts) > 0 else "N/A"

    if len(all_counts) > 50:
        chart_title = f"Categories: {col} (Top 50 of {total_unique})"
    else:
        chart_title = f"Categories: {col} ({total_unique} unique)"

    x_values = [str(x) for x in top_counts.index.tolist()]
    y_values = [int(float(y)) for y in top_counts.to_numpy(dtype=float).tolist()]

    bar_trace = {
        "type": "bar",
        "name": "Count",
        "x": x_values,
        "y": y_values,
        "text": y_values,
        "textposition": "outside",
        "textfont": {"size": 9},
        "cliponaxis": False,
        "marker": {"color": "rgb(46, 204, 113)", "opacity": 0.7, "line": {"color": "black", "width": 0.5}},
        "hovertemplate": "%{x}<br>Count: %{y}<extra></extra>"
    }

    avg_trace = {
        "type": "scatter",
        "mode": "lines",
        "name": f"Avg: {avg_count:.1f}",
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
        "name": f"Med: {med_count:.1f}",
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
        "xaxis": {"title": col, "tickangle": -45, "tickfont": {"size": 9, "color": "#b0b0b0"}, "titlefont": {"color": "#c0c0c0"}},
        "yaxis": {"title": "Count", "showgrid": True, "gridcolor": "rgba(128,128,128,0.3)", "tickfont": {"color": "#b0b0b0"}, "titlefont": {"color": "#c0c0c0"}},
        "showlegend": True,
        "legend": {
            "orientation": "v",
            "x": 1.0,
            "xanchor": "right",
            "y": 0.99,
            "yanchor": "top",
            "font": {"color": "#d0d0d0", "size": 10}
        },
        "margin": {"l": 60, "r": 160, "t": 50, "b": 120},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#d0d0d0"},
        "hoverlabel": {"bgcolor": "#1e1e1e", "font": {"color": "#e0e0e0"}, "bordercolor": "#4a4a4a"},
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
                "x": 1.01,
                "y": avg_count + (max_count * 0.015 if abs(avg_count - med_count) < max_count * 0.05 else 0),
                "xref": "paper",
                "yref": "y",
                "text": f"Avg: {avg_count:.1f}",
                "showarrow": False,
                "font": {"size": 10, "color": "#f39c12"},
                "xanchor": "left",
                "name": "avg-annot"
            },
            {
                "x": 1.01,
                "y": med_count - (max_count * 0.015 if abs(avg_count - med_count) < max_count * 0.05 else 0),
                "xref": "paper",
                "yref": "y",
                "text": f"Med: {med_count:.1f}",
                "showarrow": False,
                "font": {"size": 10, "color": "#9b59b6"},
                "xanchor": "left",
                "name": "med-annot"
            }
        ]
    }

    traces.append({
        "type": "scatter",
        "mode": "markers",
        "name": f"Most: '{most_freq}' ({max_count})",
        "x": [None],
        "y": [None],
        "marker": {"color": "#27ae60", "symbol": "triangle-up", "size": 8},
        "showlegend": True
    })
    traces.append({
        "type": "scatter",
        "mode": "markers",
        "name": f"Least: '{least_freq}' ({min_count})",
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
        try:
            rt.app.logger.debug("generate_correlation_heatmap failed (%s): %s", method, e)
        except Exception:
            pass
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

