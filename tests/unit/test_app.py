
import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import zipfile
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

# Add project root to sys.path to ensure local absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import data_analysis.runtime_app as app_module
from app import app
from data_analysis.runtime_app import (
    AI_DESCRIBE_CACHE,
    DATAFRAME_CACHE,
    INTERACTIVE_DATA_CACHE,
    NUMERIC_DF_CACHE,
    TinyLRU,
    _build_category_plotly_chart,
    _build_interactive_cache_key,
    _build_qna_cache_key,
    _compute_basic_stats,
    _compute_forecast,
    _thin_series_keep_extrema,
    allowed_file,
    convert_html_to_formatted_text,
    describe_for_ai,
    detect_anomalies,
    generate_forecast_plot,
    generate_plot,
    get_cached_anomalies,
    get_cached_numeric_df,
)


def test_allowed_file():
    """Test the allowed_file function."""
    assert allowed_file("data.csv") is True
    assert allowed_file("data.txt") is True
    assert allowed_file("data.xlsx") is True
    assert allowed_file("data.json") is True
    assert allowed_file("image.png") is False
    assert allowed_file("script.py") is False
    assert allowed_file("data") is False


def test_allowed_file_edge_cases():
    """Test allowed_file with None, empty string, and other edge cases."""
    assert allowed_file(None) is False
    assert allowed_file("") is False

def test_app_config():
    """Test basic app configuration."""
    assert app.config['UPLOAD_FOLDER'] == 'datasets'
    assert app.config['SECRET_KEY'] is not None


def test_convert_html_to_formatted_text_preserves_nested_bullets():
        """Nested list items should keep leading indentation for PDF renderer depth parsing."""
        html = """
        <h3>Data Quality Assessment</h3>
        <ul>
            <li>Missing values summary
                <ul>
                    <li>Population: 22.2%</li>
                    <li>GDP: 15.2%</li>
                </ul>
            </li>
        </ul>
        """

        text = convert_html_to_formatted_text(html)
        lines = text.splitlines()

        assert any(line.startswith("- Missing values summary") for line in lines)
        assert any(line.startswith("  - Population: 22.2%") for line in lines)
        assert any(line.startswith("  - GDP: 15.2%") for line in lines)

def test_index_route():
    """Test that the index route returns 200."""
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        # Check for expected content
        assert b"Upload Dataset" in response.data or b"Data Analysis" in response.data


def test_compute_basic_stats():
    """Ensure basic stats match pandas computations and drop NaNs."""
    series = pd.Series([1, 2, 3, 4, 5, None])
    stats = _compute_basic_stats(series)
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == 3.0
    assert stats["std"] == pytest.approx(series.dropna().std())


def test_generate_plot_returns_base64():
    """Ensure generate_plot returns a non-empty base64 string."""
    series = pd.Series([1, 2, 3, 4], index=pd.RangeIndex(4))
    img = generate_plot(series, "Trend", "Index", "Value")
    assert isinstance(img, str)
    assert len(img) > 0


def test_format_stat_value_uses_billion_trillion_suffixes():
    """Numeric labels should use k/M/B/T compact notation with trailing-zero stripping."""
    from data_analysis.analysis.plot import _format_stat_value

    assert _format_stat_value(123_456_000_000_000.0) == "123.456T"
    assert _format_stat_value(1_200_000_000.0) == "1.2B"
    assert _format_stat_value(1_234_560_000.0) == "1.235B"
    assert _format_stat_value(12_340.0) == "12.34k"
    assert _format_stat_value(2016.0) == "2.02k"
    assert _format_stat_value(2000.0) == "2k"
    assert _format_stat_value(1_500_000.0) == "1.5M"
    assert _format_stat_value(8.0) == "8"
    assert _format_stat_value(8.50) == "8.5"
    assert _format_stat_value(1e16) == "1.000e+16"


def test_resolve_static_tick_policy_distribution_prefers_horizontal_labels():
    """Distribution static policy should keep labels horizontal and densely sampled safely."""
    from data_analysis.analysis.plot import _resolve_static_tick_policy

    labels = [str(2000 + i) for i in range(24)]
    policy = _resolve_static_tick_policy(labels, chart_type="distribution")

    assert int(policy["tick_angle"]) == 0
    assert str(policy["tick_ha"]) == "center"
    assert int(policy["max_tick_labels"]) >= 10


def test_generate_forecast_plot_returns_base64():
    """Ensure generate_forecast_plot returns a non-empty base64 string."""
    history_idx = pd.date_range("2024-01-01", periods=5, freq="D")
    history = pd.Series([1, 2, 3, 4, 5], index=history_idx)
    forecast_idx = pd.date_range("2024-01-06", periods=3, freq="D")
    forecast = pd.Series([5.5, 6.0, 6.4], index=forecast_idx)
    img = generate_forecast_plot(history, forecast, "Forecast", "Timestamp", "Value")
    assert isinstance(img, str)
    assert len(img) > 0


def test_generate_forecast_plot_accepts_custom_figsize():
    """Forecast renderer should accept explicit figure-size overrides for export layouts."""
    history_idx = pd.date_range("2024-01-01", periods=8, freq="D")
    history = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=history_idx)
    forecast_idx = pd.date_range("2024-01-09", periods=3, freq="D")
    forecast = pd.Series([8.2, 8.5, 8.7], index=forecast_idx)

    img = cast(Any, generate_forecast_plot)(
        history,
        forecast,
        "Forecast",
        "Timestamp",
        "Value",
        figsize=(9.5, 6.1),
    )

    assert isinstance(img, str)
    assert len(img) > 0


def test_generate_forecast_plot_starts_y_axis_at_data_min(monkeypatch):
    """Forecast/trend y-axis lower bound should start at the true data minimum."""
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    captured_ylim: dict[str, float] = {}

    def _savefig_spy(self, *args, **kwargs):
        if not captured_ylim and self.axes:
            ymin, ymax = self.axes[0].get_ylim()
            captured_ylim["ymin"] = float(ymin)
            captured_ylim["ymax"] = float(ymax)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _savefig_spy)

    history = pd.Series([120.0, 150.0, 132.0, 141.0], index=pd.RangeIndex(4))
    forecast = pd.Series([168.0, 176.0, 210.0], index=pd.RangeIndex(4, 7))

    img = generate_forecast_plot(history, forecast, "Forecast", "Index", "Value")

    assert isinstance(img, str)
    assert len(img) > 0
    assert captured_ylim
    assert captured_ylim["ymin"] == pytest.approx(float(history.min()))
    assert captured_ylim["ymax"] > float(max(history.max(), forecast.max()))


def test_download_full_report_pdf_prefers_named_index_for_trend_forecast_xlabel(monkeypatch):
    """PDF trend/forecast charts should use the resolved index name instead of generic 'Index'."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "9" * 40 + ".csv"
    idx = pd.date_range("2024-01-01", periods=8, freq="D", name="record_date")
    df = pd.DataFrame({"value": np.linspace(10.0, 17.0, 8)}, index=idx)
    df["record_date"] = idx

    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII="
    )
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    with contextlib.suppress(Exception):
        app_module.REPORT_CACHE.clear()

    def fake_get_cached_column_forecast(_filename, _column, series, steps):
        n_steps = int(steps)
        fc_idx = pd.date_range(series.index[-1], periods=n_steps + 1, freq="D", name="record_date")[1:]
        base = float(series.iloc[-1]) if len(series) else 0.0
        fc = pd.Series(np.full(n_steps, base, dtype=float), index=fc_idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=fc_idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    def fake_generate_forecast_plot(history, forecast_series, title, xlabel, ylabel, *args, **kwargs):
        display_index = kwargs.get("display_index")
        captured_calls.append(
            {
                "title": str(title),
                "xlabel": str(xlabel),
                "ylabel": str(ylabel),
                "display_index_name": getattr(display_index, "name", None),
                "has_forecast": forecast_series is not None,
            }
        )
        return base64.b64encode(tiny_png).decode("utf-8")

    monkeypatch.setattr(app_module, "generate_forecast_plot", fake_generate_forecast_plot)
    monkeypatch.setattr(pdf_report_mod, "generate_forecast_plot", fake_generate_forecast_plot)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0.2")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    trend_calls = [call for call in captured_calls if str(call["title"]).startswith("Trend:")]
    forecast_calls = [call for call in captured_calls if str(call["title"]).startswith("Forecast:")]
    assert trend_calls
    assert forecast_calls
    assert all(call["xlabel"] == "record_date" for call in trend_calls + forecast_calls)
    assert all(call["display_index_name"] == "record_date" for call in trend_calls + forecast_calls)


def test_build_static_category_chart_uses_milder_dense_tick_angle():
    """Dense static category exports should avoid near-vertical tick angles and huge heights."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(210, 10, -1),
        index=[f"category_{i:03d}" for i in range(200)],
    )
    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        tick_labels = ax.get_xticklabels()
        rotation = float(tick_labels[0].get_rotation()) if tick_labels else 0.0
        assert rotation >= -55.0
        assert float(fig.get_size_inches()[1]) <= 28.0
    finally:
        plt.close(fig)


def test_build_static_category_chart_rotated_ticks_match_bar_centers():
    """Static categorical exports should keep each rotated tick anchored to its matching bar center."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(220, 120, -1),
        index=[f"region_{i:03d}" for i in range(100)],
    )
    built = _build_static_category_chart(counts, "region")
    assert built is not None
    fig, ax = built
    try:
        bars = list(ax.patches)
        assert len(bars) == len(counts)

        bar_centers = np.asarray([bar.get_x() + (bar.get_width() / 2.0) for bar in bars], dtype=float)
        tick_positions = np.asarray(ax.get_xticks(), dtype=float)
        assert len(tick_positions) == len(counts)
        assert np.allclose(bar_centers, tick_positions, atol=1e-8)

        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert tick_labels[0] == str(counts.index[0])
        assert tick_labels[-1] == str(counts.index[-1])

        first_rotation = float(ax.get_xticklabels()[0].get_rotation()) if ax.get_xticklabels() else 0.0
        if first_rotation != 0.0:
            assert ax.get_xticklabels()[0].get_rotation_mode() == "anchor"
            bottom_spine = ax.spines.get("bottom")
            assert bottom_spine is not None
            pos_kind, pos_value = bottom_spine.get_position()
            assert pos_kind == "outward"
            # Keep the baseline close to bars to avoid visual gap in export charts.
            assert 0.0 <= float(pos_value) <= 1.5
    finally:
        plt.close(fig)


def test_build_static_category_chart_dense_ticks_render_below_axis():
    """Dense category export labels should stay below the axis while matching bar centers."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(340, 190, -1),
        index=[f"very_long_region_name_{i:03d}" for i in range(150)],
    )
    built = _build_static_category_chart(counts, "region")
    assert built is not None
    fig, ax = built
    try:
        bars = list(ax.patches)
        bar_centers = np.asarray([bar.get_x() + (bar.get_width() / 2.0) for bar in bars], dtype=float)
        tick_positions = np.asarray(ax.get_xticks(), dtype=float)
        assert len(tick_positions) == len(bar_centers)
        assert np.allclose(tick_positions, bar_centers, atol=1e-8)

        tick_labels = list(ax.get_xticklabels())
        assert tick_labels
        first_tick = tick_labels[0]
        raw_rotation = float(first_tick.get_rotation())
        signed_rotation = ((raw_rotation + 180.0) % 360.0) - 180.0
        assert signed_rotation < 0.0
        assert first_tick.get_rotation_mode() == "anchor"
        assert first_tick.get_va() == "top"
        # Keep labels anchored below the x-axis baseline (not inside the plotting area).
        assert float(first_tick.get_position()[1]) <= 0.0
    finally:
        plt.close(fig)


def test_build_static_category_chart_legend_renders_below_xaxis_title():
    """Static category exports should keep a visible gap between xlabel and legend lanes."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(160, 40, -1),
        index=[f"category_{i:03d}" for i in range(120)],
    )
    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        fig.canvas.draw()
        legend = ax.get_legend()
        assert legend is not None
        renderer = fig.canvas.get_renderer()
        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
        legend_bbox = legend.get_window_extent(renderer)
        gap_px = float(xlabel_bbox.y0 - legend_bbox.y1)
        assert gap_px >= 8.0
    finally:
        plt.close(fig)


def test_build_static_category_chart_legend_stays_single_line_with_long_labels():
    """Static category legend should remain a single-row line, even for long category names."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(300, 120, -1),
        index=[f"very_long_category_name_for_legend_layout_{i:03d}" for i in range(180)],
    )
    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        fig.canvas.draw()
        legend = ax.get_legend()
        assert legend is not None
        texts = legend.get_texts()
        assert len(texts) >= 5
        assert all("\n" not in str(text.get_text()) for text in texts)

        renderer = fig.canvas.get_renderer()
        centers = [float((box.y0 + box.y1) / 2.0) for box in (text.get_window_extent(renderer) for text in texts)]
        assert max(centers) - min(centers) <= 2.0
    finally:
        plt.close(fig)


def test_build_static_category_chart_keeps_drawable_bar_area_stable_for_dense_labels():
    """Dense category exports should spend extra height on the footer instead of shrinking the bar area."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    sparse_counts = pd.Series(
        np.arange(90, 50, -1),
        index=[f"short_{i:02d}" for i in range(40)],
    )
    dense_counts = pd.Series(
        np.arange(260, 80, -1),
        index=[f"very_long_dense_category_label_{i:03d}" for i in range(180)],
    )

    sparse_built = _build_static_category_chart(sparse_counts, "category")
    dense_built = _build_static_category_chart(dense_counts, "category")
    assert sparse_built is not None
    assert dense_built is not None
    sparse_fig, sparse_ax = sparse_built
    dense_fig, dense_ax = dense_built
    try:
        sparse_fig.canvas.draw()
        dense_fig.canvas.draw()
        sparse_axes_height = float(sparse_ax.bbox.height)
        dense_axes_height = float(dense_ax.bbox.height)
        assert dense_axes_height >= sparse_axes_height - 2.0
    finally:
        plt.close(sparse_fig)
        plt.close(dense_fig)


def test_build_static_category_chart_keeps_footer_fonts_fixed_for_dense_labels():
    """Dense category exports should keep footer/title/stat font sizes unchanged."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    sparse_counts = pd.Series(
        np.arange(100, 60, -1),
        index=[f"short_{i:02d}" for i in range(40)],
    )
    dense_counts = pd.Series(
        np.arange(260, 80, -1),
        index=[f"very_long_dense_category_label_{i:03d}" for i in range(180)],
    )

    sparse_built = _build_static_category_chart(sparse_counts, "category")
    dense_built = _build_static_category_chart(dense_counts, "category")
    assert sparse_built is not None
    assert dense_built is not None
    sparse_fig, sparse_ax = sparse_built
    dense_fig, dense_ax = dense_built

    def _footer_metrics(ax):
        legend = ax.get_legend()
        assert legend is not None
        avg_text = next(text for text in ax.texts if str(text.get_text()).startswith("Avg:"))
        med_text = next(text for text in ax.texts if str(text.get_text()).startswith("Med:"))
        return {
            "title": float(ax.title.get_fontsize()),
            "xlabel": float(ax.xaxis.label.get_fontsize()),
            "legend": float(legend.get_texts()[0].get_fontsize()),
            "avg": float(avg_text.get_fontsize()),
            "med": float(med_text.get_fontsize()),
        }

    try:
        sparse_fig.canvas.draw()
        dense_fig.canvas.draw()
        assert _footer_metrics(dense_ax) == _footer_metrics(sparse_ax)
    finally:
        plt.close(sparse_fig)
        plt.close(dense_fig)


def test_build_static_category_chart_dense_exports_keep_render_height_when_fit_to_page_width():
    """Dense category exports should not collapse visual bar-area height when rendered at fixed page width."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    sparse_counts = pd.Series(
        np.arange(90, 50, -1),
        index=[f"short_{i:02d}" for i in range(40)],
    )
    dense_counts = pd.Series(
        np.arange(320, 140, -1),
        index=[f"very_long_dense_category_label_{i:03d}" for i in range(180)],
    )

    sparse_built = _build_static_category_chart(sparse_counts, "category")
    dense_built = _build_static_category_chart(dense_counts, "category")
    assert sparse_built is not None
    assert dense_built is not None
    sparse_fig, _sparse_ax = sparse_built
    dense_fig, _dense_ax = dense_built

    def _rendered_height_at_fixed_width(fig, width_mm: float = 180.0) -> float:
        w_in, h_in = fig.get_size_inches()
        if w_in <= 0:
            return 0.0
        return float(width_mm * (h_in / w_in))

    try:
        sparse_h = _rendered_height_at_fixed_width(sparse_fig)
        dense_h = _rendered_height_at_fixed_width(dense_fig)
        assert dense_h >= sparse_h * 0.9
    finally:
        plt.close(sparse_fig)
        plt.close(dense_fig)


def test_build_static_category_chart_dense_labels_show_all_tick_text_and_hide_bar_value_tags():
    """Dense exports should show all x tick labels while still hiding overlapping bar-value labels."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(320, 140, -1),
        index=[f"very_long_dense_category_label_{i:03d}" for i in range(180)],
    )

    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        tick_labels = [str(tick.get_text()) for tick in ax.get_xticklabels()]
        visible_labels = [label for label in tick_labels if label.strip()]
        assert visible_labels
        assert len(visible_labels) == len(tick_labels) == len(counts)
        assert tick_labels[0] == str(counts.index[0])
        assert tick_labels[-1] == str(counts.index[-1])

        bar_value_texts = [
            str(text.get_text()).strip()
            for text in ax.texts
            if str(text.get_text()).strip().isdigit()
        ]
        assert not bar_value_texts
    finally:
        plt.close(fig)


def test_build_static_category_chart_uses_uniform_single_green_bar_tone():
    """Static category export bars should keep one uniform green tone with visible separation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgba

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(220, 80, -1),
        index=[f"category_{i:03d}" for i in range(140)],
    )

    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        bars = list(ax.patches)
        assert bars
        face_hex = {to_hex(bar.get_facecolor(), keep_alpha=False).lower() for bar in bars}
        face_alpha = {round(float(to_rgba(bar.get_facecolor())[3]), 3) for bar in bars}
        edge_alpha = {round(float(to_rgba(bar.get_edgecolor())[3]), 3) for bar in bars}
        widths = {round(float(bar.get_width()), 4) for bar in bars}
        width_val = next(iter(widths))

        assert face_hex == {"#2e7d32"}
        assert face_alpha == {1.0}
        assert edge_alpha == {0.0}
        assert 0.80 <= width_val <= 0.86
    finally:
        plt.close(fig)


def test_build_static_category_chart_very_dense_uses_wider_bar_gaps():
    """Very dense category exports should keep consistent separated bar width."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(320, 120, -1),
        index=[f"category_{i:03d}" for i in range(200)],
    )

    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        bars = list(ax.patches)
        assert bars
        widths = {round(float(bar.get_width()), 4) for bar in bars}
        width_val = next(iter(widths))
        assert 0.74 <= width_val <= 0.79

        medium_counts = pd.Series(
            np.arange(220, 80, -1),
            index=[f"mid_{i:03d}" for i in range(140)],
        )
        medium_built = _build_static_category_chart(medium_counts, "category")
        assert medium_built is not None
        medium_fig, medium_ax = medium_built
        try:
            medium_bars = list(medium_ax.patches)
            assert medium_bars
            medium_width_val = float(medium_bars[0].get_width())
            assert width_val < medium_width_val
        finally:
            plt.close(medium_fig)
    finally:
        plt.close(fig)


def test_build_static_category_chart_avg_med_tags_are_right_outside_and_close_to_lines():
    """Avg/Med tags should sit in the right outside lane and stay close to their line values."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        [100, 40, 30, 20, 10],
        index=["A", "B", "C", "D", "E"],
    )

    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        avg_text = next(text for text in ax.texts if str(text.get_text()).startswith("Avg:"))
        med_text = next(text for text in ax.texts if str(text.get_text()).startswith("Med:"))

        avg_x, avg_y = avg_text.get_position()
        med_x, med_y = med_text.get_position()
        assert float(avg_x) > 1.0
        assert float(med_x) > 1.0
        assert str(avg_text.get_ha()) == "left"
        assert str(med_text.get_ha()) == "left"

        avg_count = float(counts.mean())
        med_count = float(counts.median())
        y_range = float(ax.get_ylim()[1] - ax.get_ylim()[0])
        max_allowed_offset = max(y_range * 0.012, 0.16)
        assert abs(float(avg_y) - avg_count) <= max_allowed_offset
        assert abs(float(med_y) - med_count) <= max_allowed_offset
    finally:
        plt.close(fig)


def test_build_static_category_chart_yaxis_ticks_dense_without_overlap():
    """Y-axis should show many labels while preventing overlap in static category exports."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(240, 40, -1),
        index=[f"category_{i:03d}" for i in range(200)],
    )

    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        fig.canvas.draw()
        y_labels = [label for label in ax.get_yticklabels() if str(label.get_text()).strip()]
        assert len(y_labels) >= 8

        renderer = fig.canvas.get_renderer()
        bboxes = sorted((label.get_window_extent(renderer) for label in y_labels), key=lambda box: box.y0)
        for previous, current in zip(bboxes, bboxes[1:]):
            assert float(current.y0) >= float(previous.y1) - 0.5
    finally:
        plt.close(fig)


def test_apply_dense_non_overlapping_y_ticks_balances_density_and_readability():
    """Shared y-axis helper should produce many ticks without overlapping labels."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _apply_dense_non_overlapping_y_ticks

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    try:
        values = np.linspace(0.0, 2500.0, 600)
        ax.plot(np.arange(len(values)), values)
        ax.set_ylim(0.0, 2500.0)

        _apply_dense_non_overlapping_y_ticks(
            ax,
            integer=False,
            label_fontsize=8.0,
            min_ticks=6,
            max_ticks=20,
        )

        fig.canvas.draw()
        y_labels = [label for label in ax.get_yticklabels() if str(label.get_text()).strip()]
        assert len(y_labels) >= 8

        renderer = fig.canvas.get_renderer()
        bboxes = sorted((label.get_window_extent(renderer) for label in y_labels), key=lambda box: box.y0)
        for previous, current in zip(bboxes, bboxes[1:]):
            assert float(current.y0) >= float(previous.y1) - 0.5
    finally:
        plt.close(fig)


def test_generate_plot_uses_dense_non_overlapping_y_ticks(monkeypatch):
    """Trend renderer should maximize y ticks while keeping labels readable."""
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    captured: dict[str, object] = {}

    def _savefig_spy(self, *args, **kwargs):
        if not captured and self.axes:
            ax = self.axes[0]
            self.canvas.draw()
            y_labels = [label for label in ax.get_yticklabels() if str(label.get_text()).strip()]
            renderer = self.canvas.get_renderer()
            bboxes = sorted((label.get_window_extent(renderer) for label in y_labels), key=lambda box: box.y0)
            overlap = any(float(cur.y0) < float(prev.y1) - 0.5 for prev, cur in zip(bboxes, bboxes[1:]))
            captured["count"] = len(y_labels)
            captured["overlap"] = overlap
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _savefig_spy)

    series = pd.Series(np.linspace(0.0, 5000.0, 700), index=pd.RangeIndex(700))
    img = generate_plot(series, "Trend", "Index", "Value")

    assert isinstance(img, str)
    assert len(img) > 0
    assert int(captured.get("count", 0)) >= 8
    assert bool(captured.get("overlap", True)) is False


def test_generate_forecast_plot_uses_dense_non_overlapping_y_ticks(monkeypatch):
    """Forecast renderer should maximize y ticks while avoiding y-label collisions."""
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    captured: dict[str, object] = {}

    def _savefig_spy(self, *args, **kwargs):
        if not captured and self.axes:
            ax = self.axes[0]
            self.canvas.draw()
            y_labels = [label for label in ax.get_yticklabels() if str(label.get_text()).strip()]
            renderer = self.canvas.get_renderer()
            bboxes = sorted((label.get_window_extent(renderer) for label in y_labels), key=lambda box: box.y0)
            overlap = any(float(cur.y0) < float(prev.y1) - 0.5 for prev, cur in zip(bboxes, bboxes[1:]))
            captured["count"] = len(y_labels)
            captured["overlap"] = overlap
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _savefig_spy)

    history_idx = pd.RangeIndex(320)
    history = pd.Series(np.linspace(100.0, 4200.0, len(history_idx)), index=history_idx)
    forecast_idx = pd.RangeIndex(320, 360)
    forecast = pd.Series(np.linspace(4210.0, 4620.0, len(forecast_idx)), index=forecast_idx)
    img = generate_forecast_plot(history, forecast, "Forecast", "Index", "Value")

    assert isinstance(img, str)
    assert len(img) > 0
    assert int(captured.get("count", 0)) >= 8
    assert bool(captured.get("overlap", True)) is False


def test_static_distribution_overlays_use_high_contrast_min_marker():
    """Static distribution overlays should use a dark amber/brown min marker for high contrast."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from data_analysis.analysis.plot import _add_static_distribution_overlays

    values = np.asarray([2.0, 4.0, 7.0, 9.0, 12.0], dtype=float)
    fig, ax = plt.subplots()
    try:
        ax.hist(values, bins=5)
        _add_static_distribution_overlays(ax, values)
        min_collection = next(
            coll for coll in ax.collections
            if str(coll.get_label()).startswith("Min:")
        )
        assert to_hex(min_collection.get_facecolor()[0], keep_alpha=False).lower() == "#d97706"
    finally:
        plt.close(fig)


def test_static_distribution_overlays_raise_avg_med_labels_above_chart_lane():
    """Distribution Avg/Med labels should sit just above the chart area in static exports."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _add_static_distribution_overlays

    values = np.asarray([12.0, 14.0, 17.0, 22.0, 31.0, 45.0], dtype=float)
    fig, ax = plt.subplots()
    try:
        ax.hist(values, bins=6)
        _add_static_distribution_overlays(
            ax,
            values,
            expand_xlim=False,
            right_pad_ratio=0.015,
        )
        labels = [text for text in ax.texts if str(text.get_text()).startswith(("Avg:", "Med:"))]
        assert len(labels) >= 2
        assert all(float(text.get_position()[1]) > 1.0 for text in labels)
    finally:
        plt.close(fig)


def test_static_distribution_overlays_add_tiny_right_breathing_space_only():
    """Right padding helper should add a tiny right margin without expanding left margin."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _add_static_distribution_overlays

    values = np.asarray([2.0, 5.0, 7.0, 9.0, 13.0], dtype=float)
    fig, ax = plt.subplots()
    try:
        ax.hist(values, bins=5)
        before_xlim = tuple(float(v) for v in ax.get_xlim())
        _add_static_distribution_overlays(
            ax,
            values,
            expand_xlim=False,
            right_pad_ratio=0.015,
        )
        after_xlim = tuple(float(v) for v in ax.get_xlim())
        assert after_xlim[1] > before_xlim[1]
        assert after_xlim[0] == pytest.approx(before_xlim[0])
    finally:
        plt.close(fig)


def test_build_static_category_chart_adds_tiny_right_empty_space():
    """Static category charts should keep a small empty space to the right of the last bar."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series([20, 18, 14, 11], index=["A", "B", "C", "D"])
    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        left, right = ax.get_xlim()
        assert left == pytest.approx(-0.5)
        assert right > (len(counts) - 0.5)
        assert right <= (len(counts) - 0.30 + 1e-9)
    finally:
        plt.close(fig)


def test_apply_sci_formatter_can_use_k_threshold_for_y_axis():
    """Shared formatter helper should emit k suffix on y-axis when threshold is 1e3."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _apply_sci_formatter

    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1, 2], [950.0, 1500.0, 2300.0])
        _apply_sci_formatter(ax, y_threshold=1e3, x_threshold=1e6)
        formatter = ax.yaxis.get_major_formatter()
        assert formatter(2000.0, 0) == "2k"
    finally:
        plt.close(fig)


def test_generate_forecast_plot_minmax_markers_disable_axis_clipping(monkeypatch):
    """Trend/forecast min/max markers should disable axis clipping so symbols are not cut."""
    captured: list[dict[str, object]] = []
    original_scatter = Axes.scatter

    def _scatter_spy(self, x, y, *args, **kwargs):
        label = str(kwargs.get("label") or "")
        if label.startswith(("Min:", "Max:")):
            captured.append({"label": label, "clip_on": kwargs.get("clip_on")})
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", _scatter_spy)

    history = pd.Series([1200.0, 1450.0, 1310.0, 1540.0], index=pd.RangeIndex(4))
    forecast = pd.Series([1560.0, 1600.0], index=pd.RangeIndex(4, 6))

    img = generate_forecast_plot(history, forecast, "Forecast", "Index", "Value")
    assert isinstance(img, str)
    assert len(img) > 0
    assert captured
    assert all(entry.get("clip_on") is False for entry in captured)


def test_generate_forecast_plot_positions_min_label_below_marker(monkeypatch):
    """Forecast/trend min text tag should be offset below the marker (just under x-axis lane)."""
    from data_analysis.analysis.plot import _format_stat_value

    captured_annotations: list[dict[str, object]] = []
    original_annotate = Axes.annotate

    def _annotate_spy(self, text, *args, **kwargs):
        captured_annotations.append(
            {
                "text": str(text),
                "xytext": kwargs.get("xytext"),
                "va": kwargs.get("va"),
            }
        )
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", _annotate_spy)

    history = pd.Series([90.0, 120.0, 70.0, 105.0, 130.0], index=pd.RangeIndex(5))
    forecast = pd.Series([128.0, 131.0], index=pd.RangeIndex(5, 7))

    img = generate_forecast_plot(history, forecast, "Forecast", "Index", "Value")
    assert isinstance(img, str)
    assert len(img) > 0

    min_text = _format_stat_value(float(history.min()))
    min_entries = [entry for entry in captured_annotations if str(entry.get("text")) == min_text]
    assert min_entries
    assert any(
        isinstance(entry.get("xytext"), tuple)
        and len(cast(tuple[object, ...], entry.get("xytext"))) == 2
        and float(cast(tuple[object, ...], entry.get("xytext"))[1]) < 0
        for entry in min_entries
    )
    assert any(str(entry.get("va")) == "top" for entry in min_entries)


def test_static_distribution_overlays_minmax_annotations_use_contrast_pills():
    """Static distribution min/max value annotations should use contrast pill backgrounds."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import (
        _add_static_distribution_overlays,
        _format_stat_value,
    )

    values = np.asarray([3.0, 6.0, 9.0, 15.0, 22.0], dtype=float)
    fig, ax = plt.subplots()
    try:
        ax.hist(values, bins=5)
        stats = _add_static_distribution_overlays(ax, values)
        min_label = str(_format_stat_value(float(stats["min"])))
        max_label = str(_format_stat_value(float(stats["max"])))
        minmax_texts = [
            text for text in ax.texts
            if str(text.get_text()) in {min_label, max_label}
        ]
        assert len(minmax_texts) >= 2
        assert all(text.get_bbox_patch() is not None for text in minmax_texts)
        assert all(float(text.get_fontsize()) >= 6.0 for text in minmax_texts)
    finally:
        plt.close(fig)


def test_build_static_category_chart_keeps_dense_x_tick_font_baseline():
    """Category chart should enlarge typography but keep dense (~200) x-ticks at baseline size."""
    import matplotlib.pyplot as plt

    from data_analysis.analysis.plot import _build_static_category_chart

    counts = pd.Series(
        np.arange(1, 201, dtype=np.int64),
        index=[f"category_{i}" for i in range(200)],
    )
    built = _build_static_category_chart(counts, "category")
    assert built is not None
    fig, ax = built
    try:
        x_labels = [label for label in ax.get_xticklabels() if str(label.get_text()).strip()]
        assert x_labels
        assert float(x_labels[0].get_fontsize()) == pytest.approx(6.0)
        assert float(ax.title.get_fontsize()) >= 14.0
        assert float(ax.xaxis.label.get_fontsize()) >= 14.0
        assert float(ax.yaxis.label.get_fontsize()) >= 12.0
    finally:
        plt.close(fig)


def test_get_export_chart_figsize_uses_taller_defaults_for_pdf_and_zip():
    """Shared export chart size helper should return moderately taller chart heights."""
    from data_analysis.analysis.plot import get_export_chart_figsize

    zip_trend = get_export_chart_figsize("trend", context="zip")
    zip_forecast = get_export_chart_figsize("forecast", context="zip")
    zip_dist = get_export_chart_figsize("distribution", context="zip")
    pdf_trend = get_export_chart_figsize("trend", context="pdf")
    pdf_forecast = get_export_chart_figsize("forecast", context="pdf")
    pdf_dist = get_export_chart_figsize("distribution", context="pdf")

    assert zip_trend[1] > 6.3
    assert zip_forecast[1] > 6.3
    assert zip_dist[1] > 7.0
    assert pdf_trend[1] > 5.0
    assert pdf_forecast[1] > 5.0
    assert pdf_dist[1] > 7.0


def test_generate_forecast_plot_anomaly_positions_with_duplicate_index(monkeypatch):
    """Anomaly markers should respect anomaly occurrence counts on duplicate labels."""
    history = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.Index(["A", "B", "A", "C"]))
    forecast = pd.Series([41.0, 42.0], index=pd.Index(["D", "E"]))
    anomalies_idx = pd.Index(["A"])

    captured: list[tuple[list[float], list[float]]] = []
    original_scatter = Axes.scatter

    def _scatter_spy(self, x, y, *args, **kwargs):
        if kwargs.get("label") == "Anomaly":
            x_vals = np.asarray(x).astype(float).tolist()
            y_vals = np.asarray(y).astype(float).tolist()
            captured.append((x_vals, y_vals))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", _scatter_spy)

    img = generate_forecast_plot(
        history,
        forecast,
        "Forecast",
        "Index",
        "Value",
        anomalies_idx=anomalies_idx,
    )

    assert isinstance(img, str)
    assert len(img) > 0
    assert captured, "Expected at least one anomaly scatter call"
    assert captured[0][0] == [0.0]
    assert captured[0][1] == [10.0]


def test_generate_forecast_plot_anomaly_markers_respect_display_cap(monkeypatch):
    """Anomalies should be pre-capped by caller; plot function renders exactly what it receives."""
    history = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.Index(["A", "B", "C", "D"]))
    forecast = pd.Series([41.0], index=pd.Index(["E"]))
    # Pre-cap: caller passes only 1 anomaly (simulating _cap_anomalies_for_display with max_points=1)
    anomalies_idx = pd.Index(["A"])

    captured: list[int] = []
    original_scatter = Axes.scatter

    def _scatter_spy(self, x, y, *args, **kwargs):
        if kwargs.get("label") == "Anomaly":
            captured.append(len(np.asarray(x)))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", _scatter_spy)

    img = generate_forecast_plot(
        history,
        forecast,
        "Forecast",
        "Index",
        "Value",
        anomalies_idx=anomalies_idx,
    )

    assert isinstance(img, str)
    assert len(img) > 0
    assert captured and captured[0] == 1


def test_qna_cache_key_differs_for_different_filenames_same_shape():
    """Q&A cache key must not collide for same-shaped datasets with different filenames."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    q = "what is the trend?"

    key_a = _build_qna_cache_key(df, q, filename="aaa.csv")
    key_b = _build_qna_cache_key(df, q, filename="bbb.csv")

    assert key_a != key_b


def test_qna_cache_key_differs_for_different_dataframe_schema_without_filename():
    """Q&A cache key should isolate different dataframe schemas even with same shape."""
    q = "summary"
    df_one = pd.DataFrame({"country": ["A", "B"], "value": [1, 2]})
    df_two = pd.DataFrame({"city": ["A", "B"], "value": [1, 2]})

    key_one = _build_qna_cache_key(df_one, q)
    key_two = _build_qna_cache_key(df_two, q)

    assert key_one != key_two


def test_call_gemini_raises_on_prompt_block_reason(monkeypatch):
    """Blocked prompt feedback must raise instead of returning a successful response."""
    from data_analysis.ai import engine as ai_engine_mod

    class _BlockReason:
        name = "SAFETY"

    class _PromptFeedback:
        block_reason = _BlockReason()

    class _Resp:
        prompt_feedback = _PromptFeedback()

    class _Model:
        def generate_content(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(ai_engine_mod, "ensure_ai_ready", lambda **kw: True)
    monkeypatch.setattr(ai_engine_mod, "model", _Model())

    with pytest.raises(RuntimeError, match="Content blocked"):
        app_module._call_gemini("blocked content probe", retries=0)


def test_build_category_plotly_chart_has_bar_labels():
    """Ensure category chart builder includes bar value labels."""
    series = pd.Series(["A", "A", "B", "C", "C", "C"])
    chart = _build_category_plotly_chart(series, "Category")
    assert chart is not None
    traces = cast(list[dict[str, Any]], chart.get("traces") or [])
    assert traces
    bar = traces[0]
    assert bar.get("type") == "bar"
    assert bar.get("textposition") == "outside"
    assert len(bar.get("text", [])) == len(bar.get("x", []))


def test_build_category_plotly_chart_avg_med_are_toggleable_traces():
    """Avg/Med should be legend-controlled traces and full-width layout shapes."""
    series = pd.Series(["A", "A", "A", "B", "B", "C", "D", "D"])
    chart = _build_category_plotly_chart(series, "Category")
    assert chart is not None

    traces = cast(list[dict[str, Any]], chart.get("traces") or [])
    layout = cast(dict[str, Any], chart.get("layout") or {})
    shapes = cast(list[dict[str, Any]], layout.get("shapes") or [])

    avg_trace = next((t for t in traces if t.get("meta") == "avg-control"), None)
    med_trace = next((t for t in traces if t.get("meta") == "med-control"), None)

    assert avg_trace is not None
    assert med_trace is not None
    assert avg_trace.get("type") == "scatter"
    assert med_trace.get("type") == "scatter"
    assert avg_trace.get("mode") == "lines"
    assert med_trace.get("mode") == "lines"
    assert len(shapes) >= 2
    assert shapes[0].get("xref") == "paper"
    assert shapes[0].get("x0") == 0
    assert shapes[0].get("x1") == 1
    assert shapes[1].get("xref") == "paper"
    assert shapes[1].get("x0") == 0
    assert shapes[1].get("x1") == 1


def test_compute_forecast_reproducible():
    """Ensure _compute_forecast produces identical results on repeated calls (local RNG)."""
    series = pd.Series(
        np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 50,
        index=pd.RangeIndex(100),
    )
    fc1, ci1 = _compute_forecast(series, 10)
    fc2, ci2 = _compute_forecast(series, 10)
    pd.testing.assert_series_equal(fc1, fc2)
    assert ci1 is not None
    assert ci2 is not None
    pd.testing.assert_frame_equal(ci1, ci2)


def test_compute_forecast_does_not_mutate_global_rng():
    """Verify _compute_forecast does not change numpy global random state."""
    series = pd.Series(range(50), index=pd.RangeIndex(50), dtype=float)
    # Set a known global state
    np.random.seed(999)
    state_before = np.asarray(cast(tuple[Any, ...], np.random.get_state())[1]).copy()
    _compute_forecast(series, 5)
    state_after = np.asarray(cast(tuple[Any, ...], np.random.get_state())[1]).copy()
    # Global state should be unchanged
    assert np.array_equal(state_before, state_after), "Global RNG state was mutated by _compute_forecast"


def test_detect_anomalies_returns_index():
    """Ensure detect_anomalies returns proper types."""
    series = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
    an_idx, an_score = detect_anomalies(series, contamination=0.1)
    assert isinstance(an_idx, pd.Index)
    assert isinstance(an_score, pd.Series)


def test_detect_anomalies_isolation_forest_flags_injected_outliers():
    """IsolationForest should flag the strongest injected outliers deterministically."""
    baseline = np.linspace(50.0, 60.0, 200)
    series = pd.Series(np.concatenate([baseline, [180.0, -40.0]]), index=pd.RangeIndex(202))

    an_idx, an_score = detect_anomalies(series, contamination=0.01)

    assert isinstance(an_idx, pd.Index)
    assert isinstance(an_score, pd.Series)
    assert 200 in set(an_idx)
    assert 201 in set(an_idx)
    assert len(an_idx) <= int(np.ceil(len(series) * 0.01))


def test_detect_anomalies_seasonal_prefilter_keeps_spike_outliers():
    """Seasonal pre-filter should still preserve clear spike anomalies."""
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = np.arange(n, dtype=float)
    values = 50.0 + 8.0 * np.sin((2.0 * np.pi * x) / 7.0)

    spike_positions = [60, 120, 180]
    values[60] += 25.0
    values[120] -= 30.0
    values[180] += 28.0

    series = pd.Series(values, index=idx)
    an_idx, an_score = detect_anomalies(series, contamination=0.03)

    assert isinstance(an_idx, pd.Index)
    assert isinstance(an_score, pd.Series)

    detected_positions: set[int] = set()
    for ts in an_idx:
        if ts not in series.index:
            continue
        loc = series.index.get_loc(ts)
        if isinstance(loc, (int, np.integer)):
            detected_positions.add(int(loc))
        elif isinstance(loc, slice):
            detected_positions.update(range(loc.start or 0, loc.stop or 0))
        else:
            loc_arr = np.asarray(loc)
            if loc_arr.dtype == bool:
                detected_positions.update(np.flatnonzero(loc_arr).tolist())
            else:
                detected_positions.update(loc_arr.astype(int).tolist())
    for pos in spike_positions:
        assert pos in detected_positions


def test_detect_anomalies_non_timeseries_dedupes_duplicate_index_labels():
    """Duplicate categorical index labels should keep only strongest anomaly per label."""
    idx = pd.Index(["A", "A", "A", "B", "B", "C", "C", "D", "D"], dtype=object)
    vals = pd.Series([10.0, 11.0, 250.0, 9.0, 200.0, 8.0, 180.0, 7.0, 6.0], index=idx)

    an_idx, _an_score = detect_anomalies(vals, contamination=0.3)

    assert isinstance(an_idx, pd.Index)
    counts = pd.Series(list(an_idx)).value_counts()
    assert counts.max() <= 1


def test_get_cached_anomalies_caches():
    """Ensure get_cached_anomalies returns equal results and uses cache on repeated calls."""
    series = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
    with app.app_context():
        r1 = get_cached_anomalies("test_cache_file.csv", "col1", series, 0.1)
        r2 = get_cached_anomalies("test_cache_file.csv", "col1", series, 0.1)
        # Results should be equal (same anomaly indices and scores)
        assert r1[0].equals(r2[0])
        assert r1[1].equals(r2[1])


def test_describe_for_ai_caches_by_filename():
    """Ensure describe_for_ai returns cached result when filename is provided."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Clear cache
    AI_DESCRIBE_CACHE.clear()
    with app.app_context():
        result1 = describe_for_ai(df, filename="test_describe_cache.csv")
        result2 = describe_for_ai(df, filename="test_describe_cache.csv")
        assert result1 == result2
        # Verify there's a cache entry
        assert AI_DESCRIBE_CACHE.get("test_describe_cache.csv") is not None


def test_describe_for_ai_without_filename():
    """Ensure describe_for_ai works without filename (no caching)."""
    df = pd.DataFrame({"x": [10, 20, 30]})
    with app.app_context():
        result = describe_for_ai(df)
        assert "3 rows" in result
        assert "x" in result


def test_health_endpoint():
    """Test the health check endpoint."""
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"


def test_forecast_does_not_copy_historical_segments():
    """Ensure _compute_forecast never returns a subsequence found verbatim in the input."""
    # Create a distinctive pattern that would be easy to detect if copied
    rng = np.random.default_rng(42)
    history = np.cumsum(rng.normal(0, 1, 200)) + 100
    series = pd.Series(history, index=pd.RangeIndex(200))
    fc, _ci = _compute_forecast(series, 30)
    fc_vals = fc.to_numpy(dtype=float, copy=False)

    # Slide a window of length 30 over history; none should match the forecast
    for start in range(len(history) - 30 + 1):
        segment = history[start:start + 30]
        # Even with an offset shift (like old code did), the shape shouldn't match
        # Check normalised correlation: if segment == shifted copy, corr ≈ 1.0
        seg_centered = segment - segment.mean()
        fc_centered = fc_vals - float(np.mean(fc_vals))
        seg_std = np.std(seg_centered)
        fc_std = np.std(fc_centered)
        if seg_std < 1e-9 or fc_std < 1e-9:
            continue
        corr = float(np.dot(seg_centered, fc_centered) / (seg_std * fc_std * len(segment)))
        assert corr < 0.98, (
            f"Forecast appears to be a near-copy of history[{start}:{start+30}] (corr={corr:.4f})"
        )


def test_forecast_stays_within_data_bounds_without_edge_saturation():
    """Forecast and CI should stay inside historical bounds without pegging exact edges."""
    rng = np.random.default_rng(7)
    # Data with clear bounds [20, 80]
    values = rng.uniform(20, 80, size=150)
    series = pd.Series(values, index=pd.RangeIndex(150))
    data_min = float(series.min())
    data_max = float(series.max())

    fc, ci = _compute_forecast(series, 30)
    assert ci is not None
    fc_vals = fc.to_numpy(dtype=float, copy=False)

    assert fc.min() >= data_min - 1e-9, f"Forecast min {fc.min()} < data_min {data_min}"
    assert fc.max() <= data_max + 1e-9, f"Forecast max {fc.max()} > data_max {data_max}"
    assert ci["lower"].min() >= data_min - 1e-9, f"CI lower min {ci['lower'].min()} < data_min {data_min}"
    assert ci["upper"].max() <= data_max + 1e-9, f"CI upper max {ci['upper'].max()} > data_max {data_max}"

    # Values should not sit exactly on boundaries (no saturation by clipping).
    assert not np.isclose(fc_vals, data_min, atol=1e-9).any()
    assert not np.isclose(fc_vals, data_max, atol=1e-9).any()


def test_analyze_forecast_data_range_parsing_ratio_and_rows(monkeypatch):
    """Detailed Analysis should parse data_range both as ratio and explicit row count."""
    filename = "f" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(100, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)

    captured = {}

    def fake_render_template(_template, **kwargs):
        captured["analysis"] = kwargs.get("analysis", {})
        return "ok"

    monkeypatch.setattr(app_module, "render_template", fake_render_template)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_forecast_plot", lambda *_args, **_kwargs: "x")

    def fake_get_cached_column_forecast(_filename, _column, _series, steps):
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.zeros(int(steps), dtype=float), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    with app.test_client() as client:
        response_ratio = client.get(f"/analyze/{filename}?view=forecast&forecast_pct=0.05&data_range=0.50")
        assert response_ratio.status_code == 200
        controls_ratio = captured["analysis"]["controls"]
        assert controls_ratio["data_range"] == pytest.approx(0.5)
        assert controls_ratio["data_range_rows"] == 50

        response_rows = client.get(f"/analyze/{filename}?view=forecast&forecast_pct=0.05&data_range=25")
        assert response_rows.status_code == 200
        controls_rows = captured["analysis"]["controls"]
        assert controls_rows["data_range"] == pytest.approx(0.25)
        assert controls_rows["data_range_rows"] == 25


def test_analyze_forecast_pct_does_not_overshoot_requested_share(monkeypatch):
    """For forecast_pct=5%, effective_steps should not exceed 5% visual share target."""
    filename = "u" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(100, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)

    captured = {}

    def fake_render_template(_template, **kwargs):
        captured["analysis"] = kwargs.get("analysis", {})
        return "ok"

    monkeypatch.setattr(app_module, "render_template", fake_render_template)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_forecast_plot", lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _filename, _column, _series, steps: (
            pd.Series(np.zeros(int(steps), dtype=float), index=pd.RangeIndex(int(steps))),
            None,
        ),
    )

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=forecast&forecast_pct=0.05")

    assert response.status_code == 200
    controls = captured["analysis"]["controls"]
    assert controls["effective_steps"] == 5


def test_upload_redirect_keeps_contamination_param():
    """Upload form settings should propagate contamination into analyze redirect URL."""
    payload = {
        "file": (io.BytesIO(b"x,y\n1,2\n3,4\n"), "sample.csv"),
        "forecast_pct": "0.10",
        "contamination": "0.07",
        "view": "interactive",
    }

    with app.test_client() as client:
        response = client.post("/", data=payload, content_type="multipart/form-data", follow_redirects=False)

    assert response.status_code in (302, 303)
    location = response.headers.get("Location", "")
    assert "contamination=0.07" in location
    assert "forecast_pct=0.10" in location
    assert "view=interactive" in location


@pytest.mark.parametrize(
    ("view", "control_id"),
    [
        ("interactive", "contaminationInteractive"),
        ("forecast", "contaminationForecast"),
    ],
)
def test_analysis_view_preserves_contamination_in_links_and_controls(monkeypatch, view, control_id):
    """Detailed and interactive pages should keep contamination in links and render contamination controls."""
    filename = "h" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(30, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)

    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_forecast_plot", lambda *_args, **_kwargs: "x")

    def fake_get_cached_column_forecast(_filename, _column, _series, steps):
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.zeros(int(steps), dtype=float), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view={view}&forecast_pct=0.05&contamination=0.07")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "contamination=0.07" in html
    assert f'id="{control_id}"' in html


def test_forecast_continuity_with_history():
    """Ensure the first forecast point is close to the last historical value."""
    rng = np.random.default_rng(123)
    values = np.cumsum(rng.normal(0, 0.5, 100)) + 50
    series = pd.Series(values, index=pd.RangeIndex(100))
    last_val = float(series.iloc[-1])

    fc, _ci = _compute_forecast(series, 20)
    first_fc = float(fc.iloc[0])

    # The first forecast point should be very close to the last history point
    # (within ~5% of data range) for a natural-looking transition
    data_range = float(series.max() - series.min())
    gap = abs(first_fc - last_val)
    assert gap < 0.1 * data_range, (
        f"First forecast point {first_fc:.2f} is too far from last historical value "
        f"{last_val:.2f} (gap={gap:.2f}, 10% range={0.1*data_range:.2f})"
    )


def test_forecast_matches_historical_variation():
    """Verify that the std of forecast differences is within a reasonable
    range (0.3x to 3x) of the historical data's difference std.
    This ensures the forecast doesn't become unnaturally flat or wild.
    """
    rng = np.random.default_rng(42)
    values = np.cumsum(rng.normal(0, 1.0, 200)) + 100
    series = pd.Series(values, index=pd.RangeIndex(200))

    fc, _ci = _compute_forecast(series, 50)

    hist_vals = series.to_numpy(dtype=float, copy=False)
    fc_vals = fc.to_numpy(dtype=float, copy=False)
    hist_diffs = np.diff(hist_vals)
    fc_diffs = np.diff(fc_vals)
    std_hist = float(np.std(hist_diffs, ddof=1)) if len(hist_diffs) > 1 else 0.0
    std_fc = float(np.std(fc_diffs, ddof=1)) if len(fc_diffs) > 1 else 0.0

    assert std_fc > 0, "Forecast is completely flat (zero variation)"
    ratio = std_fc / std_hist
    assert 0.3 <= ratio <= 3.0, (
        f"Forecast variation ratio {ratio:.2f} outside [0.3, 3.0]: "
        f"std_fc={std_fc:.4f}, std_hist={std_hist:.4f}"
    )


def test_forecast_seasonal_data_preserves_oscillation():
    """Test with a seasonal series to ensure the forecast exhibits meaningful
    oscillation and doesn't collapse to a flat line.
    """
    n = 120
    t = np.arange(n, dtype=float)
    # Clear seasonal pattern with period 12
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = np.random.default_rng(99).normal(0, 0.5, n)
    values = 50 + trend + seasonal + noise
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    series = pd.Series(values, index=idx)

    fc, _ci = _compute_forecast(series, 24)
    fc_vals = fc.to_numpy(dtype=float, copy=False)

    # The forecast should have meaningful variation (not flat)
    fc_range = float(np.max(fc_vals) - np.min(fc_vals))
    hist_range = float(np.max(values) - np.min(values))

    assert fc_range > 0.1 * hist_range, (
        f"Forecast range {fc_range:.2f} is too small compared to historical "
        f"range {hist_range:.2f} — forecast appears flat"
    )


def test_forecast_mean_matches_recent_history():
    """Forecast average should be close to the average of recent history,
    even when the last observations are near the data maximum (saturation guard).
    """
    rng = np.random.default_rng(77)
    values = rng.uniform(30, 70, size=100)
    # Push the last 5 values near the maximum (common saturation scenario)
    values[-5:] = values[-5:] + 25
    series = pd.Series(values, index=pd.RangeIndex(100))

    fc, _ci = _compute_forecast(series, 30)

    recent_mean = float(np.mean(values[-20:]))
    fc_vals = fc.to_numpy(dtype=float, copy=False)
    fc_mean = float(np.mean(fc_vals))
    data_range = float(np.max(values) - np.min(values))

    # The forecast mean should be within 15% of data range from recent mean
    pct_diff = abs(fc_mean - recent_mean) / data_range * 100
    assert pct_diff < 15, (
        f"Forecast mean {fc_mean:.2f} too far from recent mean {recent_mean:.2f}: "
        f"{pct_diff:.1f}% of range ({data_range:.1f}), expected < 15%"
    )


def test_tinylru_hit_miss_and_eviction_stats():
    """TinyLRU should track hits/misses and evict by max_items."""
    cache = TinyLRU(max_items=2)
    assert cache.get("missing") is None
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # hit
    cache.set("c", 3)  # evicts one key
    st = cache.stats()
    assert st["hits"] >= 1
    assert st["misses"] >= 1
    assert st["evictions"] >= 1
    assert st["size"] <= 2


def test_tinylru_size_based_eviction_for_large_values():
    """TinyLRU should evict when max_size_mb is exceeded."""
    cache = TinyLRU(max_items=10, max_size_mb=0.0001)
    cache.set("a", "x" * 200_000)
    cache.set("b", "y" * 200_000)
    st = cache.stats()
    assert st["evictions"] >= 1
    assert st["size"] == 1


def test_tinylru_size_tracking_stays_correct_on_key_replace():
    """Replacing a large entry with a small one should not keep stale size accounting."""
    cache = TinyLRU(max_items=10, max_size_mb=0.001)

    cache.set("a", "x" * 200_000)
    cache.set("a", "x" * 20)
    cache.set("b", "y" * 20)

    assert cache.get("a") == "x" * 20
    assert cache.get("b") == "y" * 20
    st = cache.stats()
    assert st["size"] == 2
    assert st["size_bytes"] > 0


def test_tinylru_clear_resets_size_bookkeeping():
    """clear() should reset both entry map and tracked total bytes."""
    cache = TinyLRU(max_items=10, max_size_mb=1)
    cache.set("a", "x" * 1000)
    cache.set("b", "y" * 1000)

    cache.clear()

    st = cache.stats()
    assert st["size"] == 0
    assert st["size_bytes"] == 0


def test_tinylru_pop_updates_size_and_respects_default():
    """pop() should update tracked bytes and preserve dict-like missing-key semantics."""
    cache = TinyLRU(max_items=10, max_size_mb=1)
    cache.set("a", "x" * 1000)
    before = cache.stats()["size_bytes"]

    popped = cache.pop("a")
    assert popped == "x" * 1000
    after = cache.stats()["size_bytes"]
    assert after < before

    assert cache.pop("missing", "fallback") == "fallback"
    with pytest.raises(KeyError):
        cache.pop("missing")


def test_cleanup_uploads_prunes_stale_ai_file_map(tmp_path):
    """Upload cleanup should remove AI_FILE_MAP entries for deleted/nonexistent files."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    stale_name = ("a" * 40) + ".csv"
    fresh_name = ("b" * 40) + ".csv"
    missing_name = ("c" * 40) + ".csv"

    stale_path = uploads_dir / stale_name
    fresh_path = uploads_dir / fresh_name
    stale_path.write_text("old", encoding="utf-8")
    fresh_path.write_text("new", encoding="utf-8")

    old_ts = time.time() - (3 * 24 * 3600)
    now_ts = time.time()
    os.utime(stale_path, (old_ts, old_ts))
    os.utime(fresh_path, (now_ts, now_ts))

    prev_retention = app_module.app.config.get("UPLOAD_RETENTION_DAYS")
    prev_uploads = app_module.app.config.get("UPLOADS_DIR")

    app_module.AI_FILE_MAP.clear()
    app_module.AI_FILE_MAP[stale_name] = object()
    app_module.AI_FILE_MAP[fresh_name] = object()
    app_module.AI_FILE_MAP[missing_name] = object()

    app_module.app.config["UPLOAD_RETENTION_DAYS"] = 1
    app_module.app.config["UPLOADS_DIR"] = str(uploads_dir)

    try:
        app_module._cleanup_uploads_if_configured()
    finally:
        app_module.app.config["UPLOAD_RETENTION_DAYS"] = prev_retention
        app_module.app.config["UPLOADS_DIR"] = prev_uploads

    assert stale_name not in app_module.AI_FILE_MAP
    assert missing_name not in app_module.AI_FILE_MAP
    assert fresh_name in app_module.AI_FILE_MAP
    assert not stale_path.exists()
    assert fresh_path.exists()


def test_thin_series_keep_extrema_includes_min_max_and_last():
    """Extrema-preserving thinning must retain min/max and final point."""
    s = pd.Series([10, 11, 12, -5, 13, 14, 99, 15, 16, 17], index=pd.RangeIndex(10))
    thinned = _thin_series_keep_extrema(s, max_points=4)

    assert s.idxmin() in thinned.index
    assert s.idxmax() in thinned.index
    assert s.index[-1] in thinned.index
    assert thinned.index.is_monotonic_increasing


def test_get_cached_numeric_df_reuses_cache_entry():
    """get_cached_numeric_df should return cached object for same filename."""
    NUMERIC_DF_CACHE.clear()
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})

    r1 = get_cached_numeric_df("numeric_cache_case.csv", df)
    r2 = get_cached_numeric_df("numeric_cache_case.csv", df)

    assert isinstance(r1, pd.DataFrame)
    assert r1 is r2
    assert "a" in r1.columns
    assert r1["a"].dtype.kind in ("f", "i")


def test_api_interactive_invalid_filename_rejected():
    """Interactive API should reject invalid filename format."""
    with app.test_client() as client:
        response = client.get('/api/interactive/not-a-valid-name.csv')
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["ok"] is False


def test_download_full_report_pdf_returns_pdf(monkeypatch):
    """PDF report route should return a valid PDF payload without page-open errors."""
    filename = "f" * 40 + ".csv"
    df = pd.DataFrame({
        "value": np.linspace(1.0, 50.0, 120),
        "category": ["A", "B", "C", "D"] * 30,
    })

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")


def test_download_full_report_pdf_small_series_includes_trend_and_forecast(monkeypatch):
    """Small numeric datasets (>=5 rows) should still produce trend and forecast charts in PDF."""
    filename = "1" * 40 + ".json"
    df = pd.DataFrame({
        "air_temp_c": [26.3, 26.5, 26.6, 26.8, 27.0],
        "soil_moisture_pct": [42.5, 42.1, 41.9, 41.7, 41.4],
    })

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")

    def fake_get_cached_column_forecast(_filename, _column, series, steps):
        n_steps = int(steps)
        idx = pd.RangeIndex(start=len(series), stop=len(series) + n_steps)
        base = float(series.iloc[-1]) if len(series) else 0.0
        vals = [base + 0.1 * (i + 1) for i in range(n_steps)]
        fc = pd.Series(vals, index=idx)
        ci = pd.DataFrame({"lower": fc - 0.2, "upper": fc + 0.2}, index=idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    original_plot = app_module.generate_forecast_plot
    titles: list[str] = []

    def spy_generate_forecast_plot(*args, **kwargs):
        title = args[2] if len(args) > 2 else kwargs.get("title", "")
        titles.append(str(title))
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(app_module, "generate_forecast_plot", spy_generate_forecast_plot)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=plant&forecast_pct=0.2")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert any(t.startswith("Trend: air_temp_c") for t in titles)
    assert any(t.startswith("Forecast: air_temp_c") for t in titles)


def test_download_full_report_pdf_forecast_steps_use_non_null_history(monkeypatch):
    """PDF forecast steps should be computed per column from non-null history rows."""
    filename = "3" * 40 + ".csv"
    df = pd.DataFrame({"value": np.concatenate([np.arange(1, 61, dtype=float), np.full(40, np.nan)])})
    captured_steps: list[int] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_get_cached_column_forecast(_filename, _column, _series, steps):
        captured_steps.append(int(steps))
        idx = pd.RangeIndex(int(steps))
        fc = pd.Series(np.zeros(int(steps), dtype=float), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0.2")

    assert response.status_code == 200
    assert captured_steps
    # 60 non-null rows => floor(60 * 0.2 / 0.8) = 15
    assert captured_steps[0] == 15


def test_download_full_report_pdf_complex_ai_summary_renders(monkeypatch):
    """Complex heading/list AI summaries should still render as PDF without errors."""
    filename = "6" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(1, 51, dtype=float)})

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)

    ai_html = (
        "<h3>Prognosis &amp; Future Outlook</h3>"
        "<p>Based on observed trends and patterns, several plausible future directions can be identified:</p>"
        "<ul>"
        "<li><strong>Expected Continuations:</strong>"
        "<ul>"
        "<li>Continuation A with confidence high.</li>"
        "<li>Continuation B with confidence medium.</li>"
        "<li>Continuation C with confidence medium.</li>"
        "</ul>"
        "</li>"
        "<li><strong>Potential Risks:</strong>"
        "<ul>"
        "<li>Risk A that may influence forecast reliability.</li>"
        "<li>Risk B related to data quality and temporal context.</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "<h3>Actionable Observations</h3>"
        "<ul><li>Recommendation 1</li><li>Recommendation 2</li></ul>"
    )
    monkeypatch.setattr(app_module, "_get_clean_ai_summary_from_cache", lambda _filename: ai_html)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0.05")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")


def test_download_full_report_pdf_overview_tables_use_auto_placement(monkeypatch):
    """Compact overview tables should stay on the first page when the full titled blocks fit."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "2" * 40 + ".csv"
    df = pd.DataFrame({
        "value": [10.0, 11.0, 12.0, 13.0, 14.0],
        "category": ["A", "B", "A", "C", "B"],
    })

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)

    original_cell = pdf_report_mod.PDFReport.cell
    title_pages: dict[str, int] = {}
    target_titles = {
        "Dataset Overview Summary:",
        "Columns & Types:",
        "First 5 Rows:",
        "Statistical Description:",
    }

    def spy_cell(self, *args, **kwargs):
        text = ""
        if len(args) >= 3:
            text = str(args[2])
        elif "text" in kwargs:
            text = str(kwargs["text"])
        if text in target_titles and text not in title_pages:
            title_pages[text] = int(self.page_no())
        return original_cell(self, *args, **kwargs)

    monkeypatch.setattr(pdf_report_mod.PDFReport, "cell", spy_cell)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test")

    assert response.status_code == 200
    assert target_titles.issubset(title_pages.keys())
    assert all(page_no == 1 for page_no in title_pages.values())


def test_api_interactive_returns_cached_payload_when_available():
    """Interactive API should return cached data without recomputing when cache is warm."""
    filename = "a" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    cached_payload = [{"column": "x", "traces": [], "layout": {}, "distribution": {"name": "x", "values": []}, "stats": None}]
    INTERACTIVE_DATA_CACHE.set(filename, cached_payload)

    with app.test_client() as client:
        response = client.get(f'/api/interactive/{filename}')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["cached"] is True
        assert payload["data"] == cached_payload


def test_api_interactive_returns_full_history_and_distribution(monkeypatch):
    """Interactive API should return full data by default and include distribution axis metadata."""
    filename = "d" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(0, 250, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0.1&contamination=0.02")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["data"]

        first = payload["data"][0]
        history_trace = first["traces"][0]
        assert len(history_trace["x"]) == 250
        assert len(history_trace["y"]) == 250
        assert len(first["distribution"]["values"]) == 250
        axis_spec = first["distribution"].get("axis_spec")
        assert isinstance(axis_spec, dict)
        assert axis_spec.get("tickvals")
        assert axis_spec.get("ticktext")


def test_api_interactive_respects_data_range_for_history_and_distribution(monkeypatch):
    """Interactive API should return range-filtered history and distribution payloads when data_range is set."""
    filename = "c" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(0, 200, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0.1&contamination=0.02&data_range=0.5")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["data"]

    first = payload["data"][0]
    history_trace = first["traces"][0]
    assert len(history_trace["x"]) == 100
    assert len(history_trace["y"]) == 100
    assert len(first["distribution"]["values"]) == 100
    assert isinstance(first["distribution"].get("axis_spec"), dict)


def test_api_interactive_cache_key_honors_data_range(monkeypatch):
    """Different data_range requests should not reuse the same interactive payload."""
    filename = "b" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(0, 120, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        full_resp = client.get(f"/api/interactive/{filename}?forecast_pct=0.05&contamination=0.02&data_range=1")
        half_resp = client.get(f"/api/interactive/{filename}?forecast_pct=0.05&contamination=0.02&data_range=0.5")

    full_payload = full_resp.get_json()
    half_payload = half_resp.get_json()
    assert full_payload["ok"] is True and half_payload["ok"] is True

    full_history_len = len(full_payload["data"][0]["traces"][0]["x"])
    half_history_len = len(half_payload["data"][0]["traces"][0]["x"])
    assert full_history_len == 120
    assert half_history_len == 60


def test_analyze_interactive_large_dataset_defers_inline_payload(monkeypatch):
    """Large interactive datasets should skip inline payload and load via async API."""
    filename = "9" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(0, 60001, dtype=float)})

    DATAFRAME_CACHE.set(filename, df)
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    match = re.search(r'<script id="interactivePayload" type="application/json">(.*?)</script>', html, flags=re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload == []


def test_analyze_interactive_template_exposes_std_legend_control(monkeypatch):
    """Interactive page template should expose the main-chart Std legend control."""
    filename = "1" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(1, 51, dtype=float)})

    DATAFRAME_CACHE.set(filename, df)
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([], dtype="int64"), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "name: formatLegendStatLabel('Std', s.std)" in html
    assert "meta: 'stat-control'" in html
    assert "showlegend: true" in html


def test_analyze_interactive_template_uses_single_row_fraction_legend_slots(monkeypatch):
    """Interactive template should use centered one-row fraction legend slot packing."""
    filename = "2" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(1, 51, dtype=float)})

    DATAFRAME_CACHE.set(filename, df)
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([], dtype="int64"), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "function resolveLegendSlotSizing" in html
    assert "function getLegendLabels" in html
    assert "entrywidthmode: 'fraction'" in html
    assert "x: 0.5" in html
    assert "xanchor: 'center'" in html
    assert "slotEpsilon: 0.010" in html
    assert "minGapPx: useCompactLegendPacking ? 6 : 8" in html


def test_analysis_template_exposes_wider_contamination_autosize_config(monkeypatch):
    """Rendered analysis pages should expose the widened contamination autosize config for both views."""
    filename = "2" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(1, 31, dtype=float)})

    DATAFRAME_CACHE.set(filename, df)
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([], dtype="int64"), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "contaminationInteractive: { min: 82, max: 120, padding: 26 }" in html
    assert "contaminationForecast: { min: 82, max: 120, padding: 26 }" in html


def test_api_interactive_forecast_visual_share_matches_pct(monkeypatch):
    """forecast_pct should map to approximately the same x-axis share in interactive charts."""
    filename = "8" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(1, 101, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def _fake_forecast(_f, _c, _s, steps):
        idx = pd.RangeIndex(int(steps))
        fc = pd.Series(np.linspace(0.0, 1.0, int(steps), dtype=float), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", _fake_forecast)

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    first = payload["data"][0]
    history_n = len(first["traces"][0]["x"])
    x_range = first["layout"]["xaxis"]["range"]
    total_extent = float(x_range[1]) if isinstance(x_range, list) and len(x_range) >= 2 else float(history_n)
    forecast_span = max(0.0, total_extent - float(history_n))
    share = forecast_span / total_extent if total_extent > 0 else 0.0
    assert 0.045 <= share <= 0.05


def test_api_interactive_forecast_share_uses_non_null_history_length(monkeypatch):
    """20% forecast should not overshoot when a column has many NaN values."""
    filename = "5" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    values = np.concatenate([np.arange(1, 61, dtype=float), np.full(40, np.nan)])
    df = pd.DataFrame({"value": values})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    captured_steps: list[int] = []

    def _fake_forecast(_f, _c, _s, steps):
        s = int(steps)
        captured_steps.append(s)
        idx = pd.RangeIndex(s)
        fc = pd.Series(np.linspace(0.0, 1.0, s, dtype=float), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", _fake_forecast)

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0.2&contamination=0.02")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    first = payload["data"][0]
    assert captured_steps

    # Non-null history length is 60; for 20% target we expect floor(60 * 0.2 / 0.8) = 15.
    assert captured_steps[0] == 15

    history_n = len(first["traces"][0]["x"])
    x_range = first["layout"]["xaxis"]["range"]
    total_extent = float(x_range[1]) if isinstance(x_range, list) and len(x_range) >= 2 else float(history_n)
    forecast_span = max(0.0, total_extent - float(history_n))
    share = forecast_span / total_extent if total_extent > 0 else 0.0
    assert share <= 0.2 + 1e-9


def test_api_interactive_returns_all_numeric_columns_not_first_eight(monkeypatch):
    """Interactive API should include all numeric columns by default."""
    filename = "7" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    data = {f"col_{i}": np.arange(1, 41, dtype=float) + i for i in range(12)}
    df = pd.DataFrame(data)
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0&contamination=0.02")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    cols = [item.get("column") for item in payload["data"]]
    assert len(cols) == 12
    assert set(cols) == set(df.columns)


def test_api_interactive_zero_pct_excludes_forecast_and_ci_traces(monkeypatch):
    """Interactive API at 0% forecast should return history/anomaly traces only (no forecast or CI)."""
    filename = "f" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(1, 101, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([], dtype="int64"), pd.Series(dtype=float)),
    )

    def _fail_if_forecast_called(*_args, **_kwargs):
        raise AssertionError("Forecast should not be requested when forecast_pct=0")

    monkeypatch.setattr(app_module, "get_cached_column_forecast", _fail_if_forecast_called)

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0&contamination=0.02")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["data"]
    trace_names = [str((t or {}).get("name", "")).lower() for t in payload["data"][0].get("traces", [])]
    assert all("forecast" not in name for name in trace_names)
    assert all(not re.search(r"\bci\b|conf|interval", name) for name in trace_names)


def test_api_interactive_anomalies_snap_to_displayed_history_points(monkeypatch):
    """Anomaly markers should stay on visible history points after downsampling."""
    filename = "6" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(0, 20001, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    an_idx = pd.Index([3, 15001], dtype="int64", name="__pos__")
    an_score = pd.Series([0.9, 0.8], index=an_idx)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (an_idx, an_score))
    monkeypatch.setattr(
        app_module,
        "get_cached_column_forecast",
        lambda _f, _c, _s, steps: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/api/interactive/{filename}?forecast_pct=0&contamination=0.02")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    first = payload["data"][0]
    traces = first.get("traces", [])
    hist = next(t for t in traces if str(t.get("name", "")).lower() == "history")
    anom = next(t for t in traces if str(t.get("name", "")).lower() == "anomaly")
    hist_x = {int(v) for v in hist.get("x", [])}
    anom_x = [int(v) for v in anom.get("x", [])]
    assert anom_x
    assert all(x in hist_x for x in anom_x)


def test_api_interactive_cache_key_includes_request_params(monkeypatch):
    """Different request params should not reuse the same interactive cached response."""
    filename = "e" * 40 + ".csv"
    INTERACTIVE_DATA_CACHE.clear()
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.arange(1, 101, dtype=float)})
    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_get_cached_column_forecast(_f, _c, _s, steps):
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.full(int(steps), float(steps)), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)

    with app.test_client() as client:
        r1 = client.get(f"/api/interactive/{filename}?forecast_pct=0.05&contamination=0.02")
        r2 = client.get(f"/api/interactive/{filename}?forecast_pct=0.2&contamination=0.02")

    p1 = r1.get_json()
    p2 = r2.get_json()
    assert p1["ok"] is True and p2["ok"] is True

    t1 = p1["data"][0]["traces"][-1]["y"][-1]
    t2 = p2["data"][0]["traces"][-1]["y"][-1]
    assert t1 != t2


def test_interactive_cache_key_varies_by_params():
    """Interactive cache key should isolate forecast/contamination/data_range variants."""
    filename = "a" * 40 + ".csv"
    k1 = _build_interactive_cache_key(filename, 0.05, 0.02)
    k2 = _build_interactive_cache_key(filename, 0.10, 0.02)
    k3 = _build_interactive_cache_key(filename, 0.05, 0.05)
    k4 = _build_interactive_cache_key(filename, 0.05, 0.02, 0.5)
    assert k1 != k2
    assert k1 != k3
    assert k1 != k4


def test_static_plots_zip_caps_forecast_steps(monkeypatch):
    """ZIP export should cap forecast horizon to avoid excessive compute."""
    filename = "c" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.linspace(1, 100, 5000)})
    captured: dict[str, int] = {}

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_get_cached_column_forecast(_filename, _column, _series, steps):
        captured["steps"] = int(steps)
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.zeros(int(steps)), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)
    monkeypatch.setattr(
        app_module,
        "generate_forecast_plot",
        lambda *_args, **_kwargs: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=",
    )

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")
        assert response.status_code == 200

    assert captured.get("steps") == 120


def test_static_plots_zip_forecast_pct_uses_non_null_history(monkeypatch):
    """ZIP forecast steps should honor forecast_pct per column history length."""
    filename = "4" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"value": np.concatenate([np.arange(1, 61, dtype=float), np.full(40, np.nan)])})
    captured: dict[str, int] = {}

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_get_cached_column_forecast(_filename, _column, _series, steps):
        captured["steps"] = int(steps)
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.zeros(int(steps)), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)
    monkeypatch.setattr(
        app_module,
        "generate_forecast_plot",
        lambda *_args, **_kwargs: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=",
    )

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip?forecast_pct=0.2")
        assert response.status_code == 200

    # 60 non-null rows => floor(60 * 0.2 / 0.8) = 15
    assert captured.get("steps") == 15


def test_static_plots_zip_includes_category_images(monkeypatch):
    """ZIP export should include *_categories.png files for true categorical columns."""
    filename = "a" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({
        "city": ["Iasi", "Iasi", "Cluj", "Bucharest", "Cluj", "Iasi"],
        "segment": ["A", "B", "A", "B", "A", "C"],
    })

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "generate_correlation_heatmap", lambda *_args, **_kwargs: None)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/zip"

        with zipfile.ZipFile(io.BytesIO(response.data), "r") as zf:
            names = zf.namelist()

    assert any(name.endswith("_categories.png") for name in names)


def test_static_plots_zip_category_exports_use_tiny_pad(monkeypatch):
    """ZIP category chart saves should use a tight crop with a tiny pad."""
    from matplotlib.figure import Figure

    filename = "c" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()
    df = pd.DataFrame({"city": ["Iasi", "Cluj", "Iasi", "Bacau", "Cluj", "Iasi"]})
    tiny_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=")
    captured_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "generate_correlation_heatmap", lambda *_args, **_kwargs: None)

    def fake_savefig(self, fp, *args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        fp.write(tiny_png)

    monkeypatch.setattr(Figure, "savefig", fake_savefig)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")

    assert response.status_code == 200
    assert captured_kwargs
    assert captured_kwargs[0].get("pad_inches") == 0.02


def test_analyze_categories_skips_active_temporal_axis_column(monkeypatch):
    """Categories view should not render a category chart for the active temporal axis column."""
    filename = "d" * 40 + ".csv"
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "record_date": dates,
            "city": ["Iasi", "Cluj", "Iasi", "Bacau", "Cluj", "Iasi"],
            "value": [10, 11, 12, 13, 14, 15],
        }
    )
    df.index = pd.DatetimeIndex(df["record_date"], name="record_date")
    DATAFRAME_CACHE.set(filename, df)

    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=categories")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    match = re.search(r'<script type="application/json" id="category-data">\s*(.*?)\s*</script>', html, flags=re.DOTALL)
    assert match is not None
    category_charts = json.loads(match.group(1))
    assert "city" in category_charts
    assert "record_date" not in category_charts


def test_static_plots_zip_skips_active_temporal_axis_categories(monkeypatch):
    """ZIP categories should exclude the active temporal axis column chart."""
    import matplotlib.pyplot as plt

    import data_analysis.routes.downloads as downloads_mod

    filename = "e" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "record_date": dates,
            "city": ["Iasi", "Cluj", "Iasi", "Bacau", "Cluj", "Iasi"],
            "value": [10, 11, 12, 13, 14, 15],
        }
    )
    df.index = pd.DatetimeIndex(df["record_date"], name="record_date")

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_static_category_chart(all_counts, _col):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([str(x) for x in all_counts.index], all_counts.to_numpy(dtype=float))
        return fig, ax

    monkeypatch.setattr(downloads_mod, "_build_static_category_chart", fake_static_category_chart)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/zip"

        with zipfile.ZipFile(io.BytesIO(response.data), "r") as zf:
            names = zf.namelist()

    assert any(name.endswith("city_categories.png") for name in names)
    assert not any(name.endswith("record_date_categories.png") for name in names)


def test_static_plots_zip_trend_uses_forecast_renderer(monkeypatch):
    """ZIP trend images should use generate_forecast_plot for consistent rendering."""
    filename = "b" * 40 + ".csv"
    NUMERIC_DF_CACHE.clear()

    df = pd.DataFrame({"year": np.linspace(2000, 2010, 60), "city": ["A"] * 60})

    calls = {"forecast": 0, "trend": 0}

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "generate_correlation_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_plot", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generate_plot should not be used for ZIP trend")))

    def fake_generate_forecast_plot(*_args, **_kwargs):
        calls["forecast"] += 1
        # 1x1 transparent PNG
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII="

    monkeypatch.setattr(app_module, "generate_forecast_plot", fake_generate_forecast_plot)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.data), "r") as zf:
            names = zf.namelist()

    assert any(name.endswith("_trend.png") for name in names)
    assert calls["forecast"] >= 1


def test_analyze_forecast_zero_pct_renders_history_only_forecast(monkeypatch):
    """Detailed Analysis at 0% should still render forecast charts using history-only traces."""
    filename = "0" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(1, 61, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)

    captured = {}

    def fake_render_template(_template, **kwargs):
        captured["analysis"] = kwargs.get("analysis", {})
        return "ok"

    calls: list[dict[str, Any]] = []

    def fake_generate_forecast_plot(history, forecast_series, title, *_args, **_kwargs):
        calls.append(
            {
                "history_len": len(history) if history is not None else 0,
                "has_forecast": forecast_series is not None,
                "title": str(title),
            }
        )
        return "x"

    monkeypatch.setattr(app_module, "render_template", fake_render_template)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_forecast_plot", fake_generate_forecast_plot)

    def fail_if_forecast_requested(_filename, _column, _series, _steps):
        raise AssertionError("get_cached_column_forecast should not be called when forecast_pct=0")

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fail_if_forecast_requested)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=forecast&forecast_pct=0")

    assert response.status_code == 200
    assert calls
    assert all(call["has_forecast"] is False for call in calls)
    grouped = captured.get("analysis", {}).get("forecast_plots_by_column", {})
    grouped_types = [p.get("type") for plots in grouped.values() for p in plots if isinstance(p, dict)]
    assert "forecast" in grouped_types


def test_download_full_report_pdf_categories_not_capped_to_top_50(monkeypatch):
    """PDF categories should render all category bars via shared helper (no top-50 truncation)."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "9" * 40 + ".csv"
    categories = [
        f"cat_{chr(65 + (i % 26))}{chr(65 + ((i // 26) % 26))}{chr(65 + ((i // (26 * 26)) % 26))}"
        for i in range(75)
    ]
    values = categories + categories
    df = pd.DataFrame({"category": values})

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    try:
        app_module.REPORT_CACHE.clear()
    except Exception:
        pass

    captured_counts: list[int] = []
    original_builder = pdf_report_mod._build_static_category_chart

    def spy_build_static_category_chart(all_counts, col):
        captured_counts.append(int(len(all_counts)))
        return original_builder(all_counts, col)

    monkeypatch.setattr(pdf_report_mod, "_build_static_category_chart", spy_build_static_category_chart)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert captured_counts
    assert max(captured_counts) == 75


def test_download_full_report_pdf_category_exports_use_tiny_pad(monkeypatch):
    """PDF category chart exports should use the same tiny crop pad as ZIP exports."""
    from matplotlib.figure import Figure

    filename = "7" * 40 + ".csv"
    df = pd.DataFrame({"city": ["Iasi", "Cluj", "Iasi", "Bacau", "Cluj", "Iasi"]})
    tiny_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=")
    captured_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    try:
        app_module.REPORT_CACHE.clear()
    except Exception:
        pass

    def fake_savefig(self, fp, *args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        fp.write(tiny_png)

    monkeypatch.setattr(Figure, "savefig", fake_savefig)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test")

    assert response.status_code == 200
    assert captured_kwargs
    assert captured_kwargs[0].get("pad_inches") == 0.02


def test_download_full_report_pdf_skips_active_temporal_axis_category_column(monkeypatch):
    """PDF should not open a blank category page for the datetime column already used as x-axis."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "8" * 40 + ".csv"
    record_date = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame({
        "record_date": record_date,
        "city": ["Iasi", "Cluj", "Iasi", "Cluj", "Bucharest", "Iasi", "Cluj", "Iasi"],
        "value": np.linspace(10.0, 17.0, 8),
    })
    df.index = pd.DatetimeIndex(df["record_date"], name="record_date")

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    def fake_get_cached_column_forecast(_filename, _column, series, steps):
        n_steps = int(steps)
        idx = pd.date_range(series.index[-1], periods=n_steps + 1, freq="D")[1:]
        base = float(series.iloc[-1]) if len(series) else 0.0
        fc = pd.Series(np.full(n_steps, base, dtype=float), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=idx)
        return fc, ci

    monkeypatch.setattr(app_module, "get_cached_column_forecast", fake_get_cached_column_forecast)
    try:
        app_module.REPORT_CACHE.clear()
    except Exception:
        pass

    original_cell = pdf_report_mod.PDFReport.cell
    rendered_columns: list[str] = []

    def spy_cell(self, *args, **kwargs):
        text = ""
        if len(args) >= 3:
            text = str(args[2])
        elif "text" in kwargs:
            text = str(kwargs["text"])
        if text.startswith("Column: "):
            rendered_columns.append(text.removeprefix("Column: "))
        return original_cell(self, *args, **kwargs)

    monkeypatch.setattr(pdf_report_mod.PDFReport, "cell", spy_cell)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0.2")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert "record_date" not in rendered_columns
    assert "city" in rendered_columns
    assert "value" in rendered_columns


def test_download_cleaned_csv_omits_duplicate_first_column_index():
    """Cleaned CSV should not duplicate the first column when index mirrors an existing data column."""
    filename = "a" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "record_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10.0, 11.5, 13.0],
        }
    )
    df.index = pd.Index(df["record_date"].tolist(), name="record_date")

    DATAFRAME_CACHE.set(filename, df)
    try:
        with app.test_client() as client:
            response = client.get(f"/download/{filename}/cleaned.csv")
    finally:
        DATAFRAME_CACHE.pop(filename, None)

    assert response.status_code == 200
    csv_text = response.get_data(as_text=True)
    parsed = pd.read_csv(io.StringIO(csv_text))
    assert list(parsed.columns) == ["record_date", "value"]
    assert not any(str(col).startswith("Unnamed") for col in parsed.columns)


def test_download_cleaned_csv_keeps_unique_named_index_metadata():
    """Cleaned CSV should include index when it is unique metadata not represented by a data column."""
    filename = "b" * 40 + ".csv"
    df = pd.DataFrame(
        {"value": [1.2, 3.4, 5.6]},
        index=pd.Index(["sample_A", "sample_B", "sample_C"], name="sample_id"),
    )

    DATAFRAME_CACHE.set(filename, df)
    try:
        with app.test_client() as client:
            response = client.get(f"/download/{filename}/cleaned.csv")
    finally:
        DATAFRAME_CACHE.pop(filename, None)

    assert response.status_code == 200
    csv_text = response.get_data(as_text=True)
    parsed = pd.read_csv(io.StringIO(csv_text))
    assert list(parsed.columns)[0] == "sample_id"
    assert list(parsed["sample_id"].astype(str)) == ["sample_A", "sample_B", "sample_C"]


def test_sample_numeric_axis_ticks_prefers_nice_steps_and_full_coverage():
    """Static distribution ticks should use nice anchored steps and cover full min/max range."""
    from data_analysis.analysis.plot import _sample_numeric_axis_ticks

    source_values = [149.0, 190.0, 231.0, 272.0, 313.0, 354.0, 395.0, 436.0, 477.0, 519.0, 560.0, 601.0, 642.0, 683.0]
    tick_values, tick_labels = _sample_numeric_axis_ticks(
        source_values,
        max_tick_labels=20,
        min_spacing_ratio=0.22,
    )

    assert len(tick_values) >= 8
    assert tick_values[0] <= min(source_values)
    assert tick_values[-1] >= max(source_values)

    diffs = np.diff(np.asarray(tick_values, dtype=float))
    assert len(diffs) >= 6
    assert np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8)

    step = float(abs(diffs[0]))
    exponent = float(np.floor(np.log10(step))) if step > 0 else 0.0
    normalized = step / (10.0 ** exponent) if step > 0 else 1.0
    assert any(abs(normalized - anchor) < 1e-6 for anchor in (1.0, 2.0, 2.5, 5.0, 10.0))

    assert len(set(tick_labels)) == len(tick_labels)
    assert all(str(label).lstrip("-").isdigit() for label in tick_labels)


def test_build_category_plotly_chart_increases_y_axis_tick_density():
    """Category page layout should request a denser, overlap-safe y-axis tick budget."""
    s_cat = pd.Series([f"group_{i % 18}" for i in range(900)])
    chart = _build_category_plotly_chart(s_cat, "group")
    assert chart is not None

    layout = cast(dict[str, Any], chart["layout"])
    yaxis = cast(dict[str, Any], layout["yaxis"])
    assert int(yaxis.get("nticks", 0)) >= 12


def test_interactive_template_keeps_fractional_tick_precision_logic():
    """Interactive axis formatter should not force integer labels for sub-unit tick values."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "formatPreciseNumericValue(numeric, 0)" not in html
    assert "formatPreciseNumericValue(numeric);" in html


def test_download_full_report_pdf_keeps_distribution_and_stl_on_same_page(monkeypatch):
    """PDF export should render distribution and STL charts together on the same page."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = ("1234abcd" * 5) + ".csv"
    idx = pd.date_range("2024-01-01", periods=64, freq="D")
    df = pd.DataFrame({"value": np.linspace(0.12, 0.88, len(idx))}, index=idx)

    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII="
    image_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    # Skip trend/forecast images so the last two image calls are distribution + STL.
    monkeypatch.setattr(app_module, "generate_forecast_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: tiny_png_b64)

    original_image = pdf_report_mod.PDFReport.image

    def spy_image(self, name, *args, **kwargs):
        width = kwargs.get("w")
        if width is None and len(args) >= 3:
            width = args[2]
        image_calls.append({"page": int(self.page_no()), "w": float(width) if width is not None else None})
        return original_image(self, name, *args, **kwargs)

    monkeypatch.setattr(pdf_report_mod.PDFReport, "image", spy_image)
    try:
        app_module.DATAFRAME_CACHE.pop(filename, None)
        app_module.NUMERIC_DF_CACHE.pop(filename, None)
        app_module.REPORT_CACHE.pop(filename, None)
    except Exception:
        pass

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0")

    try:
        app_module.DATAFRAME_CACHE.pop(filename, None)
        app_module.NUMERIC_DF_CACHE.pop(filename, None)
        app_module.REPORT_CACHE.pop(filename, None)
    except Exception:
        pass

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert len(image_calls) >= 2

    distribution_call = image_calls[-2]
    stl_call = image_calls[-1]
    assert distribution_call["page"] == stl_call["page"]


def test_interactive_template_uses_denser_yaxis_tick_helper():
    """Interactive template should densify y-axis ticks for both main and distribution charts."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "function getInteractiveYAxisTickCount(container, options = {})" in html
    assert "const mainYTickCount = getInteractiveYAxisTickCount(chartContainer" in html
    assert "const yTickCount = getInteractiveYAxisTickCount(distContainer" in html


def test_interactive_template_distribution_extrema_tags_are_theme_legible():
    """Distribution min/max annotations should include theme-aware contrast pill styling."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "function buildDistributionExtremaAnnotation(value, text, color, axisMin, axisMax" in html
    assert "bgcolor: options.bgcolor || 'rgba(255, 255, 255, 0.92)'" in html
    assert "bordercolor: options.bordercolor || color" in html
    assert "const markerLaneYMin = 0.012;" in html
    assert "const markerLaneYMax = 0.0075;" in html
    assert "const markerTagLaneY = markerLaneYMin + 0.0015;" in html
    assert "y: [markerLaneYMin]" in html
    assert "y: [markerLaneYMax]" in html
    assert "y: markerTagLaneY" in html
    assert "formatStat(s.min)" in html
    assert "formatStat(s.max)" in html
    assert "buildDistributionExtremaAnnotation(\n                s.min,\n                formatStat(s.min)," in html
    assert "buildDistributionExtremaAnnotation(\n                s.max,\n                formatStat(s.max)," in html
    assert "buildDistributionExtremaAnnotation(\n                s.min,\n                formatLegendStatLabel('Min', s.min)," not in html
    assert "buildDistributionExtremaAnnotation(\n                s.max,\n                formatLegendStatLabel('Max', s.max)," not in html
    assert "name: 'min-annot'" in html
    assert "name: 'max-annot'" in html


def test_interactive_template_distribution_avg_med_tags_restored_above_chart():
    """Interactive distribution should render plain-text Avg/Med tags in opposite outside lanes."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "function buildDistributionLineTagAnnotation" in html
    assert "name: 'avg-line-annot'" in html
    assert "name: 'med-line-annot'" in html
    assert "Number.isFinite(options.y) ? Number(options.y) : 0.992" in html
    assert "const statTagLaneY = 1.006;" in html
    assert "forceX: avgTagX" in html
    assert "forceX: medTagX" in html
    assert "xanchor: avgTagAnchor" in html
    assert "xanchor: medTagAnchor" in html
    assert "const minTagGap = Math.max(distSpan * 0.020, sideOffset * 2.4);" in html
    assert "candidateLeftTagX" in html
    assert "candidateRightTagX" in html
    assert "edgeInsetRatio: 0.014" in html
    assert "boldText: false" in html


def test_interactive_template_forecast_and_distribution_tags_drop_hard_bold_wrappers():
    """Interactive forecast/distribution stat tags should avoid hard <b> wrappers for readability."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "text: [formatStat(filteredMin)]" in html
    assert "text: [formatStat(filteredMax)]" in html
    assert "const avgText = formatLegendStatLabel('Avg', s.mean, { decimals: 1 });" in html
    assert "const medText = formatLegendStatLabel('Med', s.median, { decimals: 1 });" in html
    assert "`<b>${formatStat(filteredMin)}</b>`" not in html
    assert "`<b>${formatStat(filteredMax)}</b>`" not in html


def test_interactive_template_distribution_tick_density_targets_dense_labels():
    """Interactive distribution tick config should target denser x-label coverage when width permits."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "maxTicks: 88" in html
    assert "pxPerTick: 20" in html
    assert "function formatDistributionTickValue(value)" in html
    assert "minCompactValue: 1e3," in html


def test_interactive_template_distribution_tick_thinner_uses_collision_fit():
    """Interactive distribution tick thinning should maximize labels while preventing text collisions."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "function thinDistributionTickConfig(tickConfig, containerWidth, options = {})" in html
    assert "const labelWidths = ticktext.map((label) => {" in html
    assert "const sampleEvenTickIndexes = (targetCount) => {" in html
    assert "const hasTickLabelCollision = (indexes) => {" in html
    assert "while (low <= high)" in html


def test_interactive_template_distribution_markers_use_theme_colors_without_strokes():
    """Interactive distribution/forecast min/max markers should avoid marker stroke outlines."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "markerStroke" not in html
    assert "line: { width: 1.2, color: statPalette.markerStroke }" not in html


def test_interactive_template_distribution_marker_lane_axis_stays_pinned_on_relayout():
    """Distribution relayout handler should keep y2 fixed so min/max markers do not drift after autoscale."""
    template_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../templates/analysis.html")
    )
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    assert "distEl.on('plotly_relayout'" in html
    assert "'yaxis2.range': [0, 1]" in html
    assert "'yaxis2.autorange': false" in html
    assert "'yaxis2.fixedrange': true" in html


def test_sample_numeric_axis_ticks_uses_compact_labels_for_large_magnitudes():
    """Static distribution ticks should use compact k/M/B labels for large ranges."""
    from data_analysis.analysis.plot import _sample_numeric_axis_ticks

    source_values = [12_500.0, 18_700.0, 25_100.0, 31_400.0, 379_000.0, 4_420_000.0, 9_700_000.0]
    tick_values, tick_labels = _sample_numeric_axis_ticks(
        source_values,
        max_tick_labels=12,
        min_spacing_ratio=0.22,
    )

    assert len(tick_labels) >= 4
    assert tick_values[0] <= min(source_values)
    assert tick_values[-1] >= max(source_values)
    assert len(set(tick_labels)) == len(tick_labels)
    assert any(str(label).endswith(("k", "M", "B", "T")) for label in tick_labels)


def test_sample_numeric_axis_ticks_do_not_skip_small_integer_spans():
    """Small integer spans should show every integer tick (e.g., year-by-year axes)."""
    from data_analysis.analysis.plot import _sample_numeric_axis_ticks

    source_values = [float(v) for v in range(2000, 2016)]
    tick_values, tick_labels = _sample_numeric_axis_ticks(
        source_values,
        max_tick_labels=26,
        min_spacing_ratio=0.14,
    )

    expected = list(range(2000, 2016))
    assert [int(round(v)) for v in tick_values] == expected
    assert tick_labels == [str(v) for v in expected]


def test_sample_histogram_bin_ticks_align_to_bin_centers_for_integer_spans():
    """Distribution ticks should map to actual histogram bar centers for integer-like spans."""
    from data_analysis.analysis.plot import (
        _resolve_distribution_histogram_bins,
        _sample_histogram_bin_ticks,
    )

    values = np.repeat(np.arange(2000, 2016, dtype=float), 4)
    bins = _resolve_distribution_histogram_bins(values.tolist(), min_bins=8, max_bins=52)
    if isinstance(bins, np.ndarray):
        edges = bins.astype(float)
    else:
        edges = np.histogram_bin_edges(values, bins=int(bins)).astype(float)

    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(float)
    tick_values, tick_labels = _sample_histogram_bin_ticks(
        edges,
        max_tick_labels=24,
        min_spacing_ratio=0.14,
    )

    assert len(tick_values) >= 12
    assert len(set(tick_labels)) == len(tick_labels)
    assert all(np.any(np.isclose(tv, centers, atol=1e-9)) for tv in tick_values)


def test_sample_histogram_bin_ticks_do_not_skip_small_integer_year_spans():
    """Integer year spans should keep every year tick when max labels can fit them."""
    from data_analysis.analysis.plot import _sample_histogram_bin_ticks

    year_edges = np.arange(1999.5, 2015.5 + 1.0, 1.0, dtype=float)
    tick_values, tick_labels = _sample_histogram_bin_ticks(
        year_edges,
        max_tick_labels=16,
        min_spacing_ratio=0.14,
    )

    expected_years = [str(year) for year in range(2000, 2016)]
    assert tick_labels == expected_years
    assert [int(round(v)) for v in tick_values] == list(range(2000, 2016))


def test_sample_histogram_bin_ticks_allows_non_nice_center_values():
    """Histogram tick sampler should preserve non-nice bar-center values when needed."""
    from data_analysis.analysis.plot import _sample_histogram_bin_ticks

    edges = np.linspace(0.17, 3.89, 41, dtype=float)
    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(float)

    tick_values, _tick_labels = _sample_histogram_bin_ticks(
        edges,
        max_tick_labels=11,
        min_spacing_ratio=0.18,
    )

    assert tick_values
    assert all(np.any(np.isclose(tv, centers, atol=1e-9)) for tv in tick_values)
    assert any(abs(tv - round(tv)) > 1e-6 for tv in tick_values)


def test_generate_correlation_heatmap_uses_angled_x_labels(monkeypatch):
    """Correlation heatmap should rotate x tick labels similarly to interactive category styling."""
    from matplotlib.figure import Figure

    from data_analysis.analysis.plot import generate_correlation_heatmap

    captured: dict[str, float] = {}
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        if "rotation" not in captured and self.axes:
            ax = self.axes[0]
            self.canvas.draw()
            labels = [tick for tick in ax.get_xticklabels() if str(tick.get_text()).strip()]
            if labels:
                raw_rotation = float(labels[0].get_rotation())
                normalized = ((raw_rotation + 180.0) % 360.0) - 180.0
                captured["rotation"] = normalized
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    df = pd.DataFrame(
        {
            "very_long_feature_name_1": np.linspace(1.0, 10.0, 60),
            "very_long_feature_name_2": np.linspace(2.0, 20.0, 60),
            "very_long_feature_name_3": np.linspace(3.0, 30.0, 60),
        }
    )
    img_b64 = generate_correlation_heatmap(df, method="spearman", title="Spearman Correlation")

    assert isinstance(img_b64, str)
    assert len(img_b64) > 0
    assert "rotation" in captured
    assert -45.0 <= float(captured["rotation"]) <= -20.0


def test_generate_correlation_heatmap_export_uses_near_horizontal_labels(monkeypatch):
    """Export preset should keep x tick labels near-horizontal for readability."""
    from matplotlib.figure import Figure

    from data_analysis.analysis.plot import generate_correlation_heatmap

    captured: dict[str, float] = {}
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        if "rotation" not in captured and self.axes:
            ax = self.axes[0]
            self.canvas.draw()
            labels = [tick for tick in ax.get_xticklabels() if str(tick.get_text()).strip()]
            if labels:
                raw_rotation = float(labels[0].get_rotation())
                normalized = ((raw_rotation + 180.0) % 360.0) - 180.0
                captured["rotation"] = normalized
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    df = pd.DataFrame(
        {
            "very_long_feature_name_1": np.linspace(1.0, 10.0, 60),
            "very_long_feature_name_2": np.linspace(2.0, 20.0, 60),
            "very_long_feature_name_3": np.linspace(3.0, 30.0, 60),
            "very_long_feature_name_4": np.linspace(4.0, 40.0, 60),
            "very_long_feature_name_5": np.linspace(5.0, 50.0, 60),
            "very_long_feature_name_6": np.linspace(6.0, 60.0, 60),
            "very_long_feature_name_7": np.linspace(7.0, 70.0, 60),
            "very_long_feature_name_8": np.linspace(8.0, 80.0, 60),
        }
    )
    img_b64 = generate_correlation_heatmap(
        df,
        method="spearman",
        title="Spearman Correlation Export",
        layout_preset="export",
    )

    assert isinstance(img_b64, str)
    assert len(img_b64) > 0
    assert "rotation" in captured
    assert -24.0 <= float(captured["rotation"]) <= 0.0


def test_generate_correlation_heatmap_export_preset_is_taller(monkeypatch):
    """Export preset should render taller correlation figures than default preset."""
    from matplotlib.figure import Figure

    from data_analysis.analysis.plot import generate_correlation_heatmap

    captured_sizes: list[tuple[float, float]] = []
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        size = tuple(float(v) for v in self.get_size_inches().tolist())
        captured_sizes.append((size[0], size[1]))
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    df = pd.DataFrame(
        {
            "feature_1": np.linspace(1.0, 10.0, 120),
            "feature_2": np.linspace(2.0, 20.0, 120),
            "feature_3": np.linspace(3.0, 30.0, 120),
            "feature_4": np.linspace(4.0, 40.0, 120),
            "feature_5": np.linspace(5.0, 50.0, 120),
            "feature_6": np.linspace(6.0, 60.0, 120),
            "feature_7": np.linspace(7.0, 70.0, 120),
            "feature_8": np.linspace(8.0, 80.0, 120),
        }
    )

    default_b64 = generate_correlation_heatmap(df, method="spearman", title="Default Corr", layout_preset="default")
    export_b64 = generate_correlation_heatmap(df, method="spearman", title="Export Corr", layout_preset="export")

    assert isinstance(default_b64, str) and len(default_b64) > 0
    assert isinstance(export_b64, str) and len(export_b64) > 0
    assert len(captured_sizes) >= 2
    default_size = captured_sizes[-2]
    export_size = captured_sizes[-1]
    assert export_size[1] > default_size[1]


def test_download_full_report_pdf_keeps_both_correlations_on_same_page(monkeypatch):
    """PDF export should keep Spearman and Pearson correlation charts together on one page when possible."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "f" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "x": np.linspace(1.0, 50.0, 80),
            "y": np.linspace(10.0, 100.0, 80),
            "z": np.linspace(3.0, 300.0, 80),
        }
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [0, 1])
    ax.set_title("dummy")
    corr_buf = io.BytesIO()
    fig.savefig(corr_buf, format="png", dpi=100)
    plt.close(fig)
    corr_buf.seek(0)
    wide_png_b64 = base64.b64encode(corr_buf.read()).decode("utf-8")
    image_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: wide_png_b64)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    original_image = pdf_report_mod.PDFReport.image

    def spy_image(self, name, *args, **kwargs):
        width = kwargs.get("w")
        if width is None and len(args) >= 3:
            width = args[2]
        image_calls.append({"page": int(self.page_no()), "w": float(width) if width is not None else None})
        return original_image(self, name, *args, **kwargs)

    monkeypatch.setattr(pdf_report_mod.PDFReport, "image", spy_image)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert len(image_calls) >= 2

    first_corr = image_calls[0]
    second_corr = image_calls[1]
    assert first_corr["page"] == second_corr["page"]


def test_download_full_report_pdf_omits_correlation_caption_cells(monkeypatch):
    """PDF correlation section should render charts without Spearman/Pearson caption text rows."""
    import data_analysis.reports.pdf_report as pdf_report_mod

    filename = "a" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "x": np.linspace(1.0, 50.0, 80),
            "y": np.linspace(10.0, 100.0, 80),
            "z": np.linspace(3.0, 300.0, 80),
        }
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [0, 1])
    ax.set_title("dummy")
    corr_buf = io.BytesIO()
    fig.savefig(corr_buf, format="png", dpi=100)
    plt.close(fig)
    corr_buf.seek(0)
    corr_png_b64 = base64.b64encode(corr_buf.read()).decode("utf-8")

    cell_texts: list[str] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: False)
    monkeypatch.setattr(app_module, "describe_for_ai", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "get_cached_heatmap", lambda *_args, **_kwargs: corr_png_b64)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))

    original_cell = pdf_report_mod.PDFReport.cell

    def spy_cell(self, *args, **kwargs):
        if len(args) >= 3:
            txt = args[2]
        else:
            txt = kwargs.get("txt", kwargs.get("text", ""))
        if isinstance(txt, str):
            cell_texts.append(txt)
        return original_cell(self, *args, **kwargs)

    monkeypatch.setattr(pdf_report_mod.PDFReport, "cell", spy_cell)

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/report.pdf?display=test&forecast_pct=0")

    assert response.status_code == 200
    assert response.data.startswith(b"%PDF")
    assert all("Spearman Correlation:" not in txt for txt in cell_texts)
    assert all("Pearson Correlation:" not in txt for txt in cell_texts)
