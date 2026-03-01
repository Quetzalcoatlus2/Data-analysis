
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
    """Numeric labels should use K/M/B/T compact notation with trailing-zero stripping."""
    from data_analysis.analysis.plot import _format_stat_value

    assert _format_stat_value(123_456_000_000_000.0) == "123.456T"
    assert _format_stat_value(1_200_000_000.0) == "1.2B"
    assert _format_stat_value(1_234_560_000.0) == "1.235B"
    assert _format_stat_value(12_345.0) == "12.345K"
    assert _format_stat_value(1_500_000.0) == "1.5M"
    assert _format_stat_value(8.0) == "8"
    assert _format_stat_value(8.50) == "8.5"
    assert _format_stat_value(1e16) == "1.000e+16"


def test_generate_forecast_plot_returns_base64():
    """Ensure generate_forecast_plot returns a non-empty base64 string."""
    history_idx = pd.date_range("2024-01-01", periods=5, freq="D")
    history = pd.Series([1, 2, 3, 4, 5], index=history_idx)
    forecast_idx = pd.date_range("2024-01-06", periods=3, freq="D")
    forecast = pd.Series([5.5, 6.0, 6.4], index=forecast_idx)
    img = generate_forecast_plot(history, forecast, "Forecast", "Timestamp", "Value")
    assert isinstance(img, str)
    assert len(img) > 0


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
    std_hist = float(np.std(hist_diffs, ddof=1))
    std_fc = float(np.std(fc_diffs, ddof=1))

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
    """Interactive API should return full data; range reduction is client-side."""
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
    hist_x = set(int(v) for v in hist.get("x", []))
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
    """Interactive cache key should isolate forecast/contamination variants."""
    filename = "a" * 40 + ".csv"
    k1 = _build_interactive_cache_key(filename, 0.05, 0.02)
    k2 = _build_interactive_cache_key(filename, 0.10, 0.02)
    k3 = _build_interactive_cache_key(filename, 0.05, 0.05)
    assert k1 != k2
    assert k1 != k3


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
