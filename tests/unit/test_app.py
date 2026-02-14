
import sys
import os
import io
import zipfile

import app as app_module

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

# Add project root to sys.path to ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import (
    AI_DESCRIBE_CACHE,
    DATAFRAME_CACHE,
    INTERACTIVE_DATA_CACHE,
    NUMERIC_DF_CACHE,
    TinyLRU,
    _build_category_plotly_chart,
    _build_interactive_cache_key,
    _compute_basic_stats,
    _compute_forecast,
    _thin_series_keep_extrema,
    allowed_file,
    app,
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
    """Anomaly markers should be placed at every matching duplicate index position."""
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
    assert captured[0][0] == [0.0, 2.0]
    assert captured[0][1] == [10.0, 30.0]


def test_build_category_plotly_chart_has_bar_labels():
    """Ensure category chart builder includes bar value labels."""
    series = pd.Series(["A", "A", "B", "C", "C", "C"])
    chart = _build_category_plotly_chart(series, "Category")
    assert chart is not None
    traces = chart.get("traces") or []
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

    traces = chart.get("traces") or []
    layout = chart.get("layout") or {}
    shapes = layout.get("shapes") or []

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
    pd.testing.assert_frame_equal(ci1, ci2)


def test_compute_forecast_does_not_mutate_global_rng():
    """Verify _compute_forecast does not change numpy global random state."""
    series = pd.Series(range(50), index=pd.RangeIndex(50), dtype=float)
    # Set a known global state
    np.random.seed(999)
    state_before = np.random.get_state()[1].copy()
    _compute_forecast(series, 5)
    state_after = np.random.get_state()[1].copy()
    # Global state should be unchanged
    assert np.array_equal(state_before, state_after), "Global RNG state was mutated by _compute_forecast"


def test_detect_anomalies_returns_index():
    """Ensure detect_anomalies returns proper types."""
    series = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
    an_idx, an_score = detect_anomalies(series, contamination=0.1)
    assert isinstance(an_idx, pd.Index)
    assert isinstance(an_score, pd.Series)


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
    fc_vals = fc.values

    # Slide a window of length 30 over history; none should match the forecast
    for start in range(len(history) - 30 + 1):
        segment = history[start:start + 30]
        # Even with an offset shift (like old code did), the shape shouldn't match
        # Check normalised correlation: if segment == shifted copy, corr ≈ 1.0
        seg_centered = segment - segment.mean()
        fc_centered = fc_vals - fc_vals.mean()
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

    assert fc.min() >= data_min - 1e-9, f"Forecast min {fc.min()} < data_min {data_min}"
    assert fc.max() <= data_max + 1e-9, f"Forecast max {fc.max()} > data_max {data_max}"
    assert ci["lower"].min() >= data_min - 1e-9, f"CI lower min {ci['lower'].min()} < data_min {data_min}"
    assert ci["upper"].max() <= data_max + 1e-9, f"CI upper max {ci['upper'].max()} > data_max {data_max}"

    # Values should not sit exactly on boundaries (no saturation by clipping).
    assert not np.isclose(fc.values, data_min, atol=1e-9).any()
    assert not np.isclose(fc.values, data_max, atol=1e-9).any()


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

    hist_diffs = np.diff(series.values)
    fc_diffs = np.diff(fc.values.astype(float))
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
    fc_vals = fc.values.astype(float)

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
    fc_mean = float(np.mean(fc.values.astype(float)))
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
