
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path to ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import (
    AI_DESCRIBE_CACHE,
    _build_category_plotly_chart,
    _compute_basic_stats,
    _compute_forecast,
    allowed_file,
    app,
    describe_for_ai,
    detect_anomalies,
    generate_forecast_plot,
    generate_plot,
    get_cached_anomalies,
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


def test_forecast_stays_within_data_bounds():
    """Ensure forecast values and CI never exceed historical min/max."""
    rng = np.random.default_rng(7)
    # Data with clear bounds [20, 80]
    values = rng.uniform(20, 80, size=150)
    series = pd.Series(values, index=pd.RangeIndex(150))
    data_min = float(series.min())
    data_max = float(series.max())

    fc, ci = _compute_forecast(series, 30)

    # Forecast values must stay within [data_min, data_max]
    assert fc.min() >= data_min - 1e-9, f"Forecast min {fc.min()} < data_min {data_min}"
    assert fc.max() <= data_max + 1e-9, f"Forecast max {fc.max()} > data_max {data_max}"

    # CI bounds must also stay within [data_min, data_max]
    assert ci["lower"].min() >= data_min - 1e-9, f"CI lower min {ci['lower'].min()} < data_min {data_min}"
    assert ci["upper"].max() <= data_max + 1e-9, f"CI upper max {ci['upper'].max()} > data_max {data_max}"


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
