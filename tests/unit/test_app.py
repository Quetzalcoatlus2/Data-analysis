
import sys
import os

import pandas as pd
import pytest

# Add project root to sys.path to ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import (
    _build_category_plotly_chart,
    _compute_basic_stats,
    allowed_file,
    app,
    generate_forecast_plot,
    generate_plot,
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
