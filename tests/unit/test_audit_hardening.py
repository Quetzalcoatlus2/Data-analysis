import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root import resolution is stable in all runners.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_analysis.routes import upload as upload_mod
from data_analysis.runtime_app import FORECAST_CACHE, _compute_forecast, _match_amplitude


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_upload_mime_type_resolution_for_supported_extensions():
    """Upload MIME selection should match supported file extensions."""
    assert upload_mod._mime_type_for_upload("dataset.csv") == "text/csv"
    assert upload_mod._mime_type_for_upload("dataset.txt") == "text/plain"
    assert upload_mod._mime_type_for_upload("dataset.json") == "application/json"
    assert (
        upload_mod._mime_type_for_upload("dataset.xlsx")
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def test_compute_forecast_cache_key_avoids_same_stats_collision():
    """Different series with same first/last/mean/std should not share forecast cache entries."""
    FORECAST_CACHE.clear()
    try:
        # Same length, same first/last, same mean and std; only order differs.
        s1 = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], index=pd.RangeIndex(6))
        s2 = pd.Series([0.0, 8.0, 6.0, 4.0, 2.0, 10.0], index=pd.RangeIndex(6))

        _compute_forecast(s1, 8)
        _compute_forecast(s2, 8)

        assert len(FORECAST_CACHE) == 2
    finally:
        FORECAST_CACHE.clear()


def test_match_amplitude_flat_forecast_is_reproducible():
    """The local RNG fallback for flat forecasts should be deterministic."""
    history = pd.Series([0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0])
    forecast = pd.Series([8.0] * 6, index=pd.RangeIndex(10, 16))

    first, first_ci = _match_amplitude(history, forecast)
    second, second_ci = _match_amplitude(history, forecast)

    pd.testing.assert_series_equal(first, second)
    assert first_ci is second_ci is None


def test_compute_forecast_flat_and_short_series_are_reproducible():
    for series in (
        pd.Series([5.0] * 8, index=pd.RangeIndex(8)),
        pd.Series([2.0, 2.0], index=pd.RangeIndex(2)),
    ):
        FORECAST_CACHE.clear()
        first_fc, first_ci = _compute_forecast(series, 4)
        FORECAST_CACHE.clear()
        second_fc, second_ci = _compute_forecast(series, 4)

        pd.testing.assert_series_equal(first_fc, second_fc)
        pd.testing.assert_frame_equal(first_ci, second_ci)
    FORECAST_CACHE.clear()


def test_gitignore_covers_runtime_artifacts():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in (
        ".venv/",
        "__pycache__/",
        "*.py[cod]",
        ".env",
        "app.log",
        "data-analysis-*.json",
        "datasets/",
    ):
        assert pattern in gitignore


def test_no_tracked_python_bytecode_files():
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git ls-files unavailable: {exc}")

    tracked = result.stdout.splitlines()
    bytecode = [
        path for path in tracked
        if path.endswith(".pyc") or "__pycache__/" in path.replace("\\", "/")
    ]
    assert bytecode == []

