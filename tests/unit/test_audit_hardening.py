import os
import sys

import numpy as np
import pandas as pd

# Ensure project root import resolution is stable in all runners.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_analysis.analysis import anomaly as anomaly_mod
from data_analysis.routes import upload as upload_mod
from data_analysis.runtime_app import FORECAST_CACHE, _compute_forecast


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


def test_detect_anomalies_skips_stl_when_series_exceeds_cap(monkeypatch):
    """STL stage should be skipped when max_stl_points threshold is exceeded."""
    stl_calls = {"count": 0}

    def _fake_get_stl():
        stl_calls["count"] += 1
        raise AssertionError("STL should not be called above cap")

    class _FakeIF:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, x):
            return np.ones(len(x), dtype=int)

        def decision_function(self, x):
            return np.zeros(len(x), dtype=float)

    monkeypatch.setattr(anomaly_mod, "get_stl", _fake_get_stl)
    monkeypatch.setattr(anomaly_mod, "get_isolation_forest", lambda: _FakeIF)

    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    series = pd.Series(np.sin(np.arange(200, dtype=float)), index=idx)

    an_idx, an_score = anomaly_mod.detect_anomalies(
        series,
        contamination=0.05,
        is_reliable_timeseries_index=lambda _idx: True,
        infer_seasonal_period=lambda _idx: 7,
        max_stl_points=50,
    )

    assert stl_calls["count"] == 0
    assert isinstance(an_idx, pd.Index)
    assert isinstance(an_score, pd.Series)
