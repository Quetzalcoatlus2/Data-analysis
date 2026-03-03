import os
import sys

import pandas as pd

# Ensure project root import resolution is stable in all runners.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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



