import numpy as np
import pandas as pd
import pytest

import data_analysis.analysis.research as research_module
import data_analysis.runtime_app as app_module
from app import app


def _seed_dataframe(filename: str, df: pd.DataFrame) -> None:
    app_module.DATAFRAME_CACHE.set(filename, df)
    app_module.NUMERIC_DF_CACHE.pop(filename, None)


def _clear_dataframe(filename: str) -> None:
    app_module.DATAFRAME_CACHE.pop(filename, None)
    app_module.NUMERIC_DF_CACHE.pop(filename, None)


def _install_fast_forecast_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_cached_forecast(_filename, _column, series, steps):
        horizon = int(max(0, steps))
        if horizon <= 0:
            return pd.Series(dtype=float), None
        base = float(series.iloc[-1]) if len(series) else 0.0
        idx = pd.RangeIndex(start=len(series), stop=len(series) + horizon)
        fc = pd.Series(base + np.linspace(0.1, 0.1 * horizon, horizon), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.2, "upper": fc + 0.2}, index=idx)
        return fc, ci

    def _fake_compute_forecast(series, steps):
        horizon = int(max(0, steps))
        if horizon <= 0:
            return pd.Series(dtype=float), None
        base = float(series.iloc[-1]) if len(series) else 0.0
        idx = pd.RangeIndex(start=len(series), stop=len(series) + horizon)
        fc = pd.Series(base + np.linspace(0.05, 0.05 * horizon, horizon), index=idx)
        ci = pd.DataFrame({"lower": fc - 0.1, "upper": fc + 0.1}, index=idx)
        return fc, ci

    def _fake_cached_anomalies(_filename, _column, series, _contamination=0.02):
        if len(series) < 5:
            return pd.Index([]), pd.Series(dtype=float)
        idx = pd.Index([series.index[2], series.index[-2]])
        scores = pd.Series([0.73, 0.89], index=idx)
        return idx, scores

    monkeypatch.setattr(app_module, "get_cached_column_forecast", _fake_cached_forecast)
    monkeypatch.setattr(app_module, "_compute_forecast", _fake_compute_forecast)
    monkeypatch.setattr(app_module, "get_cached_anomalies", _fake_cached_anomalies)


@pytest.mark.parametrize(
    "lab_key",
    [
        "forecast",
        "anomaly",
        "quality",
        "change-points",
        "conformal",
        "shap",
        "multivariate",
    ],
)
def test_research_labs_api_endpoints_return_payloads(monkeypatch: pytest.MonkeyPatch, lab_key: str):
    filename = "a" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "target": np.linspace(10.0, 40.0, 80),
            "feature_1": np.sin(np.linspace(0, 12, 80)),
            "feature_2": np.cos(np.linspace(0, 8, 80)) * 3.5,
            "feature_3": np.linspace(0.0, 1.0, 80),
            "category": ["A", "B", "C", "D"] * 20,
        }
    )
    _seed_dataframe(filename, df)
    _install_fast_forecast_mocks(monkeypatch)
    monkeypatch.setattr(research_module, "get_shap", lambda: None)

    try:
        with app.test_client() as client:
            response = client.get(
                f"/api/labs/{filename}/{lab_key}?column=target&forecast_pct=0.12&contamination=0.03"
            )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["lab"] == lab_key
        assert isinstance(payload["data"], dict)
        assert "schema_version" in payload
        assert payload.get("selected_col") in ("target", None)
    finally:
        _clear_dataframe(filename)


def test_research_labs_meta_endpoint_exposes_dataset_summary():
    filename = "b" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "sales": [10.0, 12.0, 14.0, 15.0],
            "cost": [6.0, 6.5, 7.1, 7.9],
            "segment": ["Retail", "SMB", "Enterprise", "Retail"],
        }
    )
    _seed_dataframe(filename, df)

    try:
        with app.test_client() as client:
            response = client.get(f"/api/labs/{filename}/meta?column=sales")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["lab"] == "meta"
        assert payload["data"]["dataset"]["rows"] == 4
        assert payload["data"]["dataset"]["columns"] == 3
        assert "sales" in payload["data"]["numeric_columns"]
        assert payload["selected_col"] == "sales"
    finally:
        _clear_dataframe(filename)


def test_research_labs_endpoint_rejects_unsupported_lab_key():
    filename = "c" * 40 + ".csv"
    with app.test_client() as client:
        response = client.get(f"/api/labs/{filename}/unknown-lab")

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["ok"] is False


def test_research_labs_cache_marks_second_response_as_cached(monkeypatch: pytest.MonkeyPatch):
    filename = "d" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "target": np.linspace(5.0, 20.0, 64),
            "feat": np.linspace(1.0, 3.0, 64),
        }
    )
    _seed_dataframe(filename, df)
    _install_fast_forecast_mocks(monkeypatch)

    try:
        with app.test_client() as client:
            first = client.get(f"/api/labs/{filename}/forecast?column=target&forecast_pct=0.1&contamination=0.02")
            second = client.get(f"/api/labs/{filename}/forecast?column=target&forecast_pct=0.1&contamination=0.02")

        p1 = first.get_json()
        p2 = second.get_json()
        assert p1["ok"] is True and p2["ok"] is True
        assert p1["cached"] is False
        assert p2["cached"] is True
    finally:
        _clear_dataframe(filename)
