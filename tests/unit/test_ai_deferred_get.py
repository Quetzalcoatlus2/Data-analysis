import numpy as np
import pandas as pd

import app as app_module
from app import AI_SUMMARY_CACHE, DATAFRAME_CACHE, app


def _install_fast_forecast_stubs(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(app_module, "get_cached_stl_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "generate_forecast_plot", lambda *_args, **_kwargs: "x")

    def _fake_get_cached_column_forecast(_filename, _column, _series, steps):
        idx = pd.RangeIndex(int(steps))
        return pd.Series(np.zeros(int(steps), dtype=float), index=idx), None

    monkeypatch.setattr(app_module, "get_cached_column_forecast", _fake_get_cached_column_forecast)


def test_analyze_get_defers_ai_summary_generation_for_forecast_view(monkeypatch):
    filename = "g" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(100, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)
    AI_SUMMARY_CACHE.pop(filename, None)

    calls = {"summary": 0}

    def _fake_summary(*_args, **_kwargs):
        calls["summary"] += 1
        return "<p>AI Summary</p>"

    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: True)
    monkeypatch.setattr(app_module, "get_ai_summary_with_file", _fake_summary)
    _install_fast_forecast_stubs(monkeypatch)

    with app.test_client() as client:
        response = client.get(f"/analyze/{filename}?view=forecast&forecast_pct=0.05&contamination=0.02")

    assert response.status_code == 200
    assert calls["summary"] == 0


def test_analyze_post_still_generates_ai_summary(monkeypatch):
    filename = "p" * 40 + ".csv"
    df = pd.DataFrame({"value": np.arange(80, dtype=float)})
    DATAFRAME_CACHE.set(filename, df)
    AI_SUMMARY_CACHE.pop(filename, None)

    calls = {"summary": 0}

    def _fake_summary(*_args, **_kwargs):
        calls["summary"] += 1
        return "<p>AI Summary</p>"

    monkeypatch.setattr(app_module, "ensure_ai_ready", lambda: True)
    monkeypatch.setattr(app_module, "get_ai_summary_with_file", _fake_summary)
    _install_fast_forecast_stubs(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            f"/analyze/{filename}?view=forecast&forecast_pct=0.05&contamination=0.02",
            data={},
        )

    assert response.status_code == 200
    assert calls["summary"] >= 1
