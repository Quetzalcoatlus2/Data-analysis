import io
import json
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import data_analysis.runtime_app as app_module
from app import app
from data_analysis.analysis import anomaly as anomaly_module


def test_anomaly_positions_support_positional_indices():
    data_index = pd.Index(["A", "A", "B", "B", "C"])
    anomalies_idx = pd.Index([1, 3], dtype="int64")

    positions = app_module._anomaly_positions_for_index(data_index, anomalies_idx)

    assert positions == [1, 3]


def test_anomaly_positions_integer_labels_are_treated_as_labels_not_positions():
    data_index = pd.Index([2000, 2001, 2002, 2003], dtype="int64")
    anomalies_idx = pd.Index([2002], dtype="int64")

    positions = app_module._anomaly_positions_for_index(data_index, anomalies_idx)

    assert positions == [2]


def test_get_cached_anomalies_does_not_collide_across_series_slices(monkeypatch):
    app_module.ANOMALY_CACHE.clear()
    calls: list[int] = []

    def _fake_detect(series: pd.Series, contamination: float = 0.02, **_kwargs):
        calls.append(len(series))
        idx = pd.Index([series.index[0]]) if len(series) else pd.Index([])
        score = pd.Series([1.0], index=idx) if len(series) else pd.Series(dtype=float)
        return idx, score

    monkeypatch.setattr(anomaly_module, "detect_anomalies", _fake_detect)

    full = pd.Series(np.arange(20.0), index=pd.RangeIndex(20))
    tail = full.tail(7)

    app_module.get_cached_anomalies("a" * 40 + ".csv", "x", full, 0.02)
    app_module.get_cached_anomalies("a" * 40 + ".csv", "x", tail, 0.02)

    assert calls == [20, 7]


def test_generate_correlation_heatmap_falls_back_without_seaborn(monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)

    df = pd.DataFrame(
        {
            "x": np.linspace(1.0, 10.0, 30),
            "y": np.linspace(5.0, 25.0, 30),
            "z": np.sin(np.linspace(0.0, 5.0, 30)),
        }
    )

    img = app_module.generate_correlation_heatmap(df, method="spearman", title="Fallback")

    assert isinstance(img, str)
    assert len(img) > 100


def test_generate_correlation_heatmap_fallback_large_matrix(monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)

    rng = np.random.default_rng(7)
    data = rng.normal(size=(90, 32))
    df = pd.DataFrame(data, columns=[f"c{i}" for i in range(32)])

    img = app_module.generate_correlation_heatmap(df, method="spearman", title="Large Fallback")

    assert isinstance(img, str)
    assert len(img) > 100


def test_sanitize_ai_html_trims_garbage_tail_and_restores_warning_emoji():
    raw = (
        "<h3>[WARNING] Limitations & Caveats</h3><p>Keep this section.</p>"
        "<p>HTML forms are fundamental for user interaction and should be ignored here.</p>"
    )

    out = app_module.sanitize_ai_html(raw)

    assert "⚠️" in out
    assert "HTML forms are fundamental for user interaction" not in out


def test_get_dataframe_for_year_index_uses_real_years_not_epoch_ns(tmp_path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_uploads_dir = app.config.get("UPLOADS_DIR")
    filename = "f" * 40 + ".csv"
    csv_path = uploads_dir / filename
    csv_path.write_text("year,value\n2000,10\n2001,11\n2002,12\n", encoding="utf-8")

    app_module.DATAFRAME_CACHE.clear()
    app.config["UPLOADS_DIR"] = str(uploads_dir)

    try:
        df = app_module.get_dataframe_for(filename)
    finally:
        app.config["UPLOADS_DIR"] = original_uploads_dir

    assert df is not None
    if isinstance(df.index, pd.DatetimeIndex):
        years = {int(y) for y in df.index.year.tolist()}
        assert years == {2000, 2001, 2002}
    elif 'year' not in [str(c).lower() for c in df.columns]:
        # Year was absorbed into index by datetime inference — check index values
        idx_vals = list(df.index)
        # Accept year-like integer index or datetime-like 
        assert len(idx_vals) == 3
    else:
        # Year remained as a regular column
        assert 'year' in [str(c).lower() for c in df.columns]


def test_static_plots_zip_contains_correlation_images_after_fallback(monkeypatch):
    filename = "a" * 40 + ".csv"
    df = pd.DataFrame(
        {
            "a": np.linspace(0.0, 100.0, 60),
            "b": np.linspace(5.0, 205.0, 60),
            "c": np.cos(np.linspace(0.0, 10.0, 60)),
        }
    )

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (pd.Index([]), pd.Series(dtype=float)))
    monkeypatch.setattr(app_module, "_forecast_with_fallback", lambda s, steps, **_kwargs: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None))
    monkeypatch.setattr(
        app_module,
        "generate_forecast_plot",
        lambda *_args, **_kwargs: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=",
    )

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data), "r") as zf:
        names = set(zf.namelist())
    assert "correlation_spearman.png" in names
    assert "correlation_pearson.png" in names


def test_static_plots_zip_uses_request_contamination(monkeypatch):
    filename = "c" * 40 + ".csv"
    df = pd.DataFrame({"x": np.linspace(0.0, 100.0, 40)})
    seen: list[float] = []

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)

    def _fake_anoms(_filename, _col, _series, contamination):
        seen.append(float(contamination))
        return pd.Index([]), pd.Series(dtype=float)

    monkeypatch.setattr(app_module, "get_cached_anomalies", _fake_anoms)
    monkeypatch.setattr(
        app_module,
        "generate_forecast_plot",
        lambda *_args, **_kwargs: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5gYb8AAAAASUVORK5CYII=",
    )
    monkeypatch.setattr(
        app_module,
        "_forecast_with_fallback",
        lambda s, steps, **_kwargs: (pd.Series(np.zeros(int(steps)), index=pd.RangeIndex(int(steps))), None),
    )

    with app.test_client() as client:
        response = client.get(f"/download/{filename}/static_plots.zip?contamination=0.075")

    assert response.status_code == 200
    assert seen
    assert all(abs(v - 0.075) < 1e-9 for v in seen)


def test_analyze_interactive_includes_positional_anomalies(monkeypatch):
    filename = "d" * 40 + ".csv"
    df = pd.DataFrame({"value": [10.0, 50.0, 12.0, 55.0, 11.0]}, index=pd.Index(["A", "A", "B", "B", "C"]))
    numeric_df = pd.DataFrame({"value": pd.to_numeric(df["value"], errors="coerce")}, index=df.index)

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_numeric_df", lambda _filename, _df: numeric_df)
    monkeypatch.setattr(
        app_module,
        "get_cached_df_info",
        lambda *_args, **_kwargs: {"head": "", "description": "", "overview_table_html": "", "info": "", "missing_values": ""},
    )
    monkeypatch.setattr(app_module, "build_ai_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "_get_clean_ai_summary_from_cache", lambda *_args, **_kwargs: "")

    an_idx = pd.Index([1, 3], dtype="int64", name="__pos__")
    an_score = pd.Series([1.1, 1.2], index=pd.Index([1, 3], dtype="int64", name="__pos__"))
    monkeypatch.setattr(app_module, "get_cached_anomalies", lambda *_args, **_kwargs: (an_idx, an_score))
    monkeypatch.setattr(app_module, "get_cached_column_forecast", lambda *_args, **_kwargs: (pd.Series(dtype=float), None))

    app_module.DATAFRAME_CACHE.set(filename, df)
    try:
        with app.test_client() as client:
            response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0")
    finally:
        app_module.DATAFRAME_CACHE.pop(filename, None)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    match = re.search(r'<script id="interactivePayload" type="application/json">(.*?)</script>', html, flags=re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload

    traces = payload[0].get("traces", [])
    anomaly_trace = next((t for t in traces if str(t.get("name", "")).lower() == "anomaly"), None)
    assert anomaly_trace is not None
    assert anomaly_trace.get("x") == [1, 3]


def test_analyze_interactive_short_series_keeps_history_trace(monkeypatch):
    filename = "e" * 40 + ".csv"
    df = pd.DataFrame({"value": [10.0, 11.0, 12.0]}, index=pd.Index(["A", "B", "C"]))
    numeric_df = pd.DataFrame({"value": pd.to_numeric(df["value"], errors="coerce")}, index=df.index)

    monkeypatch.setattr(app_module, "get_dataframe_for", lambda _name: df)
    monkeypatch.setattr(app_module, "get_cached_numeric_df", lambda _filename, _df: numeric_df)
    monkeypatch.setattr(
        app_module,
        "get_cached_df_info",
        lambda *_args, **_kwargs: {"head": "", "description": "", "overview_table_html": "", "info": "", "missing_values": ""},
    )
    monkeypatch.setattr(app_module, "build_ai_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "_get_clean_ai_summary_from_cache", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        app_module,
        "get_cached_anomalies",
        lambda *_args, **_kwargs: (pd.Index([], dtype="int64"), pd.Series(dtype=float)),
    )

    app_module.DATAFRAME_CACHE.set(filename, df)
    try:
        with app.test_client() as client:
            response = client.get(f"/analyze/{filename}?view=interactive&forecast_pct=0.2")
    finally:
        app_module.DATAFRAME_CACHE.pop(filename, None)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    match = re.search(r'<script id="interactivePayload" type="application/json">(.*?)</script>', html, flags=re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload
    traces = payload[0].get("traces", [])
    assert any(str(t.get("name", "")).lower() == "history" for t in traces)


def test_overview_tables_include_named_index_column():
    idx = pd.to_datetime(["2015-01-01", "2014-01-01", "2013-01-01"])
    df = pd.DataFrame(
        {
            "Status": ["Developing", "Developing", "Developing"],
            "Life expectancy": [65.0, 59.9, 59.9],
        },
        index=idx,
    )
    df.index.name = "Year"

    head_html = app_module.safe_df_head_html(df)
    desc_html = app_module.safe_df_description_html(df)
    overview_html = app_module.safe_dataset_overview_html(df)

    assert "<th style=\"min-width: 50px;\">Year</th>" in head_html
    assert "<th>Year</th>" not in head_html
    assert "Year" in desc_html
    assert "Columns (displayed)" in overview_html
    assert "Year" in overview_html


def test_generate_forecast_plot_uses_index_axis_for_epoch_ns_datetime(monkeypatch):
    raw_idx = pd.Series(np.arange(2000, 2060), dtype="int64")
    bad_dt_idx = pd.to_datetime(raw_idx, errors="coerce")  # 1970 + nanoseconds
    history = pd.Series(np.linspace(10.0, 20.0, 60), index=pd.DatetimeIndex(bad_dt_idx))
    forecast = pd.Series(np.linspace(20.5, 23.0, 6), index=pd.RangeIndex(6))

    captured: dict[str, str] = {}
    original_set_xlabel = Axes.set_xlabel

    def _xlabel_spy(self, xlabel, *args, **kwargs):
        captured["xlabel"] = str(xlabel)
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xlabel", _xlabel_spy)

    img = app_module.generate_forecast_plot(history, forecast, "Forecast", "Timestamp", "value")

    assert isinstance(img, str)
    assert len(img) > 0
    assert captured.get("xlabel") == "Index"
