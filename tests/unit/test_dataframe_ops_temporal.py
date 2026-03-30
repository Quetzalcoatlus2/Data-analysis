import importlib.util
from pathlib import Path

import pandas as pd


def _load_dataframe_ops_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "data_analysis" / "analysis" / "dataframe_ops.py"
    spec = importlib.util.spec_from_file_location("test_dataframe_ops_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load dataframe_ops module for tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_looks_temporal_series_rejects_numeric_only_with_time_name_hint():
    module = _load_dataframe_ops_module()
    series = pd.Series(["1", "2", "3", "4"], name="event_time")

    assert module._looks_temporal_series(series) is False


def test_looks_temporal_series_samples_head_only():
    module = _load_dataframe_ops_module()
    values = [str(i) for i in range(200)] + ["2024-01-01", "2024-01-02"]
    series = pd.Series(values, name="date")

    assert module._looks_temporal_series(series) is False
