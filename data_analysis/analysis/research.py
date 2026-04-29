from __future__ import annotations

from collections.abc import Callable
import importlib
import importlib.util
import math
from typing import Any, cast

import numpy as np
import pandas as pd

from data_analysis.analysis.controls import forecast_steps_for_history
from data_analysis.core.lazy_imports import get_shap


DEFAULT_CONFIDENCE_LEVELS: tuple[float, float, float] = (0.80, 0.90, 0.95)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_index_position(index: pd.Index, label: Any) -> int | None:
    """Resolve a safe integer position for a label in an index."""
    try:
        if label not in index:
            return None
    except Exception:
        return None

    try:
        positions = index.get_indexer_for([label])
    except Exception:
        return None

    if positions.size == 0:
        return None

    try:
        pos = int(positions[0])
    except Exception:
        return None

    return pos if pos >= 0 else None


def _sample_positions(length: int, max_points: int) -> list[int]:
    if length <= 0:
        return []
    if max_points <= 0 or length <= max_points:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=max_points, dtype=float)
    positions = sorted({int(round(v)) for v in raw})
    if positions and positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def _series_to_points(series: pd.Series, *, max_points: int = 400) -> list[dict[str, Any]]:
    if series is None or series.empty:
        return []
    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    idx = series.index
    out: list[dict[str, Any]] = []
    for pos in _sample_positions(len(series), max_points=max_points):
        y_val = _safe_float(values[pos])
        if y_val is None:
            continue
        out.append({"x": str(idx[pos]), "y": y_val, "pos": int(pos)})
    return out


def _mean_diff_std(values: np.ndarray) -> float | None:
    if values.size < 3:
        return None
    diffs = np.diff(values)
    if diffs.size < 2:
        return None
    out = float(np.nanstd(diffs, ddof=1))
    return out if math.isfinite(out) else None


def _stats_from_series(series: pd.Series) -> dict[str, Any]:
    if series is None or series.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
        }
    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    count = int(values.size)
    return {
        "count": count,
        "min": _safe_float(np.nanmin(values)),
        "max": _safe_float(np.nanmax(values)),
        "mean": _safe_float(np.nanmean(values)),
        "median": _safe_float(np.nanmedian(values)),
        "std": _safe_float(np.nanstd(values, ddof=1)) if count > 1 else 0.0,
        "p05": _safe_float(np.nanquantile(values, 0.05)),
        "p95": _safe_float(np.nanquantile(values, 0.95)),
    }


def resolve_numeric_column(
    numeric_df: pd.DataFrame,
    requested_column: str | None = None,
) -> tuple[str | None, str]:
    """Pick a numeric column, preferring the requested one when valid.

    Returns:
        (column_name, source) where source is one of: requested, auto, none
    """
    if numeric_df is None or numeric_df.empty:
        return None, "none"

    columns = [str(c) for c in numeric_df.columns]
    if requested_column:
        req = str(requested_column)
        if req in columns:
            return req, "requested"

    for col in columns:
        s = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
        if len(s) >= 5:
            return col, "auto"

    return columns[0], "auto"


def _coerce_numeric_series(numeric_df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(numeric_df[column], errors="coerce").dropna().astype(float)


def _optional_dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _quantile_safe(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    try:
        out = float(np.nanquantile(values, q))
        return out if math.isfinite(out) else None
    except Exception:
        return None


def build_labs_meta_payload(
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    *,
    requested_column: str | None,
) -> dict[str, Any]:
    selected_col, selected_source = resolve_numeric_column(numeric_df, requested_column)
    numeric_columns = [str(c) for c in list(numeric_df.columns)]
    all_columns = [str(c) for c in list(df.columns)]

    missing_cells = int(df.isna().sum().sum()) if isinstance(df, pd.DataFrame) else 0
    total_cells = int(df.shape[0] * df.shape[1]) if isinstance(df, pd.DataFrame) else 0
    missing_rate = (float(missing_cells) / float(total_cells)) if total_cells else 0.0

    return {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": int(len(numeric_columns)),
            "missing_cells": missing_cells,
            "missing_rate": float(round(missing_rate, 6)),
            "duplicate_rows": int(df.duplicated().sum()) if not df.empty else 0,
        },
        "columns": all_columns,
        "numeric_columns": numeric_columns,
        "selected_col": selected_col,
        "selected_col_source": selected_source,
        "capabilities": {
            "shap": bool(get_shap() is not None),
            "ruptures": _optional_dependency_available("ruptures"),
            "statsmodels": _optional_dependency_available("statsmodels"),
        },
        "labs": [
            {"key": "forecast", "name": "Forecast"},
            {"key": "anomaly", "name": "Anomaly"},
            {"key": "quality", "name": "Quality"},
            {"key": "change-points", "name": "Change Points"},
            {"key": "conformal", "name": "Conformal"},
            {"key": "shap", "name": "SHAP"},
            {"key": "multivariate", "name": "Multivariate"},
        ],
    }


def _compute_recent_trend(series: pd.Series) -> dict[str, Any]:
    if series is None or series.empty:
        return {"slope": None, "window": 0, "pct_change": None}

    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    window = int(min(values.size, max(12, values.size // 4)))
    y = values[-window:]
    if y.size < 2:
        return {"slope": None, "window": int(y.size), "pct_change": None}
    x = np.arange(y.size, dtype=float)
    slope = _safe_float(np.polyfit(x, y, 1)[0])
    first = float(y[0])
    change = float(y[-1] - y[0])
    pct = (change / (abs(first) + 1e-12)) * 100.0
    return {"slope": slope, "window": window, "pct_change": _safe_float(pct)}


def _compute_forecast_steps(history_len: int, forecast_pct: float) -> int:
    return forecast_steps_for_history(history_len, forecast_pct)


def _backtest_forecast(
    series: pd.Series,
    *,
    compute_forecast_fn: Callable[[pd.Series, int], tuple[pd.Series, pd.DataFrame | None]],
) -> dict[str, Any] | None:
    if series is None or len(series) < 28:
        return None

    holdout = int(min(60, max(8, len(series) // 5)))
    train = series.iloc[:-holdout]
    truth = series.iloc[-holdout:]
    if len(train) < 12 or len(truth) < 5:
        return None

    pred, conf_df = compute_forecast_fn(train, holdout)
    pred_vals = np.asarray(pd.Series(pred).to_numpy(dtype=float), dtype=float)
    truth_vals = np.asarray(pd.Series(truth).to_numpy(dtype=float), dtype=float)
    m = int(min(pred_vals.size, truth_vals.size))
    if m < 3:
        return None

    pred_vals = pred_vals[:m]
    truth_vals = truth_vals[:m]
    err = truth_vals - pred_vals
    abs_err = np.abs(err)

    mae = float(np.nanmean(abs_err))
    rmse = float(np.sqrt(np.nanmean(np.square(err))))

    non_zero = np.abs(truth_vals) > 1e-9
    if np.any(non_zero):
        mape = float(np.nanmean(abs_err[non_zero] / np.abs(truth_vals[non_zero])) * 100.0)
    else:
        mape = None

    coverage = None
    if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
        lower = np.asarray(conf_df.iloc[:, 0].to_numpy(dtype=float), dtype=float)[:m]
        upper = np.asarray(conf_df.iloc[:, 1].to_numpy(dtype=float), dtype=float)[:m]
        inside = (truth_vals >= lower) & (truth_vals <= upper)
        coverage = float(np.mean(inside)) if inside.size else None

    return {
        "holdout_size": int(m),
        "mae": _safe_float(mae),
        "rmse": _safe_float(rmse),
        "mape": _safe_float(mape),
        "ci_coverage": _safe_float(coverage),
    }


def build_forecast_lab_payload(
    *,
    filename: str,
    column: str,
    numeric_df: pd.DataFrame,
    forecast_pct: float,
    contamination: float,
    max_points: int,
    get_cached_column_forecast_fn: Callable[[str, str, pd.Series, int], tuple[pd.Series | None, pd.DataFrame | None]],
    get_cached_anomalies_fn: Callable[[str, str, pd.Series, float], tuple[pd.Index, pd.Series]],
    compute_forecast_fn: Callable[[pd.Series, int], tuple[pd.Series, pd.DataFrame | None]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    series = _coerce_numeric_series(numeric_df, column)
    if len(series) < 8:
        return {
            "column": column,
            "history_count": int(len(series)),
            "forecast_steps": 0,
            "message": "Not enough numeric observations for forecasting (need at least 8).",
        }, ["Insufficient data for robust forecast diagnostics."]

    history_values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    history_stats = _stats_from_series(series)
    history_trend = _compute_recent_trend(series)
    steps = _compute_forecast_steps(len(series), forecast_pct)

    forecast_series: pd.Series | None = None
    conf_df: pd.DataFrame | None = None
    if steps > 0:
        forecast_series, conf_df = get_cached_column_forecast_fn(filename, column, series, steps)
        if forecast_series is None or len(forecast_series) == 0:
            warnings.append("Forecast model returned no future points; using history diagnostics only.")
            steps = 0

    anomaly_idx, anomaly_scores = get_cached_anomalies_fn(filename, column, series, contamination)
    anomaly_rows: list[dict[str, Any]] = []
    if anomaly_idx is not None and len(anomaly_idx):
        for idx in anomaly_idx[:25]:
            pos = _safe_index_position(series.index, idx)
            try:
                val = _safe_float(series.loc[idx]) if idx in series.index else None
            except Exception:
                val = None
            score = _safe_float(anomaly_scores.loc[idx]) if idx in getattr(anomaly_scores, "index", []) else None
            anomaly_rows.append(
                {
                    "index": str(idx),
                    "pos": pos,
                    "value": val,
                    "score": score,
                }
            )

    forecast_stats = _stats_from_series(forecast_series) if isinstance(forecast_series, pd.Series) else None

    forecast_values: np.ndarray | None = None
    if isinstance(forecast_series, pd.Series) and not forecast_series.empty:
        forecast_values = np.asarray(forecast_series.to_numpy(dtype=float), dtype=float)

    history_diff_std = _mean_diff_std(history_values)
    forecast_diff_std = _mean_diff_std(forecast_values) if forecast_values is not None else None
    volatility_ratio = None
    if history_diff_std and forecast_diff_std is not None and history_diff_std > 1e-9:
        volatility_ratio = float(forecast_diff_std / history_diff_std)

    backtest = _backtest_forecast(series, compute_forecast_fn=compute_forecast_fn)
    if backtest is None:
        warnings.append("Backtest metrics omitted because the series is too short for a stable holdout split.")

    payload: dict[str, Any] = {
        "column": column,
        "history_count": int(len(series)),
        "forecast_steps": int(steps),
        "forecast_pct": float(forecast_pct),
        "contamination": float(contamination),
        "history_stats": history_stats,
        "forecast_stats": forecast_stats,
        "recent_trend": history_trend,
        "volatility_ratio": _safe_float(volatility_ratio),
        "backtest": backtest,
        "series": {
            "history": _series_to_points(series, max_points=max_points),
            "forecast": _series_to_points(forecast_series, max_points=max(120, max_points // 2))
            if isinstance(forecast_series, pd.Series)
            else [],
            "confidence": [],
        },
        "anomalies": {
            "count": int(len(anomaly_rows)),
            "rows": anomaly_rows,
        },
        "insights": [],
    }

    if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2 and isinstance(forecast_series, pd.Series):
        ci_rows: list[dict[str, Any]] = []
        lower = np.asarray(conf_df.iloc[:, 0].to_numpy(dtype=float), dtype=float)
        upper = np.asarray(conf_df.iloc[:, 1].to_numpy(dtype=float), dtype=float)
        idx = list(forecast_series.index)
        for pos in _sample_positions(len(idx), max_points=max(120, max_points // 2)):
            lo = _safe_float(lower[pos]) if pos < len(lower) else None
            hi = _safe_float(upper[pos]) if pos < len(upper) else None
            y = _safe_float(forecast_series.iloc[pos])
            if lo is None or hi is None or y is None:
                continue
            ci_rows.append({"x": str(idx[pos]), "y": y, "lower": lo, "upper": hi, "pos": int(pos)})
        payload["series"]["confidence"] = ci_rows

    insights: list[str] = []
    slope = payload["recent_trend"].get("slope")
    if slope is not None:
        if slope > 0:
            insights.append("Recent trend is upward.")
        elif slope < 0:
            insights.append("Recent trend is downward.")
        else:
            insights.append("Recent trend is flat.")
    if payload.get("volatility_ratio") is not None:
        vr = float(payload["volatility_ratio"])
        if vr < 0.7:
            insights.append("Forecast variation is smoother than recent history.")
        elif vr > 1.4:
            insights.append("Forecast variation is more volatile than recent history.")
    if anomaly_rows:
        insights.append(f"Detected {len(anomaly_rows)} anomalies at contamination={contamination:.3f}.")
    payload["insights"] = insights

    return payload, warnings


def build_anomaly_lab_payload(
    *,
    filename: str,
    column: str,
    numeric_df: pd.DataFrame,
    contamination: float,
    max_points: int,
    get_cached_anomalies_fn: Callable[[str, str, pd.Series, float], tuple[pd.Index, pd.Series]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    series = _coerce_numeric_series(numeric_df, column)
    if len(series) < 8:
        return {
            "column": column,
            "count": 0,
            "message": "Not enough points to run anomaly analysis (need at least 8).",
            "series": _series_to_points(series, max_points=max_points),
            "anomalies": [],
        }, ["Series too short for stable anomaly scoring."]

    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    if not math.isfinite(mad) or mad < 1e-9:
        mad = float(np.nanstd(values, ddof=1)) if values.size > 1 else 1.0
    scale = mad if mad > 1e-9 else 1.0

    anomaly_idx, anomaly_scores = get_cached_anomalies_fn(filename, column, series, contamination)
    rows: list[dict[str, Any]] = []
    for idx in anomaly_idx[:80]:
        try:
            val = float(series.loc[idx])
        except Exception:
            continue
        try:
            score = _safe_float(anomaly_scores.loc[idx]) if idx in anomaly_scores.index else None
        except Exception:
            score = None
        z_like = abs((val - med) / (scale + 1e-9))
        severity = "high" if z_like >= 3.5 else "medium" if z_like >= 2.5 else "low"
        rows.append(
            {
                "index": str(idx),
                "value": _safe_float(val),
                "score": score,
                "robust_z": _safe_float(z_like),
                "severity": severity,
                "direction": "high" if val >= med else "low",
            }
        )

    rows.sort(key=lambda item: (item.get("score") is None, -(item.get("score") or 0.0)))
    top_rows = rows[:40]

    score_values = np.asarray([r["score"] for r in rows if r.get("score") is not None], dtype=float)
    score_stats = {
        "p50": _quantile_safe(score_values, 0.50),
        "p90": _quantile_safe(score_values, 0.90),
        "p95": _quantile_safe(score_values, 0.95),
        "max": _safe_float(np.nanmax(score_values)) if score_values.size else None,
    }

    payload = {
        "column": column,
        "contamination": float(contamination),
        "count": int(len(rows)),
        "series": _series_to_points(series, max_points=max_points),
        "anomalies": top_rows,
        "score_stats": score_stats,
    }

    if not rows:
        warnings.append("No anomalies detected at current contamination threshold.")

    return payload, warnings


def build_quality_lab_payload(
    *,
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    missing_rate = float(missing_cells / total_cells) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    duplicate_rate = float(duplicate_rows / rows) if rows else 0.0

    constant_cols: list[str] = []
    low_variance_cols: list[str] = []
    high_cardinality_cols: list[str] = []
    coercion_candidate_cols: list[str] = []

    for col in df.columns:
        s = df[col]
        non_na = s.dropna()
        nunique = int(non_na.nunique(dropna=True)) if len(non_na) else 0
        if len(non_na) and nunique <= 1:
            constant_cols.append(str(col))

        if col in numeric_df.columns:
            num = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
            if len(num) >= 5:
                var = _safe_float(num.var(ddof=1))
                if var is not None and var <= 1e-10:
                    low_variance_cols.append(str(col))
        else:
            ratio = (nunique / len(non_na)) if len(non_na) else 0.0
            if len(non_na) >= 20 and ratio > 0.85:
                high_cardinality_cols.append(str(col))
            if str(s.dtype) in {"object", "string"}:
                parsed = pd.to_numeric(non_na, errors="coerce")
                parsed_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
                if 0.35 <= parsed_ratio <= 0.95:
                    coercion_candidate_cols.append(str(col))

    column_issues: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        non_na = s.dropna()
        missing_pct = float((1.0 - (len(non_na) / rows)) * 100.0) if rows else 0.0
        issue_flags: list[str] = []
        if str(col) in constant_cols:
            issue_flags.append("constant")
        if str(col) in low_variance_cols:
            issue_flags.append("low_variance")
        if str(col) in high_cardinality_cols:
            issue_flags.append("high_cardinality")
        if str(col) in coercion_candidate_cols:
            issue_flags.append("mixed_type_numeric_like")
        if missing_pct >= 25.0:
            issue_flags.append("high_missingness")
        if issue_flags:
            column_issues.append(
                {
                    "column": str(col),
                    "dtype": str(s.dtype),
                    "missing_pct": _safe_float(missing_pct),
                    "non_null": int(len(non_na)),
                    "unique": int(non_na.nunique(dropna=True)) if len(non_na) else 0,
                    "issues": issue_flags,
                }
            )

    col_count = max(cols, 1)
    score = 100.0
    score -= missing_rate * 50.0
    score -= duplicate_rate * 20.0
    score -= (len(constant_cols) / col_count) * 15.0
    score -= (len(low_variance_cols) / col_count) * 10.0
    score -= (len(high_cardinality_cols) / col_count) * 5.0
    quality_score = max(0.0, min(100.0, score))

    recommendations: list[str] = []
    if missing_rate > 0.10:
        recommendations.append("Address missing values using imputation or targeted filtering.")
    if duplicate_rows > 0:
        recommendations.append("Review duplicate rows and de-duplicate if they are accidental repeats.")
    if constant_cols:
        recommendations.append("Drop or deprioritize constant columns; they carry no predictive signal.")
    if low_variance_cols:
        recommendations.append("Inspect low-variance numeric columns; consider scaling or removal.")
    if coercion_candidate_cols:
        recommendations.append("Normalize mixed numeric-like text columns via explicit type coercion rules.")
    if not recommendations:
        recommendations.append("Quality baseline looks good; proceed with feature engineering.")

    payload = {
        "quality_score": _safe_float(round(quality_score, 2)),
        "summary": {
            "rows": rows,
            "columns": cols,
            "missing_cells": missing_cells,
            "missing_rate": _safe_float(missing_rate),
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": _safe_float(duplicate_rate),
            "constant_columns": int(len(constant_cols)),
            "low_variance_columns": int(len(low_variance_cols)),
            "high_cardinality_columns": int(len(high_cardinality_cols)),
        },
        "issue_columns": column_issues[:80],
        "recommendations": recommendations,
    }

    if rows < 20:
        warnings.append("Quality metrics are estimated on a small dataset; treat scores as indicative.")

    return payload, warnings


def _detect_change_points_baseline(series: pd.Series) -> tuple[list[int], np.ndarray, int]:
    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    n = len(values)
    if n < 16:
        return [], np.zeros(n, dtype=float), 0

    window = int(max(5, min(60, n // 10)))
    roll_mean = pd.Series(values).rolling(window=window, min_periods=window).mean()
    roll_std = pd.Series(values).rolling(window=window, min_periods=window).std(ddof=0)
    signal = (roll_mean.diff().abs() / (roll_std + 1e-9)).fillna(0.0)
    signal_values = np.asarray(signal.to_numpy(dtype=float), dtype=float)

    positive = signal_values[signal_values > 0]
    threshold = float(np.quantile(positive, 0.97)) if positive.size else float("inf")
    candidates = np.where(signal_values >= threshold)[0].tolist() if math.isfinite(threshold) else []

    # De-cluster nearby points and keep highest-scoring representative.
    min_gap = max(3, window // 2)
    selected: list[int] = []
    for pos in candidates:
        if not selected:
            selected.append(int(pos))
            continue
        if pos - selected[-1] < min_gap:
            if signal_values[pos] > signal_values[selected[-1]]:
                selected[-1] = int(pos)
            continue
        selected.append(int(pos))
    return selected, signal_values, window


def _detect_change_points_ruptures(series: pd.Series) -> list[int]:
    if not _optional_dependency_available("ruptures"):
        return []
    try:
        rpt = cast(Any, importlib.import_module("ruptures"))
    except Exception:
        return []

    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    if values.size < 24:
        return []

    try:
        algo = rpt.Pelt(model="rbf").fit(values.reshape(-1, 1))
        penalty = max(3.0, float(np.log(values.size) * np.nanstd(values)))
        points = algo.predict(pen=penalty)
        out = sorted({int(p - 1) for p in points if int(p) < values.size and int(p) > 0})
        return out[:30]
    except Exception:
        return []


def build_change_points_lab_payload(
    *,
    column: str,
    numeric_df: pd.DataFrame,
    max_points: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    series = _coerce_numeric_series(numeric_df, column)
    if len(series) < 16:
        return {
            "column": column,
            "change_points": [],
            "message": "Need at least 16 observations for change-point diagnostics.",
            "series": _series_to_points(series, max_points=max_points),
        }, ["Series too short for change-point detection."]

    base_points, base_signal, window = _detect_change_points_baseline(series)
    ruptures_points = _detect_change_points_ruptures(series)

    merged_points = sorted(set(base_points) | set(ruptures_points))[:40]
    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    rows: list[dict[str, Any]] = []
    for pos in merged_points:
        if pos < 0 or pos >= len(series):
            continue
        score = _safe_float(base_signal[pos]) if pos < len(base_signal) else None
        rows.append(
            {
                "pos": int(pos),
                "index": str(series.index[pos]),
                "value": _safe_float(values[pos]),
                "score": score,
                "sources": [
                    s
                    for s, ok in (
                        ("baseline", pos in base_points),
                        ("ruptures", pos in ruptures_points),
                    )
                    if ok
                ],
            }
        )

    # Segment summaries.
    boundaries = [0] + [p + 1 for p in merged_points if p + 1 < len(series)] + [len(series)]
    boundaries = sorted(set(boundaries))
    segments: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        chunk = values[a:b]
        if chunk.size == 0:
            continue
        segments.append(
            {
                "segment": int(i + 1),
                "start_pos": int(a),
                "end_pos": int(b - 1),
                "start_index": str(series.index[a]),
                "end_index": str(series.index[b - 1]),
                "count": int(chunk.size),
                "mean": _safe_float(np.nanmean(chunk)),
                "std": _safe_float(np.nanstd(chunk, ddof=1)) if chunk.size > 1 else 0.0,
            }
        )

    payload = {
        "column": column,
        "window": int(window),
        "change_points": rows,
        "series": _series_to_points(series, max_points=max_points),
        "segments": segments,
    }

    if not rows:
        warnings.append("No strong change points detected with the current baseline settings.")
    if not ruptures_points:
        warnings.append("Optional 'ruptures' engine unavailable or yielded no candidates; baseline detector used.")

    return payload, warnings


def build_conformal_lab_payload(
    *,
    filename: str,
    column: str,
    numeric_df: pd.DataFrame,
    forecast_pct: float,
    max_points: int,
    get_cached_column_forecast_fn: Callable[[str, str, pd.Series, int], tuple[pd.Series | None, pd.DataFrame | None]],
    compute_forecast_fn: Callable[[pd.Series, int], tuple[pd.Series, pd.DataFrame | None]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    series = _coerce_numeric_series(numeric_df, column)
    if len(series) < 30:
        return {
            "column": column,
            "message": "Need at least 30 observations for conformal calibration.",
            "series": _series_to_points(series, max_points=max_points),
            "levels": [],
            "forecast": [],
            "bands": {},
        }, ["Series too short for conformal diagnostics."]

    cal_size = int(min(80, max(10, len(series) // 5)))
    train = series.iloc[:-cal_size]
    calibration_truth = series.iloc[-cal_size:]

    if len(train) < 12 or len(calibration_truth) < 8:
        return {
            "column": column,
            "message": "Insufficient train/calibration split for conformal intervals.",
            "series": _series_to_points(series, max_points=max_points),
            "levels": [],
            "forecast": [],
            "bands": {},
        }, ["Unable to compute stable calibration residuals."]

    cal_pred, _ = compute_forecast_fn(train, len(calibration_truth))
    truth_vals = np.asarray(calibration_truth.to_numpy(dtype=float), dtype=float)
    pred_vals = np.asarray(pd.Series(cal_pred).to_numpy(dtype=float), dtype=float)
    m = int(min(len(truth_vals), len(pred_vals)))
    if m < 5:
        warnings.append("Conformal calibration used a minimal overlap between truth and predictions.")
    truth_vals = truth_vals[:m]
    pred_vals = pred_vals[:m]
    residuals = np.abs(truth_vals - pred_vals)

    steps = _compute_forecast_steps(len(series), forecast_pct)
    if steps <= 0:
        steps = min(12, max(6, len(series) // 10))
        warnings.append("forecast_pct produced zero horizon; using default conformal horizon.")

    future_forecast, _ = get_cached_column_forecast_fn(filename, column, series, steps)
    if future_forecast is None or future_forecast.empty:
        future_forecast, _ = compute_forecast_fn(series, steps)

    future_vals = np.asarray(pd.Series(future_forecast).to_numpy(dtype=float), dtype=float)
    future_idx = list(pd.Series(future_forecast).index)

    coverage_rows: list[dict[str, Any]] = []
    bands: dict[str, list[dict[str, Any]]] = {}

    for level in DEFAULT_CONFIDENCE_LEVELS:
        q = float(np.quantile(residuals, level)) if residuals.size else 0.0
        empirical = float(np.mean(residuals <= q)) if residuals.size else None
        level_label = str(int(level * 100))

        coverage_rows.append(
            {
                "level": level_label,
                "target": _safe_float(level),
                "empirical": _safe_float(empirical),
                "quantile": _safe_float(q),
            }
        )

        band_rows: list[dict[str, Any]] = []
        for pos in _sample_positions(len(future_vals), max_points=max(100, max_points // 2)):
            y = _safe_float(future_vals[pos])
            if y is None:
                continue
            band_rows.append(
                {
                    "x": str(future_idx[pos]),
                    "y": y,
                    "lower": _safe_float(y - q),
                    "upper": _safe_float(y + q),
                    "pos": int(pos),
                }
            )
        bands[level_label] = band_rows

    payload = {
        "column": column,
        "calibration_size": int(m),
        "residual_stats": {
            "mean_abs_error": _safe_float(np.nanmean(residuals)) if residuals.size else None,
            "median_abs_error": _safe_float(np.nanmedian(residuals)) if residuals.size else None,
            "p90_abs_error": _safe_float(np.nanquantile(residuals, 0.90)) if residuals.size else None,
        },
        "levels": coverage_rows,
        "forecast": _series_to_points(pd.Series(future_forecast), max_points=max(100, max_points // 2)),
        "bands": bands,
    }
    return payload, warnings


def build_shap_lab_payload(
    *,
    column: str,
    numeric_df: pd.DataFrame,
    max_features: int = 12,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    numeric_columns = [str(c) for c in numeric_df.columns]
    if column not in numeric_columns:
        return {
            "column": column,
            "mode": "unavailable",
            "feature_importance": [],
            "message": "Target column is not numeric in current dataset.",
        }, ["SHAP lab requires a numeric target column."]

    feature_cols = [c for c in numeric_columns if c != column]
    if len(feature_cols) < 1:
        return {
            "column": column,
            "mode": "unavailable",
            "feature_importance": [],
            "message": "Need at least one additional numeric feature for explainability.",
        }, ["Not enough numeric features for SHAP/surrogate explainability."]

    frame = numeric_df[[column] + feature_cols].dropna()
    if len(frame) < 20:
        return {
            "column": column,
            "mode": "unavailable",
            "feature_importance": [],
            "message": "Need at least 20 complete rows for explainability.",
        }, ["Insufficient complete rows for model fitting."]

    # Keep computations bounded.
    if len(frame) > 1200:
        frame = frame.sample(n=1200, random_state=42)
        warnings.append("Explainability sampled 1200 rows for performance.")

    # Use only top-N variance features to keep model and SHAP stable.
    variances = frame[feature_cols].var(numeric_only=True).sort_values(ascending=False)
    use_features = [str(c) for c in variances.index[:max_features]]
    X = frame[use_features]
    y = frame[column]

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except Exception:
        return {
            "column": column,
            "mode": "unavailable",
            "feature_importance": [],
            "message": "scikit-learn is unavailable.",
        }, ["scikit-learn is required for SHAP lab fallback model."]

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )
    model.fit(X, y)

    perm_result = cast(Any, permutation_importance(model, X, y, n_repeats=4, random_state=42))
    perm_importances = getattr(perm_result, "importances_mean", None)
    perm_values = perm_importances if perm_importances is not None else []
    perm_map = {f: _safe_float(v) for f, v in zip(use_features, perm_values, strict=False)}

    mode = "surrogate"
    shap_map: dict[str, float | None] = {}
    shap_module = get_shap()
    if shap_module is not None:
        try:
            sample_X = X.sample(n=min(len(X), 250), random_state=42)
            explainer = shap_module.TreeExplainer(model)
            raw_values = explainer.shap_values(sample_X)
            arr = np.asarray(raw_values)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[1] == len(use_features):
                mean_abs = np.mean(np.abs(arr), axis=0)
                shap_map = {
                    feat: _safe_float(val)
                    for feat, val in zip(use_features, mean_abs, strict=False)
                }
                mode = "shap"
            else:
                warnings.append("SHAP returned an unexpected shape; used surrogate importance.")
        except Exception:
            warnings.append("SHAP computation failed; used surrogate importance.")
    else:
        warnings.append("SHAP package not installed; used surrogate importance.")

    rows: list[dict[str, Any]] = []
    model_importances = getattr(model, "feature_importances_", None)
    model_map: dict[str, float | None] = {}
    if model_importances is not None:
        model_map = {
            feat: _safe_float(val)
            for feat, val in zip(use_features, model_importances, strict=False)
        }

    for feat in use_features:
        shap_val = shap_map.get(feat)
        perm_val = perm_map.get(feat)
        model_val = model_map.get(feat)
        primary = shap_val if shap_val is not None else perm_val if perm_val is not None else model_val
        rows.append(
            {
                "feature": feat,
                "importance": _safe_float(primary),
                "shap_importance": _safe_float(shap_val),
                "permutation_importance": _safe_float(perm_val),
                "model_importance": _safe_float(model_val),
            }
        )

    rows.sort(key=lambda item: -(item.get("importance") or 0.0))
    payload = {
        "column": column,
        "mode": mode,
        "rows_used": int(len(frame)),
        "feature_importance": rows[:20],
    }
    return payload, warnings


def build_multivariate_lab_payload(
    *,
    numeric_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if numeric_df is None or numeric_df.empty or len(numeric_df.columns) < 2:
        return {
            "message": "Need at least two numeric columns for multivariate diagnostics.",
            "top_correlations": [],
            "pca": None,
            "vif": [],
            "joint_anomalies": [],
        }, ["Insufficient numeric columns for multivariate analysis."]

    sub = numeric_df.copy()
    # Keep at most 12 columns by variance to avoid heavy matrix computations.
    var_rank = sub.var(numeric_only=True).sort_values(ascending=False)
    keep_cols = [str(c) for c in var_rank.index[:12]]
    sub = sub[keep_cols]

    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    pairs: list[dict[str, Any]] = []
    cols = list(sub.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = str(cols[i])
            c2 = str(cols[j])
            p = _safe_float(pearson.loc[c1, c2]) if c1 in pearson.index and c2 in pearson.columns else None
            s = _safe_float(spearman.loc[c1, c2]) if c1 in spearman.index and c2 in spearman.columns else None
            abs_corr = max(abs(p or 0.0), abs(s or 0.0))
            pairs.append(
                {
                    "feature_a": c1,
                    "feature_b": c2,
                    "pearson": p,
                    "spearman": s,
                    "abs_corr": _safe_float(abs_corr),
                }
            )
    pairs.sort(key=lambda item: -(item.get("abs_corr") or 0.0))

    pca_payload: dict[str, Any] | None = None
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        pca_frame = sub.dropna()
        if len(pca_frame) >= 8:
            x = StandardScaler().fit_transform(pca_frame.to_numpy(dtype=float))
            pca = PCA(n_components=min(5, x.shape[1]))
            pca.fit(x)
            pca_payload = {
                "components": int(pca.n_components_),
                "explained_variance_ratio": [
                    _safe_float(v) for v in pca.explained_variance_ratio_.tolist()
                ],
                "cumulative_explained_ratio": [
                    _safe_float(v)
                    for v in np.cumsum(pca.explained_variance_ratio_).tolist()
                ],
            }
        else:
            warnings.append("PCA skipped due to limited complete rows.")
    except Exception:
        warnings.append("PCA unavailable (scikit-learn missing or failed).")

    vif_rows: list[dict[str, Any]] = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_frame = sub.dropna()
        if len(vif_frame) >= 12 and vif_frame.shape[1] >= 2:
            # Restrict to top 8 columns for stability.
            vif_cols = list(vif_frame.columns[:8])
            x = np.asarray(vif_frame[vif_cols].to_numpy(dtype=float), dtype=float)
            # Drop degenerate columns.
            non_constant: list[int] = []
            for idx, col in enumerate(vif_cols):
                if _safe_float(np.nanstd(x[:, idx], ddof=1)) not in (None, 0.0):
                    non_constant.append(idx)
            if len(non_constant) >= 2:
                x2 = x[:, non_constant]
                cols2 = [vif_cols[i] for i in non_constant]
                for i, col in enumerate(cols2):
                    vif_rows.append(
                        {
                            "feature": str(col),
                            "vif": _safe_float(variance_inflation_factor(x2, i)),
                        }
                    )
                vif_rows.sort(key=lambda row: -(row.get("vif") or 0.0))
    except Exception:
        warnings.append("VIF diagnostics unavailable (statsmodels missing or failed).")

    joint_anomalies: list[dict[str, Any]] = []
    try:
        dist_frame = sub.dropna()
        if len(dist_frame) >= 20 and dist_frame.shape[1] >= 2:
            x = np.asarray(dist_frame.to_numpy(dtype=float), dtype=float)
            mu = np.mean(x, axis=0)
            cov = np.cov(x, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            diff = x - mu
            d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
            distances = np.sqrt(np.maximum(d2, 0.0))
            threshold = float(np.quantile(distances, 0.99))
            idx = dist_frame.index
            hits = np.where(distances >= threshold)[0].tolist()
            for pos in hits[:25]:
                joint_anomalies.append(
                    {
                        "index": str(idx[pos]),
                        "distance": _safe_float(distances[pos]),
                    }
                )
    except Exception:
        warnings.append("Joint anomaly screening skipped due to covariance instability.")

    payload = {
        "numeric_columns": [str(c) for c in cols],
        "top_correlations": pairs[:30],
        "pca": pca_payload,
        "vif": vif_rows[:20],
        "joint_anomalies": joint_anomalies,
    }
    return payload, warnings


def build_consolidated_lab_payload(
    *,
    lab_key: str,
    filename: str,
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    requested_column: str | None,
    forecast_pct: float,
    contamination: float,
    max_points: int,
    get_cached_column_forecast_fn: Callable[[str, str, pd.Series, int], tuple[pd.Series | None, pd.DataFrame | None]],
    get_cached_anomalies_fn: Callable[[str, str, pd.Series, float], tuple[pd.Index, pd.Series]],
    compute_forecast_fn: Callable[[pd.Series, int], tuple[pd.Series, pd.DataFrame | None]],
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Dispatch helper used by API routes.

    Returns:
        payload, warnings, selected_column, selected_column_source
    """
    selected_col, col_source = resolve_numeric_column(numeric_df, requested_column)

    if lab_key == "quality":
        payload, warnings = build_quality_lab_payload(df=df, numeric_df=numeric_df)
        return payload, warnings, selected_col, col_source

    if lab_key == "multivariate":
        payload, warnings = build_multivariate_lab_payload(numeric_df=numeric_df)
        return payload, warnings, selected_col, col_source

    if not selected_col:
        return (
            {
                "message": "No numeric columns are available for this lab.",
            },
            ["This lab requires numeric columns, but none were detected."],
            None,
            "none",
        )

    if lab_key == "forecast":
        payload, warnings = build_forecast_lab_payload(
            filename=filename,
            column=selected_col,
            numeric_df=numeric_df,
            forecast_pct=forecast_pct,
            contamination=contamination,
            max_points=max_points,
            get_cached_column_forecast_fn=get_cached_column_forecast_fn,
            get_cached_anomalies_fn=get_cached_anomalies_fn,
            compute_forecast_fn=compute_forecast_fn,
        )
        return payload, warnings, selected_col, col_source

    if lab_key == "anomaly":
        payload, warnings = build_anomaly_lab_payload(
            filename=filename,
            column=selected_col,
            numeric_df=numeric_df,
            contamination=contamination,
            max_points=max_points,
            get_cached_anomalies_fn=get_cached_anomalies_fn,
        )
        return payload, warnings, selected_col, col_source

    if lab_key == "change-points":
        payload, warnings = build_change_points_lab_payload(
            column=selected_col,
            numeric_df=numeric_df,
            max_points=max_points,
        )
        return payload, warnings, selected_col, col_source

    if lab_key == "conformal":
        payload, warnings = build_conformal_lab_payload(
            filename=filename,
            column=selected_col,
            numeric_df=numeric_df,
            forecast_pct=forecast_pct,
            max_points=max_points,
            get_cached_column_forecast_fn=get_cached_column_forecast_fn,
            compute_forecast_fn=compute_forecast_fn,
        )
        return payload, warnings, selected_col, col_source

    if lab_key == "shap":
        payload, warnings = build_shap_lab_payload(
            column=selected_col,
            numeric_df=numeric_df,
        )
        return payload, warnings, selected_col, col_source

    return ({"message": f"Unsupported lab key: {lab_key}"}, ["Unsupported lab key."], selected_col, col_source)


__all__ = [
    "DEFAULT_CONFIDENCE_LEVELS",
    "resolve_numeric_column",
    "build_labs_meta_payload",
    "build_forecast_lab_payload",
    "build_anomaly_lab_payload",
    "build_quality_lab_payload",
    "build_change_points_lab_payload",
    "build_conformal_lab_payload",
    "build_shap_lab_payload",
    "build_multivariate_lab_payload",
    "build_consolidated_lab_payload",
]
