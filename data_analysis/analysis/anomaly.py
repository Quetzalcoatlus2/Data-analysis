from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from data_analysis.core.lazy_imports import get_isolation_forest, get_stl


def _anomaly_series_signature(series: pd.Series) -> tuple[Any, ...]:
    """Build a lightweight, hashable signature to avoid cache collisions across series slices."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        s = pd.Series(dtype=float)
    if s is None or s.empty:
        return ("empty", 0)
    try:
        first_idx = str(s.index[0])
    except Exception:
        first_idx = "?"
    try:
        last_idx = str(s.index[-1])
    except Exception:
        last_idx = "?"
    try:
        first_val = float(s.iloc[0])
    except Exception:
        first_val = float("nan")
    try:
        last_val = float(s.iloc[-1])
    except Exception:
        last_val = float("nan")
    try:
        mean_val = float(s.mean())
    except Exception:
        mean_val = float("nan")

    return (
        int(len(s)),
        first_idx,
        last_idx,
        None if not np.isfinite(first_val) else round(first_val, 8),
        None if not np.isfinite(last_val) else round(last_val, 8),
        None if not np.isfinite(mean_val) else round(mean_val, 8),
    )


def detect_anomalies(
    series: pd.Series,
    contamination: float = 0.02,
    *,
    is_reliable_timeseries_index: Callable[[pd.Index], bool] | None = None,
    infer_seasonal_period: Callable[[pd.Index], int | None] | None = None,
    logger: Any | None = None,
) -> tuple[pd.Index, pd.Series]:
    """Detect anomalies with IsolationForest and optional STL prefiltering for time series."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
    except Exception as e:
        if logger is not None:
            logger.debug("detect_anomalies numeric coercion fallback used: %s", e)
        s = pd.Series(dtype=float)

    if s is None or len(s) < 5:
        return pd.Index([]), pd.Series([], dtype=float)

    try:
        cont = float(contamination) if contamination is not None else 0.02
    except Exception as e:
        if logger is not None:
            logger.debug("detect_anomalies contamination parse fallback used: %s", e)
        cont = 0.02
    if not (0.0 < cont < 0.5):
        cont = 0.02

    is_ts_fn = is_reliable_timeseries_index or (lambda idx: isinstance(idx, pd.DatetimeIndex))
    infer_sp_fn = infer_seasonal_period or (lambda _idx: None)

    try:
        s_for_model = s
        if is_ts_fn(s.index):
            seasonal_period = infer_sp_fn(s.index)
            if isinstance(seasonal_period, int) and seasonal_period >= 2 and len(s) >= max(28, seasonal_period * 2):
                try:
                    stl_cls = get_stl()
                    stl_result = stl_cls(s.astype(float), period=int(seasonal_period), robust=True).fit()
                    resid = pd.to_numeric(pd.Series(stl_result.resid, index=s.index), errors="coerce").dropna()
                    if len(resid) >= max(20, seasonal_period * 2):
                        s_for_model = resid
                except Exception as e:
                    if logger is not None:
                        logger.debug("detect_anomalies STL residual fallback used: %s", e)
                    s_for_model = s

        vals = np.asarray(s_for_model.to_numpy(dtype=float), dtype=float)
        n = len(vals)
        if n < 5:
            return pd.Index([]), pd.Series([], dtype=float)

        isolation_forest_cls = get_isolation_forest()
        n_estimators = 300 if n >= 200 else 200
        model = isolation_forest_cls(
            contamination=cont,
            random_state=42,
            n_estimators=n_estimators,
            max_samples=min(256, n),
            n_jobs=1,
        )
        x = vals.reshape(-1, 1)
        labels = model.fit_predict(x)
        if labels is None or len(labels) != n:
            return pd.Index([]), pd.Series([], dtype=float)

        anomaly_positions = np.where(np.asarray(labels) == -1)[0]
        if anomaly_positions.size == 0:
            return pd.Index([]), pd.Series([], dtype=float)

        # decision_function: higher means more normal; invert so higher means more anomalous.
        raw_scores = -np.asarray(model.decision_function(x), dtype=float)
        finite_scores = raw_scores[np.isfinite(raw_scores)]
        if finite_scores.size == 0:
            return pd.Index([]), pd.Series([], dtype=float)

        score_med = float(np.nanmedian(finite_scores))
        score_mad = float(np.nanmedian(np.abs(finite_scores - score_med)))
        is_timeseries = is_ts_fn(s_for_model.index)
        if np.isfinite(score_mad) and score_mad > 1e-12:
            mad_multiplier = 2.5 if is_timeseries else 3.0
            min_keep_score = score_med + (mad_multiplier * score_mad)
        else:
            fallback_q = max(0.90, 1.0 - cont * 0.5)
            if not is_timeseries:
                fallback_q = max(fallback_q, 0.97)
            min_keep_score = float(np.nanquantile(finite_scores, fallback_q))

        keep_positions = [
            int(pos)
            for pos in anomaly_positions.tolist()
            if np.isfinite(raw_scores[int(pos)]) and raw_scores[int(pos)] >= min_keep_score
        ]
        if not keep_positions:
            anomaly_scores = raw_scores[anomaly_positions]
            best_rel = int(np.nanargmax(anomaly_scores))
            keep_positions = [int(anomaly_positions[best_rel])]

        # For non-time-series data with duplicate index labels, keep strongest anomaly per label.
        if not is_timeseries and len(keep_positions) > 1:
            best_by_label: dict[Any, tuple[int, float]] = {}
            for pos in keep_positions:
                label = s_for_model.index[int(pos)]
                score = float(raw_scores[int(pos)])
                prev = best_by_label.get(label)
                if prev is None or score > prev[1]:
                    best_by_label[label] = (int(pos), score)
            keep_positions = [v[0] for v in best_by_label.values()]

        selected_positions = sorted(keep_positions)
        if not s_for_model.index.is_unique:
            an_idx = pd.Index(selected_positions, dtype="int64", name="__pos__")
            an_score = pd.Series(
                raw_scores[selected_positions],
                index=pd.Index(selected_positions, dtype="int64", name="__pos__"),
                dtype=float,
            )
        else:
            an_idx = s_for_model.index[selected_positions]
            an_score = pd.Series(raw_scores[selected_positions], index=an_idx, dtype=float)
        return an_idx, an_score
    except Exception as e:
        if logger is not None:
            logger.debug("detect_anomalies failed; returning empty anomalies: %s", e)
        return pd.Index([]), pd.Series([], dtype=float)


def get_cached_anomalies(
    filename: str,
    column: str,
    series: pd.Series,
    contamination: float = 0.02,
    *,
    cache: Any | None = None,
    logger: Any | None = None,
    is_reliable_timeseries_index: Callable[[pd.Index], bool] | None = None,
    infer_seasonal_period: Callable[[pd.Index], int | None] | None = None,
) -> tuple[pd.Index, pd.Series]:
    """Get anomaly detection results from cache or compute and cache them."""
    cache_key = (
        filename,
        str(column),
        round(float(contamination), 6),
        _anomaly_series_signature(series),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            if logger is not None:
                logger.debug("Anomaly cache HIT: %s/%s", filename[:8], column)
            return cached
    if logger is not None:
        logger.debug("Anomaly cache MISS: %s/%s - computing", filename[:8], column)

    result = detect_anomalies(
        series,
        contamination=contamination,
        is_reliable_timeseries_index=is_reliable_timeseries_index,
        infer_seasonal_period=infer_seasonal_period,
        logger=logger,
    )
    if cache is not None:
        cache.set(cache_key, result)
    return result


__all__ = [
    "_anomaly_series_signature",
    "detect_anomalies",
    "get_cached_anomalies",
]
