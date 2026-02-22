# mypy: ignore-errors
# ruff: noqa: F401,F403,F405
from __future__ import annotations

from data_analysis.runtime_app import *

_LOCAL_SYMBOLS = {
    '_LOCAL_SYMBOLS',
    '_bind_runtime_globals',
    '_thin_series',
    '_thin_series_keep_extrema',
    'normalize_timeseries',
    '_infer_future_index',
    '_infer_seasonal_period',
    '_compute_basic_stats',
    '_match_amplitude',
    '_compute_forecast',
    '_recent_slope_forecast',
    '_forecast_with_fallback',
    'get_cached_column_forecast',
    '__all__',
}


def _bind_runtime_globals():
    import data_analysis.runtime_app as rt

    g = globals()
    for key, value in rt.__dict__.items():
        if key.startswith("__") or key in _LOCAL_SYMBOLS:
            continue
        g[key] = value
    return rt


def _thin_series(s: pd.Series, max_points: int) -> pd.Series:
    _bind_runtime_globals()
    try:
        if not isinstance(s, pd.Series):
            return s
        n = len(s)
        if max_points and max_points > 0 and n > max_points:
            step = max(1, n // max_points)
            out = s.iloc[::step]
            # Ensure the last point is included for continuity
            try:
                if out.index[-1] != s.index[-1]:
                    out = pd.concat([out, s.iloc[np.array([-1], dtype=int)]])
            except Exception:
                pass
            return out
        return s
    except Exception:
        return s


def _thin_series_keep_extrema(s: pd.Series, max_points: int, keep_idx: pd.Index | None = None) -> pd.Series:
    """Thin a series while always keeping min/max, last points, and specific requested indices in original order."""
    _bind_runtime_globals()
    try:
        if not isinstance(s, pd.Series):
            return s
        n = len(s)
        if not max_points or max_points <= 0 or n <= max_points:
            return s

        step = max(1, n // max_points)
        keep_pos = list(range(0, n, step))
        keep_pos.append(n - 1)

        vals = np.asarray(s.to_numpy(dtype=float), dtype=float)
        min_pos = int(np.argmin(vals))
        max_pos = int(np.argmax(vals))
        keep_pos.extend([min_pos, max_pos])

        if keep_idx is not None and len(keep_idx) > 0:
            try:
                bonus_pos = s.index.get_indexer(keep_idx)
                keep_pos.extend([int(p) for p in bonus_pos if p >= 0])
            except Exception:
                pass

        keep_pos = sorted(set(keep_pos))
        return s.iloc[np.array(keep_pos, dtype=int)]
    except Exception:
        return _thin_series(s, max_points)


def normalize_timeseries(series: pd.Series) -> pd.Series:
    """
    Ensure a numeric series with a clean, timezone-naive, sorted DatetimeIndex when possible.
    - Coerces values to numeric and drops NaNs.
    - If index is DatetimeIndex, make it tz-naive, sort it, and drop duplicate index entries (keep last).
    - Otherwise, return the numeric series as-is.
    """
    _bind_runtime_globals()
    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
    except Exception:
        try:
            s = pd.Series(series).dropna()
        except Exception:
            return series
    try:
        idx = s.index
        if isinstance(idx, pd.DatetimeIndex):
            try:
                idx = idx.tz_convert(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
            try:
                s = s.copy()
                s.index = idx
            except Exception:
                pass
            try:
                s = s.sort_index()
            except Exception:
                pass
            try:
                if not s.index.is_unique:
                    s = s[~s.index.duplicated(keep='last')]
            except Exception:
                pass
    except Exception:
        pass
    return s


def _infer_future_index(idx, steps):
    _bind_runtime_globals()
    if _is_reliable_timeseries_index(idx):
        # Always calculate the base interval for most accurate forecasting
        if len(idx) > 1:
            # Calculate all intervals in the entire dataset for accurate detection
            diffs = pd.Series(idx).diff().dropna()
            
            if not diffs.empty:
                # Find the minimum non-zero interval (the base sampling rate)
                # This handles datasets with gaps better than median or mode
                non_zero_diffs = diffs[diffs > pd.Timedelta(0)]
                if len(non_zero_diffs) > 0:
                    offset = non_zero_diffs.min()
                else:
                    offset = diffs.median()
            else:
                # Fallback: average interval across entire dataset
                total_duration = idx[-1] - idx[0]
                offset = total_duration / (len(idx) - 1)
        else:
            offset = pd.Timedelta(hours=1)
        
        # Debug logging
        try:
            app.logger.debug("Forecast: %d steps, offset=%s, last_date=%s, forecast_end=%s", steps, offset, idx[-1], idx[-1] + offset * steps)
        except Exception:
            pass
        
        # Generate future timestamps manually to ensure correct spacing
        start = idx[-1]
        future_dates = [start + offset * (i + 1) for i in range(steps)]
        return pd.DatetimeIndex(future_dates)
    
    try:
        ser_idx = pd.Series(idx.astype('int64') if hasattr(idx, 'astype') else list(idx))
    except Exception:
        ser_idx = pd.Series(range(len(idx)))
    diffs = ser_idx.diff().dropna()
    step = int(diffs.median()) if not diffs.empty else 1
    last = int(ser_idx.iloc[-1])
    return pd.Index([last + step * (i + 1) for i in range(steps)])


def _infer_seasonal_period(idx, min_seasons=2):
    _bind_runtime_globals()
    if not _is_reliable_timeseries_index(idx):
        return None
    freq = (idx.freqstr or pd.infer_freq(idx)) or ""
    f = freq.upper()
    if f.startswith("H"):
        period = 24
    elif f.startswith("T") or f.startswith("MIN"):
        period = 60
    elif f.startswith("S"):
        period = 60
    elif f.startswith("D"):
        period = 7
    elif f.startswith("W"):
        period = 52
    elif f.startswith("M"):
        period = 12
    elif f.startswith("Q"):
        period = 4
    else:
        period = None
    try:
        n = len(idx)
        if period is None or n < period * min_seasons:
            return None
        return period
    except Exception:
        return None


def _compute_basic_stats(series: pd.Series) -> dict[str, float]:
    """Compute basic statistics for a series (min, max, mean, median, std)."""
    _bind_runtime_globals()
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        nan = float('nan')
        return {"min": nan, "max": nan, "mean": nan, "median": nan, "std": nan}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
    }


def _match_amplitude(
    history: pd.Series,
    forecast_series: pd.Series,
    conf_df: pd.DataFrame | None = None,
    seasonal_period: int | None = None,
    min_scale: float = 0.85,
    max_scale: float = 2.5,
):
    """Scale forecast deviations to better match recent history amplitude.
    - Compute std of recent history increments vs forecast increments.
    - If forecast variance is too low, scale deviations around a linear baseline.
    - Adjust conf intervals by same scale.
    Returns (forecast_series, conf_df).
    """
    _bind_runtime_globals()
    try:
        y = pd.to_numeric(history, errors='coerce').dropna()
        fc = pd.to_numeric(forecast_series, errors='coerce')
        if len(y) < 6 or len(fc) < 2:
            return forecast_series, conf_df
        n = len(y)
        w = seasonal_period if (isinstance(seasonal_period, int) and seasonal_period >= 2) else max(12, n // 4)
        y_win = y.tail(min(n, int(w)))
        y_win_arr = np.asarray(y_win.to_numpy(dtype=float), dtype=float)
        fc_arr = np.asarray(fc.to_numpy(dtype=float), dtype=float)
        hist_diffs = np.diff(y_win_arr)
        fc_diffs = np.diff(fc_arr)
        std_hist = float(np.nanstd(hist_diffs, ddof=1)) if len(hist_diffs) else 0.0
        std_fc = float(np.nanstd(fc_diffs, ddof=1)) if len(fc_diffs) else 0.0
        if not np.isfinite(std_hist) or not np.isfinite(std_fc) or std_hist <= 0:
            return forecast_series, conf_df
        # If forecast is perfectly flat, synthesize deviations from historical increments
        if std_fc <= 1e-12:
            rng = np.random.default_rng()
            incs = rng.choice(hist_diffs, size=len(fc), replace=True).astype(float)
            incs = incs - np.median(hist_diffs)
            dev = np.cumsum(incs)
            x = np.arange(len(fc), dtype=float)
            slope, intercept = np.polyfit(x, fc_arr, 1)
            baseline = slope * x + intercept
            fc2_vals = baseline + dev
            fc2 = pd.Series(fc2_vals, index=fc.index)
            c2 = None
            if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                c2 = conf_df.copy()
            return fc2, (c2 if c2 is not None else conf_df)
        ratio = std_hist / (std_fc + 1e-12)
        # Only scale if forecast is notably flatter than history
        if ratio < 1.0:
            return forecast_series, conf_df
        scale = float(np.clip(ratio, min_scale, max_scale))
        # Build linear baseline for forecast
        x = np.arange(len(fc), dtype=float)
        slope, intercept = np.polyfit(x, fc_arr, 1)
        baseline = slope * x + intercept
        deviations = fc_arr - baseline
        fc_scaled = baseline + scale * deviations
        fc2 = pd.Series(fc_scaled, index=fc.index)
        c2 = None
        if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
            # Scale CI bounds relative to new center
            lower = np.asarray(conf_df.iloc[:, 0].to_numpy(dtype=float), dtype=float)
            upper = np.asarray(conf_df.iloc[:, 1].to_numpy(dtype=float), dtype=float)
            lower_dev = lower - fc_arr
            upper_dev = upper - fc_arr
            fc2_arr = np.asarray(fc2.to_numpy(dtype=float), dtype=float)
            lower2 = fc2_arr + scale * lower_dev
            upper2 = fc2_arr + scale * upper_dev
            c2 = pd.DataFrame({"lower": lower2, "upper": upper2}, index=fc.index)
        return fc2, (c2 if c2 is not None else conf_df)
    except Exception:
        return forecast_series, conf_df


def _compute_forecast(series: pd.Series, steps: int):
    """Natural-looking forecast that preserves realistic patterns and variance.
    - Uses historical pattern matching for realistic continuations
    - Adds realistic noise based on historical volatility
    - Keeps forecast and CI within historical min/max by construction
    Returns (fc_mean, conf_df).
    """
    _bind_runtime_globals()
    def _natural_forecast(s: pd.Series, k: int):

        """Generate a synthetic forecast that matches the variations and dynamics

        of the historical data.  The forecast is **never** a verbatim copy of any

        past segment – instead it is built from:

          1. A gently damped trend baseline (no mean-reversion to avoid flattening).

          2. An STL-learned seasonal cycle replayed from real data shape (not a

             synthetic sine wave).

          3. AR(1) noise calibrated to the empirical residual variance from STL

             decomposition — matching the actual data's noise amplitude.

          4. Soft boundary compression near historical extremes (no hard flatline clipping).

        """

        s = pd.to_numeric(s, errors='coerce').dropna()

        idx = _infer_future_index(s.index if hasattr(s, 'index') else pd.RangeIndex(0, 1), k)

        if s.empty or k <= 0:

            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)

            ci = pd.DataFrame({"lower": zero, "upper": zero})

            return zero, ci



        n = len(s)

        values = np.asarray(s.to_numpy(dtype=float), dtype=float)

        last = float(values[-1])



        # --- Historical statistics (bounds) ---

        data_min = float(np.min(values))

        data_max = float(np.max(values))

        data_range = max(data_max - data_min, 1e-12)



        window_size = max(20, min(200, n // 2 if n >= 40 else n))

        recent = np.asarray(values[-window_size:], dtype=float)



        # Weighted trend: exponentially weight recent increments

        changes = np.diff(recent)

        if len(changes) >= 1:

            w = np.exp(np.linspace(-1, 0, len(changes)))

            w /= w.sum()

            trend = float(np.average(changes, weights=w))

        else:

            trend = 0.0



        # Increment volatility (σ of diff)

        inc_std = float(np.std(changes, ddof=1)) if len(changes) > 2 else data_range * 0.05

        if inc_std < 1e-9:

            inc_std = data_range * 0.05



        # --- Deterministic seed from data (reproducible per column) ---

        series_hash = hash((

            float(np.sum(values)), float(np.mean(values)), float(np.std(values)),

            float(values[0]), float(values[len(values) // 2]),

            last, trend, n, data_min, data_max

        ))

        rng = np.random.default_rng(int(abs(series_hash) % (2**31)))



        # ---- 1. Damped trend baseline anchored on recent mean ----
        # Anchor on the recent window mean instead of `last` to prevent
        # the forecast from saturating when the last observed value is
        # near a data boundary.
        step_k = np.arange(1, k + 1, dtype=float)
        damping = 0.97  # per-step decay
        damped_k = np.cumsum(damping ** np.arange(k))
        recent_mean = float(np.mean(recent))
        baseline = recent_mean + trend * damped_k



        # ---- 2. STL-learned seasonal component ----

        # Extract the ACTUAL seasonal shape from the data via STL decomposition

        # and replay it cyclically, instead of using a synthetic sine wave.

        seasonal = np.zeros(k, dtype=float)

        resid_from_stl = None  # will hold STL residuals if available

        sp = _infer_seasonal_period(s.index) if _is_reliable_timeseries_index(s.index) else None

        if isinstance(sp, int) and sp >= 2 and n >= sp * 2:

            try:

                stl_result = STL(s.astype(float), period=sp, robust=True).fit()

                # Get the learned seasonal pattern from the last full cycle

                stl_seasonal = stl_result.seasonal.values

                resid_from_stl = stl_result.resid.values



                # Build a single representative cycle by averaging last 2-3 cycles

                n_cycles = min(3, n // sp)

                tail_seasonal = stl_seasonal[-(n_cycles * sp):]

                cycle = np.zeros(sp, dtype=float)

                counts = np.zeros(sp, dtype=float)

                for i, v in enumerate(tail_seasonal):

                    cycle[i % sp] += v

                    counts[i % sp] += 1

                counts[counts == 0] = 1

                cycle /= counts



                # Replay cycle starting from where history left off

                last_cycle_pos = n % sp

                for i in range(k):

                    seasonal[i] = cycle[(last_cycle_pos + i) % sp]

            except Exception:

                # Fallback: no seasonal component

                seasonal = np.zeros(k, dtype=float)



        # ---- 3. AR(1) noise calibrated to empirical residuals ----

        # Use STL residuals if available; otherwise fall back to increment std.

        # This ensures the forecast noise amplitude matches the actual data's

        # non-seasonal, non-trend variability.

        if resid_from_stl is not None:

            noise_sigma = float(np.nanstd(resid_from_stl, ddof=1))

            resid_std = noise_sigma  # also use for CI

        else:

            # Residual std around linear fit (fallback)

            x_fit = np.arange(len(recent), dtype=float)

            try:

                slope_lr, intercept = np.polyfit(x_fit, np.asarray(recent, dtype=float), 1)

                residuals = recent - (slope_lr * x_fit + intercept)

                resid_std = float(np.std(residuals, ddof=1))

            except Exception:

                resid_std = inc_std

            noise_sigma = inc_std



        if not np.isfinite(noise_sigma) or noise_sigma < 1e-9:

            noise_sigma = inc_std

        if not np.isfinite(resid_std) or resid_std < 1e-9:

            resid_std = inc_std



        rho = 0.6  # AR(1) persistence for smooth variation

        noise_scale = noise_sigma * 0.50  # match ~50% of residual std

        noise = np.zeros(k, dtype=float)

        if noise_scale > 0:

            for i in range(k):

                shock = rng.normal(0.0, noise_scale)

                noise[i] = rho * (noise[i - 1] if i > 0 else 0.0) + shock



        # ---- Combine components ----

        forecast_vals = baseline + seasonal + noise

        # ---- Mean-matching + bounded transform ----
        # Shift the forecast so its average stays close to the recent history
        # mean, then clamp.  Clamping can asymmetrically pull the mean toward
        # bounds, so we iterate: shift → clamp → re-check → shift → …
        edge_eps = max(1e-9, data_range * 0.005)
        low_inner = data_min + edge_eps
        high_inner = data_max - edge_eps

        if high_inner <= low_inner:
            low_inner = data_min
            high_inner = data_max

        def _bound_inside(vals):
            """Affinely fit values inside (data_min, data_max) while preserving variation."""
            out = np.asarray(vals, dtype=float).copy()
            if out.size == 0:
                return out
            if high_inner <= low_inner:
                anchor = float(np.clip(np.mean(out), data_min, data_max))
                return np.full_like(out, fill_value=anchor, dtype=float)

            out_min = float(np.min(out))
            out_max = float(np.max(out))
            if out_max - out_min < 1e-12:
                anchor = float(np.clip(float(np.mean(out)), low_inner, high_inner))
                return np.full_like(out, fill_value=anchor, dtype=float)

            target_span = max(high_inner - low_inner, 1e-12)
            current_span = max(out_max - out_min, 1e-12)
            scale = min(1.0, target_span / current_span)
            mean_val = float(np.mean(out))
            out = (out - mean_val) * scale + mean_val

            if float(np.min(out)) < low_inner:
                out += (low_inner - float(np.min(out)))
            if float(np.max(out)) > high_inner:
                out -= (float(np.max(out)) - high_inner)
            return out

        # Iterative mean-matching with clamping (converges in 2-3 passes)
        for _pass in range(3):
            fc_mean = float(np.mean(forecast_vals))
            # Blend: 85% recent history mean, 15% raw forecast mean
            target_mean = 0.85 * recent_mean + 0.15 * fc_mean
            shift = target_mean - fc_mean
            if abs(shift) < 1e-9:
                break
            forecast_vals += shift
            forecast_vals = _bound_inside(forecast_vals)

        # ---- 5. Smooth junction with history ----
        # Short blend (2-3 points) so the forecast starts from `last`
        # for visual continuity, without pulling the whole series to an extreme.
        blend_len = min(max(2, k // 12), 3)
        blend_anchor = float(np.clip(last, low_inner, high_inner))
        for i in range(blend_len):
            alpha = (i + 1) / (blend_len + 1)
            forecast_vals[i] = (1 - alpha) * blend_anchor + alpha * forecast_vals[i]
        forecast_vals = _bound_inside(forecast_vals)



        fc = pd.Series(forecast_vals, index=idx)



        # --- Expanding confidence interval (bounded by construction) ---

        expanding_uncertainty = resid_std * np.sqrt(step_k)
        ci_target = 1.96 * expanding_uncertainty

        down_room = np.maximum(forecast_vals - data_min, 0.0)
        up_room = np.maximum(data_max - forecast_vals, 0.0)
        lower = forecast_vals - np.minimum(ci_target, down_room * 0.98)
        upper = forecast_vals + np.minimum(ci_target, up_room * 0.98)

        lower = np.minimum(lower, forecast_vals)
        upper = np.maximum(upper, forecast_vals)

        ci = pd.DataFrame({"lower": lower, "upper": upper}, index=idx)



        return fc, ci



    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
        if s.empty:
            idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            return zero, pd.DataFrame({"lower": zero, "upper": zero})
        
        # Build cache key from series shape, steps, AND actual values
        # Include summary stats to ensure different columns get different forecasts
        try:
            values_hash = hash((
                float(s.iloc[0]) if len(s) > 0 else 0.0,
                float(s.iloc[-1]) if len(s) > 0 else 0.0,
                float(s.mean()),
                float(s.std())
            ))
            cache_key = (tuple(s.shape) if hasattr(s, 'shape') else (len(s),), int(steps), values_hash)
            cached = FORECAST_CACHE.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            cache_key = None
        
        max_in = int(app.config.get('FORECAST_MAX_INPUT_POINTS', 2000))  # Balance speed and quality
        if max_in and len(s) > max_in:
            s = _thin_series(s, max_points=max_in)
        fc, ci = _natural_forecast(s, steps)
        
        # Post-processing: scale forecast amplitude to match recent history dynamics
        fc, ci = _match_amplitude(series, fc, conf_df=ci)

        data_min = float(s.min())
        data_max = float(s.max())
        data_range = max(data_max - data_min, 1e-12)
        edge_eps = max(1e-9, data_range * 0.005)
        low_inner = data_min + edge_eps
        high_inner = data_max - edge_eps

        if high_inner > low_inner and len(fc) > 0:
            fc_vals = np.asarray(fc.to_numpy(dtype=float), dtype=float)
            fc_min = float(np.min(fc_vals))
            fc_max = float(np.max(fc_vals))
            if fc_max - fc_min < 1e-12:
                anchor = float(np.clip(np.mean(fc_vals), low_inner, high_inner))
                fc_vals = np.full_like(fc_vals, fill_value=anchor, dtype=float)
            else:
                target_span = max(high_inner - low_inner, 1e-12)
                current_span = max(fc_max - fc_min, 1e-12)
                scale = min(1.0, target_span / current_span)
                mean_val = float(np.mean(fc_vals))
                fc_vals = (fc_vals - mean_val) * scale + mean_val
                if float(np.min(fc_vals)) < low_inner:
                    fc_vals += (low_inner - float(np.min(fc_vals)))
                if float(np.max(fc_vals)) > high_inner:
                    fc_vals -= (float(np.max(fc_vals)) - high_inner)
            fc = pd.Series(fc_vals, index=fc.index)

        if isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2 and len(fc) > 0:
            lower_raw = np.asarray(ci.iloc[:, 0].to_numpy(dtype=float), dtype=float)
            upper_raw = np.asarray(ci.iloc[:, 1].to_numpy(dtype=float), dtype=float)
            fc_arr = np.asarray(fc.to_numpy(dtype=float), dtype=float)
            down_width = np.maximum(fc_arr - lower_raw, 0.0)
            up_width = np.maximum(upper_raw - fc_arr, 0.0)
            down_cap = np.maximum(fc_arr - data_min, 0.0) * 0.98
            up_cap = np.maximum(data_max - fc_arr, 0.0) * 0.98
            lower_new = fc_arr - np.minimum(down_width, down_cap)
            upper_new = fc_arr + np.minimum(up_width, up_cap)
            ci = pd.DataFrame({"lower": lower_new, "upper": upper_new}, index=ci.index)

        # Cache the result
        try:
            if cache_key is not None:
                FORECAST_CACHE.set(cache_key, (fc, ci))
        except Exception:
            pass
        
        return fc, ci
    except Exception:
        # deterministic fallback: use simple trend
        try:
            s = pd.to_numeric(series, errors='coerce').dropna()
            idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)
            if len(s) >= 2:
                # Simple linear trend fallback
                s_arr = np.asarray(s.to_numpy(dtype=float), dtype=float)
                trend = float(np.mean(np.diff(s_arr[-min(20, len(s_arr)): ])))
                last = float(s.iloc[-1])
                vals = [last + trend * (i + 1) for i in range(steps)]
            else:
                last = float(s.iloc[-1]) if len(s) else 0.0
                vals = [last] * steps
            fc = pd.Series(vals, index=idx)
            s_arr = np.asarray(s.to_numpy(dtype=float), dtype=float)
            std = float(np.std(s_arr, ddof=1)) if len(s_arr) > 1 else 1.0
            fc_arr = np.asarray(fc.to_numpy(dtype=float), dtype=float)
            ci = pd.DataFrame({"lower": fc_arr - 1.96 * std, "upper": fc_arr + 1.96 * std}, index=idx)
            return fc, ci
        except Exception:
            idx = _infer_future_index(pd.RangeIndex(0, 1), steps)
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            return zero, pd.DataFrame({"lower": zero, "upper": zero})


def _recent_slope_forecast(series: pd.Series, steps: int = 10, lookback: int = 20):
    """Fallback forecast based on recent average slope.

    Returns:
      - forecast mean series
      - confidence interval DataFrame with columns: lower, upper
    """
    _bind_runtime_globals()
    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
        idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)

        if len(s) == 0:
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            return zero, pd.DataFrame({"lower": zero, "upper": zero}, index=idx)

        if len(s) >= 2:
            recent = np.asarray(s.to_numpy(dtype=float), dtype=float)[-min(int(lookback), len(s)):]
            diffs = np.diff(recent)
            slope = float(np.mean(diffs)) if len(diffs) else 0.0
        else:
            slope = 0.0

        last = float(s.iloc[-1])
        fc_vals = [last + slope * (i + 1) for i in range(int(steps))]
        fc = pd.Series(fc_vals, index=idx)

        s_arr = np.asarray(s.to_numpy(dtype=float), dtype=float)
        base_std = float(np.std(np.diff(s_arr), ddof=1)) if len(s_arr) > 2 else float(np.std(s_arr, ddof=1)) if len(s_arr) > 1 else 1.0
        if not np.isfinite(base_std) or base_std <= 0:
            base_std = 1.0
        horizon = np.sqrt(np.arange(1, int(steps) + 1, dtype=float))
        width = 1.96 * base_std * horizon
        fc_arr = np.asarray(fc.to_numpy(dtype=float), dtype=float)
        ci = pd.DataFrame({"lower": fc_arr - width, "upper": fc_arr + width}, index=idx)
        return fc, ci
    except Exception:
        idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)
        zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
        return zero, pd.DataFrame({"lower": zero, "upper": zero}, index=idx)


def _forecast_with_fallback(series: pd.Series, steps: int, filename: str | None = None, col: str | None = None):
    """Unified forecast with cascading fallbacks.

    Tries, in order:
      1. Cached column forecast (if filename and col provided)
      2. _compute_forecast (the full model pipeline)
      3. _recent_slope_forecast (simple linear trend)
      4. Flat-line forecast (last value repeated)

    Always returns (fc_mean, conf_df) — never None.
    """
    _bind_runtime_globals()
    fc_mean, ci = None, None

    # 1. Try cached column forecast
    if filename and col:
        try:
            rt = _bind_runtime_globals()
            cached_fc_fn = getattr(rt, "get_cached_column_forecast", None)
            if callable(cached_fc_fn):
                fc_mean, ci = cached_fc_fn(filename, col, series, steps)
            else:
                fc_mean, ci = get_cached_column_forecast(filename, col, series, steps)
        except Exception:
            fc_mean, ci = None, None

    # 2. Try full forecast pipeline
    if fc_mean is None or len(fc_mean) == 0:
        try:
            fc_mean, ci = _compute_forecast(series, steps)
        except Exception:
            fc_mean, ci = None, None

    # 3. Try simple slope fallback
    if fc_mean is None or len(fc_mean) == 0:
        try:
            fc_mean, ci = _recent_slope_forecast(series, steps=steps)
        except Exception:
            fc_mean, ci = None, None

    # 4. Flat-line fallback (always succeeds)
    if fc_mean is None or len(fc_mean) == 0:
        idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)
        last = float(series.iloc[-1]) if len(series) else 0.0
        fc_mean = pd.Series([last] * len(idx), index=idx)
        ci = None

    return fc_mean, ci


def get_cached_column_forecast(filename: str, column: str, series: pd.Series, steps: int):
    """Get forecast from cache or compute and cache it.
    
    This is the primary entry point for getting forecasts - ensures each
    (filename, column, steps) combination is only computed once, then reused
    across Forecast view, Interactive view, and PDF generation.
    """
    _bind_runtime_globals()
    if steps <= 0:
        return None, None
    cache_key = (filename, str(column), int(steps))
    cached = COLUMN_FORECAST_CACHE.get(cache_key)
    if cached is not None:
        app.logger.debug("Forecast cache HIT: %s/%s/%d", filename[:8], column, steps)
        return cached
    app.logger.debug("Forecast cache MISS: %s/%s/%d - computing", filename[:8], column, steps)
    fc_mean, conf_df = _compute_forecast(series, steps)
    COLUMN_FORECAST_CACHE.set(cache_key, (fc_mean, conf_df))
    return fc_mean, conf_df


__all__ = [
    '_thin_series',
    '_thin_series_keep_extrema',
    'normalize_timeseries',
    '_infer_future_index',
    '_infer_seasonal_period',
    '_compute_basic_stats',
    '_match_amplitude',
    '_compute_forecast',
    '_recent_slope_forecast',
    '_forecast_with_fallback',
    'get_cached_column_forecast',
]
