# ruff: noqa: F401,F403,F405
from __future__ import annotations

from data_analysis.runtime_app import *

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "handle_health",
    "handle_api_ai_summary",
    "handle_api_interactive_data",
    "handle_full_history_json",
    "health",
    "api_ai_summary",
    "api_interactive_data",
    "full_history_json",
    "__all__",
}



def _bind_runtime_globals():
    import data_analysis.runtime_app as rt

    sync = getattr(rt, "_sync_ai_engine_state", None)
    if callable(sync):
        sync()

    g = globals()
    for key, value in rt.__dict__.items():
        if key.startswith("__") or key in _LOCAL_SYMBOLS:
            continue
        g[key] = value
    return rt


def handle_health():
    _bind_runtime_globals()
    return jsonify({"status": "ok"}), 200

def handle_api_ai_summary(filename):
    _bind_runtime_globals()
    """Async endpoint for fetching AI summary via AJAX for faster page loads."""
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "html": "<p>Invalid file.</p>"}), 400
    
    # Check cache first for instant response
    cached = _get_clean_ai_summary_from_cache(filename)
    if cached:
        m_name = None
        _m = re.search(r'<!--\s*model:(.*?)\s*-->', str(cached))
        if _m:
            m_name = _m.group(1).strip()
            if m_name.startswith('models/'):
                m_name = m_name[7:]
        model_display = m_name or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3.0-flash'
        display_html = re.sub(r'<!--.*?-->', '', str(cached), flags=re.DOTALL)
        return jsonify({"ok": True, "html": display_html, "cached": True, "model": model_display})
    
    # Generate new summary
    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return jsonify({"ok": False, "html": "<p>Dataset not found.</p>"}), 404
    

    ai_context = describe_for_ai(df, filename=filename)
    
    try:
        if ensure_ai_ready():
            summary = get_or_cache_ai_summary_for(filename, df, extra_context=ai_context)
            
            # Extract model for AJAX
            m_name = None
            if summary and isinstance(summary, str):
                _m = re.search(r'<!--\s*model:(.*?)\s*-->', summary)
                if _m:
                    m_name = _m.group(1).strip()
                    if m_name.startswith('models/'):
                        m_name = m_name[7:]
            
            model_display = m_name or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3.0-flash'
            
            display_html = re.sub(r'<!--.*?-->', '', str(summary), flags=re.DOTALL) if isinstance(summary, str) else summary
            
            return jsonify({"ok": True, "html": display_html, "cached": False, "model": model_display})
        else:
            fallback = offline_answer(df, "summary", error="AI not ready", filename=filename)
            return jsonify({"ok": True, "html": fallback, "cached": False})
    except Exception as e:
        app.logger.warning("API AI summary failed: %s", e)
        fallback = offline_answer(df, "summary", error=e, filename=filename)
        return jsonify({"ok": True, "html": fallback, "cached": False})

# Maximum points sent to the browser for interactive chart history and distribution.
# For 700k+ row datasets, sending the full series as JSON would be 10-100MB, causing
# the browser to hang or fail. We uniformly subsample to keep payloads manageable.
_MAX_INTERACTIVE_POINTS = 10_000  # max history trace points
_MAX_DIST_POINTS = 5_000          # max distribution values
_MAX_INTERACTIVE_FORECAST_POINTS = 2_000  # max forecast/CI trace points


def _downsample_series(x_list: list, y_list: list, max_pts: int) -> tuple[list, list]:
    """Uniformly downsample parallel x/y lists to at most max_pts points."""
    n = len(x_list)
    if n <= max_pts:
        return x_list, y_list
    step = max(1, n // max_pts)
    xs = x_list[::step]
    ys = y_list[::step]
    # Always keep the last point for continuity
    if xs[-1] != x_list[-1]:
        xs = xs + [x_list[-1]]
        ys = ys + [y_list[-1]]
    return xs, ys


def _downsample_indices(length: int, max_pts: int) -> list[int]:
    """Return sampled indices that always include the last point."""
    if length <= 0:
        return []
    if length <= max_pts:
        return list(range(length))
    step = max(1, length // max_pts)
    out = list(range(0, length, step))
    if out[-1] != length - 1:
        out.append(length - 1)
    return out


def handle_api_interactive_data(filename):
    _bind_runtime_globals()
    """Return interactive chart data via AJAX for faster page load.
    
    This endpoint returns pre-computed or freshly computed interactive chart traces
    as JSON, allowing the page to load quickly and fetch chart data asynchronously.
    For large datasets (700k+ rows), history/distribution values are uniformly
    downsampled to keep JSON payloads manageable.
    """
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "error": "Invalid filename"}), 400
    request_start = time.perf_counter()

    # Get parameters - default to 5%
    raw_pct = request.args.get('forecast_pct', '0.05')
    try:
        user_contam = float(request.args.get('contamination', '0.02'))
    except Exception as e:
        app.logger.debug("Interactive API contamination parse fallback used for %s: %s", filename, e)
        user_contam = 0.02
    user_contam = max(0.001, min(0.2, user_contam))

    def _safe_number(value):
        try:
            out = float(value)
            return out if math.isfinite(out) else None
        except Exception:
            return None
    
    try:
        pct = float(raw_pct) if raw_pct else 0.05
        # Allow 0 for no-forecast mode, clamp max to 0.5
        pct = max(0, min(0.5, pct))
    except Exception as e:
        app.logger.debug("Interactive API forecast_pct parse fallback used for %s: %s", filename, e)
        pct = 0.05

    # Check cache first for instant response (keyed by request parameters)
    cache_key = _build_interactive_cache_key(filename, pct, user_contam)
    cached = INTERACTIVE_DATA_CACHE.get(cache_key)
    if cached is None:
        # Backward compatibility with legacy filename-only key for default params
        if abs(pct - 0.05) < 1e-9 and abs(user_contam - 0.02) < 1e-9:
            cached = INTERACTIVE_DATA_CACHE.get(filename)
    if cached is not None:
        _log_cache_stats_if_needed("interactive-cached")
        elapsed = time.perf_counter() - request_start
        app.logger.info("Interactive API cache HIT file=%s pct=%.4f contam=%.4f elapsed=%.3fs", filename, pct, user_contam, elapsed)
        return jsonify({"ok": True, "data": cached, "cached": True})

    # Load DataFrame
    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return jsonify({"ok": False, "error": "Dataset not found"}), 404
    
    def _steps_for_history_length(history_len: int) -> int:
        """Map forecast_pct to steps for a concrete visible history length."""
        if pct <= 0 or history_len <= 0:
            return 0
        pct_den = max(1e-9, 1.0 - float(pct))
        return max(1, int(math.floor(float(history_len) * float(pct) / pct_den)))
    is_timeseries = _is_reliable_timeseries_index(df.index)
    numeric_df_cached = get_cached_numeric_df(filename, df)
    
    interactive = []
    cols_processed = 0
    # Default: no column cap (return all numeric columns). Set env var >0 to cap.
    try:
        configured_max_cols = int(os.getenv("INTERACTIVE_API_MAX_COLS", "0"))
    except Exception:
        configured_max_cols = 0
    max_cols = configured_max_cols if configured_max_cols > 0 else max(1, len(numeric_df_cached.columns))
    
    for column in df.columns:
        if cols_processed >= max_cols:
            break

        if column not in numeric_df_cached.columns:
            continue
        series = numeric_df_cached[column].dropna()
        if series.empty or len(series) < 5:
            continue
        
        cols_processed += 1
        
        # Reuse shared anomaly cache helper.
        an_idx, an_score = get_cached_anomalies(filename, column, series, user_contam)
        
        # Build traces using NUMERIC indices for proportional X-axis display
        # This ensures forecast (e.g. 20%) takes proportionally 20% of chart width
        # Always provide full series to interactive API.
        # The UI Data Range control applies any user-requested reduction.
        s_tail = series
        n_hist = len(s_tail)
        
        # Use numeric indices 0, 1, 2... for history
        x_hist_numeric_full = list(range(n_hist))
        y_hist_full = [_safe_number(v) for v in s_tail.values]
        original_labels_full = [str(i) for i in s_tail.index]
        
        # Downsample history for large series to keep JSON payload manageable.
        # This prevents browser hangs on 700k+ row datasets.  The downsampled
        # positions still represent the same proportional numeric x-axis so
        # the forecast proportion is preserved.
        x_hist_numeric, y_hist = _downsample_series(
            x_hist_numeric_full, y_hist_full, _MAX_INTERACTIVE_POINTS
        )
        # Reconstruct original_labels for hover text at the sampled positions
        ds_step = max(1, n_hist // _MAX_INTERACTIVE_POINTS)
        original_labels = original_labels_full[::ds_step]
        if original_labels[-1] != original_labels_full[-1]:
            original_labels = original_labels + [original_labels_full[-1]]
        # Trim to match x_hist_numeric length (rounding may cause ±1 off)
        min_len = min(len(x_hist_numeric), len(y_hist), len(original_labels))
        x_hist_numeric = x_hist_numeric[:min_len]
        y_hist = y_hist[:min_len]
        original_labels = original_labels[:min_len]
        sampled_y_by_pos = {int(xv): y_hist[i] for i, xv in enumerate(x_hist_numeric)}
        
        traces = [{
            "type": "scatter",
            "mode": "lines",
            "name": "History",
            "x": x_hist_numeric,
            "y": y_hist,
            "text": original_labels,  # Show original labels on hover
            "hovertemplate": "%{text}<br>%{y}<extra></extra>",
            "line": {"color": "rgb(31,119,180)", "width": 1.5},
            "marker": {"size": 3, "opacity": 0.5}
        }]
        
        # Add anomaly markers using numeric positions
        if len(an_idx):
            an_display = _cap_anomalies_for_display(an_idx, an_score)
            an_positions = _anomaly_positions_for_index(s_tail.index, an_display)
            an_values = []
            an_positions_display = []
            an_labels = []
            tail_vals = np.asarray(s_tail.to_numpy(dtype=float), dtype=float)
            tail_median = float(np.nanmedian(tail_vals)) if tail_vals.size else 0.0
            seen_positions: set[int] = set()

            score_buckets: dict[Any, list[float]] = {}
            try:
                for idx_val, score_val in an_score.items():
                    if pd.notna(score_val):
                        score_buckets.setdefault(idx_val, []).append(float(score_val))
            except Exception as e:
                app.logger.debug("Interactive anomaly score bucketing skipped for %s: %s", column, e)
                score_buckets = {}

            for i in an_positions:
                pos_raw = int(i)
                if pos_raw < 0 or pos_raw >= n_hist:
                    continue

                if pos_raw in sampled_y_by_pos:
                    pos_display = pos_raw
                elif x_hist_numeric:
                    # Keep anomaly markers on displayed history points for downsampled traces.
                    pos_display = min(x_hist_numeric, key=lambda p: abs(int(p) - pos_raw))
                else:
                    continue

                if pos_display in seen_positions:
                    continue
                seen_positions.add(pos_display)

                idx = s_tail.index[pos_raw]
                an_positions_display.append(pos_display)
                an_values.append(sampled_y_by_pos.get(pos_display, _safe_number(s_tail.iloc[pos_raw])))
                score_list = score_buckets.get(idx, [])
                score_val = score_list.pop(0) if score_list else None
                score_buckets[idx] = score_list

                if score_val is None:
                    try:
                        score_val = float(an_score.loc[pos_raw]) if pos_raw in an_score.index else None
                    except Exception:
                        score_val = None

                if score_val is None or not math.isfinite(float(score_val)):
                    reason = "Outlier"
                elif float(s_tail.iloc[pos_raw]) >= tail_median:
                    reason = f"High outlier (IF score={float(score_val):.3f})"
                else:
                    reason = f"Low outlier (IF score={float(score_val):.3f})"
                an_labels.append(f"{idx} | {reason}")
            
            if an_positions_display:
                traces.append({
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Anomaly",
                    "x": an_positions_display,
                    "y": an_values,
                    "text": an_labels,
                    "hovertemplate": "Anomaly<br>%{text}<br>%{y}<extra></extra>",
                    "marker": {"color": "#ef4444", "size": 5, "opacity": 0.9}
                })
        
        # Add forecast using numeric positions continuing from history.
        split_x = None
        n_fc_total = 0
        fc_labels: list[str] = []
        fc_y: list[float | None] = []
        column_steps = _steps_for_history_length(n_hist)
        if column_steps > 0 and len(series) >= 10:
            try:
                # Use cached forecast - reuses computation from analyze_file if available
                fc_mean, conf_df = get_cached_column_forecast(filename, column, series, column_steps)
                if fc_mean is None:
                    raise ValueError("No forecast generated")
                split_x = n_hist - 0.5  # Split line between last history and first forecast
                
                # Forecast uses indices n_hist, n_hist+1, n_hist+2...
                n_fc_total = int(len(fc_mean))
                fc_x_numeric_full = list(range(n_hist, n_hist + n_fc_total))
                fc_y_full = [_safe_number(v) for v in fc_mean.values]
                fc_labels_full = [str(i) for i in fc_mean.index]

                sample_idx = _downsample_indices(n_fc_total, _MAX_INTERACTIVE_FORECAST_POINTS)
                fc_x_numeric = [fc_x_numeric_full[i] for i in sample_idx]
                fc_y = [fc_y_full[i] for i in sample_idx]
                fc_labels = [fc_labels_full[i] for i in sample_idx]
                
                if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                    ci_lower_full = [_safe_number(v) for v in conf_df.iloc[:, 0].values]
                    ci_upper_full = [_safe_number(v) for v in conf_df.iloc[:, 1].values]
                    ci_lower = [ci_lower_full[i] for i in sample_idx]
                    ci_upper = [ci_upper_full[i] for i in sample_idx]
                    
                    # Add CI traces
                    traces.append({
                        "type": "scatter", "mode": "lines", "name": "95% CI",
                        "x": fc_x_numeric, "y": ci_lower, "line": {"width": 0},
                        "hoverinfo": "skip", "showlegend": True
                    })
                    traces.append({
                        "type": "scatter", "mode": "lines", "name": "95% CI",
                        "x": fc_x_numeric, "y": ci_upper, "line": {"width": 0},
                        "fill": "tonexty", "fillcolor": "rgba(255,69,0,0.18)",
                        "hoverinfo": "skip", "showlegend": False
                    })
                
                # Forecast line with connection to last history point
                x_plot = [n_hist - 1] + list(fc_x_numeric)
                y_plot = [_safe_number(series.iloc[-1])] + list(fc_y)
                text_plot = [original_labels[-1]] + list(fc_labels)
                    
                traces.append({
                    "type": "scatter", "mode": "lines+markers", "name": "Forecast",
                    "x": x_plot, "y": y_plot,
                    "text": text_plot,
                    "hovertemplate": "%{text}<br>%{y}<extra></extra>",
                    "line": {"color": "orangered", "width": 3},
                    "marker": {"size": 3}
                })
            except Exception as e:
                app.logger.debug("Forecast skipped for %s: %s", column, e)
        
        # Total x-axis extent: 0..n_hist for history, n_hist..n_hist+n_fc for forecast.
        # Setting xaxis.range explicitly ensures forecast occupies exactly pct% of
        # the x-axis regardless of data filtering.
        total_x_extent = n_hist + (n_fc_total if split_x else 0)
        x_axis_range = [0, total_x_extent]

        # Build uniform tick values across the full numeric x range (20 ticks max)
        tick_count = 20
        extent_int = int(max(1, total_x_extent))
        if extent_int > tick_count:
            raw_ticks = np.linspace(0, extent_int - 1, num=tick_count, dtype=float).tolist()
            tv_num = sorted(set(int(round(v)) for v in raw_ticks))
            if tv_num[-1] != extent_int - 1:
                tv_num.append(extent_int - 1)
        else:
            tv_num = list(range(extent_int))

        # Map numeric positions back to original index labels for tick text
        def _idx_label(pos: int) -> str:
            if pos < n_hist:
                # map sampled position back to original: pos is in numeric full-history space
                orig_pos = min(pos, len(s_tail.index) - 1)
                return str(s_tail.index[orig_pos])
            else:
                if split_x and n_fc_total > 0 and fc_labels:
                    fc_pos = min(max(int(pos - n_hist), 0), n_fc_total - 1)
                    rel = fc_pos / max(1, n_fc_total - 1)
                    label_idx = min(int(round(rel * (len(fc_labels) - 1))), len(fc_labels) - 1)
                    return fc_labels[label_idx]
                return str(pos)

        tv = tv_num
        tt = [_idx_label(p) for p in tv_num]

        def _format_large_tick(v: float) -> str:
            try:
                value = float(v)
                mag = abs(value)
                if mag >= 1e12:
                    return f"{value / 1e12:.3f}T"
                if mag >= 1e9:
                    return f"{value / 1e9:.3f}B"
                return f"{value:.2f}"
            except Exception:
                return str(v)

        # Detect if axis values are large enough for compact B/T labels.
        try:
            y_scale_vals = [v for v in y_hist if v is not None]
            y_scale_vals.extend(v for v in fc_y if v is not None)
            y_max_abs = max((abs(v) for v in y_scale_vals), default=0.0)
            use_suffix = y_max_abs >= 1e9
            y_min_val = min(y_scale_vals) if y_scale_vals else 0.0
            y_max_val = max(y_scale_vals) if y_scale_vals else 0.0
            if use_suffix:
                if abs(y_max_val - y_min_val) <= 1e-12:
                    y_tick_vals = [float(y_min_val)]
                else:
                    y_tick_vals = [
                        float(v) for v in np.linspace(float(y_min_val), float(y_max_val), num=8)
                    ]
                y_tick_text = [_format_large_tick(v) for v in y_tick_vals]
            else:
                y_tick_vals = []
                y_tick_text = []
        except Exception:
            use_suffix = False
            y_tick_vals = []
            y_tick_text = []

        yaxis_cfg: dict[str, Any] = {
            "title": column,
            "showgrid": True,
        }
        if use_suffix and y_tick_vals:
            yaxis_cfg.update({
                "tickmode": "array",
                "tickvals": y_tick_vals,
                "ticktext": y_tick_text,
            })

        # Build layout
        layout = {
            "title": {"text": f"{column} (interactive)", "x": 0.02},
            "xaxis": {
                "title": str(df.index.name) if df.index.name else ("Timestamp" if is_timeseries else "Index"),
                "showgrid": True,
                "tickmode": "array",
                "tickvals": tv,
                "ticktext": tt,
                "range": x_axis_range,
            },
            "yaxis": yaxis_cfg,
            "shapes": [] if not split_x else [{
                "type": "line", "xref": "x", "yref": "paper",
                "x0": split_x, "x1": split_x, "y0": 0, "y1": 1,
                "line": {"color": "gray", "width": 1, "dash": "dot"}
            }],
            "legend": {"orientation": "h"},
            "margin": {"l": 40, "r": 10, "t": 40, "b": 40}
        }
        
        if is_timeseries:
            layout["xaxis"]["rangeslider"] = {"visible": True}
        
        # Downsample distribution values to keep JSON payload small
        dist_raw = [v for v in (_safe_number(x) for x in series.dropna().values) if v is not None]
        if len(dist_raw) > _MAX_DIST_POINTS:
            dist_step = max(1, len(dist_raw) // _MAX_DIST_POINTS)
            dist_raw = dist_raw[::dist_step]
        dist = {
            "name": column,
            "values": dist_raw
        }
        
        # Compute statistics for the column
        try:
            stats = {
                "min": _safe_number(series.min()),
                "max": _safe_number(series.max()),
                "mean": _safe_number(series.mean()),
                "median": _safe_number(series.median()),
                "std": _safe_number(series.std())
            }
        except Exception as e:
            app.logger.debug("Interactive stats computation skipped for %s: %s", column, e)
            stats = None
        
        interactive.append({"column": column, "traces": traces, "layout": layout, "distribution": dist, "stats": stats})
    
    # Cache the result
    INTERACTIVE_DATA_CACHE.set(cache_key, interactive)
    _log_cache_stats_if_needed("interactive")
    elapsed = time.perf_counter() - request_start
    app.logger.info("Interactive API cache MISS file=%s pct=%.4f contam=%.4f cols=%d elapsed=%.3fs", filename, pct, user_contam, cols_processed, elapsed)
    
    return jsonify({"ok": True, "data": interactive, "cached": False})

def handle_full_history_json():
    _bind_runtime_globals()
    """
    Return full-history numeric series for interactive charts.
    Query params:
      - filename: required, hashed upload filename (e.g., 40-hex.csv/xlsx/json/txt)
      - display: optional, original filename for UI context
      - max_points: optional int; if provided and smaller than length, uniformly thin the points
    Response JSON:
      {
        "ok": true/false,
        "message": optional on errors,
        "is_timeseries": bool,
        "display": str,
        "length": int (number of rows serialized),
        "columns": [str, ...],  
        "x": [ ... ],           
        "series": { "col": [y0, y1, ...], ... }
      }
    """
    try:
        filename = request.args.get('filename', '').strip()
        display = (request.args.get('display') or filename or '').strip()
        max_points = request.args.get('max_points', type=int)

        if not filename:
            return jsonify({"ok": False, "message": "Missing 'filename' parameter."}), 400
        if not HASHED_UPLOAD_RE.match(filename):
            return jsonify({"ok": False, "message": "Invalid filename format."}), 400

        df = get_dataframe_for(filename)  
        if df is None or df.empty:
            return jsonify({"ok": False, "message": "Dataset not found or empty."}), 404

        
        try:
            df = df.sort_index()
        except Exception as sort_err:
            app.logger.debug("full_history_json sort_index skipped for %s: %s", filename, sort_err)

        is_ts = _is_reliable_timeseries_index(df.index)

        
        if is_ts:
            idx = pd.DatetimeIndex(df.index)
            
            try:
                idx = idx.tz_convert(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception as tz_err:
                    app.logger.debug("full_history_json timezone normalization skipped for %s: %s", filename, tz_err)
            try:
                x_all = [ts.isoformat() for ts in idx.to_pydatetime()]
            except Exception as e:
                app.logger.debug("full_history_json ISO timestamp conversion fallback for %s: %s", filename, e)
                
                x_all = [str(v) for v in idx.astype('datetime64[ns]').tolist()]
        else:
            
            try:
                x_all_raw = df.index.tolist()
                x_all = []
                for v in x_all_raw:
                    if isinstance(v, (int, float, str)):
                        x_all.append(v)
                    elif isinstance(v, (np.integer,)):
                        x_all.append(int(v))
                    elif isinstance(v, (np.floating,)):
                        x_all.append(float(v))
                    else:
                        x_all.append(str(v))
            except Exception as e:
                app.logger.debug("full_history_json index serialization fallback for %s: %s", filename, e)
                x_all = list(range(len(df)))

        n = len(x_all)
        
        step = 1
        if max_points and max_points > 0 and n > max_points:
            
            step = max(1, n // max_points)

        
        num_df = get_cached_numeric_df(filename, df)
        numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]
        
        if not numeric_cols:
            for c in df.columns:
                try:
                    parsed = _try_parse_numeric_series(df[c])
                    num_df[c] = parsed
                except Exception as e:
                    app.logger.debug("full_history_json numeric parse skipped for %s.%s: %s", filename, c, e)
                    continue
            numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]

        
        x_vals = x_all[::step] if step > 1 else x_all
        series = {}
        for c in numeric_cols:
            try:
                y_all = num_df[c].astype(float).tolist()
            except Exception as e:
                app.logger.debug("full_history_json float cast fallback for %s.%s: %s", filename, c, e)
                
                y_all = [float(v) if pd.notna(v) else None for v in num_df[c].tolist()]
            y_vals = y_all[::step] if step > 1 else y_all

            
            if len(y_vals) != len(x_vals):
                m = min(len(y_vals), len(x_vals))
                y_vals = y_vals[:m]
                # Note: x_vals is truncated per-column, don't mutate the shared list
            series[c] = y_vals

        payload = {
            "ok": True,
            "message": None,
            "is_timeseries": bool(is_ts),
            "display": display,
            "length": len(x_vals),
            "columns": numeric_cols,
            "x": x_vals,
            "series": series,
        }
        return jsonify(payload), 200

    except Exception as e:
        app.logger.exception("full_history_json failed for %s: %s", locals().get('filename', '<unknown>'), e)
        return jsonify({"ok": False, "message": "An internal error occurred. Check server logs for details."}), 500


health = handle_health
api_ai_summary = handle_api_ai_summary
api_interactive_data = handle_api_interactive_data
full_history_json = handle_full_history_json

__all__ = [
    "handle_health",
    "handle_api_ai_summary",
    "handle_api_interactive_data",
    "handle_full_history_json",
    "health",
    "api_ai_summary",
    "api_interactive_data",
    "full_history_json",
]
