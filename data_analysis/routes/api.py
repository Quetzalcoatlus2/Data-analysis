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
        model_display = m_name or CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3.0-flash'
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
            
            model_display = m_name or CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3.0-flash'
            
            display_html = re.sub(r'<!--.*?-->', '', str(summary), flags=re.DOTALL) if isinstance(summary, str) else summary
            
            return jsonify({"ok": True, "html": display_html, "cached": False, "model": model_display})
        else:
            fallback = offline_answer(df, "summary", error="AI not ready", filename=filename)
            return jsonify({"ok": True, "html": fallback, "cached": False})
    except Exception as e:
        app.logger.warning("API AI summary failed: %s", e)
        fallback = offline_answer(df, "summary", error=e, filename=filename)
        return jsonify({"ok": True, "html": fallback, "cached": False})

def handle_api_interactive_data(filename):
    _bind_runtime_globals()
    """Return interactive chart data via AJAX for faster page load.
    
    This endpoint returns pre-computed or freshly computed interactive chart traces
    as JSON, allowing the page to load quickly and fetch chart data asynchronously.
    """
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "error": "Invalid filename"}), 400
    request_start = time.perf_counter()

    # Get parameters - default to 5%
    raw_pct = request.args.get('forecast_pct', '0.05')
    try:
        user_contam = float(request.args.get('contamination', '0.02'))
    except Exception:
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
    except Exception:
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
    
    total_rows = len(df)
    # 0% means no forecast
    effective_steps = 0 if pct == 0 else max(2, int(math.ceil(total_rows * pct)))
    is_timeseries = _is_reliable_timeseries_index(df.index)
    numeric_df_cached = get_cached_numeric_df(filename, df)
    
    interactive = []
    cols_processed = 0
    max_cols = 8  # Limit columns for performance
    
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
        x_hist_numeric = list(range(n_hist))
        y_hist = [_safe_number(v) for v in s_tail.values]
        
        # Store original labels for hover text
        original_labels = [str(i) for i in s_tail.index]
        
        traces = [{
            "type": "scatter",
            "mode": "lines+markers",
            "name": "History",
            "x": x_hist_numeric,
            "y": y_hist,
            "text": original_labels,  # Show original labels on hover
            "hovertemplate": "%{text}<br>%{y}<extra></extra>",
            "line": {"color": "rgb(31,119,180)", "width": 2},
            "marker": {"size": 4, "opacity": 0.6}
        }]
        
        # Add anomaly markers using numeric positions
        if len(an_idx):
            an_display = _cap_anomalies_for_display(an_idx, an_score)
            an_positions = _anomaly_positions_for_index(s_tail.index, an_display)
            an_values = []
            an_labels = []
            tail_vals = np.asarray(s_tail.to_numpy(dtype=float), dtype=float)
            tail_median = float(np.nanmedian(tail_vals)) if tail_vals.size else 0.0

            score_buckets: dict[Any, list[float]] = {}
            try:
                for idx_val, score_val in an_score.items():
                    if pd.notna(score_val):
                        score_buckets.setdefault(idx_val, []).append(float(score_val))
            except Exception:
                score_buckets = {}

            for i in an_positions:
                idx = s_tail.index[int(i)]
                an_values.append(_safe_number(s_tail.iloc[int(i)]))
                score_list = score_buckets.get(idx, [])
                score_val = score_list.pop(0) if score_list else None
                score_buckets[idx] = score_list

                if score_val is None:
                    try:
                        score_val = float(an_score.loc[int(i)]) if int(i) in an_score.index else None
                    except Exception:
                        score_val = None

                if score_val is None or not math.isfinite(float(score_val)):
                    reason = "Outlier"
                elif float(s_tail.iloc[int(i)]) >= tail_median:
                    reason = f"High outlier (IF score={float(score_val):.3f})"
                else:
                    reason = f"Low outlier (IF score={float(score_val):.3f})"
                an_labels.append(f"{idx} | {reason}")
            
            if an_positions:
                traces.append({
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Anomaly",
                    "x": an_positions,
                    "y": an_values,
                    "text": an_labels,
                    "hovertemplate": "Anomaly<br>%{text}<br>%{y}<extra></extra>",
                    "marker": {"color": "#ef4444", "size": 5, "opacity": 0.9}
                })
        
        # Add forecast using numeric positions continuing from history
        split_x = None
        if effective_steps > 0 and len(series) >= 10:
            try:
                # Use cached forecast - reuses computation from analyze_file if available
                fc_mean, conf_df = get_cached_column_forecast(filename, column, series, effective_steps)
                if fc_mean is None:
                    raise ValueError("No forecast generated")
                split_x = n_hist - 0.5  # Split line between last history and first forecast
                
                # Forecast uses indices n_hist, n_hist+1, n_hist+2...
                n_fc = len(fc_mean)
                fc_x_numeric = list(range(n_hist, n_hist + n_fc))
                fc_y = [_safe_number(v) for v in fc_mean.values]
                fc_labels = [str(i) for i in fc_mean.index]
                
                if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                    ci_lower = [_safe_number(v) for v in conf_df.iloc[:, 0].values]
                    ci_upper = [_safe_number(v) for v in conf_df.iloc[:, 1].values]
                    
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
        
        all_x_num = x_hist_numeric + (fc_x_numeric if split_x else [])
        all_x_text = original_labels + (fc_labels if split_x else [])
        
        if len(all_x_num) > 20:
            step = len(all_x_num) // 20
            tv = all_x_num[::step]
            tt = all_x_text[::step]
            if tv[-1] != all_x_num[-1]:
                tv.append(all_x_num[-1])
                tt.append(all_x_text[-1])
        else:
            tv = all_x_num
            tt = all_x_text

        # Build layout
        layout = {
            "title": {"text": f"{column} (interactive)", "x": 0.02},
            "xaxis": {
                "title": str(df.index.name) if df.index.name else ("Timestamp" if is_timeseries else "Index"), 
                "showgrid": True,
                "tickmode": "array",
                "tickvals": tv,
                "ticktext": tt
            },
            "yaxis": {"title": column, "showgrid": True},
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
        
        dist = {
            "name": column,
            "values": [v for v in (_safe_number(x) for x in series.dropna().values) if v is not None]
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
        except Exception:
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
            except Exception:
                
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
            except Exception:
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
                except Exception:
                    continue
            numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]

        
        x_vals = x_all[::step] if step > 1 else x_all
        series = {}
        for c in numeric_cols:
            try:
                y_all = num_df[c].astype(float).tolist()
            except Exception:
                
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
