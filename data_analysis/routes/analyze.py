# ruff: noqa: F401,F403,F405
from __future__ import annotations

from typing import Any

from data_analysis.analysis.plot import (
    _add_static_distribution_overlays,
    _build_non_timeseries_tick_labels,
    _resolve_plot_display_axis,
    _resolve_static_tick_policy,
    _sample_numeric_axis_ticks,
)
from data_analysis.core.runtime_bind import bind_runtime_globals
from data_analysis.runtime_app import *

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "handle_analyze_file",
    "analyze_file",
    "__all__",
}



def _bind_runtime_globals():
    return bind_runtime_globals(globals(), _LOCAL_SYMBOLS)


def _extract_model_and_strip_comments(html: str | None) -> tuple[str | None, str]:
    """Return embedded model marker (if present) and HTML with comments removed."""
    raw = "" if html is None else str(html)
    model_name = None
    model_match = re.search(r'<!--\s*model:(.*?)\s*-->', raw)
    if model_match:
        model_name = model_match.group(1).strip()
    cleaned = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL)
    return model_name, cleaned


def _clean_model_name(model_name: str | None) -> str:
    model_str = str(model_name) if model_name else 'gemini-3-flash-preview'
    return model_str[7:] if model_str.startswith('models/') else model_str


def handle_analyze_file(filename):
    _bind_runtime_globals()
    filepath = os.path.join(app.config['UPLOADS_DIR'], filename)
    display_name = request.args.get('display') or request.form.get('display') or filename
    active_view = (request.args.get('view') or request.form.get('view') or 'overview').strip().lower()

    default_steps = int(app.config.get('DEFAULT_FORECAST_STEPS', 40))
    default_contam = float(app.config.get('DEFAULT_CONTAMINATION', 0.02))
    # New percentage-based horizon; fallback to numeric steps if pct absent.
    raw_pct = request.args.get('forecast_pct') or request.form.get('forecast_pct')
    pct = None
    try:
        if raw_pct not in (None, ""):
            pct = float(raw_pct)
            # Validate percentage - allow 0 for no-forecast mode, max 0.5
            if pct < 0:
                pct = 0
            elif pct > 0.5:
                app.logger.warning("forecast_pct out of range (%.4f), clamping to 0.5", pct)
                pct = 0.5
        else:
            # Default to 5% if not specified
            pct = 0.05
    except Exception:
        pct = 0.05  # Default to 5%
    raw_data_range = request.args.get("data_range") or request.form.get("data_range")
    user_steps = _get_arg_int("forecast_horizon", default_steps)  # legacy param support
    user_contam = _get_arg_float("contamination", default_contam)
    # Validate contamination is in valid range for IsolationForest
    try:
        if user_contam < 0.001 or user_contam > 0.2:
            app.logger.warning("contamination out of range (%.4f), clamping to [0.001, 0.2]", user_contam)
            user_contam = max(0.001, min(0.2, user_contam))
    except Exception:
        user_contam = default_contam

    if not os.path.exists(filepath) and filename not in DATAFRAME_CACHE:
        flash("The uploaded file is no longer available. Please re-upload it.")
        return redirect(url_for('pages.upload_file'))

    df = get_dataframe_for(filename)
    if df is None:
        flash('Could not read the uploaded file. Please re-upload.')
        return redirect(url_for('pages.upload_file'))

    # File deletion is handled by @after_this_request at end of analyze_file
    _cleanup_uploads_if_configured()

    # Use the uploaded asset (if available) for AI features within this request
    file_asset = AI_FILE_MAP.get(filename) if 'AI_FILE_MAP' in globals() else None

    user_question = None
    ai_answer = None
    if request.method == 'POST':
        user_question = (request.form.get('question') or '').strip()
        ai_answer_html = ""
        if user_question:
            ai_answer_html = get_ai_answer_with_file(df, user_question, file_asset=file_asset, filename=filename)
        ai_answer = ai_answer_html  

    analysis = {}
    forecast_plots = []
    anomalies_found = {}
    is_timeseries = _is_reliable_timeseries_index(df.index)
    used_cols = []

    # Reuse cached numeric coercion once per request to avoid repeated pd.to_numeric work.
    numeric_df_cached = get_cached_numeric_df(filename, df)

    # Correlation: only compute for views that actually need it (overview & correlation)
    corr_payload = None
    if active_view in ('overview', 'correlation'):
        corr_payload = CORRELATION_CACHE.get(filename)
        if corr_payload is None:
            try:
                num_df = numeric_df_cached
                if num_df is not None and not num_df.empty:
                    valid = [c for c in num_df.columns if num_df[c].notna().sum() >= 3]
                    num_df = num_df[valid]
                    keep = []
                    for c in num_df.columns:
                        s = num_df[c].dropna()
                        if s.empty:
                            continue
                        if float(s.max()) == float(s.min()):
                            continue
                        keep.append(c)
                    num_df = num_df[keep] if keep else num_df
                if num_df is not None and not num_df.empty and len(num_df.columns) >= 2:
                    cols = list(num_df.columns)
                    payload = {}
                    try:
                        spearman = num_df.corr(method='spearman')
                    except Exception:
                        spearman = None
                    try:
                        pearson = num_df.corr(method='pearson')
                    except Exception:
                        pearson = None
                    if spearman is not None:
                        spearman_m = spearman.reindex(index=cols, columns=cols).to_numpy(dtype=float)
                        payload["x"] = cols
                        payload["y"] = cols
                        payload["z"] = [[float(v) if np.isfinite(v) else None for v in row] for row in spearman_m]
                    if pearson is not None:
                        pearson_m = pearson.reindex(index=cols, columns=cols).to_numpy(dtype=float)
                        payload["pearson"] = {
                            "x": cols,
                            "y": cols,
                            "z": [[float(v) if np.isfinite(v) else None for v in row] for row in pearson_m]
                        }
                    corr_payload = payload if ("z" in payload or "pearson" in payload) else None
                    # Cache the computed correlation for faster subsequent view loads
                    if corr_payload:
                        CORRELATION_CACHE.set(filename, corr_payload)
                else:
                    corr_payload = None
            except Exception as e:
                app.logger.warning("Correlation computation failed: %s", e)
                corr_payload = None

    interactive = []

    # Apply percentage-based forecast horizon early (if provided) so all downstream forecast logic uses updated user_steps.
    total_rows = int(getattr(df, 'shape', (0,))[0]) if hasattr(df, 'shape') else 0
    data_range_ratio = 1.0
    data_range_rows = 0
    try:
        if raw_data_range not in (None, ""):
            dr = float(raw_data_range)
            if dr <= 0:
                data_range_ratio = 1.0
                data_range_rows = 0
            elif dr <= 1.0:
                data_range_ratio = dr
                if total_rows > 0:
                    rows = int(math.ceil(total_rows * dr))
                    data_range_rows = max(1, min(rows, total_rows))
                    if data_range_rows >= total_rows:
                        data_range_rows = 0
                        data_range_ratio = 1.0
            else:
                rows = int(dr)
                if rows <= 0:
                    data_range_ratio = 1.0
                    data_range_rows = 0
                elif total_rows > 0:
                    rows = min(rows, total_rows)
                    if rows >= total_rows:
                        data_range_ratio = 1.0
                        data_range_rows = 0
                    else:
                        data_range_rows = rows
                        data_range_ratio = rows / total_rows
                else:
                    data_range_ratio = 1.0
                    data_range_rows = 0
    except Exception:
        data_range_ratio = 1.0
        data_range_rows = 0
    
    # pct is always defined now (defaulting to 0.05).
    # Use pct as a VISUAL x-axis share target:
    # forecast_share = steps / (history + steps) ~= pct
    def _steps_for_history_rows(history_rows: int) -> int:
        if pct <= 0 or history_rows <= 0:
            return 0
        pct_den = max(1e-9, 1.0 - float(pct))
        return max(1, int(math.floor(float(history_rows) * float(pct) / pct_den)))

    if pct == 0:
        user_steps = 0
        effective_steps = 0
    elif pct > 0 and total_rows > 0:
        base_rows = total_rows
        if active_view == "forecast" and data_range_rows > 0:
            base_rows = data_range_rows
        user_steps = _steps_for_history_rows(base_rows)
        effective_steps = user_steps
    else:
        effective_steps = max(1, int(user_steps)) if user_steps else 1

    # Determine whether to build forecast/interactive content based on active_view.
    # IMPORTANT: To keep upload/overview fast, run heavy forecasting only in the explicit Forecast view.
    # Note: overview, correlation, and categories all use the fast path above.
    build_forecast = active_view == "forecast"
    build_interactive = active_view == "interactive"
    numeric_cols_all = list(numeric_df_cached.columns) if numeric_df_cached is not None else []
    inline_rows_limit = int(os.getenv("INTERACTIVE_INLINE_MAX_ROWS", "50000"))
    inline_cells_limit = int(os.getenv("INTERACTIVE_INLINE_MAX_CELLS", "1200000"))
    interactive_force_async = bool(
        build_interactive
        and (
            total_rows > inline_rows_limit
            or (len(numeric_cols_all) * max(total_rows, 1)) > inline_cells_limit
        )
    )
    
    # Initialize timing early (used by both fast path and full path)
    request_start = time.perf_counter()

    # PERFORMANCE OPTIMIZATION: For views that don't need the heavy column loop
    # (overview, correlation, categories), use the fast path.
    # - overview/correlation: skip plots, forecasts, interactive traces entirely
    # - categories: skip forecasts/anomalies; only compute category charts
    # Note: interactive & forecast still use the full column loop for server-side data.
    if active_view in ("overview", "correlation", "categories") or interactive_force_async:
        # Use cached info for fast response
        if interactive_force_async:
            cached_info = {
                'head': '',
                'description': '',
                'overview_table_html': '',
                'info': '',
                'missing_values': '',
            }
        else:
            cached_info = get_cached_df_info(filename, df)
        
        # Use cached numeric DF to avoid per-column pd.to_numeric calls
        _num_df = numeric_df_cached
        if build_interactive:
            used_cols = list(_num_df.columns) if _num_df is not None and not _num_df.empty else list(df.columns)
        else:
            used_cols = list(_num_df.columns)[:20] if _num_df is not None and not _num_df.empty else list(df.columns)[:20]
        
        # Skip describe_for_ai on the fast path — overview defers AI summary
        # to the AJAX endpoint, so building this expensive context is unnecessary.
        ai_context = ""
        
        ai_summary = _get_clean_ai_summary_from_cache(filename)
        if ai_summary is None and active_view == 'overview':
            # Defer AI summary - will load via AJAX
            ai_summary = ""
        elif ai_summary is None:
            ai_summary = ""
        
        analysis.update({
            'head': cached_info['head'],
            'description': cached_info['description'],
            'overview_table': cached_info.get('overview_table_html', ''),
            'info': cached_info['info'],
            'missing_values': cached_info['missing_values'],
            'plots': [],
            'forecast_plots': [],
            'forecast_plots_by_column': {},
            'anomalies': {},
            'ai_summary': ai_summary,
            'user_question': user_question,
            'ai_answer': ai_answer,
            'interactive': [],  # Interactive data loaded via AJAX from /api/interactive/
            'columns': used_cols,
            'corr': corr_payload,
            'controls': {
                'forecast_horizon': user_steps,
                'effective_steps': effective_steps,
                'forecast_pct': pct if pct is not None else None,
                'total_rows': total_rows,
                'contamination': user_contam,
                'data_range': data_range_ratio,
                'data_range_rows': data_range_rows
            }
        })
        
        # Categories view: build category charts (lightweight, no forecasts/anomalies)
        if active_view == 'categories':
            category_charts = {}
            numeric_non_na_counts: dict[Any, int] = {}
            if _num_df is not None and not _num_df.empty:
                numeric_non_na_counts = {c: int(_num_df[c].notna().sum()) for c in _num_df.columns}
            for col in df.columns:
                try:
                    if numeric_non_na_counts.get(col, 0) >= 3:
                        continue  # Skip - numeric column
                    s_cat = df[col].dropna()
                    if len(s_cat) < 3:
                        continue
                    chart_data = _build_category_plotly_chart(s_cat, col)
                    if chart_data is not None:
                        category_charts[col] = chart_data
                except Exception as cat_err:
                    app.logger.debug("Category chart build skipped for %s: %s", col, cat_err)
            analysis['category_charts'] = category_charts
        
        total_dt = time.perf_counter() - request_start
        app.logger.info(
            "Analyze FAST PATH done file=%s view=%s elapsed=%.2fs interactive_force_async=%s",
            filename, active_view, total_dt, interactive_force_async,
        )
        _log_cache_stats_if_needed("analyze-fast")
        
        summary_fallback_model = AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3-flash-preview'
        answer_fallback_model = CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3-flash-preview'
        ai_summary_model, ai_summary_clean = _extract_model_and_strip_comments(ai_summary)
        ai_answer_model, ai_answer_clean = _extract_model_and_strip_comments(ai_answer)

        analysis['ai_summary'] = ai_summary_clean
        analysis['ai_answer'] = ai_answer_clean

        ai_summary_model_display = _clean_model_name(ai_summary_model or summary_fallback_model)
        ai_answer_model_display = _clean_model_name(ai_answer_model or answer_fallback_model)
        ai_summary_is_valid = bool(
            isinstance(ai_summary_clean, str)
            and ai_summary_clean
            and not _is_offline_html(ai_summary_clean)
        )
        ai_answer_is_valid = bool(
            isinstance(ai_answer_clean, str)
            and ai_answer_clean
            and not _is_offline_html(ai_answer_clean)
        )
        _ai_is_valid = (
            ai_summary_is_valid
            or ai_answer_is_valid
        )
        
        return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name,
                               ai_summary_model_name=ai_summary_model_display,
                               ai_answer_model_name=ai_answer_model_display,
                               ai_summary_is_valid=ai_summary_is_valid,
                               ai_answer_is_valid=ai_answer_is_valid,
                               ai_is_valid=_ai_is_valid)

    # Per-request timing and budgets.
    # Forecast view intentionally has no budget/column cap so all charts are rendered.
    overview_budget_s = float(os.getenv("OVERVIEW_TIME_BUDGET_SEC", "6.0"))
    interactive_budget_s = float(os.getenv("INTERACTIVE_TIME_BUDGET_SEC", "90.0"))
    budget_s = interactive_budget_s if build_interactive else overview_budget_s
    overview_cols_max = int(os.getenv("OVERVIEW_FORECAST_COLS_MAX", "100"))
    interactive_cols_max = int(os.getenv("INTERACTIVE_FORECAST_COLS_MAX", "80"))
    cols_limit = interactive_cols_max if build_interactive else overview_cols_max
    forecast_force_full = bool(build_forecast)
    forecast_done = 0
    skip_forecasts = False

    for column in df.columns:
            if column not in numeric_df_cached.columns:
                continue

            try:
                full_series = numeric_df_cached[column].dropna()
            except Exception:
                full_series = pd.Series(dtype=float)
            series = full_series
            if build_forecast and data_range_rows > 0:
                series = full_series.tail(data_range_rows)
            if series.empty:
                continue
            used_cols.append(column)
            
            # Optimization: Only run anomaly detection if we are in a view that needs it
            # 'overview' doesn't show anomalies in the UI directly, only in AI context.
            # We skip it for overview on large datasets to speed up transitions.
            an_idx = pd.Index([])
            an_score = pd.Series([], dtype=float)
            if build_forecast or build_interactive or len(df.columns) < 10:
                 # Keep anomaly detection input consistent with PDF/ZIP exports:
                 # detect on the full numeric series, then render only visible anomalies per view.
                 raw_an_idx, raw_an_score = get_cached_anomalies(filename, column, full_series, user_contam)
                 if len(raw_an_idx):
                     try:
                         # Force cap at max_points early to ensure Interactive and Export/Plotly match
                         try:
                             max_points = int(app.config.get('ANOMALY_MARKER_CAP', 20))
                         except Exception:
                             max_points = 20
                         
                         an_idx = _cap_anomalies_for_display(raw_an_idx, raw_an_score, max_points=max_points)
                         an_score = raw_an_score[an_idx] if not raw_an_score.empty else raw_an_score
                         
                         anomalies_found[str(column)] = [str(i) for i in an_idx]
                     except Exception as cap_err:
                         app.logger.debug("Anomaly display cap skipped for %s: %s", column, cap_err)


            # Stop forecasting if time/column limits are exceeded
            # Generate forecasts for any numeric series with sufficient length
            if build_forecast and not skip_forecasts and len(series) >= 5:
                try:
                    t0 = time.perf_counter()
                    steps = _steps_for_history_rows(len(series))
                    has_future_forecast = steps > 0

                    if has_future_forecast:
                        app.logger.info("Forecast start col=%s steps=%s rows=%s pct=%s", column, steps, len(series), pct)

                    title_fc = f"Forecast for {column} (with anomalies)"
                    fc_mean_thin = None
                    conf_df_thin = None

                    if has_future_forecast:
                        # Unified pipeline - use cached helper for cross-view performance
                        fc_mean, conf_df = get_cached_column_forecast(filename, column, series, steps)
                        
                        # Do not thin history for display to ensure anomaly positions 
                        # perfectly match Interactive, PDF, and ZIP exports which use full series.
                        fc_mean_thin = fc_mean
                        conf_df_thin = conf_df
                    else:
                        title_fc = f"Forecast for {column} (history only)"

                    s_hist = series
                    # Calculate stats (min, max, mean, median, std) for consistency with distribution chart
                    fc_stats = _compute_basic_stats(series)

                    xlab, display_index = _resolve_plot_display_axis(
                        series,
                        source_df=df,
                        fallback_label='Timestamp' if _is_reliable_timeseries_index(series.index) else 'Index',
                    )
                    try:
                        # Pass anomaly indices to the forecast plot generation
                        img_fc = generate_forecast_plot(
                            s_hist,
                            fc_mean_thin,
                            title_fc,
                            xlab,
                            column,
                            conf_int=conf_df_thin,
                            history_tail=None,
                            anomalies_idx=an_idx,  # Add anomaly markers to forecast plot
                            anomalies_score=an_score,
                            stats=fc_stats,        # Pass stats for visualization
                            display_index=display_index,
                        )
                        forecast_plots.append({"img": img_fc, "title": title_fc, "column": column, "type": "forecast"})
                    except Exception as _e:
                        app.logger.warning("Could not render forecast image for %s: %s", column, _e)
                    try:
                        fc_points = len(fc_mean_thin) if isinstance(fc_mean_thin, pd.Series) else 0
                        app.logger.info("Forecast plot ready col=%s forecast_points=%d", column, fc_points)
                    except Exception as fc_log_err:
                        app.logger.debug("Forecast points logging skipped for %s: %s", column, fc_log_err)

                    dt = time.perf_counter() - t0
                    forecast_done += 1
                    app.logger.info("Forecast done col=%s took=%.2fs steps=%s points=%s", column, dt, steps, len(series))
                    # Enforce per-column forecast budget to skip ONLY repeated forecasting;
                    # distribution plots and remaining columns still run.
                    if (
                        (not forecast_force_full)
                        and has_future_forecast
                        and ((time.perf_counter() - request_start) > budget_s or forecast_done >= cols_limit)
                    ):
                        skip_forecasts = True  # Skip forecast computation for subsequent columns
                        app.logger.info(
                            "Forecast budget reached: elapsed=%.2fs limit=%.2fs cols=%d/%d - "
                            "distribution plots for remaining columns will continue.",
                            time.perf_counter() - request_start, budget_s, forecast_done, cols_limit
                        )
                except Exception as e:
                    app.logger.warning("Could not generate forecast for %s: %s", column, e)

            if build_forecast and not skip_forecasts and len(series) >= 5:
                try:
                    if _is_reliable_timeseries_index(series.index):
                        sp = _infer_seasonal_period(series.index)
                        if sp:
                            # Use cached STL plot for performance
                            stl_img = get_cached_stl_plot(filename, column, series, sp)
                            if stl_img:
                                forecast_plots.append({"img": stl_img, "title": f"STL decomposition for {column}", "column": column, "type": "stl"})
                except Exception as e:
                    app.logger.warning("STL plot failed for %s: %s", column, e)
                
                # Generate distribution histogram for this column
                try:
                    fig, ax = plt.subplots(figsize=(7.2, 4.2))
                    series_arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
                    ax.hist(series_arr, bins=min(50, max(10, len(series) // 10)), color='tab:blue', alpha=0.7, edgecolor='black', linewidth=0.5, label=column)
                    ax.set_title(f"Distribution: {column}", fontsize=10)
                    ax.set_xlabel(column, fontsize=9, labelpad=2)
                    ax.set_ylabel("Frequency", fontsize=9)
                    ax.grid(True, alpha=0.3)

                    try:
                        from matplotlib.ticker import MaxNLocator
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, min_n_ticks=6))
                    except Exception:
                        pass

                    finite_unique_values = np.unique(series_arr[np.isfinite(series_arr)]) if series_arr.size else np.asarray([], dtype=float)
                    tick_policy = _resolve_static_tick_policy(
                        finite_unique_values.tolist(),
                        chart_type='distribution',
                    )
                    tick_values, tick_labels = _sample_numeric_axis_ticks(
                        series_arr.tolist(),
                        max_tick_labels=int(tick_policy['max_tick_labels']),
                        min_spacing_ratio=float(tick_policy['min_spacing_ratio']),
                    )
                    if tick_values:
                        ax.set_xticks(tick_values)
                        ax.set_xticklabels(
                            tick_labels,
                            rotation=int(tick_policy['tick_angle']),
                            ha=str(tick_policy['tick_ha']),
                            fontsize=float(tick_policy['tick_fontsize']),
                        )

                    _add_static_distribution_overlays(
                        ax,
                        series_arr,
                        legend_fontsize=6,
                        legend_columns=6,
                        legend_y=-0.18,
                    )
                    fig.subplots_adjust(bottom=0.27, right=0.95, top=0.90)
                    
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    dist_img = base64.b64encode(buf.read()).decode('utf-8')
                    forecast_plots.append({"img": dist_img, "title": f"Distribution: {column}", "column": column, "type": "distribution"})
                except Exception as dist_e:
                    app.logger.warning("Distribution plot failed for %s: %s", column, dist_e)

            if build_interactive:
                # Always provide full series to interactive view.
                # Data-range filtering is handled client-side by the Data Range selector.
                s_tail = series

                # Use NUMERIC X-axis for proportional display (like in PDF)
                n_hist = len(s_tail)
                x_axis_title, display_history_index = _resolve_plot_display_axis(
                    s_tail,
                    source_df=df,
                    fallback_label='Timestamp' if _is_reliable_timeseries_index(s_tail.index) else 'Index',
                    prefer_first_column=True,
                )
                x_hist_numeric = list(range(n_hist))
                y_hist = [float(v) for v in s_tail.values]
                original_labels = [str(i) for i in display_history_index]
                fc_labels: list[str] = []
                n_fc_total = 0
                
                traces = [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "History",
                    "x": x_hist_numeric,
                    "y": y_hist,
                    "text": original_labels,
                    "hovertemplate": "%{text}<br>%{y}<extra></extra>",
                    "line": {"color": "rgb(31,119,180)", "width": 2},
                    "marker": {"size": 4, "opacity": 0.6}
                }]

                if not skip_forecasts and len(series) >= 5:
                    try:
                        sp = _infer_seasonal_period(series.index) if _is_reliable_timeseries_index(series.index) else None
                        if sp:
                            # Use cached STL plot - may already be computed in forecast view
                            stl_img = get_cached_stl_plot(filename, column, series, sp)
                            if stl_img:
                                forecast_plots.append({"img": stl_img, "title": f"STL decomposition for {column}", "column": column, "type": "stl"})
                    except Exception as stl_err:
                        app.logger.debug("Interactive STL plot skipped for %s: %s", column, stl_err)

            
            if build_interactive and len(an_idx):
                # an_idx is already capped uniformly above
                an_display = an_idx
                an_positions = _anomaly_positions_for_index(s_tail.index, an_display)
                an_y_values = []
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

                for pos in an_positions:
                    i = int(pos)
                    idx = s_tail.index[i]
                    val = float(s_tail.iloc[i])
                    an_y_values.append(val)
                    score_list = score_buckets.get(idx, [])
                    score_val = score_list.pop(0) if score_list else None
                    score_buckets[idx] = score_list

                    if score_val is None:
                        try:
                            score_key = int(i)
                            score_val = float(an_score.loc[score_key]) if score_key in an_score.index else None
                        except Exception:
                            score_val = None

                    if score_val is None or not math.isfinite(float(score_val)):
                        reason = "Outlier"
                    elif float(s_tail.iloc[i]) >= tail_median:
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
                        "y": an_y_values,
                        "text": an_labels,
                        "marker": {"color": "#ef4444", "size": 5, "opacity": 0.9},
                        "hovertemplate": "Anomaly<br>%{text}<br>%{y}<extra></extra>"
                    })
            
            # fc_x removed as it was unused
            fc_y = ci_lower = ci_upper = split_x = None
            # Generate forecasts and CI for interactive plots (removed is_timeseries requirement)
            if build_interactive and not skip_forecasts and len(series) >= 5 and pct > 0:
                try:
                    steps = _steps_for_history_rows(len(series))
                    if steps <= 0:
                        continue
                    # Use cached forecast - may already be computed in forecast view
                    fc_mean, conf_df = get_cached_column_forecast(filename, column, series, steps)
                    
                    # Use numeric X-axis continuing from history
                    if fc_mean is None or len(fc_mean) == 0:
                        raise ValueError("Empty forecast")
                    n_fc = len(fc_mean)
                    n_fc_total = n_fc
                    fc_x_numeric = list(range(n_hist, n_hist + n_fc))
                    fc_y = [float(v) for v in fc_mean.to_numpy(dtype=float)]
                    fc_labels = [str(i) for i in fc_mean.index]
                    split_x = n_hist - 0.5  # Numeric position for split line
                    
                    if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                        ci_lower = [float(v) for v in conf_df.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df.iloc[:, 1].values]

                    # Add interactive traces using numeric X-axis
                    if fc_x_numeric and ci_lower and ci_upper:
                        ci_group = f"ci-{re.sub(r'[^A-Za-z0-9_-]+', '', str(column))}"
                        traces.append({
                            "type": "scatter",
                            "mode": "lines",
                            "name": "95% CI",
                            "x": fc_x_numeric, "y": ci_lower,
                            "line": {"width": 0},
                            "hoverinfo": "skip",
                            "showlegend": True,
                            "legendgroup": ci_group
                        })
                        traces.append({
                            "type": "scatter",
                            "mode": "lines",
                            "name": "95% CI",
                            "x": fc_x_numeric, "y": ci_upper,
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(255,69,0,0.18)",
                            "hoverinfo": "skip",
                            "showlegend": False,
                            "legendgroup": ci_group,
                            "legendgrouptitle": {"text": "95% CI"}
                        })

                    if fc_x_numeric and fc_y:
                        # Connect to last history point
                        x_plot = [n_hist - 1] + list(fc_x_numeric)
                        y_plot = [float(series.iloc[-1])] + list(fc_y)
                        text_plot = [original_labels[-1]] + list(fc_labels)
                        traces.append({
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Forecast",
                            "x": x_plot, "y": y_plot,
                            "text": text_plot,
                            "hovertemplate": "%{text}<br>%{y}<extra></extra>",
                            "line": {"color": "orangered", "width": 3},
                            "marker": {"size": 3}
                        })
                except Exception as e:
                    app.logger.warning("Interactive forecast build failed for %s: %s", column, e)

            # Build layout and append to interactive list (moved outside the forecast condition)
            # This ensures interactive charts work even for non-timeseries or short series
            if build_interactive:
                total_x_extent = n_hist + (n_fc_total if split_x is not None else 0)
                x_axis_range = [0, total_x_extent]

                history_prefers_text_labels = False
                try:
                    history_prefers_text_labels = (
                        not _is_reliable_timeseries_index(display_history_index)
                        and not bool(pd.api.types.is_numeric_dtype(display_history_index))
                        and split_x is not None
                        and n_fc_total > 0
                        and bool(fc_labels)
                    )
                except Exception:
                    history_prefers_text_labels = False

                if history_prefers_text_labels:
                    tickvals, ticktext = _build_non_timeseries_tick_labels(
                        display_history_index,
                        fc_mean.index if 'fc_mean' in locals() and fc_mean is not None else None,
                        max_tick_labels=20,
                    )
                else:
                    tick_count = 20
                    extent_int = int(max(1, total_x_extent))
                    if extent_int > tick_count:
                        raw_ticks = np.linspace(0, extent_int - 1, num=tick_count, dtype=float).tolist()
                        tickvals = sorted({int(round(v)) for v in raw_ticks})
                        if tickvals[-1] != extent_int - 1:
                            tickvals.append(extent_int - 1)
                    else:
                        tickvals = list(range(extent_int))

                    def _idx_label(pos: int) -> str:
                        if pos < n_hist:
                            orig_pos = min(pos, len(s_tail.index) - 1)
                            return str(display_history_index[orig_pos])
                        if split_x is not None and n_fc_total > 0 and fc_labels:
                            fc_pos = min(max(int(pos - n_hist), 0), n_fc_total - 1)
                            rel = fc_pos / max(1, n_fc_total - 1)
                            label_idx = min(int(round(rel * (len(fc_labels) - 1))), len(fc_labels) - 1)
                            return fc_labels[label_idx]
                        return str(pos)

                    ticktext = [_idx_label(pos) for pos in tickvals]
                xaxis = {
                    "title": x_axis_title,
                    "showgrid": True,
                    "tickmode": "array",
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                    "range": x_axis_range,
                }
                layout = {
                    "title": {"text": f"{column} (interactive)", "x": 0.02},
                    "xaxis": xaxis,
                    "yaxis": {"title": column, "showgrid": True},
                    "shapes": [] if not split_x else [{
                        "type": "line", "xref": "x", "yref": "paper",
                        "x0": split_x, "x1": split_x, "y0": 0, "y1": 1,
                        "line": {"color": "gray", "width": 1, "dash": "dot"}
                    }],
                    "legend": {"orientation": "h", "groupclick": "togglegroup"},
                    "margin": {"l": 40, "r": 10, "t": 40, "b": 40}
                }

                dist = {"name": column, "values": [float(v) for v in series.dropna().values]}
                
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

    # If in forecast view and no forecast plots were generated (due to errors or strict budgets),
    # render a fallback forecast for the first eligible numeric column to avoid an empty page.
    try:
        if build_forecast and not forecast_plots:
            for column in df.columns:
                if column not in numeric_df_cached.columns:
                    continue
                full_series = numeric_df_cached[column].dropna()
                series = full_series
                if data_range_rows > 0:
                    series = full_series.tail(data_range_rows)
                if len(series) >= 5:
                    # Detect anomalies for fallback forecast
                    an_idx_fb, _ = get_cached_anomalies(filename, column, full_series, user_contam)
                    steps = _steps_for_history_rows(len(series))
                    fc_mean = None
                    conf_df = None
                    title_fc = f"Forecast for {column}"
                    if steps > 0:
                        fc_mean, conf_df = get_cached_column_forecast(filename, column, series, steps)
                    else:
                        title_fc = f"Forecast for {column} (history only)"
                    s_hist = series
                    fallback_xlab, fallback_display_index = _resolve_plot_display_axis(
                        series,
                        source_df=df,
                        fallback_label='Timestamp' if _is_reliable_timeseries_index(series.index) else 'Index',
                    )
                    forecast_plots.append({
                        "img": generate_forecast_plot(
                            s_hist,
                            fc_mean,
                            title_fc,
                            fallback_xlab,
                            column,
                            conf_int=conf_df,
                            history_tail=None,
                            anomalies_idx=an_idx_fb,
                            display_index=fallback_display_index,
                        ),
                        "title": title_fc
                    })
                    break
    except Exception as e:
        app.logger.warning("Fallback static forecast failed: %s", e)

    # Use cached DataFrame info to avoid recomputation on view switches
    cached_info = get_cached_df_info(filename, df)
    info_string = cached_info['info']
    missing_values_html = cached_info['missing_values']


    used_cols = used_cols or list(df.columns)
    ai_context = build_ai_context(
        df=df,
        anomalies_found=anomalies_found,
        corr_payload=corr_payload,
        used_cols=used_cols,
        is_timeseries=is_timeseries,
        forecast_horizon=user_steps,
        contamination=user_contam
    )

    ai_summary = _get_clean_ai_summary_from_cache(filename)
    if ai_summary is None:
        # Defer AI summary on all GET requests and load via AJAX to keep navigation latency low.
        # Keep blocking behavior for POST requests (e.g., Q&A flows that need immediate context).
        defer_ai_on_get = (request.method == 'GET')
        
        if not defer_ai_on_get and ensure_ai_ready():
            try:
                generated = get_ai_summary_with_file(df, file_asset, extra_context=ai_context)
                ai_summary = generated
                if isinstance(generated, str) and not _is_offline_html(generated):
                    AI_SUMMARY_CACHE[filename] = generated
            except Exception as _e:
                try:
                    reason = _sanitize_error_message(getattr(_e, 'message', None) or str(_e)) or (AI_STATUS.get('message') or '')
                except Exception:
                    reason = ''
                detail = f"<p class=\"muted\"><small>Reason: {htmllib.escape(str(reason))}</small></p>" if reason else ""
                ai_summary = f"<p>AI summary temporarily unavailable.</p>{detail}"
        elif defer_ai_on_get:
            # Leave ai_summary empty/None - frontend will load it async via AJAX
            ai_summary = ""
        else:
            # ensure_ai_ready() failed - report the actual reason
            reason = AI_STATUS.get('message') or ("AI disabled or not configured." if not AI_ENABLED else "")
            detail = f"<p class=\"muted\"><small>Reason: {htmllib.escape(str(reason))}</small></p>" if reason else ""
            ai_summary = f"<p>AI summary temporarily unavailable.</p>{detail}"

    
    # Log forecast_plots length and per-column stats
    try:
        app.logger.info("Static forecast_plots count: %d", len(forecast_plots))
        for fp in forecast_plots:
            if isinstance(fp, dict) and 'title' in fp:
                app.logger.info("Forecast plot: %s", fp['title'])
    except Exception as fp_log_err:
        app.logger.debug("Forecast plot logging skipped: %s", fp_log_err)

    # Organize forecast_plots by column for grouped display
    forecast_plots_by_column = {}
    type_order = {'forecast': 0, 'distribution': 1, 'stl': 2}
    for fp in forecast_plots:
        if isinstance(fp, dict):
            col = fp.get('column', 'Other')
            if col not in forecast_plots_by_column:
                forecast_plots_by_column[col] = []
            forecast_plots_by_column[col].append(fp)
    # Sort plots within each column by type order
    for col in forecast_plots_by_column:
        forecast_plots_by_column[col].sort(key=lambda x: type_order.get(x.get('type', ''), 99))

    analysis.update({
        'head': cached_info['head'],
        'description': cached_info['description'],
        'overview_table': cached_info.get('overview_table_html', ''),
        'info': info_string,
        'missing_values': missing_values_html,
        'plots': [],
        'forecast_plots': _ensure_plot_dicts(forecast_plots) if build_forecast else [],
        'forecast_plots_by_column': forecast_plots_by_column if build_forecast else {},
        'anomalies': anomalies_found,
        'ai_summary': ai_summary,
        'user_question': user_question,
        'ai_answer': ai_answer,
        'interactive': interactive if build_interactive else [],
        'columns': used_cols,
        'corr': corr_payload,
        'controls': {
            'forecast_horizon': user_steps,
            'effective_steps': effective_steps,
            'forecast_pct': pct if pct is not None else None,
            'total_rows': total_rows,
            'contamination': user_contam,
            'data_range': data_range_ratio,
            'data_range_rows': data_range_rows
        }
    })
    

    
    if (
        app.config.get('DELETE_UPLOADED_AFTER_PROCESSING', False)
        and HASHED_UPLOAD_RE.match(os.path.basename(filepath))
        and os.path.exists(filepath)
    ):
        @after_this_request
        def _delete_hashed_upload(response):
            try:
                success, error_msg = _safe_delete(filepath)
                if success:
                    app.logger.info("Deferred delete of %s done", filepath)
                else:
                    app.logger.warning("Deferred delete of %s failed: %s", filepath, error_msg or "unknown error")
            except Exception as e:
                app.logger.warning("Deferred delete callback failed for %s: %s", filepath, e)
            return response

    total_dt = time.perf_counter() - request_start
    app.logger.info("Analyze done file=%s view=%s elapsed=%.2fs cols=%d", filename, active_view, total_dt, len(df.columns))
    
    # Get current AI model name for attribution display
    # Priority: embedded model in AI summary > actual model used > configured default > generic fallback
    ai_summary = analysis.get('ai_summary', '')
    ai_answer = analysis.get('ai_answer', '')

    ai_summary_model, ai_summary = _extract_model_and_strip_comments(ai_summary)
    ai_answer_model, ai_answer = _extract_model_and_strip_comments(ai_answer)
    analysis['ai_summary'] = ai_summary
    analysis['ai_answer'] = ai_answer
    
    summary_fallback_model = AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3-flash-preview'
    answer_fallback_model = CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3-flash-preview'
    ai_summary_model = ai_summary_model or summary_fallback_model
    ai_answer_model = ai_answer_model or answer_fallback_model
    
    ai_summary_model_display = _clean_model_name(ai_summary_model)
    ai_answer_model_display = _clean_model_name(ai_answer_model)
    ai_summary_is_valid = bool(
        isinstance(ai_summary, str)
        and ai_summary
        and not _is_offline_html(ai_summary)
    )
    ai_answer_is_valid = bool(
        isinstance(ai_answer, str)
        and ai_answer
        and not _is_offline_html(ai_answer)
    )
    
    # Check if AI summary is a valid AI response (not offline/error)
    ai_is_valid = (
        ai_summary_is_valid
        or ai_answer_is_valid
    )
    
    app.logger.debug("AI model attribution: display_sum=%s, display_ans=%s", ai_summary_model_display, ai_answer_model_display)
    
    _log_cache_stats_if_needed("analyze")
    return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name, 
                           ai_summary_model_name=ai_summary_model_display,
                           ai_answer_model_name=ai_answer_model_display,
                           ai_summary_is_valid=ai_summary_is_valid,
                           ai_answer_is_valid=ai_answer_is_valid,
                           ai_is_valid=ai_is_valid)

analyze_file = handle_analyze_file

__all__ = [
    "handle_analyze_file",
    "analyze_file",
]
