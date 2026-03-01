# ruff: noqa: F401,F403,F405
from __future__ import annotations

from data_analysis.runtime_app import *
from data_analysis.runtime_app import (
    _cap_anomalies_for_display,
    _forecast_with_fallback,
    _get_clean_ai_summary_from_cache,
    _infer_seasonal_period,
    _is_offline_html,
    _is_reliable_timeseries_index,
    _try_parse_numeric_series,
)

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "handle_download_cleaned_csv",
    "handle_download_ai_summary_html",
    "handle_download_static_plots_zip",
    "handle_download_full_report_html",
    "download_cleaned_csv",
    "download_ai_summary_html",
    "download_static_plots_zip",
    "download_full_report_html",
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


def handle_download_cleaned_csv(filename):
    _bind_runtime_globals()
    
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)
    df = DATAFRAME_CACHE.get(filename)
    if df is None:
        uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
        path = os.path.join(uploads_dir, filename)
        if not os.path.exists(path):
            return ("Not found", 404)
        
        df = get_dataframe_for(filename)
        if df is None:
            return ("Not found", 404)

    
    cleaned = df.copy()
    try:
        for col in cleaned.columns:
            ser = cleaned[col]
            if pd.api.types.is_numeric_dtype(ser):
                cleaned[col] = pd.to_numeric(ser, errors='coerce')
            else:
                coerced = _try_parse_numeric_series(ser)
                

                if coerced.notna().sum() >= pd.to_numeric(ser, errors='coerce').notna().sum():
                    cleaned[col] = coerced
        
        if isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned = cleaned.sort_index()
        
        cleaned = cleaned.dropna(axis=1, how='all')
    except Exception as clean_err:
        app.logger.warning("Cleaned CSV normalization fallback for %s: %s", filename, clean_err)

    csv = cleaned.to_csv(index=True)
    resp = make_response(csv)
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_cleaned.csv")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

def handle_download_ai_summary_html(filename):
    _bind_runtime_globals()
    
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    
    ai_html = _get_clean_ai_summary_from_cache(filename)
    if ai_html is None:
        
        df = get_dataframe_for(filename)
        if df is None:
            return ("Not found", 404)
        try:
            file_asset = AI_FILE_MAP.get(filename)
            ai_html = get_ai_summary_with_file(df, file_asset, extra_context=describe_for_ai(df))
        except Exception:
            reason = AI_STATUS.get('message') or ''
            detail = f"<p class=\"muted\"><small>Reason: {htmllib.escape(str(reason))}</small></p>" if reason else ""
            ai_html = f"<p>AI summary temporarily unavailable.</p>{detail}"
        if isinstance(ai_html, str) and not _is_offline_html(ai_html):
            AI_SUMMARY_CACHE[filename] = ai_html

    resp = make_response(ai_html if isinstance(ai_html, str) else str(ai_html))
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_ai_summary.html")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

def handle_download_static_plots_zip(filename):
    _bind_runtime_globals()
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    request_start = time.perf_counter()

    df = get_dataframe_for(filename)
    if df is None or df.empty:
        return ("Not found", 404)

    try:
        user_contam = float(request.args.get('contamination', app.config.get('DEFAULT_CONTAMINATION', 0.02)))
    except Exception:
        user_contam = float(app.config.get('DEFAULT_CONTAMINATION', 0.02))
    user_contam = max(0.001, min(0.2, user_contam))

    raw_pct = request.args.get('forecast_pct', '0.05')
    try:
        forecast_pct = float(raw_pct) if raw_pct not in (None, "") else 0.05
        forecast_pct = max(0.0, min(0.5, forecast_pct))
    except Exception:
        forecast_pct = 0.05

    def _steps_for_history_rows(history_rows: int) -> int:
        if forecast_pct <= 0 or history_rows <= 0:
            return 0
        pct_den = max(1e-9, 1.0 - float(forecast_pct))
        return max(1, int(math.floor(float(history_rows) * float(forecast_pct) / pct_den)))


    is_timeseries = _is_reliable_timeseries_index(df.index)
    numeric_df_cached = get_cached_numeric_df(filename, df)
    numeric_cols = {
        col for col in numeric_df_cached.columns
        if numeric_df_cached[col].notna().sum() >= 3
    }
    bio = io.BytesIO()
    
    total_rows = len(df)
    static_zip_max_forecast_steps = int(os.getenv("STATIC_ZIP_MAX_FORECAST_STEPS", "120"))
    max_forecast_steps_used = 0
    processed_numeric_cols = 0

    try:
        from data_analysis.analysis.plot import (
            _apply_sci_formatter as _plot_apply_sci_formatter,
            _format_stat_value as _plot_format_stat_value,
        )
    except Exception:
        def _plot_format_stat_value(v: float) -> str:
            try:
                value = float(v)
                mag = abs(value)
                if mag >= 1e15:
                    return f"{value:.3e}"
                if mag >= 1e12:
                    raw = f"{value / 1e12:.3f}"
                    return raw.rstrip("0").rstrip(".") + "T"
                if mag >= 1e9:
                    raw = f"{value / 1e9:.3f}"
                    return raw.rstrip("0").rstrip(".") + "B"
                if mag >= 1e6:
                    raw = f"{value / 1e6:.3f}"
                    return raw.rstrip("0").rstrip(".") + "M"
                if mag >= 1e3:
                    raw = f"{value / 1e3:.3f}"
                    return raw.rstrip("0").rstrip(".") + "K"
                raw = f"{value:.2f}"
                return raw.rstrip("0").rstrip(".")
            except Exception:
                return str(v)

        def _plot_apply_sci_formatter(ax: Any) -> None:
            return None
    
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Generate correlation heatmaps
        try:
            spearman_heatmap = get_cached_heatmap(filename, df, method='spearman')
            if spearman_heatmap:
                raw = base64.b64decode(spearman_heatmap.encode('utf-8'))
                zf.writestr("correlation_spearman.png", raw)
        except Exception as heatmap_err:
            app.logger.debug("ZIP spearman heatmap skipped for %s: %s", filename, heatmap_err)
        
        try:
            pearson_heatmap = get_cached_heatmap(filename, df, method='pearson')
            if pearson_heatmap:
                raw = base64.b64decode(pearson_heatmap.encode('utf-8'))
                zf.writestr("correlation_pearson.png", raw)
        except Exception as heatmap_err:
            app.logger.debug("ZIP pearson heatmap skipped for %s: %s", filename, heatmap_err)
        
        # Generate plots for each numeric column
        for col in df.columns:
            if col not in numeric_cols:
                continue

            try:
                s = numeric_df_cached[col].dropna()
            except Exception as series_err:
                app.logger.debug("ZIP numeric series extraction failed for %s/%s: %s", filename, col, series_err)
                s = pd.Series(dtype=float)
            if s.empty or len(s) < 3:
                continue

            processed_numeric_cols += 1
            
            # Detect anomalies for this column
            raw_an_idx, raw_an_score = get_cached_anomalies(filename, col, s, user_contam)
            try:
                max_pts = int(app.config.get('ANOMALY_MARKER_CAP', 20))
            except Exception:
                max_pts = 20
            an_idx = _cap_anomalies_for_display(raw_an_idx, raw_an_score, max_points=max_pts)
            an_score = raw_an_score[an_idx] if not raw_an_score.empty else raw_an_score
            
            # Trend plot with anomalies
            try:
                title = f"Trend for {col}"
                img_b64 = generate_forecast_plot(
                    s,
                    None,
                    title,
                    'Timestamp' if is_timeseries else 'Index',
                    col,
                    conf_int=None,
                    history_tail=None,
                    anomalies_idx=an_idx,
                    anomalies_score=an_score,
                    legend_y=-0.38,
                    xlabel_labelpad=10
                )
                raw = base64.b64decode(img_b64.encode('utf-8'))
                zf.writestr(f"{secure_filename(str(col))}_trend.png", raw)
            except Exception as trend_err:
                app.logger.debug("ZIP trend plot skipped for %s/%s: %s", filename, col, trend_err)
            
            # Distribution histogram - HIGH QUALITY: Larger figure and more bins
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                s_arr = np.asarray(pd.to_numeric(s, errors='coerce').dropna().to_numpy(dtype=float), dtype=float)
                ax.hist(s_arr, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', linewidth=0.5, label=col)
                ax.set_title(f"Distribution: {col}", fontsize=10)
                ax.set_xlabel(col, fontsize=9, labelpad=8)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add vertical stat lines on the histogram
                try:
                    stats_min = float(s.min())
                    stats_max = float(s.max())
                    stats_median = float(s.median())
                    stats_mean = float(s.mean())
                    stats_std = float(s.std())
                    
                    # Draw vertical lines for Avg and Median
                    ax.axvline(x=stats_mean, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {_plot_format_stat_value(stats_mean)}')
                    ax.axvline(x=stats_median, color='#9b59b6', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'Median: {_plot_format_stat_value(stats_median)}')
                    
                    # Avg/Median value tags - both on RIGHT side of their lines, staggered vertically to avoid overlap
                    ylim = ax.get_ylim()
                    xlim = ax.get_xlim()
                    x_offset = (xlim[1] - xlim[0]) * 0.01  # 1% offset to position text just right of line
                    y_pos = ylim[1] * 0.985
                    # Place Avg and Med at the same height on opposite sides
                    if stats_mean <= stats_median:
                        ax.text(stats_mean - x_offset, y_pos, f'Avg: {_plot_format_stat_value(stats_mean)}', va='top', ha='right', fontsize=8, color='#f39c12', fontweight='bold')
                        ax.text(stats_median + x_offset, y_pos, f'Med: {_plot_format_stat_value(stats_median)}', va='top', ha='left', fontsize=8, color='#9b59b6', fontweight='bold')
                    else:
                        ax.text(stats_median - x_offset, y_pos, f'Med: {_plot_format_stat_value(stats_median)}', va='top', ha='right', fontsize=8, color='#9b59b6', fontweight='bold')
                        ax.text(stats_mean + x_offset, y_pos, f'Avg: {_plot_format_stat_value(stats_mean)}', va='top', ha='left', fontsize=8, color='#f39c12', fontweight='bold')
                    
                    # Min/Max markers at bottom - BOTH tags ABOVE their symbols
                    y_lim = ax.get_ylim()
                    marker_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05
                    
                    min_color = '#ff3b30'
                    max_color = '#00BCD4'  # Cyan - works on both light and dark backgrounds
                    edge_color = '#0b1220'
                    ax.scatter([stats_min], [marker_y], color=min_color, s=30, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {_plot_format_stat_value(stats_min)}')
                    ax.scatter([stats_max], [marker_y], color=max_color, s=30, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {_plot_format_stat_value(stats_max)}')

                    # Match Detailed Analysis: min tag slightly left, max tag slightly right.
                    xlim = ax.get_xlim()
                    min_xytext, min_ha = (-3, 12), 'right'
                    max_xytext, max_ha = (3, 12), 'left'
                    if abs(stats_max - stats_min) <= (xlim[1] - xlim[0]) * 0.03:
                        min_xytext = (min_xytext[0], 12)
                        max_xytext = (max_xytext[0], 22)

                    ax.annotate(f'{_plot_format_stat_value(stats_min)}', (stats_min, marker_y), textcoords='offset points', xytext=min_xytext, ha=min_ha, fontsize=7, color=min_color, fontweight='bold', annotation_clip=False, clip_on=False)
                    ax.annotate(f'{_plot_format_stat_value(stats_max)}', (stats_max, marker_y), textcoords='offset points', xytext=max_xytext, ha=max_ha, fontsize=7, color=max_color, fontweight='bold', annotation_clip=False, clip_on=False)
                    
                    # Std in legend only
                    ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {_plot_format_stat_value(stats_std)}')
                    
                    # Legend on single line - just below x-axis title
                    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6, frameon=False, columnspacing=0.5)
                    fig.subplots_adjust(bottom=0.30)
                    
                    # Apply compact B/T axis labels for large values.
                    _plot_apply_sci_formatter(ax)
                    try:
                        import matplotlib.ticker as _mticker
                        xmin, xmax = ax.get_xlim()
                        if max(abs(xmin), abs(xmax)) >= 1e9:
                            xfmt = _mticker.FuncFormatter(lambda val, _pos: _plot_format_stat_value(float(val)))
                            ax.xaxis.set_major_formatter(xfmt)
                    except Exception:
                        pass
                except Exception as stats_err:
                    app.logger.debug("ZIP distribution stats overlay skipped for %s/%s: %s", filename, col, stats_err)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                zf.writestr(f"{secure_filename(str(col))}_distribution.png", buf.read())
            except Exception as dist_err:
                app.logger.debug("ZIP distribution plot skipped for %s/%s: %s", filename, col, dist_err)
            
            # STL decomposition (for timeseries with sufficient data)
            if is_timeseries and len(s) >= 28:
                try:
                    s_norm = normalize_timeseries(s)
                    sp = _infer_seasonal_period(s_norm.index)
                    if sp and isinstance(sp, int) and sp >= 2 and len(s_norm) >= sp * 2:
                        stl_img = get_cached_stl_plot(filename, col, s_norm, sp)
                        if stl_img:
                            raw = base64.b64decode(stl_img.encode('utf-8'))
                            zf.writestr(f"{secure_filename(str(col))}_stl.png", raw)
                except Exception as stl_err:
                    app.logger.debug("ZIP STL plot skipped for %s/%s: %s", filename, col, stl_err)
            
            # Forecast (for numeric series)
            col_forecast_steps = min(static_zip_max_forecast_steps, _steps_for_history_rows(len(s)))
            if len(s) >= 10 and col_forecast_steps > 0:
                try:
                    fc_mean, ci = _forecast_with_fallback(s, col_forecast_steps, filename=filename, col=col)
                    max_forecast_steps_used = max(max_forecast_steps_used, int(col_forecast_steps))

                    xlab = 'Timestamp' if is_timeseries else 'Index'

                    fc_b64 = generate_forecast_plot(
                        s,
                        fc_mean,
                        f"Forecast: {col} ({col_forecast_steps} steps)",
                        xlab,
                        col,
                        conf_int=ci,
                        history_tail=None,
                        anomalies_idx=an_idx,
                        anomalies_score=an_score,
                        legend_y=-0.38
                    )
                    raw = base64.b64decode(fc_b64.encode('utf-8'))
                    zf.writestr(f"{secure_filename(str(col))}_forecast.png", raw)
                except Exception as fc_err:
                    app.logger.debug("ZIP forecast plot skipped for %s/%s: %s", filename, col, fc_err)

        # Generate Categories bar charts for non-numeric columns (top 50)
        for col in df.columns:
            try:
                # Skip numeric columns already processed above
                if col in numeric_cols:
                    continue  # Skip - already processed as numeric
                
                # Process as categorical
                s_cat = df[col].astype(str).dropna()
                if len(s_cat) < 3:
                    continue
                
                # Generate Categories bar chart (top 50 for readability)
                all_counts = s_cat.value_counts()
                top_counts = all_counts.head(50)
                if len(top_counts) < 2:
                    continue
                
                # Calculate stats for annotation
                total_unique = len(all_counts)
                max_count = int(all_counts.max())
                min_count = int(all_counts.min())
                avg_count = float(all_counts.mean())
                med_count = float(all_counts.median())
                most_freq = str(all_counts.index[0])[:20]
                    
                fig, ax = plt.subplots(figsize=(12, 5))
                top_counts.plot(kind='bar', ax=ax, color='tab:green', alpha=0.7, edgecolor='black', label='Count')

                # Add value labels above each bar
                try:
                    if ax.containers and isinstance(ax.containers[0], BarContainer):
                        ax.bar_label(
                            ax.containers[0],
                            labels=[str(int(v)) for v in top_counts.values],
                            padding=2,
                            fontsize=7
                        )
                except Exception as bar_label_err:
                    app.logger.debug("ZIP categories bar labels skipped for %s/%s: %s", filename, col, bar_label_err)
                
                if len(all_counts) > 50:
                    ax.set_title(f"Categories: {col} (Top 50 of {total_unique})", fontsize=12)
                else:
                    ax.set_title(f"Categories: {col} ({total_unique} unique)", fontsize=12)
                
                ax.set_xlabel(col, fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                
                # Add horizontal avg/med lines for counts
                ax.axhline(y=avg_count, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {avg_count:.1f}')
                ax.axhline(y=med_count, color='#9b59b6', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Med: {med_count:.1f}')
                
                # Add text labels for avg/med lines next to the chart
                ylim = ax.get_ylim()
                y_range = ylim[1] - ylim[0]
                threshold = y_range * 0.03  # keep labels closer to their lines

                if abs(avg_count - med_count) < threshold:
                    offset = threshold * 0.4
                    if avg_count >= med_count:
                        avg_y = avg_count + offset
                        med_y = med_count - offset
                    else:
                        avg_y = avg_count - offset
                        med_y = med_count + offset
                else:
                    avg_y = avg_count
                    med_y = med_count

                ax.text(1.005, avg_y, f'Avg: {avg_count:.1f}', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8, color='#f39c12', fontweight='bold')
                ax.text(1.005, med_y, f'Med: {med_count:.1f}', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8, color='#9b59b6', fontweight='bold')
                
                # Get least frequent item name
                least_freq = str(all_counts.index[-1])[:20] if len(all_counts) > 0 else "N/A"
                
                # Add Most/Least as legend entries (invisible traces)
                ax.plot([], [], color='#27ae60', marker='s', linestyle='', markersize=8, label=f"Most: '{most_freq}' ({max_count})")
                ax.plot([], [], color='#e74c3c', marker='s', linestyle='', markersize=8, label=f"Least: '{least_freq}' ({min_count})")
                
                # Legend on top-right, vertical (line by line)
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                zf.writestr(f"{secure_filename(str(col))}_categories.png", buf.read())
            except Exception as cat_err:
                app.logger.debug("ZIP categories plot skipped for %s/%s: %s", filename, col, cat_err)


    bio.seek(0)
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_all_plots.zip")
    resp = make_response(bio.read())
    resp.headers['Content-Type'] = 'application/zip'
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    elapsed = time.perf_counter() - request_start
    app.logger.info(
        "Static plots ZIP ready file=%s rows=%d forecast_pct=%.3f max_forecast_steps=%d elapsed=%.2fs",
        filename,
        total_rows,
        forecast_pct,
        max_forecast_steps_used,
        elapsed,
    )
    return resp

def handle_download_full_report_html(filename):
    _bind_runtime_globals()
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ("Not found", 404)

    # Basic dataset info
    head_html = safe_df_head_html(df)
    desc_html = safe_df_description_html(df)
    buf = io.StringIO()
    try:
        df.info(buf=buf)
        info_str = buf.getvalue()
    except Exception:
        info_str = "Unable to render DataFrame info()."
    try:
        mv = df.isnull().sum()
        mvf = mv[mv > 0]
        missing_html = mvf.to_frame('missing_count').to_html() if not mvf.empty else ""
    except Exception as mv_err:
        app.logger.debug("HTML report missing-values rendering skipped for %s: %s", filename, mv_err)
        missing_html = ""

    # AI summary
    ai_html = _get_clean_ai_summary_from_cache(filename)
    if ai_html is None:
        try:
            file_asset = AI_FILE_MAP.get(filename)
            ai_html = get_ai_summary_with_file(df, file_asset, extra_context=describe_for_ai(df))
            AI_SUMMARY_CACHE[filename] = ai_html
        except Exception:
            ai_html = "<p>AI summary temporarily unavailable.</p>"

    # Correlation matrix (table and heatmaps)
    corr_html = ""
    corr_heatmap_spearman = None
    corr_heatmap_pearson = None
    try:
        df_num = get_cached_numeric_df(filename, df)
        sel = df_num.select_dtypes(include='number')
        if not sel.empty:
            nunique = sel.nunique(dropna=True)
            sel = sel.loc[:, nunique > 1]
        if sel.shape[1] >= 2:
            corr = sel.corr(method='spearman').round(3)
            corr_html = corr.to_html()
            # Generate heatmaps
            corr_heatmap_spearman = generate_correlation_heatmap(df, method='spearman', title='Spearman Correlation Heatmap')
            corr_heatmap_pearson = generate_correlation_heatmap(df, method='pearson', title='Pearson Correlation Heatmap')
    except Exception as corr_err:
        app.logger.debug("HTML report correlation section skipped for %s: %s", filename, corr_err)

    # Generate plots for each numeric column
    is_ts = _is_reliable_timeseries_index(df.index)
    try:
        user_contam = float(request.args.get('contamination', app.config.get('DEFAULT_CONTAMINATION', 0.02)))
    except Exception:
        user_contam = float(app.config.get('DEFAULT_CONTAMINATION', 0.02))
    user_contam = max(0.001, min(0.2, user_contam))

    raw_pct = request.args.get('forecast_pct', '0.05')
    try:
        forecast_pct = float(raw_pct) if raw_pct not in (None, "") else 0.05
    except Exception:
        forecast_pct = 0.05
    forecast_pct = max(0.0, min(0.5, forecast_pct))

    def _steps_for_history_rows(history_rows: int) -> int:
        if forecast_pct <= 0 or history_rows <= 0:
            return 0
        pct_den = max(1e-9, 1.0 - float(forecast_pct))
        return max(1, int(math.floor(float(history_rows) * float(forecast_pct) / pct_den)))

    distribution_sections = []
    stl_sections = []
    forecast_sections = []
    forecast_pct_label = f"{forecast_pct * 100:.0f}%"
    
    for col in df.columns:
        try:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) < 3:
                continue
        except Exception:
            continue
        
        # Ensure series has proper index from dataframe
        # This is critical for forecast to work correctly with DatetimeIndex
        if not isinstance(s.index, type(df.index)):
            try:
                # Re-align with df to maintain proper index
                s_temp = df[col].copy()
                s = pd.to_numeric(s_temp, errors='coerce').dropna()
            except Exception as realign_err:
                app.logger.debug("HTML report index realign skipped for %s/%s: %s", filename, col, realign_err)
        
        # Detect anomalies for forecast plots
        an_idx, an_score = get_cached_anomalies(filename, col, s, user_contam)
        
        # Generate distribution histogram for this column
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            s_arr = np.asarray(pd.to_numeric(s, errors='coerce').dropna().to_numpy(dtype=float), dtype=float)
            ax.hist(s_arr, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            dist_img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            distribution_sections.append(f'<figure><figcaption><strong>Distribution: {col}</strong></figcaption><img style="max-width:100%" src="data:image/png;base64,{dist_img}" /></figure>')
        except Exception:
            try:
                plt.close(fig)
            except Exception as fig_close_err:
                app.logger.debug("HTML report figure cleanup skipped for %s/%s: %s", filename, col, fig_close_err)
        
        # STL decomposition (for timeseries with sufficient data)
        if is_ts and len(s) >= 28:
            try:
                s_norm = normalize_timeseries(s)
                sp = _infer_seasonal_period(s_norm.index)
                if sp and isinstance(sp, int) and sp >= 2 and len(s_norm) >= sp * 2:
                    stl_img = generate_stl_plot(s_norm, f"STL Decomposition: {col}", seasonal_period=sp)
                    if stl_img:
                        stl_sections.append(f'<figure><figcaption><strong>STL Decomposition: {col}</strong></figcaption><img style="max-width:100%" src="data:image/png;base64,{stl_img}" /></figure>')
            except Exception as stl_err:
                app.logger.debug("HTML report STL section skipped for %s/%s: %s", filename, col, stl_err)
        
        # Forecast (for timeseries) - use selected forecast_pct as horizon share.
        col_forecast_steps = _steps_for_history_rows(len(s))
        if is_ts and len(s) >= 10 and col_forecast_steps > 0:
            fc_mean, ci = _forecast_with_fallback(s, col_forecast_steps, filename=filename, col=col)
            
            if fc_mean is not None and len(fc_mean) > 0:
                try:
                    app.logger.debug("Creating forecast plot for %s", col)
                    fc_b64 = generate_forecast_plot(
                        s,
                        fc_mean,
                        f"Forecast: {col} ({col_forecast_steps} steps = {forecast_pct_label})",
                        'Timestamp',
                        col,
                        conf_int=ci,
                        history_tail=None,
                        anomalies_idx=an_idx,
                        anomalies_score=an_score,
                    )
                    forecast_sections.append(
                        f'<figure><figcaption><strong>Forecast: {col}</strong> '
                        f'({col_forecast_steps} steps, {forecast_pct_label} of data)</figcaption>'
                        f'<img style="max-width:100%" src="data:image/png;base64,{fc_b64}" /></figure>'
                    )
                    app.logger.debug("Successfully created forecast plot for %s", col)
                except Exception:
                    app.logger.exception("Forecast plot error for %s", col)
            else:
                app.logger.debug("Forecast is None or empty for %s", col)

    # Build HTML report
    app.logger.debug("Report generation complete: distributions=%d, stl=%d, forecasts=%d",
                      len(distribution_sections), len(stl_sections), len(forecast_sections))
    if len(forecast_sections) == 0:
        app.logger.debug("No forecast sections were generated")
    
    display = request.args.get('display') or filename
    title = f"Analysis Report — {display}"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:#0f172a; background:#ffffff; margin: 0; padding: 20px; }}
  h1 {{ color:#0f172a; border-bottom: 2px solid #0ea5e9; padding-bottom: 8px; }}
  h2 {{ color:#0f172a; margin-top: 32px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
  h3 {{ color:#334155; margin-top: 24px; }}
  .muted {{ color:#475569; font-style: italic; }}
  figure {{ margin: 24px 0; page-break-inside: avoid; }}
  figcaption {{ margin: 0 0 8px 0; font-weight: 600; font-size: 0.95em; color: #0f172a; }}
  pre {{ white-space: pre-wrap; background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  td, th {{ border:1px solid #cbd5e1; padding:6px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 4px; }}
  .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(800px, 1fr)); gap: 24px; }}
  @media print {{
    @page {{ size: A4; margin: 14mm; }}
    figure {{ page-break-inside: avoid; }}
    h2 {{ page-break-after: avoid; }}
  }}
</style></head>
<body>
  <h1>{title}</h1>
  <p class="muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

  <h2>📊 Dataset Overview</h2>
  <h3>Preview</h3>{head_html}
  <h3>Statistical Description</h3>{desc_html}
  <h3>Dataset Info</h3><pre>{info_str}</pre>
  {"<h3>Missing Values</h3>" + missing_html if missing_html else ""}

  <h2>🤖 AI Analysis Summary</h2>
  {ai_html}

  {"<h2>📊 Value Distributions</h2><div class='section-grid'>" + ''.join(distribution_sections) + "</div>" if distribution_sections else ""}

  {"<h2>🔄 STL Decompositions</h2><div class='section-grid'>" + ''.join(stl_sections) + "</div>" if stl_sections else ""}

  {"<h2>🔮 Forecasts (with Anomaly Detection)</h2><div class='section-grid'>" + ''.join(forecast_sections) + "</div>" if forecast_sections else ""}

  <h2>📈 Correlation Matrix</h2>
  {corr_html if corr_html else '<p class="muted">Not enough numeric columns to compute correlation.</p>'}

  <h2>📊 Correlation Heatmaps</h2>
  {"<div class='section-grid'>" + 
   (f"<figure><figcaption><strong>Spearman Correlation</strong></figcaption><img style='max-width:100%' src='data:image/png;base64,{corr_heatmap_spearman}' /></figure>" if corr_heatmap_spearman else "") +
   (f"<figure><figcaption><strong>Pearson Correlation</strong></figcaption><img style='max-width:100%' src='data:image/png;base64,{corr_heatmap_pearson}' /></figure>" if corr_heatmap_pearson else "") +
   "</div>" if (corr_heatmap_spearman or corr_heatmap_pearson) else '<p class="muted">No correlation heatmaps available (requires 2+ numeric columns).</p>'}

</body></html>
"""
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_complete_report.html")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp


download_cleaned_csv = handle_download_cleaned_csv
download_ai_summary_html = handle_download_ai_summary_html
download_static_plots_zip = handle_download_static_plots_zip
download_full_report_html = handle_download_full_report_html

__all__ = [
    "handle_download_cleaned_csv",
    "handle_download_ai_summary_html",
    "handle_download_static_plots_zip",
    "handle_download_full_report_html",
    "download_cleaned_csv",
    "download_ai_summary_html",
    "download_static_plots_zip",
    "download_full_report_html",
]
