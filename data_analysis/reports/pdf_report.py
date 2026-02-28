# ruff: noqa: F401,F403,F405
from __future__ import annotations

import base64
import io
import re
from html.parser import HTMLParser

import emoji
from fpdf import FPDF

from data_analysis.runtime_app import *

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "PDFReport",
    "handle_download_full_report_pdf",
    "download_full_report_pdf",
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


_bind_runtime_globals()


def _select_pdf_font(available_fonts: set[str]) -> str:
    """Pick stable body font from loaded TTF families."""
    normalized = {str(name).lower(): str(name) for name in available_fonts}

    for cand in ("segoeui", "arial"):
        if cand in normalized:
            return normalized[cand]

    return "helvetica"
class PDFReport(FPDF):  # type: ignore[no-redef]
    def __init__(self, title_str, display_name):
        super().__init__()
        self.report_title = title_str
        self.display_name = display_name
    
    def header(self):
        # Only show header on first page
        if self.page_no() == 0:
            return
        if self.page_no() != 1:
            return
        try:
            # Determine strictness based on available fonts
            font_keys = {str(k).lower() for k in self.fonts.keys()}
            font_family = "Arial" if "arial" in font_keys else "helvetica"
            
            # Defensive: Always sanitize for now to rule out encoding issues causing crashes
            # We can relax this later once we confirm stability.
            # Even with Arial, fpdf2 might have issues with some chars if not fully compatible.
            safe_title = self.report_title.encode('latin-1', 'replace').decode('latin-1')
            safe_display = f"Dataset: {self.display_name}".encode('latin-1', 'replace').decode('latin-1')
            
            # Override if we are confident (testing phase: use safe versions first)
            # If Arial is prevalent, we can try to use raw strings in a sub-try
            if font_family == "Arial":
                 # Use raw strings but catch error
                 try:
                     self.set_font("Arial", 'B', 15)
                     self.cell(0, 10, self.report_title, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
                     self.set_font("Arial", 'I', 10)
                     self.cell(0, 5, f"Dataset: {self.display_name}", border=False, new_x="LMARGIN", new_y="NEXT", align='C')
                 except Exception as e:
                     app.logger.error(f"Header Arial rendering failed: {e}")
                     # Fallback to safe/helvetica logic locally
                     self.set_font("helvetica", 'B', 15)
                     self.cell(0, 10, safe_title, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
                     self.set_font("helvetica", 'I', 10)
                     self.cell(0, 5, safe_display, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
            else:
                self.set_font("helvetica", 'B', 15)
                self.cell(0, 10, safe_title, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
                
                self.set_font("helvetica", 'I', 10)
                self.cell(0, 5, safe_display, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
            
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(10)
        except Exception as e:
            app.logger.error(f"CRITICAL HEADER FAILURE: {e}")
            # Do nothing more to avoid crashing add_page context
            pass

    def footer(self):
        if self.page_no() == 0:
            return
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%H:%M %d.%m.%Y")}', align='R')


def handle_download_full_report_pdf(filename):
    _bind_runtime_globals()
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ("Not found", 404)

    display = request.args.get('display') or filename
    
    # Get forecast percentage from request (default 5%)
    try:
        forecast_pct = float(request.args.get('forecast_pct', 0.05))
    except (TypeError, ValueError):
        forecast_pct = 0.05

    try:
        user_contam = float(request.args.get('contamination', app.config.get('DEFAULT_CONTAMINATION', 0.02)))
    except Exception:
        user_contam = float(app.config.get('DEFAULT_CONTAMINATION', 0.02))
    user_contam = max(0.001, min(0.2, user_contam))
    
    # Ensure AI summary is generated if not already cached
    if _get_clean_ai_summary_from_cache(filename) is None:
        try:
            if ensure_ai_ready():
                # Build context for AI
                try:
                    ai_context = describe_for_ai(df, filename=filename)
                except Exception:
                    ai_context = ""
                
                # Get file asset if available
                file_asset = AI_FILE_MAP.get(filename)
                
                # Generate AI summary
                generated = get_ai_summary_with_file(df, file_asset, extra_context=ai_context)
                if isinstance(generated, str) and generated.strip():
                    AI_SUMMARY_CACHE[filename] = generated
                    app.logger.info(f"Generated AI summary for PDF: {filename}")
        except Exception as e:
            app.logger.warning(f"Could not generate AI summary for PDF: {e}")
    
    try:
        app.logger.info(f"Starting PDF generation for {filename}, display={display}")
        pdf = PDFReport("Data Analysis Report", display)
        pdf.alias_nb_pages()
        
        # Load font candidates before add_page because header() depends on font availability.
        loaded_font_families: set[str] = set()
        
        font_candidates = [
            ("SegoeUI", "C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\segoeuii.ttf", "C:\\Windows\\Fonts\\segoeuiz.ttf"),
            ("Arial", "C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\ariali.ttf", "C:\\Windows\\Fonts\\arialbi.ttf"),
        ]
        
        for family, reg, bd, it, bi in font_candidates:
            try:
                if not os.path.exists(reg):
                    continue
                pdf.add_font(family, "", reg)
                if bd and os.path.exists(bd):
                    pdf.add_font(family, "B", bd)
                if it and os.path.exists(it):
                    pdf.add_font(family, "I", it)
                if bi and os.path.exists(bi):
                    pdf.add_font(family, "BI", bi)
                
                if family.lower() in {str(k).lower() for k in pdf.fonts.keys()}:
                    loaded_font_families.add(family)
                    app.logger.info("Loaded PDF font: %s", family)
            except Exception as e:
                app.logger.debug("PDF font candidate failed %s: %s", family, e)

        body_font = _select_pdf_font(loaded_font_families)
        body_font_is_unicode = body_font.lower() != "helvetica"
        
        app.logger.info(
            "PDF font context body=%s unicode_body=%s",
            body_font,
            body_font_is_unicode,
        )

        # CRITICAL: Must add first page IMMEDIATELY after initialization
        # This prevents "No page open" errors from fpdf2
        app.logger.info("Adding initial page...")
        pdf.add_page()
        app.logger.info(f"Initial page added. Page No: {pdf.page_no()}")

        def ensure_page():
            if pdf.page_no() == 0:
                pdf.add_page()

        app.logger.info("Continuing PDF generation...")
        ensure_page()
        app.logger.info(f"Page added. Page No: {pdf.page_no()}")

        default_font = body_font

        def _safe_pdf_text(value: str) -> str:
            out = str(value)
            return out if body_font_is_unicode else out.encode('latin-1', 'replace').decode('latin-1')

        pdf.set_font(default_font, size=12)
        def _restore_emoji_placeholders(text: str) -> str:
            """Strip emojis entirely."""
            return emoji.replace_emoji(text, replace='')

        # Helper for adding sections
        def add_section_title(title, new_page=True):
            ensure_page()
            if new_page:
                pdf.add_page()
            else:
                remaining = float(pdf.h - pdf.b_margin - pdf.get_y())
                # Require generous space (60 units) to avoid title ending up orphaned at page bottom
                if remaining < 60.0:
                    pdf.add_page()
                pdf.ln(5)

            pdf.set_font(default_font, style="B", size=13)
            pdf.set_text_color(0, 0, 0)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)
            pdf.set_font(default_font, size=10)

        def add_text_block(text, courier=False, is_html=False):
            font_family = default_font

            # Ensure we have a page
            if pdf.page_no() == 0:
                app.logger.warning("No page open, adding one.")
                ensure_page()
            
            if is_html:
                pdf.set_font(font_family, size=10)
                
                html_text = str(text)
                html_text = _restore_emoji_placeholders(html_text)
                
                # Strip HTML comments (e.g. model markers)
                html_text = re.sub(r'<!--.*?-->', '', html_text, flags=re.DOTALL)
                
                try:
                    pdf.write_html(html_text)
                except Exception as e:
                    app.logger.error(f"Native HTMLParser PDF rendering failed: {e}")
                    # Ultimate fallback
                    fallback_text = re.sub(r'<[^>]+>', ' ', html_text)
                    fallback_text = re.sub(r'\s+', ' ', fallback_text).strip()
                    if body_font_is_unicode:
                        pdf.multi_cell(0, 5, fallback_text)
                    else:
                        pdf.multi_cell(0, 5, _safe_pdf_text(fallback_text))
                pdf.ln(5)
                return

            if courier:
                pdf.set_font("Courier", size=9)
            else:
                pdf.set_font(default_font, size=10)
            
            text = _restore_emoji_placeholders(text)
            
            if body_font_is_unicode:
                pdf.multi_cell(0, 5, text)
            else:
                pdf.multi_cell(0, 5, _safe_pdf_text(text))
            pdf.ln(5)
            
            pdf.set_font(font_family, size=10)

        def add_df_table(df_table, title=None, new_page=True):
            """Renders a pandas DataFrame as a table in the PDF.
               If new_page=True (default), ALWAYS starts on a fresh page.
            """
            ensure_page()
            # ALWAYS start on fresh page to prevent mid-page tables
            if new_page:
                pdf.add_page()
            
            # Use fpdf2's built-in table context if available, otherwise manual
            try:
                # Basic data prep: keep full columns/values and convert to string
                # Apply _display_df_with_index to show the index as a proper named column
                # (same as the overview page does)
                try:
                    df_display = _display_df_with_index(df_table)
                except Exception:
                    df_display = df_table.copy()
                
                # Convert all to string
                df_display = df_display.astype(str)
                
                # Use actual column names (no generic 'Index' header)
                full_headers = [str(c) for c in df_display.columns]

                row_count = len(df_display)
                total_cols = len(full_headers)

                # Adaptive readability settings.
                if total_cols >= 18:
                    font_size = 6.0
                    line_h = 3.8
                    cols_per_chunk = 8
                elif total_cols >= 12:
                    font_size = 6.6
                    line_h = 4.1
                    cols_per_chunk = 10
                else:
                    font_size = 7.5
                    line_h = 4.6
                    cols_per_chunk = 12

                if row_count >= 20:
                    font_size = max(5.6, font_size - 0.7)
                    line_h = max(3.4, line_h - 0.4)

                # Keep selected data columns per chunk.
                # The first column (from _display_df_with_index, e.g. stat labels or
                # meaningful index like Country) is repeated in every chunk.
                data_columns = list(df_display.columns)
                label_col = data_columns[0] if data_columns else None
                remaining_cols = data_columns[1:] if len(data_columns) > 1 else []
                
                if remaining_cols and label_col is not None:
                    per_chunk = max(1, cols_per_chunk - 1)  # -1 to account for label column
                    chunks = [
                        [label_col] + remaining_cols[i:i + per_chunk]
                        for i in range(0, len(remaining_cols), per_chunk)
                    ]
                else:
                    chunks = [data_columns] if data_columns else [[]]

                # Print Title at top of fresh page
                if title:
                    pdf.set_font(default_font, 'B', 11)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)

                for chunk_idx, chunk_cols in enumerate(chunks, start=1):
                    chunk_headers = [str(c) for c in chunk_cols]

                    # Subtitle for multi-part wide tables.
                    if len(chunks) > 1:
                        pdf.set_font(default_font, 'I', 8)
                        pdf.cell(
                            0,
                            6,
                            f"Columns {chunk_idx}/{len(chunks)}",
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )

                    available_width = pdf.w - pdf.l_margin - pdf.r_margin

                    # Compute content-aware widths so narrow columns use less space
                    # and long/header-heavy columns get more space.
                    width_scores: list[float] = []
                    max_rows_for_width = 150

                    wide_table = len(chunk_headers) >= 10

                    for col_pos, col_name in enumerate(chunk_headers):
                        source_col = chunk_cols[col_pos]
                        series_values = [str(v) for v in df_display[source_col].iloc[:max_rows_for_width]]

                        max_cell_len = max([len(str(col_name))] + [len(v) for v in series_values])
                        if wide_table:
                            width_scores.append(max(4.0, min(24.0, float(max_cell_len))))
                        else:
                            width_scores.append(max(6.0, min(36.0, float(max_cell_len))))

                    score_sum = sum(width_scores) if width_scores else 1.0
                    raw_widths = [(w / score_sum) * available_width for w in width_scores]

                    if wide_table:
                        min_widths = [12.0] + [7.0] * max(0, len(chunk_headers) - 1)
                    else:
                        min_widths = [16.0] + [9.0] * max(0, len(chunk_headers) - 1)
                    col_widths = [max(raw, min_w) for raw, min_w in zip(raw_widths, min_widths)]

                    total_width = sum(col_widths)
                    if total_width > available_width and total_width > 0:
                        shrink_factor = available_width / total_width
                        col_widths = [w * shrink_factor for w in col_widths]

                    def _safe_text(val):
                        return _safe_pdf_text(val)

                    def _ensure_row_space():
                        # Keep a small bottom margin; repeat the table header on a new page.
                        if pdf.get_y() <= 274:
                            return
                        pdf.add_page()
                        if title:
                            pdf.set_font(default_font, 'B', 11)
                            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(1)
                        if len(chunks) > 1:
                            pdf.set_font(default_font, 'I', 8)
                            pdf.cell(
                                0,
                                6,
                                f"Columns {chunk_idx}/{len(chunks)}",
                                new_x="LMARGIN",
                                new_y="NEXT",
                            )
                        pdf.set_font(default_font, 'B', font_size)
                        pdf.set_text_color(0, 0, 0)
                        for i, h in enumerate(chunk_headers):
                            align = 'L' if i == 0 else 'C'
                            pdf.cell(col_widths[i], line_h, _safe_text(str(h)), border=1, align=align)
                        pdf.ln(line_h)

                    # Header row
                    pdf.set_font(default_font, 'B', font_size)
                    pdf.set_text_color(0, 0, 0)
                    for i, h in enumerate(chunk_headers):
                        align = 'L' if i == 0 else 'C'
                        pdf.cell(col_widths[i], line_h, _safe_text(str(h)), border=1, align=align)
                    pdf.ln(line_h)

                    # Data rows
                    pdf.set_text_color(0, 0, 0)
                    for _, row_vals in df_display[chunk_cols].iterrows():
                        _ensure_row_space()
                        rendered = [str(v) for v in row_vals.values]
                        for i, item in enumerate(rendered):
                            # First column bold as row header
                            if i == 0:
                                pdf.set_font(default_font, 'B', font_size)
                                align = 'L'
                            else:
                                pdf.set_font(default_font, '', font_size)
                                align = 'R'
                            pdf.cell(col_widths[i], line_h, _safe_text(item), border=1, align=align)
                        pdf.ln(line_h)

                    if chunk_idx < len(chunks):
                        pdf.ln(2)
                
                pdf.ln(4)
            except Exception as e:
                app.logger.warning("Table rendering failed, using compact text fallback: %s", e)
                # Keep a predictable fallback that still includes tabular cues.
                if title:
                    pdf.set_font(default_font, 'B', 11)
                    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                add_text_block(df_table.to_string(max_rows=None, max_cols=None), courier=True)

        # Basic Info
        # First section doesn't need a new page (already on p1)
        add_section_title("1. Dataset Overview", new_page=False)

        font_family = default_font

        try:
            summary_df, columns_df = get_dataset_overview_tables(df)
            add_df_table(summary_df, title="Dataset Overview Summary:", new_page=False)
            add_df_table(columns_df, title="Columns & Types:", new_page=False)
        except Exception as e:
            app.logger.warning("Dataset overview table rendering failed, fallback to info text: %s", e)
            buf = io.StringIO()
            df.info(buf=buf)
            add_text_block(buf.getvalue(), courier=True)
        


        
        # Use new table function for head
        add_df_table(df.head(), title="First 5 Rows:")

        # Use new table function for describe
        add_df_table(df.describe(), title="Statistical Description:")

        # Missing Values - also on fresh page like other tables
        try:
            mv = df.isnull().sum()
            mvf = mv[mv > 0]
            if not mvf.empty:
                # Convert to DataFrame and use add_df_table for consistent formatting
                mv_df = mvf.to_frame('Missing Count')
                add_df_table(mv_df, title="Missing Values:")
        except Exception:
            pass

        # AI Summary
        ai_html = _get_clean_ai_summary_from_cache(filename)
        if ai_html:
            add_section_title("2. AI Analysis Summary")
            
            # Extract embedded model name
            model_name_used = None
            model_match = re.search(r'<!--\s*model:(.*?)\s*-->', ai_html)
            if model_match:
                model_name_used = model_match.group(1).strip()
            
            # Remove the comment before rendering
            clean_html = re.sub(r'<!--.*?-->', '', ai_html, flags=re.DOTALL)
            add_text_block(clean_html, is_html=True)

            # Always include model attribution at the bottom of the summary section.
            try:
                model_name = model_name_used or CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3.0-flash'
                if isinstance(model_name, str) and model_name.startswith('models/'):
                    model_name = model_name[7:]
                is_valid_summary = not _is_offline_html(clean_html)
                label = "Model used for AI summary" if is_valid_summary else "Configured AI model"
                pdf.set_font(font_family, 'I', 9)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 6, f"{label}: {model_name}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
                pdf.set_font(font_family, size=10)
                pdf.ln(2)
            except Exception:
                pass

        # Correlation Heatmaps
        try:
            corr_header_added = False
            
            # Helper to check/add header
            def ensure_corr_header():
                nonlocal corr_header_added
                if not corr_header_added:
                    add_section_title("3. Correlation Analysis")
                    corr_header_added = True
            # Use cached heatmaps for performance
            corr_heatmap_spearman = get_cached_heatmap(filename, df, method='spearman')
            if corr_heatmap_spearman:
                ensure_corr_header()
                # Keep-with-next logic: If near bottom, page break
                if pdf.page_no() > 0 and pdf.get_y() > 200: # Approx 297mm height, safety margin
                    pdf.add_page()
                    
                pdf.set_font(font_family, 'B', 10)
                pdf.cell(0, 8, "Spearman Correlation:", new_x="LMARGIN", new_y="NEXT")
                img_data = base64.b64decode(corr_heatmap_spearman)
                # Keep image within page width
                pdf.image(io.BytesIO(img_data), w=150, x=30)
                pdf.ln(5)

            corr_heatmap_pearson = get_cached_heatmap(filename, df, method='pearson')
            if corr_heatmap_pearson:
                ensure_corr_header()
                # Keep label and image together - add page if not enough space for both
                # Image height is approximately 100-120mm, so break earlier
                if pdf.page_no() > 0 and pdf.get_y() > 120:
                    pdf.add_page()
                    
                pdf.set_font(font_family, 'B', 10)
                pdf.cell(0, 8, "Pearson Correlation:", new_x="LMARGIN", new_y="NEXT")
                img_data = base64.b64decode(corr_heatmap_pearson)
                pdf.image(io.BytesIO(img_data), w=150, x=30)
                pdf.ln(5)
        except Exception as e:
            app.logger.error(f"Error adding correlation heatmaps to PDF: {e}")

        # Plots
        add_section_title("4. Column Analysis", new_page=True)
        
        # Show forecast setting used (Moved here)
        font_family = default_font
        pdf.set_font(font_family, 'B', 10)
        forecast_pct_display = int(forecast_pct * 100)
        if forecast_pct == 0:
            pdf.cell(0, 8, "Forecast Setting: 0% (data and anomalies only)", new_x="LMARGIN", new_y="NEXT")
        else:
            total_rows = len(df)
            forecast_steps = max(2, int(total_rows * forecast_pct))
            pdf.cell(0, 8, f"Forecast Setting: {forecast_pct_display}% ({forecast_steps} steps of {total_rows} rows)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_family, size=10)
        pdf.ln(2)
        
        is_ts = _is_reliable_timeseries_index(df.index)
        numeric_df_cached = get_cached_numeric_df(filename, df)
        numeric_cols = {col for col in numeric_df_cached.columns}
        x_axis_label = 'Timestamp' if is_ts else 'Index'
        total_rows = len(df)
        forecast_steps = max(2, int(total_rows * forecast_pct)) if forecast_pct > 0 else 0
        first_col = True
        img_width = 180
        img_x = 15

        def _add_base64_plot(plot_b64: str | None) -> bool:
            if not plot_b64:
                return False
            pdf.image(io.BytesIO(base64.b64decode(plot_b64)), w=img_width, x=img_x)
            pdf.ln(30)
            return True
        
        for col in df.columns:
            numeric_series = numeric_df_cached[col].dropna() if col in numeric_cols else pd.Series(dtype=float)
            is_numeric = len(numeric_series) >= 3
            series = numeric_series if is_numeric else df[col].astype(str)
            if not is_numeric and series.empty:
                continue

            # Force new page for EACH column to ensure clean layout
            # We don't use add_section_title here because we want a specific format
            # FIX: Only add page if it's NOT the first column (title page covers it)
            if first_col:
                first_col = False
                # Ensure we have a page before writing
                if pdf.page_no() == 0:
                    pdf.add_page()
            else:
                pdf.add_page()
            
            font_family = default_font
            pdf.set_font(font_family, 'B', 12)
            pdf.set_fill_color(245, 245, 245)
            # Column Title
            pdf.cell(0, 8, f"Column: {col}", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)
            pdf.set_font(font_family, size=10)
            
            pdf.ln(5)
            # Visual separator for column
            pdf.set_draw_color(100, 100, 100)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            
            pdf.set_font(font_family, size=10)

            # 1. TREND CHART (History + Anomalies, no forecast) - always first
            if is_numeric and len(numeric_series) >= 10:
                try:
                    raw_an_idx, raw_an_score = get_cached_anomalies(filename, col, numeric_series, user_contam)
                    
                    # Cap anomalies to match interactive page display
                    try:
                        max_points = int(app.config.get('ANOMALY_MARKER_CAP', 20))
                    except Exception:
                        max_points = 20
                    an_idx = _cap_anomalies_for_display(raw_an_idx, raw_an_score, max_points=max_points)
                    an_score = raw_an_score[an_idx] if not raw_an_score.empty else raw_an_score
                    
                    # Generate trend plot (no forecast, just history + anomalies)
                    trend_title = f"Trend: {col}"
                    trend_b64 = generate_forecast_plot(
                        numeric_series,
                        None,  # No forecast
                        trend_title,
                        x_axis_label,
                        col,
                        conf_int=None,
                        history_tail=None,
                        anomalies_idx=an_idx,
                        anomalies_score=an_score,
                        legend_y=-0.36,
                        xlabel_labelpad=6
                    )
                    _add_base64_plot(trend_b64)
                except Exception as e:
                    app.logger.error(f"Error adding trend plot to PDF for {col}: {e}")
                    an_idx = pd.Index([])
                    an_score = pd.Series([], dtype=float)
            
            # 2. FORECAST CHART (with forecast if pct > 0)
            if is_numeric and len(numeric_series) >= 10 and forecast_steps > 0:
                try:
                    # Use cached forecast
                    try:
                        fc_mean, ci = get_cached_column_forecast(filename, col, numeric_series, forecast_steps)
                    except Exception:
                        fc_mean, ci = None, None
                    
                    if fc_mean is not None:
                        fc_title = f"Forecast: {col} ({forecast_steps} steps)"
                        fc_b64 = generate_forecast_plot(
                            numeric_series,
                            fc_mean,
                            fc_title,
                            x_axis_label,
                            col,
                            conf_int=ci,
                            history_tail=None,
                            anomalies_idx=an_idx,
                            legend_y=-0.40,
                            xlabel_labelpad=6
                        )
                        _add_base64_plot(fc_b64)
                except Exception as e:
                    app.logger.error(f"Error adding forecast plot to PDF for {col}: {e}")
                    pass

            # 3. DISTRIBUTION chart
            try:
                fig, ax = plt.subplots(figsize=(10, 5))  # Wider figure for full-width image
                if is_numeric:
                    s_num = numeric_series
                    s_arr = np.asarray(s_num.to_numpy(dtype=float), dtype=float)
                    ax.hist(s_arr, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', label='Distribution')
                    ax.set_title(f"Distribution: {col}")
                    ax.set_ylabel("Frequency")
                    
                    # Add stat markers like web version
                    stats_min, stats_max = float(s_num.min()), float(s_num.max())
                    stats_mean, stats_median = float(s_num.mean()), float(s_num.median())
                    stats_std = float(s_num.std())
                    
                    # Avg/Med vertical lines (with labels for legend)
                    ax.axvline(x=stats_mean, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {stats_mean:.2f}')
                    ax.axvline(x=stats_median, color='#9b59b6', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'Med: {stats_median:.2f}')
                    
                    # Min/Max markers at bottom with annotations
                    ylim = ax.get_ylim()
                    xlim = ax.get_xlim()
                    marker_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
                    min_color = '#ff3b30'
                    max_color = '#00e5ff'
                    edge_color = '#0b1220'
                    ax.scatter([stats_min], [marker_y], color=min_color, s=80, zorder=10, marker='v', edgecolors=edge_color, linewidths=1.5, label=f'Min: {stats_min:.2f}')
                    ax.scatter([stats_max], [marker_y], color=max_color, s=80, zorder=10, marker='^', edgecolors=edge_color, linewidths=1.5, label=f'Max: {stats_max:.2f}')

                    # Match Detailed Analysis: min tag slightly left, max tag slightly right.
                    min_xytext, min_ha = (-3, 12), 'right'
                    max_xytext, max_ha = (3, 12), 'left'
                    if abs(stats_max - stats_min) <= (xlim[1] - xlim[0]) * 0.03:
                        min_xytext = (min_xytext[0], 12)
                        max_xytext = (max_xytext[0], 22)
                    ax.annotate(f'{stats_min:.2f}', (stats_min, marker_y), textcoords='offset points', xytext=min_xytext, ha=min_ha, fontsize=7, color=min_color, fontweight='bold', annotation_clip=False, clip_on=False)
                    ax.annotate(f'{stats_max:.2f}', (stats_max, marker_y), textcoords='offset points', xytext=max_xytext, ha=max_ha, fontsize=7, color=max_color, fontweight='bold', annotation_clip=False, clip_on=False)
                    
                    # Avg/Med tags
                    x_offset = (xlim[1] - xlim[0]) * 0.02
                    if stats_mean <= stats_median:
                        ax.text(stats_mean - x_offset, ylim[1] * 0.985, f'Avg: {stats_mean:.2f}', va='top', ha='right', fontsize=8, color='#f39c12', fontweight='bold')
                        ax.text(stats_median + x_offset, ylim[1] * 0.985, f'Med: {stats_median:.2f}', va='top', ha='left', fontsize=8, color='#9b59b6', fontweight='bold')
                    else:
                        ax.text(stats_median - x_offset, ylim[1] * 0.985, f'Med: {stats_median:.2f}', va='top', ha='right', fontsize=8, color='#9b59b6', fontweight='bold')
                        ax.text(stats_mean + x_offset, ylim[1] * 0.985, f'Avg: {stats_mean:.2f}', va='top', ha='left', fontsize=8, color='#f39c12', fontweight='bold')
                    
                    # Std in legend only (always add this, outside the if/else)
                    ax.plot([], [], color='#94a3b8', linestyle=':', label=f'Std: {stats_std:.2f}')
                    
                    # Legend with all stats (Min, Max, Avg, Med, Std) - always show
                    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False)
                    fig.subplots_adjust(bottom=0.22)
                else:
                    # Categorical bar chart (top 50)
                    all_counts = series.value_counts()
                    top_counts = all_counts.head(50)
                    top_counts.plot(kind='bar', ax=ax, color='tab:green', alpha=0.7, edgecolor='black')

                    # Add value labels above each bar
                    try:
                        if ax.containers and isinstance(ax.containers[0], BarContainer):
                            ax.bar_label(
                                ax.containers[0],
                                labels=[str(int(v)) for v in top_counts.values],
                                padding=2,
                                fontsize=7
                            )
                    except Exception:
                        pass
                    
                    total_unique = len(all_counts)
                    if len(all_counts) > 50:
                        ax.set_title(f"Categories: {col} (Top 50 of {total_unique})")
                    else:
                        ax.set_title(f"Categories: {col} ({total_unique} unique)")
                    ax.set_ylabel("Count")
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
                    
                    # Stats for annotation
                    max_count = int(all_counts.max())
                    min_count = int(all_counts.min())
                    avg_count = float(all_counts.mean())
                    med_count = float(all_counts.median())
                    most_freq = str(all_counts.index[0])[:15]
                    
                    # Add horizontal avg/med lines for counts
                    ax.axhline(y=avg_count, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label=f'Avg: {avg_count:.1f}')
                    ax.axhline(y=med_count, color='#9b59b6', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Med: {med_count:.1f}')
                    
                    # Add text labels for avg/med lines next to the chart, with extra separation if close
                    cat_ylim = ax.get_ylim()
                    y_range = cat_ylim[1] - cat_ylim[0]
                    threshold = y_range * 0.03
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

                    ax.text(1.01, avg_y, f'Avg: {avg_count:.1f}', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8, color='#f39c12', fontweight='bold')
                    ax.text(1.01, med_y, f'Med: {med_count:.1f}', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8, color='#9b59b6', fontweight='bold')
                    
                    # Stats in TOP-RIGHT corner
                    # Get least frequent item name
                    least_freq = str(all_counts.index[-1])[:15] if len(all_counts) > 0 else "N/A"
                    
                    # Add Most/Least as legend entries
                    ax.plot([], [], color='#27ae60', marker='s', linestyle='', markersize=8, label=f"Most: '{most_freq}' ({max_count})")
                    ax.plot([], [], color='#e74c3c', marker='s', linestyle='', markersize=8, label=f"Least: '{least_freq}' ({min_count})")
                    
                    # Legend on top-right, vertical (line by line)
                    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

                ax.set_xlabel(col)
                ax.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                pdf.image(buf, w=img_width, x=img_x)
                pdf.ln(30)
            except Exception:
                pass
                
            # 4. STL DECOMPOSITION - last (for timeseries only)
            if is_numeric and is_ts and len(numeric_series) >= 28:
                try:
                    sp = _infer_seasonal_period(numeric_series.index)
                    if sp and isinstance(sp, int) and sp >= 2:
                        # Use cached STL plot - may already be computed from web view
                        stl_b64 = get_cached_stl_plot(filename, col, numeric_series, sp)
                        _add_base64_plot(stl_b64)
                except Exception:
                    pass
    except Exception as e:
        app.logger.error(f"Error generating PDF: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"ok": False, "message": f"PDF generation failed: {str(e)}"}), 500

    # Output PDF with safeguard
    try:
        # Ensure at least one page exists before output
        if pdf.page_no() == 0:
            app.logger.warning("No pages in PDF, adding a blank page")
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            pdf.cell(0, 10, "No content available for this report.", new_x="LMARGIN", new_y="NEXT")
        
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1')
        elif isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        out = io.BytesIO(pdf_bytes)
        out.seek(0)
    except Exception as e:
        app.logger.error(f"PDF output failed: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"ok": False, "message": f"PDF output failed: {str(e)}"}), 500
    
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_report.pdf")
    
    return make_response(out.read(), 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'attachment; filename="{out_name}"'
    })

download_full_report_pdf = handle_download_full_report_pdf

__all__ = ["PDFReport", "handle_download_full_report_pdf", "download_full_report_pdf"]
