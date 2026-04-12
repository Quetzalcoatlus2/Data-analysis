# ruff: noqa: F401,F403,F405
from __future__ import annotations

import base64
import html as htmllib
import io
import math
import re
from html.parser import HTMLParser
from typing import Any, cast

import emoji
from fpdf import FPDF

from data_analysis.analysis.plot import (
    _add_static_distribution_overlays,
    _apply_dense_non_overlapping_y_ticks,
    _apply_sci_formatter,
    _build_static_category_chart,
    _format_stat_value,
    apply_distribution_axis_spec,
    apply_static_distribution_compact_layout,
    build_distribution_axis_spec,
    get_export_chart_figsize,
)
from data_analysis.core.runtime_bind import bind_runtime_globals
from data_analysis.runtime_app import *  # pyright: ignore[reportAssignmentType]
from data_analysis.runtime_app import (
    _cap_anomalies_for_display,
    _display_df_with_index,
    _get_clean_ai_summary_from_cache,
    _infer_seasonal_period,
    _is_active_temporal_axis_column,
    _is_offline_html,
    _is_reliable_timeseries_index,
)

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "PDFReport",
    "handle_download_full_report_pdf",
    "download_full_report_pdf",
    "__all__",
}



def _bind_runtime_globals():
    return bind_runtime_globals(globals(), _LOCAL_SYMBOLS)


_bind_runtime_globals()


def _select_pdf_font(available_fonts: set[str]) -> str:
    """Pick stable body font from loaded TTF families."""
    normalized = {str(name).lower(): str(name) for name in available_fonts}

    for cand in ("segoeui", "arial"):
        if cand in normalized:
            return normalized[cand]

    return "helvetica"


def _collapse_pdf_html_spacing(html: str) -> str:
    """Normalize AI-summary HTML spacing to reduce false page skips."""
    text = str(html or "")
    # Remove comments (e.g., model markers) before layout checks.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Collapse repeated empty paragraphs into a single separator paragraph.
    text = re.sub(
        r"(?is)(?:\s*<p[^>]*>\s*(?:&nbsp;|\s|<br\s*/?>)*\s*</p>\s*){2,}",
        "<p></p>",
        text,
    )
    # Collapse repeated line breaks.
    text = re.sub(r"(?is)(?:<br\s*/?>\s*){2,}", "<br/>", text)
    # Remove whitespace-only gaps between tags that can add hidden vertical space.
    text = re.sub(r">\s+<", "><", text)
    return text.strip()


def _extract_first_context_text(body_html: str) -> tuple[str, bool]:
    """Return first meaningful paragraph/list-item text and whether it's from a list item."""
    html_body = str(body_html or "").strip()
    if not html_body:
        return "", False

    candidates: list[tuple[int, str, bool]] = []
    for is_list, pattern in (
        (True, r"(?is)<li[^>]*>(.*?)</li>"),
        (False, r"(?is)<p[^>]*>(.*?)</p>"),
    ):
        for match in re.finditer(pattern, html_body):
            snippet = re.sub(r"<[^>]+>", " ", match.group(1))
            snippet = htmllib.unescape(snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if snippet:
                candidates.append((match.start(), snippet, is_list))
                break

    if candidates:
        candidates.sort(key=lambda c: c[0])
        _, text, is_list = candidates[0]
        return text, is_list

    plain = re.sub(r"<[^>]+>", " ", html_body)
    plain = htmllib.unescape(plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain, False


def _estimate_wrapped_lines(pdf: FPDF, text: str, width_mm: float) -> int:
    """Estimate wrapped line count using current PDF font metrics."""
    if width_mm <= 0:
        return 1
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return 1

    words = normalized.split(" ")
    if not words:
        return 1

    line_count = 1
    current_line = ""

    for word in words:
        if not current_line:
            if pdf.get_string_width(word) <= width_mm:
                current_line = word
                continue

            # Hard-wrap a single long token.
            chunk = ""
            segments = 1
            for ch in word:
                trial = chunk + ch
                if not chunk or pdf.get_string_width(trial) <= width_mm:
                    chunk = trial
                else:
                    segments += 1
                    chunk = ch
            line_count += max(0, segments - 1)
            current_line = chunk
            continue

        candidate = f"{current_line} {word}"
        if pdf.get_string_width(candidate) <= width_mm:
            current_line = candidate
            continue

        line_count += 1
        if pdf.get_string_width(word) <= width_mm:
            current_line = word
            continue

        chunk = ""
        segments = 1
        for ch in word:
            trial = chunk + ch
            if not chunk or pdf.get_string_width(trial) <= width_mm:
                chunk = trial
            else:
                segments += 1
                chunk = ch
        line_count += max(0, segments - 1)
        current_line = chunk

    return max(1, line_count)


def _should_break_before_heading(y_mm: float, remaining_mm: float, required_mm: float) -> bool:
    """Break only when a heading is near page bottom and context space is insufficient."""
    if y_mm <= 46.0:
        return False
    return remaining_mm < required_mm


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

        def _remaining_page_space() -> float:
            return float(pdf.h - pdf.b_margin - pdf.get_y())

        pdf.set_font(default_font, size=12)
        def _restore_emoji_placeholders(text: str) -> str:
            """Strip emojis entirely."""
            return emoji.replace_emoji(text, replace='')

        # Helper for adding sections
        def add_section_title(title, new_page=True):
            ensure_page()
            if new_page:
                # Avoid creating a near-empty page when we're already at the top.
                if float(pdf.get_y()) > 46.0:
                    pdf.add_page()
            else:
                # Keep title + at least ~2 body lines together without being overly aggressive.
                # 8mm title + 2mm spacing + 2*5mm body lines + small cushion ~= 24mm.
                if _remaining_page_space() < 24.0:
                    pdf.add_page()
                else:
                    pdf.ln(4)

            pdf.set_font(default_font, style="B", size=13)
            pdf.set_text_color(0, 0, 0)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)
            pdf.set_font(default_font, size=10)

        def _truncate_cell(text: str, col_width_mm: float, font_size_pt: int = 9) -> str:
            """Clip text to fit within a PDF cell of the given width.

            Uses pdf.get_string_width() for exact measurement. When the text
            is too wide for the cell it is progressively shortened until it
            fits. A short ellipsis is appended when clipping occurs.
            """
            if not text:
                return text
            text = str(text).replace("\r", " ").replace("\n", " ").strip()
            # Measure actual rendered width using the current font
            try:
                # Save current font state
                cur_family = pdf.font_family
                cur_style = pdf.font_style
                cur_size = pdf.font_size_pt
                cur_size_int = max(1, int(round(float(cur_size))))
                pdf.set_font(cur_family, style=cast(Any, cur_style), size=font_size_pt)
                tw = pdf.get_string_width(text)
                # Keep text strictly inside the inner content box:
                # full cell width minus left/right cell margins and a tiny safety gap.
                inner_margin = float(getattr(pdf, "c_margin", 1.0))
                target_w = max(0.0, col_width_mm - (2.0 * inner_margin) - 0.05)
                if tw <= target_w or target_w <= 0:
                    pdf.set_font(cur_family, style=cast(Any, cur_style), size=cur_size_int)
                    return text
                # Binary search for the longest prefix that fits
                lo, hi = 0, len(text)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if pdf.get_string_width(text[:mid]) <= target_w:
                        lo = mid
                    else:
                        hi = mid - 1
                clipped = text[:max(1, lo)].rstrip()
                if lo < len(text):
                    ell = "..."
                    if len(clipped) > len(ell):
                        while clipped and pdf.get_string_width(clipped + ell) > target_w:
                            clipped = clipped[:-1]
                        if clipped:
                            clipped = clipped + ell
                pdf.set_font(cur_family, style=cast(Any, cur_style), size=cur_size_int)
                return clipped
            except Exception:
                # Fallback: rough char estimate
                char_w = font_size_pt * 0.8 * 0.352778
                max_chars = max(3, int(col_width_mm / char_w))
                if len(text) <= max_chars:
                    return text
                if max_chars <= 3:
                    return text[:max_chars]
                return text[:max_chars - 3] + "..."

        def _format_table_cell_value(value: object) -> str:
            """Compact table text to avoid cell overflow."""
            text = str(value).replace("\r", " ").replace("\n", " ").strip()
            if not text:
                return text
            try:
                n = float(text)
                if math.isfinite(n):
                    mag = abs(n)
                    if mag >= 1e12:
                        return f"{n / 1e12:.3f}T"
                    if mag >= 1e9:
                        return f"{n / 1e9:.3f}B"
            except Exception:
                pass
            return text

        def _format_plot_value(value: float) -> str:
            """Compact stat labels for charts with K/M/B/T suffixes."""
            try:
                v = float(value)
                if not math.isfinite(v):
                    return f"{v:.2f}"
                mag = abs(v)
                if mag >= 1e15:
                    return f"{v:.3e}"
                if mag >= 1e12:
                    raw = f"{v / 1e12:.3f}"
                    return raw.rstrip("0").rstrip(".") + "T"
                if mag >= 1e9:
                    raw = f"{v / 1e9:.3f}"
                    return raw.rstrip("0").rstrip(".") + "B"
                if mag >= 1e6:
                    raw = f"{v / 1e6:.3f}"
                    return raw.rstrip("0").rstrip(".") + "M"
                if mag >= 1e3:
                    raw = f"{v / 1e3:.2f}"
                    return raw.rstrip("0").rstrip(".") + "K"
                raw = f"{v:.2f}"
                return raw.rstrip("0").rstrip(".")
            except Exception:
                return str(value)

        def _ensure_img_space(min_mm: float = 60.0) -> None:
            """Start a new page if there isn't enough vertical space for an image."""
            if _remaining_page_space() < min_mm:
                pdf.add_page()

        def add_text_block(text, courier=False, is_html=False):
            font_family = default_font

            # Ensure we have a page
            if pdf.page_no() == 0:
                app.logger.warning("No page open, adding one.")
                ensure_page()
            
            if is_html:
                pdf.set_font(font_family, size=10)
                
                html_text = _restore_emoji_placeholders(str(text))
                html_text = _collapse_pdf_html_spacing(html_text)

                def _write_html_fragment(fragment: str) -> None:
                    fragment = str(fragment or "").strip()
                    if not fragment:
                        return
                    try:
                        pdf.write_html(fragment)
                    except Exception as e:
                        app.logger.error(f"Native HTMLParser PDF rendering failed: {e}")
                        fallback_text = re.sub(r'<[^>]+>', ' ', fragment)
                        fallback_text = re.sub(r'\s+', ' ', fallback_text).strip()
                        if body_font_is_unicode:
                            pdf.multi_cell(0, 5, fallback_text)
                        else:
                            pdf.multi_cell(0, 5, _safe_pdf_text(fallback_text))

                # Keep heading blocks from being stranded at page bottom.
                heading_re = re.compile(r'(?is)<h[1-6][^>]*>.*?</h[1-6]>')
                heading_matches = list(heading_re.finditer(html_text))
                if not heading_matches:
                    _write_html_fragment(html_text)
                else:
                    cursor = 0
                    for idx_match, match in enumerate(heading_matches):
                        start = match.start()
                        end_of_block = heading_matches[idx_match + 1].start() if idx_match + 1 < len(heading_matches) else len(html_text)

                        # Render content before heading (if any).
                        if start > cursor:
                            _write_html_fragment(html_text[cursor:start])

                        # Heading + following content block.
                        heading_html = html_text[start:match.end()]
                        body_html = html_text[match.end():end_of_block]
                        heading_html = heading_html.strip()
                        body_html = body_html.strip()
                        body_plain = re.sub(r"<[^>]+>", " ", body_html)
                        body_plain = re.sub(r"\s+", " ", body_plain).strip()

                        first_context_text, is_list_context = _extract_first_context_text(body_html)
                        if not first_context_text:
                            first_context_text = body_plain

                        available_width = float(pdf.w - pdf.l_margin - pdf.r_margin - 1.0)
                        if first_context_text:
                            first_context_lines = _estimate_wrapped_lines(pdf, first_context_text, available_width)
                            first_context_lines = max(1, min(6, first_context_lines))
                        else:
                            first_context_lines = 0

                        heading_mm = 8.0
                        heading_gap_mm = 2.0
                        line_mm = 5.0
                        if first_context_lines > 0:
                            # Extra context guard: if only heading + one short paragraph fits,
                            # defer section to next page.
                            extra_context_mm = 4.0 if is_list_context else 5.0
                            min_block_space = (
                                heading_mm
                                + heading_gap_mm
                                + (line_mm * float(first_context_lines))
                                + extra_context_mm
                            )
                        else:
                            # Heading-only blocks should still render, but avoid orphaning near footer.
                            min_block_space = heading_mm + heading_gap_mm + 8.0

                        # Preserve nested list structure but keep at least one bullet with
                        # heading+intro when body pattern is paragraph -> list.
                        first_para = re.search(r"(?is)<p\b[^>]*>.*?</p>", body_html)
                        first_list = re.search(r"(?is)<(?:ul|ol)\b[^>]*>", body_html)
                        if first_para and first_list and first_list.start() > first_para.start():
                            first_bullet_text = ""
                            bullet_match = re.search(r"(?is)<li\b[^>]*>(.*?)</li>", body_html)
                            if bullet_match:
                                snippet = re.sub(r"<[^>]+>", " ", bullet_match.group(1))
                                snippet = htmllib.unescape(snippet)
                                first_bullet_text = re.sub(r"\s+", " ", snippet).strip()
                            bullet_width = max(10.0, available_width - 6.0)
                            if first_bullet_text:
                                bullet_lines = _estimate_wrapped_lines(
                                    pdf,
                                    first_bullet_text,
                                    bullet_width,
                                )
                                bullet_lines = max(1, min(3, bullet_lines))
                                min_block_space += 2.0 + (line_mm * float(bullet_lines))
                            else:
                                min_block_space += 6.0

                        remaining = _remaining_page_space()
                        if _should_break_before_heading(
                            y_mm=float(pdf.get_y()),
                            remaining_mm=remaining,
                            required_mm=min_block_space,
                        ):
                            pdf.add_page()

                        # Render heading first, then body. This avoids native HTML pagination
                        # splitting the heading from the first sentence/list item.
                        _write_html_fragment(heading_html)
                        _write_html_fragment(body_html)
                        cursor = end_of_block
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

        def add_df_table(
            df_table,
            title=None,
            new_page=True,
            placement_mode: str | None = None,
        ):
            """Renders a pandas DataFrame as a table in the PDF.
               placement_mode="always" forces a fresh page, "auto" keeps the titled
               block together when it fits the remaining space, and "flow" preserves
               the older lightweight guard behavior.
            """
            ensure_page()
            
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
                    font_size = 5.5
                    line_h = 3.4
                    cols_per_chunk = 6  # Fewer cols → more width → less overflow
                elif total_cols >= 12:
                    font_size = 6.0
                    line_h = 3.8
                    cols_per_chunk = 7
                else:
                    font_size = 7.0
                    line_h = 4.2
                    cols_per_chunk = 8

                if row_count >= 20:
                    font_size = max(5.4, font_size - 0.8)
                    line_h = max(3.8, line_h - 0.2)

                font_size_int = max(5, int(round(font_size)))

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

                resolved_mode = str(
                    placement_mode if placement_mode is not None else ("always" if new_page else "flow")
                ).strip().lower()
                if resolved_mode not in {"always", "auto", "flow"}:
                    resolved_mode = "always"

                def _estimate_table_block_height_mm() -> float:
                    title_height = 13.0 if title else 0.0
                    chunk_title_height = 6.0 if len(chunks) > 1 else 0.0
                    inter_chunk_gap = 2.0
                    header_height = float(line_h)
                    data_height = float(max(0, row_count)) * float(line_h)
                    total = title_height
                    for chunk_idx in range(len(chunks)):
                        total += chunk_title_height + header_height + data_height
                        if chunk_idx < len(chunks) - 1:
                            total += inter_chunk_gap
                    total += 4.0
                    return total

                estimated_block_height = _estimate_table_block_height_mm()
                if resolved_mode == "always":
                    pdf.add_page()
                elif resolved_mode == "auto":
                    if _remaining_page_space() < estimated_block_height:
                        pdf.add_page()
                elif _remaining_page_space() < 34.0:
                    pdf.add_page()

                # Print Title at top of fresh page
                if title:
                    pdf.set_font(default_font, 'B', 11)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)

                for chunk_idx, chunk_cols in enumerate(chunks, start=1):
                    chunk_headers = [str(c) for c in chunk_cols]

                    # Avoid orphaned chunk title/header at page bottom.
                    min_chunk_space = 24.0
                    if _remaining_page_space() < min_chunk_space:
                        pdf.add_page()

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

                    # -------------------------------------------------------
                    # Content-aware column widths using MEASURED string widths
                    # pdf.get_string_width() accounts for actual glyph sizes.
                    # -------------------------------------------------------
                    pdf.set_font(default_font, 'B', font_size_int)  # headers are bold
                    header_widths = [pdf.get_string_width(h) + 3.0 for h in chunk_headers]  # +3 padding

                    pdf.set_font(default_font, '', font_size_int)  # data is regular
                    max_rows_for_width = 150
                    data_widths: list[float] = []
                    for col_pos, _ in enumerate(chunk_headers):
                        source_col = chunk_cols[col_pos]
                        series_values = [str(v) for v in df_display[source_col].iloc[:max_rows_for_width]]
                        max_data_w = max((pdf.get_string_width(v) for v in series_values), default=0.0) + 2.0
                        data_widths.append(max_data_w)

                    # Each column's ideal width is the max of its header and data content.
                    ideal_widths = [max(hw, dw) for hw, dw in zip(header_widths, data_widths)]

                    # Hard per-column bounds:
                    # - keep the first (label/index) column readable
                    # - prevent long free-text columns from starving neighbors.
                    def _is_compact_index_like_first_col() -> bool:
                        if not chunk_headers:
                            return False
                        h0 = str(chunk_headers[0]).strip().lower()
                        if not (h0 == "index" or h0 == "level_0" or h0.startswith("unnamed:")):
                            return False
                        values = [str(v).strip() for v in df_display[chunk_cols[0]].iloc[:max_rows_for_width].tolist()]
                        vals = [v for v in values if v]
                        if not vals:
                            return True
                        lengths = sorted(len(v) for v in vals)
                        p90_len = lengths[min(len(lengths) - 1, int(0.9 * (len(lengths) - 1)))]
                        numericish = sum(1 for v in vals if re.fullmatch(r"[-+]?\d+", v) is not None)
                        numericish_ratio = numericish / max(1, len(vals))
                        return p90_len <= 6 and numericish_ratio >= 0.85

                    compact_first_col = _is_compact_index_like_first_col()
                    first_min = 9.0 if compact_first_col else 12.0
                    first_max = 16.0 if compact_first_col else 42.0
                    min_widths = [first_min] + [9.0] * max(0, len(chunk_headers) - 1)
                    max_widths = [first_max] + [64.0] * max(0, len(chunk_headers) - 1)
                    bounded = [
                        max(min_w, min(w, max_w))
                        for w, min_w, max_w in zip(ideal_widths, min_widths, max_widths)
                    ]

                    total_width = sum(bounded)
                    if total_width > available_width:
                        # Shrink only the part above minimum widths.
                        min_total = sum(min_widths)
                        if min_total >= available_width:
                            col_widths = [available_width / max(1, len(chunk_headers))] * len(chunk_headers)
                        else:
                            overflow = total_width - available_width
                            reducible = [max(0.0, w - m) for w, m in zip(bounded, min_widths)]
                            reducible_total = sum(reducible)
                            if reducible_total > 0:
                                col_widths = [
                                    w - overflow * (r / reducible_total)
                                    for w, r in zip(bounded, reducible)
                                ]
                            else:
                                col_widths = list(bounded)
                    else:
                        # Grow columns with remaining headroom.
                        surplus = available_width - total_width
                        headroom = [max(0.0, mx - w) for w, mx in zip(bounded, max_widths)]
                        headroom_total = sum(headroom)
                        if headroom_total > 0 and surplus > 0:
                            col_widths = [
                                w + surplus * (h / headroom_total)
                                for w, h in zip(bounded, headroom)
                            ]
                        else:
                            col_widths = list(bounded)

                    def _safe_text(val):
                        return _safe_pdf_text(val)

                    def _ensure_row_space():
                        # Keep a small bottom margin; repeat the table header on a new page.
                        if (pdf.get_y() + line_h) <= (pdf.h - pdf.b_margin):
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
                        pdf.set_font(default_font, 'B', font_size_int)
                        pdf.set_text_color(0, 0, 0)
                        for i, h in enumerate(chunk_headers):
                            align = 'L' if i == 0 else 'C'
                            pdf.cell(col_widths[i], line_h,
                                     _safe_text(_truncate_cell(str(h), col_widths[i], font_size_int)),
                                     border=1, align=align)
                        pdf.ln(line_h)

                    # Header row — clip header text to column width (no ellipsis)
                    pdf.set_font(default_font, 'B', font_size_int)
                    pdf.set_text_color(0, 0, 0)
                    for i, h in enumerate(chunk_headers):
                        align = 'L' if i == 0 else 'C'
                        pdf.cell(col_widths[i], line_h,
                                 _safe_text(_truncate_cell(str(h), col_widths[i], font_size_int)),
                                 border=1, align=align)
                    pdf.ln(line_h)

                    # Data rows
                    pdf.set_text_color(0, 0, 0)
                    chunk_df = df_display[chunk_cols]
                    for row_vals in chunk_df.itertuples(index=False, name=None):
                        _ensure_row_space()
                        rendered = [_format_table_cell_value(v) for v in row_vals]
                        for i, item in enumerate(rendered):
                            # First column bold as row header
                            if i == 0:
                                pdf.set_font(default_font, 'B', font_size_int)
                                align = 'L'
                            else:
                                pdf.set_font(default_font, '', font_size_int)
                                align = 'R'
                            pdf.cell(
                                col_widths[i], line_h,
                                _safe_text(_truncate_cell(item, col_widths[i], font_size_int)),
                                border=1, align=align,
                            )
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
            add_df_table(summary_df, title="Dataset Overview Summary:", placement_mode="auto")
            add_df_table(columns_df, title="Columns & Types:", placement_mode="auto")
        except Exception as e:
            app.logger.warning("Dataset overview table rendering failed, fallback to info text: %s", e)
            buf = io.StringIO()
            df.info(buf=buf)
            add_text_block(buf.getvalue(), courier=True)
        

        
        
        # Use new table function for head
        add_df_table(df.head(), title="First 5 Rows:", placement_mode="auto")

        # Use new table function for describe
        add_df_table(df.describe(), title="Statistical Description:", placement_mode="auto")

        # Missing Values - also on fresh page like other tables
        try:
            mv = df.isnull().sum()
            mvf = mv[mv > 0]
            if not mvf.empty:
                # Convert to DataFrame and use add_df_table for consistent formatting
                mv_df = mvf.to_frame('Missing Count')
                add_df_table(mv_df, title="Missing Values:")
        except Exception as e:
            app.logger.debug("Missing values table skipped in PDF generation: %s", e)

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
                model_name = model_name_used or CURRENT_MODEL_NAME or AI_STATUS.get('model') or DEFAULT_AI_MODEL or 'gemini-3-flash-preview'
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
            except Exception as e:
                app.logger.debug("AI summary model attribution skipped in PDF: %s", e)

        # Correlation Heatmaps
        try:
            corr_header_added = False
            
            # Helper to check/add header
            def ensure_corr_header():
                nonlocal corr_header_added
                if not corr_header_added:
                    add_section_title("3. Correlation Analysis")
                    corr_header_added = True

            def _estimate_b64_image_height_mm(plot_b64: str, width_mm: float) -> float:
                default_height = 90.0
                try:
                    from PIL import Image

                    raw = base64.b64decode(plot_b64)
                    with Image.open(io.BytesIO(raw)) as img_obj:
                        width_px, height_px = img_obj.size
                    if width_px <= 0:
                        return default_height
                    ratio = float(height_px) / float(width_px)
                    return max(30.0, float(width_mm) * ratio)
                except Exception:
                    return default_height

            corr_specs: list[str] = []
            corr_heatmap_spearman = get_cached_heatmap(
                filename,
                df,
                method='spearman',
                layout_preset='export',
            )
            if corr_heatmap_spearman:
                corr_specs.append(corr_heatmap_spearman)
            corr_heatmap_pearson = get_cached_heatmap(
                filename,
                df,
                method='pearson',
                layout_preset='export',
            )
            if corr_heatmap_pearson:
                corr_specs.append(corr_heatmap_pearson)

            if corr_specs:
                ensure_corr_header()
                corr_img_width_mm = 150.0
                corr_gap_mm = 2.0

                if len(corr_specs) >= 2:
                    required_space_mm = 0.0
                    for idx, plot_b64 in enumerate(corr_specs[:2]):
                        required_space_mm += _estimate_b64_image_height_mm(plot_b64, corr_img_width_mm)
                        if idx < 1:
                            required_space_mm += corr_gap_mm
                    if _remaining_page_space() < (required_space_mm + 4.0):
                        pdf.add_page()

                for idx, plot_b64 in enumerate(corr_specs):
                    img_height_mm = _estimate_b64_image_height_mm(plot_b64, corr_img_width_mm)
                    _ensure_img_space(img_height_mm + 1.0)
                    img_data = base64.b64decode(plot_b64)
                    pdf.image(io.BytesIO(img_data), w=corr_img_width_mm, x=30)
                    if idx < len(corr_specs) - 1:
                        pdf.ln(corr_gap_mm)
                    else:
                        pdf.ln(4)
        except Exception as e:
            app.logger.error(f"Error adding correlation heatmaps to PDF: {e}")

        # Plots
        add_section_title("4. Column Analysis", new_page=True)

        def _steps_for_history_rows(history_rows: int) -> int:
            if forecast_pct <= 0 or history_rows <= 0:
                return 0
            pct_den = max(1e-9, 1.0 - float(forecast_pct))
            return max(1, int(math.floor(float(history_rows) * float(forecast_pct) / pct_den)))
        
        # Show forecast setting used (Moved here)
        font_family = default_font
        pdf.set_font(font_family, 'B', 10)
        forecast_pct_display = int(forecast_pct * 100)
        if forecast_pct == 0:
            pdf.cell(0, 8, "Forecast Setting: 0% (data and anomalies only)", new_x="LMARGIN", new_y="NEXT")
        else:
            total_rows = len(df)
            forecast_steps = _steps_for_history_rows(total_rows)
            pdf.cell(
                0,
                8,
                (
                    f"Forecast Setting: {forecast_pct_display}% "
                    f"(up to {forecast_steps} steps for {total_rows} rows; per-column non-null history)"
                ),
                new_x="LMARGIN",
                new_y="NEXT",
            )
        pdf.set_font(font_family, size=10)
        pdf.ln(2)
        
        is_ts = _is_reliable_timeseries_index(df.index)
        numeric_df_cached = get_cached_numeric_df(filename, df)
        numeric_cols = {col for col in numeric_df_cached.columns}
        x_axis_label = 'Timestamp' if is_ts else 'Index'
        first_col = True
        img_width = 180
        img_x = 15

        def _add_base64_plot(
            plot_b64: str | None,
            *,
            min_space_mm: float = 96.0,
            gap_after_mm: float = 8.0,
            width_mm: float | None = None,
            ensure_space: bool = True,
        ) -> bool:
            if not plot_b64:
                return False
            if ensure_space:
                _ensure_img_space(min_space_mm)

            render_width = float(width_mm) if width_mm is not None else float(img_width)
            if not math.isfinite(render_width) or render_width <= 0:
                render_width = float(img_width)

            if render_width < float(img_width):
                render_x = max(float(pdf.l_margin), (float(pdf.w) - render_width) / 2.0)
            else:
                render_x = float(img_x)

            pdf.image(io.BytesIO(base64.b64decode(plot_b64)), w=render_width, x=render_x)
            pdf.ln(gap_after_mm)
            return True

        def _resolve_paired_plot_width(
            remaining_space_mm: float,
            *,
            first_plot_ratio: float,
            second_plot_ratio: float,
            gap_mm: float,
            min_width_mm: float = 118.0,
            max_width_mm: float = 180.0,
        ) -> float | None:
            available_space = float(remaining_space_mm) - float(gap_mm)
            if available_space <= 0:
                return None

            total_ratio = max(1e-6, float(first_plot_ratio) + float(second_plot_ratio))
            width = available_space / total_ratio
            if width < float(min_width_mm):
                return None
            return min(float(max_width_mm), width)
        
        for col in df.columns:
            numeric_series = numeric_df_cached[col].dropna() if col in numeric_cols else pd.Series(dtype=float)
            is_numeric = len(numeric_series) >= 3
            series = numeric_series if is_numeric else df[col].astype(str)
            if not is_numeric and series.empty:
                continue
            if not is_numeric and _is_active_temporal_axis_column(df, col):
                continue
            col_forecast_steps = _steps_for_history_rows(len(numeric_series))

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

            # Keep PDF chart gating aligned with interactive/analyze flow for small datasets.
            min_trend_forecast_points = 5
            an_idx = pd.Index([])
            an_score = pd.Series([], dtype=float)

            # 1. TREND CHART (History + Anomalies, no forecast) - always first
            if is_numeric and len(numeric_series) >= min_trend_forecast_points:
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
                        figsize=get_export_chart_figsize("trend", context="pdf"),
                    )
                    _add_base64_plot(trend_b64)
                except Exception as e:
                    app.logger.error(f"Error adding trend plot to PDF for {col}: {e}")
                    an_idx = pd.Index([])
                    an_score = pd.Series([], dtype=float)
            
            # 2. FORECAST CHART (with forecast if pct > 0)
            if is_numeric and len(numeric_series) >= min_trend_forecast_points and col_forecast_steps > 0:
                try:
                    # Use cached forecast
                    try:
                        fc_mean, ci = get_cached_column_forecast(filename, col, numeric_series, col_forecast_steps)
                    except Exception as e:
                        app.logger.debug("Cached forecast unavailable for PDF column '%s': %s", col, e)
                        fc_mean, ci = None, None
                    
                    if fc_mean is not None:
                        fc_title = f"Forecast: {col} ({col_forecast_steps} steps)"
                        fc_b64 = generate_forecast_plot(
                            numeric_series,
                            fc_mean,
                            fc_title,
                            x_axis_label,
                            col,
                            conf_int=ci,
                            history_tail=None,
                            anomalies_idx=an_idx,
                            figsize=get_export_chart_figsize("forecast", context="pdf"),
                        )
                        _add_base64_plot(fc_b64)
                except Exception as e:
                    app.logger.error(f"Error adding forecast plot to PDF for {col}: {e}")
                    pass

            stl_b64: str | None = None
            if is_numeric and is_ts and len(numeric_series) >= 28:
                try:
                    sp = _infer_seasonal_period(numeric_series.index)
                    if sp and isinstance(sp, int) and sp >= 2:
                        # Use cached STL plot - may already be computed from web view
                        stl_b64 = get_cached_stl_plot(filename, col, numeric_series, sp)
                except Exception as e:
                    app.logger.debug("STL plot skipped in PDF for '%s': %s", col, e)

            pair_with_stl = bool(stl_b64)
            paired_plot_width_mm: float | None = None
            distribution_rendered = False

            # 3. DISTRIBUTION chart
            try:
                fig, ax = plt.subplots(figsize=get_export_chart_figsize("distribution", context="pdf"))
                if is_numeric:
                    s_num = numeric_series
                    s_arr = np.asarray(s_num.to_numpy(dtype=float), dtype=float)
                    axis_spec = build_distribution_axis_spec(
                        s_arr.tolist(),
                        min_bins=max(8, min(12, len(s_arr) // 5)) if len(s_arr) >= 20 else 8,
                        max_bins=52,
                        integer_span_threshold=260,
                    )
                    hist_bins = axis_spec.get('hist_bins') if isinstance(axis_spec, dict) else None
                    if not hist_bins:
                        hist_bins = max(8, min(52, int(len(s_arr) // 10) if len(s_arr) >= 20 else 8))
                    _hist_counts, hist_edges, _hist_patches = ax.hist(
                        s_arr,
                        bins=hist_bins,
                        color='tab:blue',
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5,
                        label='Distribution',
                    )
                    ax.set_title(f"Distribution: {col}", pad=16)
                    ax.set_xlabel(col, fontsize=9, labelpad=0)
                    ax.set_ylabel("Frequency", labelpad=8)
                    ax.grid(True, alpha=0.3)
                    _apply_dense_non_overlapping_y_ticks(
                        ax,
                        integer=True,
                        label_fontsize=8.0,
                        min_ticks=6,
                        max_ticks=18,
                    )
                    if isinstance(axis_spec, dict) and axis_spec:
                        apply_distribution_axis_spec(ax, axis_spec)
                    elif len(hist_edges) >= 2:
                        ax.set_xlim(float(hist_edges[0]), float(hist_edges[-1]))

                    _add_static_distribution_overlays(
                        ax,
                        s_arr,
                        value_formatter=_format_stat_value,
                        legend_fontsize=6,
                        legend_columns=6,
                        legend_y=-0.12,
                        expand_xlim=False,
                    )
                    _apply_sci_formatter(ax)
                    apply_static_distribution_compact_layout(fig, ax, right=0.95, top=0.90)
                else:
                    # Categorical bar chart (all categories)
                    all_counts = series.value_counts()
                    built_chart = _build_static_category_chart(all_counts, col)
                    if built_chart is None:
                        raise ValueError(f"Could not build categories chart for {col}")
                    plt.close(fig)
                    fig, ax = built_chart
                
                buf = io.BytesIO()
                savefig_kwargs = {"format": "png", "bbox_inches": "tight", "dpi": 150}
                savefig_kwargs["pad_inches"] = 0.0
                if not is_numeric:
                    savefig_kwargs["dpi"] = 120
                    savefig_kwargs["pad_inches"] = 0.0
                fig.savefig(buf, **savefig_kwargs)
                plt.close(fig)
                buf.seek(0)
                distribution_b64 = base64.b64encode(buf.read()).decode('utf-8')

                if pair_with_stl and is_numeric:
                    dist_figsize = get_export_chart_figsize("distribution", context="pdf")
                    dist_ratio = max(0.45, float(dist_figsize[1]) / max(1e-6, float(dist_figsize[0])))
                    # STL generator currently renders with figsize=(10, 7)
                    stl_ratio = 0.70
                    pair_gap_mm = 6.0

                    paired_plot_width_mm = _resolve_paired_plot_width(
                        _remaining_page_space(),
                        first_plot_ratio=dist_ratio,
                        second_plot_ratio=stl_ratio,
                        gap_mm=pair_gap_mm,
                        min_width_mm=118.0,
                        max_width_mm=float(img_width),
                    )
                    if paired_plot_width_mm is None:
                        pdf.add_page()
                        paired_plot_width_mm = _resolve_paired_plot_width(
                            _remaining_page_space(),
                            first_plot_ratio=dist_ratio,
                            second_plot_ratio=stl_ratio,
                            gap_mm=pair_gap_mm,
                            min_width_mm=118.0,
                            max_width_mm=float(img_width),
                        )
                    if paired_plot_width_mm is None:
                        paired_plot_width_mm = 118.0

                    distribution_rendered = _add_base64_plot(
                        distribution_b64,
                        min_space_mm=0.0,
                        gap_after_mm=pair_gap_mm,
                        width_mm=paired_plot_width_mm,
                        ensure_space=False,
                    )
                else:
                    distribution_rendered = _add_base64_plot(distribution_b64)
            except Exception as e:
                app.logger.debug("Distribution chart skipped in PDF for '%s': %s", col, e)
                
            # 4. STL DECOMPOSITION - last (for timeseries only)
            if stl_b64:
                _add_base64_plot(
                    stl_b64,
                    min_space_mm=0.0 if (pair_with_stl and distribution_rendered and paired_plot_width_mm is not None) else 96.0,
                    gap_after_mm=8.0,
                    width_mm=paired_plot_width_mm if (pair_with_stl and distribution_rendered and paired_plot_width_mm is not None) else None,
                    ensure_space=not (pair_with_stl and distribution_rendered and paired_plot_width_mm is not None),
                )
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
