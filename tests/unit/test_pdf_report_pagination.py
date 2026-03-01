from __future__ import annotations

import re

from fpdf import FPDF

from data_analysis.reports.pdf_report import (
    _collapse_pdf_html_spacing,
    _estimate_wrapped_lines,
    _extract_first_context_text,
    _should_break_before_heading,
)


def _required_block_space(pdf: FPDF, first_text: str) -> float:
    lines = _estimate_wrapped_lines(pdf, first_text, width_mm=170.0)
    lines = max(1, min(6, lines))
    return 8.0 + 2.0 + (5.0 * float(lines)) + 5.0


def test_collapses_excess_blank_html_spacing_between_headings():
    raw = (
        "<!-- model:gemini-2.5-flash-lite -->"
        "<h3>Temporal Patterns &amp; Trends</h3>"
        "<p>Not applicable (no datetime index).</p>"
        "<p> </p><p>&nbsp;</p><p>\n\n</p>"
        "<br><br>"
        "    <h3>Correlations &amp; Relationships</h3>"
    )

    collapsed = _collapse_pdf_html_spacing(raw)

    assert "<!--" not in collapsed
    assert re.search(r"(?is)(?:<br\s*/?>\s*){2,}", collapsed) is None
    assert "</p><p></p><p></p>" not in collapsed
    assert "Not applicable (no datetime index)." in collapsed


def test_breaks_when_only_heading_plus_one_paragraph_fit_near_bottom():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)

    first_text = "Not applicable (no datetime index)."
    required = _required_block_space(pdf, first_text)

    should_break = _should_break_before_heading(
        y_mm=120.0,
        remaining_mm=required - 0.1,
        required_mm=required,
    )

    assert should_break is True


def test_does_not_break_mid_page_when_space_is_ample():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)

    first_text = "While specific correlation coefficients are not provided."
    required = _required_block_space(pdf, first_text)

    assert _should_break_before_heading(
        y_mm=110.0,
        remaining_mm=required + 8.0,
        required_mm=required,
    ) is False


def test_extract_first_context_text_prefers_earliest_meaningful_block():
    body_html = "<p>  </p><li><strong>Strong Positive Correlations</strong></li><p>Fallback paragraph</p>"

    text, is_list = _extract_first_context_text(body_html)

    assert "Strong Positive Correlations" in text
    assert is_list is True
