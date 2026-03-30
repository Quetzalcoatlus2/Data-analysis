# mypy: ignore-errors
"""Standalone text / HTML processing for AI outputs.

Converts raw Gemini output into safe HTML, converts HTML to formatted text,
and handles emoji ↔ placeholder conversions for PDF compatibility.
"""
from __future__ import annotations

import html as htmllib
import re
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

CODE_FENCE_START_RE = re.compile(r'^\s*```(?:\w+)?\s*\n?', re.I | re.M)
CODE_FENCE_END_RE = re.compile(r'\n?\s*```\s*$', re.M)
HTML_ENTITY_TAG_RE = re.compile(r'&lt;/?[a-zA-Z]')
HTML_STRUCTURE_RE = re.compile(
    r'</?(h[1-6]|p|ul|ol|li|strong|em|b|i|br|table|thead|tbody|tr|th|td|a)\b',
    re.I,
)
HTML_BODY_TAG_RE = re.compile(r'</?\s*(html|body)[^>]*>', re.I)
SCRIPT_STYLE_RE = re.compile(r'<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>', re.I | re.S)
EVENT_ATTR_RE = re.compile(r'\s+on\w+\s*=\s*(".*?"|\'.*?\'|\w+)', re.I)
JS_PROTOCOL_RE = re.compile(r'javascript\s*:', re.I)
QUOTED_TERM_RE = re.compile(r"'([^'<>\n]{2,80})'")


# ---------------------------------------------------------------------------
# Emoji / placeholder maps
# ---------------------------------------------------------------------------

PLACEHOLDER_TO_EMOJI: dict[str, str] = {
    "[CHART]": "📊",
    "[SEARCH]": "🔍",
    "[TREND UP]": "📈",
    "[TREND DOWN]": "📉",
    "[TIME]": "⏱️",
    "[LINK]": "🔗",
    "[WARNING]": "⚠️",
    "[PREDICTION]": "🔮",
    "[TIP]": "💡",
    "[OK]": "✅",
    "[X]": "❌",
    "[TARGET]": "🎯",
    "[LIST]": "📋",
    "[NUM]": "🔢",
    "[NOTE]": "📝",
    "[ROCKET]": "🚀",
    "[STAR]": "⭐",
    "[HOT]": "🔥",
    "[MONEY]": "💰",
    "[DATE]": "📅",
    "[CLOCK]": "🕐",
}

EMOJI_REPLACEMENTS: dict[str, str] = {
    '📊': '\u25A4',     # Square with horizontal fill (like a chart)
    '🔍': '\u2315',     # Search icon
    '📈': '\u2197',     # North East Arrow
    '⏱️': '\u23F1',     # Stopwatch
    '🔗': '\u221E',     # Infinity
    '⚠️': '\u26A0',     # Warning Sign
    '🔮': '\u25C9',     # Fisheye
    '💡': '\u002A',     # Asterisk
    '✅': '\u2713',     # Check mark
    '❌': '\u2717',     # Cross mark
    '📉': '\u2198',     # South East Arrow
    '🎯': '\u25CE',     # Bullseye
    '📋': '\u25A4',     # Document
    '🔢': '#',        # Hash
    '📝': '\u270E',     # Pencil
    '🚀': '\u21E1',     # Upwards arrow with two tips
    '⭐': '\u2605',     # Black star
    '🔥': '\u263C',     # Sun
    '💰': '$',        # Dollar sign
    '📅': '\u25A3',     # Square with vertical fill
    '🕐': '\u231A',     # Watch
    '➡️': '->',
    '⬆️': '^',
    '⬇️': 'v',
    '✓': '\u2713',
    '•': '-',
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _restore_emoji_placeholders(text: str) -> str:
    if not text:
        return ""
    out = str(text)
    for tag, emoji in PLACEHOLDER_TO_EMOJI.items():
        out = re.sub(re.escape(tag), emoji, out, flags=re.IGNORECASE)
    out = re.sub(r'(?i)\bwarning\]', '⚠️', out)
    out = re.sub(r'(?i)\bchart\]', '📊', out)
    return out


def _trim_ai_garbage_tail(html_text: str) -> str:
    """Trim obvious model-generated HTML tutorial/code tail artifacts."""
    if not html_text:
        return html_text
    s = str(html_text)
    lower = s.lower()
    markers = [
        "html forms are fundamental for user interaction",
        "for single-line text entry",
        'input type="radio"',
        'input type="checkbox"',
        'input type="submit"',
        "html5 element",
    ]
    cut_points = [
        lower.find(m)
        for m in markers
        if lower.find(m) >= max(0, int(len(lower) * 0.35))
    ]
    if cut_points:
        s = s[:min(cut_points)].rstrip()
    return s


def _apply_text_segment_emphasis(html_text: str) -> str:
    """Apply emphasis to text nodes only, keeping HTML tags/attributes untouched."""
    if not html_text:
        return html_text

    parts = re.split(r'(<[^>]+>)', html_text)
    out_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith('<') and part.endswith('>'):
            out_parts.append(part)
            continue
        text_part = QUOTED_TERM_RE.sub(lambda m: f"<em>{m.group(1).strip()}</em>", part)
        out_parts.append(text_part)

    return ''.join(out_parts)


def sanitize_ai_html(raw: str) -> str:
    """Coerce Gemini output into a safe, clean HTML snippet."""
    if raw is None:
        return "<p></p>"
    s = str(raw)
    s = CODE_FENCE_START_RE.sub('', s)
    s = CODE_FENCE_END_RE.sub('', s)
    s = s.replace("```", "")
    if HTML_ENTITY_TAG_RE.search(s) or "&amp;lt;" in s.lower():
        try:
            for _ in range(3):
                newer = htmllib.unescape(s)
                if newer == s:
                    break
                s = newer
        except Exception:
            pass
    s = HTML_BODY_TAG_RE.sub('', s)
    s = SCRIPT_STYLE_RE.sub('', s)
    s = EVENT_ATTR_RE.sub('', s)
    s = JS_PROTOCOL_RE.sub('', s)
    s = _trim_ai_garbage_tail(s)
    s = s.strip()
    if not HTML_STRUCTURE_RE.search(s):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = "<p>" + "</p><p>".join(lines) + "</p>" if lines else "<p></p>"
    s = _restore_emoji_placeholders(s)
    s = _apply_text_segment_emphasis(s)
    return s


def convert_html_to_formatted_text(html: str) -> str:
    """Convert HTML to structured text preserving headings, nested lists, and emphasis."""
    if not html:
        return ""

    class _SummaryHtmlParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.lines: list[str] = []
            self.paragraph_parts: list[str] = []
            self.heading_parts: list[str] = []
            self.in_heading = False
            self.list_stack: list[dict[str, int | str]] = []
            self.li_parts: list[str] | None = None
            self.style_stack: list[str] = []
            self.in_cell = False
            self.cell_parts: list[str] = []
            self.row_cells: list[str] = []

        def _append_part(self, target: list[str], text: str) -> None:
            txt = re.sub(r'\s+', ' ', text)
            if not txt.strip():
                return
            txt = txt.strip()
            if "b" in self.style_stack:
                txt = f"**{txt}**"
            if "i" in self.style_stack:
                txt = f"*{txt}*"
            if target and not target[-1].endswith((" ", "\n", "(")) and not txt.startswith((".", ",", ":", ";", "?", "!", ")")):
                target.append(" ")
            target.append(txt)

        def _flush_paragraph(self, add_blank: bool = True) -> None:
            if self.in_heading or self.li_parts is not None or self.in_cell:
                return
            text = "".join(self.paragraph_parts).strip()
            self.paragraph_parts = []
            if text:
                self.lines.append(text)
                if add_blank:
                    self.lines.append("")

        def _flush_heading(self) -> None:
            text = "".join(self.heading_parts).strip()
            self.heading_parts = []
            self.in_heading = False
            if not text:
                return
            plain_len = len(re.sub(r'[*]', '', text))
            separator = '=' * min(max(plain_len, 3), 50)
            self.lines.extend(["", separator, text.upper(), separator, ""])

        def _flush_li(self) -> None:
            if self.li_parts is None:
                return
            text = "".join(self.li_parts).strip()
            self.li_parts = None
            if not text:
                return
            depth = max(0, len(self.list_stack) - 1)
            indent = "  " * depth
            prefix = "- "
            if self.list_stack and self.list_stack[-1].get("type") == "ol":
                idx_val = int(self.list_stack[-1].get("index", 1))
                prefix = f"{idx_val}. "
            self.lines.append(f"{indent}{prefix}{text}")

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            t = tag.lower()
            if t in ("strong", "b"):
                self.style_stack.append("b")
                return
            if t in ("em", "i"):
                self.style_stack.append("i")
                return
            if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self._flush_paragraph(add_blank=True)
                self._flush_li()
                self.in_heading = True
                self.heading_parts = []
                return
            if t == "p":
                self._flush_paragraph(add_blank=False)
                return
            if t in ("ul", "ol"):
                if self.li_parts is not None and "".join(self.li_parts).strip():
                    self._flush_li()
                self._flush_paragraph(add_blank=True)
                self.list_stack.append({"type": t, "index": 1})
                return
            if t == "li":
                if self.list_stack and self.list_stack[-1].get("type") == "ol":
                    cur = int(self.list_stack[-1].get("index", 1))
                    self.list_stack[-1]["index"] = cur
                self.li_parts = []
                return
            if t == "br":
                if self.li_parts is not None:
                    self.li_parts.append("\n")
                elif self.in_heading:
                    self.heading_parts.append(" ")
                else:
                    self._flush_paragraph(add_blank=False)
                return
            if t == "tr":
                self.row_cells = []
                return
            if t in ("td", "th"):
                self.in_cell = True
                self.cell_parts = []
                return

        def handle_endtag(self, tag: str) -> None:
            t = tag.lower()
            if t in ("strong", "b"):
                if "b" in self.style_stack:
                    self.style_stack.remove("b")
                return
            if t in ("em", "i"):
                if "i" in self.style_stack:
                    self.style_stack.remove("i")
                return
            if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self._flush_heading()
                return
            if t == "p":
                self._flush_paragraph(add_blank=True)
                return
            if t == "li":
                self._flush_li()
                if self.list_stack and self.list_stack[-1].get("type") == "ol":
                    cur = int(self.list_stack[-1].get("index", 1))
                    self.list_stack[-1]["index"] = cur + 1
                return
            if t in ("ul", "ol"):
                if self.li_parts is not None:
                    self._flush_li()
                if self.list_stack:
                    self.list_stack.pop()
                self.lines.append("")
                return
            if t in ("td", "th"):
                self.in_cell = False
                cell_text = "".join(self.cell_parts).strip()
                self.cell_parts = []
                if cell_text:
                    self.row_cells.append(cell_text)
                return
            if t == "tr":
                if self.row_cells:
                    self.lines.append(" | ".join(self.row_cells))
                self.row_cells = []
                return
            if t == "table":
                self.lines.append("")

        def handle_data(self, data: str) -> None:
            if not data or not data.strip():
                return
            if self.in_cell:
                self._append_part(self.cell_parts, data)
            elif self.li_parts is not None:
                self._append_part(self.li_parts, data)
            elif self.in_heading:
                self._append_part(self.heading_parts, data)
            else:
                self._append_part(self.paragraph_parts, data)

        def get_text(self) -> str:
            self._flush_li()
            self._flush_paragraph(add_blank=False)
            if self.in_heading:
                self._flush_heading()
            text = "\n".join(self.lines)
            normalized_lines: list[str] = []
            for raw_line in text.splitlines():
                match = re.match(r'^([ \t]*)(.*)$', raw_line)
                if not match:
                    normalized_lines.append(raw_line)
                    continue
                indent = match.group(1).replace('\t', '    ')
                body = re.sub(r'[ \t]+', ' ', match.group(2)).strip()
                if body:
                    html_token_count = len(re.findall(r'</?[a-zA-Z][^>]{0,40}>', body))
                    if html_token_count >= 3:
                        continue
                    if body.count('&lt;') >= 2 or body.count('&gt;') >= 2:
                        continue
                normalized_lines.append(f"{indent}{body}" if body else "")
            text = "\n".join(normalized_lines)
            text = re.sub(r'[ \t]+\n', '\n', text)
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            return text.strip()

    parser = _SummaryHtmlParser()
    parser.feed(htmllib.unescape(str(html)))
    parser.close()
    return parser.get_text()


def replace_emojis_for_pdf(text: str) -> str:
    """Replace emojis with text equivalents for PDF compatibility."""
    if not text:
        return ""
    result = text
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        result = result.replace(emoji, replacement)
    result = re.sub(r'[\U00010000-\U0010FFFF]', '', result)
    # Replaced stripping of basic symbols (\u2600-\u26FF and \u2700-\u27BF) so our replacements render!
    result = re.sub(r'  +', ' ', result)
    return result


__all__ = [
    # Regex constants (for backward compat)
    "CODE_FENCE_START_RE",
    "CODE_FENCE_END_RE",
    "HTML_ENTITY_TAG_RE",
    "HTML_STRUCTURE_RE",
    "HTML_BODY_TAG_RE",
    "SCRIPT_STYLE_RE",
    "EVENT_ATTR_RE",
    "JS_PROTOCOL_RE",
    "QUOTED_TERM_RE",
    # Data
    "PLACEHOLDER_TO_EMOJI",
    "EMOJI_REPLACEMENTS",
    # Functions
    "_restore_emoji_placeholders",
    "_trim_ai_garbage_tail",
    "_apply_text_segment_emphasis",
    "sanitize_ai_html",
    "convert_html_to_formatted_text",
    "replace_emojis_for_pdf",
]
