import os
import io
import base64
import zipfile
from flask import Flask, request, render_template, redirect, url_for, flash, after_this_request, make_response, jsonify

from datetime import datetime, timedelta
import pandas as pd
from werkzeug.utils import secure_filename

# --- Add these two lines ---
try:
    from dotenv import load_dotenv
    load_dotenv(".env.public")   # committed, non-sensitive defaults
    load_dotenv(".env")          # local, secrets override
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AI & ML Libraries
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
import hashlib
import uuid
import json  # <— add
import numpy as np  # add
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # add
from statsmodels.tsa.seasonal import STL  # <-- add this import
from collections import OrderedDict  # add
import re
import html as htmllib  # add
import math  # add

# Optional security/rate limiting (enabled via env)
try:
    from flask_limiter import Limiter  # add
    from flask_limiter.util import get_remote_address  # add
except Exception:
    Limiter = None
try:
    from flask_talisman import Talisman  # add
except Exception:
    Talisman = None

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# New: dedicated subfolder for hashed uploads
app.config['UPLOADS_SUBDIR'] = os.getenv("UPLOADS_SUBDIR", "uploaded")
app.config['UPLOADS_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOADS_SUBDIR'])
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY") or "dev-secret-change-me"
# Enable immediate deletion by default; override with .env
app.config['DELETE_UPLOADED_AFTER_PROCESSING'] = os.getenv("DELETE_UPLOADED_AFTER_PROCESSING", "true").strip().lower() in ("1", "true", "yes", "on")
# Optional retention cleanup window (set in .env to auto-clean stragglers)
if "UPLOAD_RETENTION_DAYS" in os.environ:
    try:
        app.config['UPLOAD_RETENTION_DAYS'] = int(os.getenv("UPLOAD_RETENTION_DAYS"))
    except Exception:
        app.logger.warning("Invalid UPLOAD_RETENTION_DAYS; ignoring")

# Defaults for cache and analysis settings
app.config.setdefault('MAX_CACHE_ITEMS', int(os.getenv("MAX_CACHE_ITEMS", "6")))
app.config.setdefault('DEFAULT_FORECAST_STEPS', int(os.getenv("DEFAULT_FORECAST_STEPS", "30")))
app.config.setdefault('DEFAULT_CONTAMINATION', float(os.getenv("DEFAULT_CONTAMINATION", "0.02")))
app.config.setdefault('PLOTLY_TAIL', int(os.getenv("PLOTLY_TAIL", "800")))
# Add AI call behavior knobs
app.config.setdefault('AI_TIMEOUT_SECONDS', int(os.getenv("AI_TIMEOUT_SECONDS", "30")))
app.config.setdefault('AI_RETRY_ATTEMPTS', int(os.getenv("AI_RETRY_ATTEMPTS", "2")))
app.config.setdefault('AI_RETRY_BACKOFF_SECONDS', float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "2.0")))

import logging
import re
from logging.handlers import RotatingFileHandler
import time  # add

# Optional: disable CLI coloring globally (helps Werkzeug/Click)
os.environ.setdefault("NO_COLOR", "1")

class StripAnsiFormatter(logging.Formatter):
    _ansi = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    def format(self, record):
        s = super().format(record)
        return self._ansi.sub('', s)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# File handler (rotating) with ANSI stripping
file_handler = RotatingFileHandler("app.log", maxBytes=2_000_000, backupCount=3)
file_handler.setLevel(log_level)
file_handler.setFormatter(StripAnsiFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

# Console handler (stdout) for on-screen logs
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter("%(message)s"))  # simple, readable in terminal

# Attach once
for h in (file_handler, console_handler):
    if not any(type(x) is type(h) for x in app.logger.handlers):
        app.logger.addHandler(h)

app.logger.setLevel(log_level)

# Also forward Werkzeug logs to both handlers (once)
werk = logging.getLogger("werkzeug")
werk.setLevel(log_level)
for h in (file_handler, console_handler):
    if not any(type(x) is type(h) for x in werk.handlers):
        werk.addHandler(h)

# --- AI model selection & caching ---
DEFAULT_AI_MODEL = (
    os.getenv("GENAI_MODEL")
    or os.getenv("GOOGLE_MODEL")
    or "models/gemini-2.5-pro-preview-06-05"   # default (typo-safe normalization below)
)
MODEL_CACHE = {}          # name -> GenerativeModel instance
CURRENT_MODEL_NAME = None # the last-resolved working model name

def _normalize_model_aliases(name: str) -> list[str]:
    """
    Produce a robust list of candidate identifiers for a requested model name:
    - fix common 'odels/' typo to 'models/'
    - try both with and without 'models/' prefix
    - include a couple stable fallbacks at the end
    """
    if not name:
        return []
    n = name.strip()
    if n.startswith("odels/"):  # fix missing 'm'
        n = "m" + n
    candidates = [n]
    # try toggling the 'models/' prefix
    if n.startswith("models/"):
        candidates.append(n.replace("models/", "", 1))
    else:
        candidates.append("models/" + n)
    # stable fallbacks (only if not already present)
    for fb in ("gemini-1.5-pro", "models/gemini-1.5-pro", "gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-pro", "models/gemini-pro"):
        if fb not in candidates:
            candidates.append(fb)
    # de-dup preserving order
    seen = set(); out = []
    for c in candidates:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

def _make_model(name: str):
    m = genai.GenerativeModel(name)
    # quick ping to confirm it actually responds
    resp = m.generate_content("OK", request_options={"timeout": 15}, generation_config={"response_mime_type": "text/plain"})
    _ = _extract_text_from_gemini_response(resp)  # ignore the content
    return m

def get_or_create_model(preferred: str | None = None):
    """
    Return a working GenerativeModel instance for the preferred name, trying
    normalized aliases and a few stable fallbacks. Caches instances.
    """
    global CURRENT_MODEL_NAME
    if not AI_ENABLED:
        raise RuntimeError("AI is disabled or not configured.")
    for nm in _normalize_model_aliases(preferred or DEFAULT_AI_MODEL):
        try:
            if nm in MODEL_CACHE:
                CURRENT_MODEL_NAME = nm
                return MODEL_CACHE[nm]
            m = _make_model(nm)
            MODEL_CACHE[nm] = m
            CURRENT_MODEL_NAME = nm
            app.logger.info("Using Gemini model: %s", nm)
            return m
        except Exception as e:
            app.logger.warning("Model candidate failed (%s): %s", nm, e)
            continue
    raise RuntimeError("No working Gemini model available. Check API key/network or try a different model.")

# --- AI Configuration (lazy model init) ---
app.logger.info("Attempting to configure AI...")

def configure_ai():
    global model, AI_ENABLED
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        AI_ENABLED = True
        # lazily initialize the default model; if it fails, defer to first call
        try:
            model = get_or_create_model(DEFAULT_AI_MODEL)
        except Exception as e:
            model = None
            app.logger.warning("Deferred model init; will attempt on first request. %s", e)
        app.logger.info("AI configured successfully.")
    except Exception:
        app.logger.exception("AI configuration failed")
        model = None
        AI_ENABLED = False

configure_ai()

def _call_gemini(prompt: str, file_asset=None, *, timeout: int | None = None, retries: int | None = None, generation_config: dict | None = None):
    """
    Thin wrapper around Gemini calls that supports optional file context,
    retry/backoff, a request timeout, and generation config, returning the raw response object.
    """
    if not AI_ENABLED or model is None:
        raise RuntimeError("AI is disabled or not configured.")
    attempts = int(app.config.get('AI_RETRY_ATTEMPTS', 2)) if retries is None else int(retries)
    backoff = float(app.config.get('AI_RETRY_BACKOFF_SECONDS', 2.0))
    timeout = int(app.config.get('AI_TIMEOUT_SECONDS', 30)) if timeout is None else int(timeout)
    last_err = None
    for i in range(max(1, attempts + 1)):
        try:
            content = [file_asset, prompt] if file_asset is not None else [prompt]
            resp = model.generate_content(
                content,
                request_options={"timeout": timeout},
                generation_config=(generation_config or {})
            )
            # If blocked, bubble up to trigger fallback
            try:
                pf = getattr(resp, "prompt_feedback", None)
                if pf and getattr(pf, "block_reason", None):
                    br = pf.block_reason
                    raise RuntimeError(f"Content blocked: {getattr(br, 'name', br)}")
            except Exception:
                pass
            return resp
        except Exception as e:
            last_err = e
            if i < attempts:
                try:
                    time.sleep(backoff * (i + 1))
                except Exception:
                    pass
                continue
            raise last_err

# Replace plain dict with tiny LRU using OrderedDict
class TinyLRU(OrderedDict):
    def __init__(self, max_items=6):
        super().__init__()
        self.max_items = max_items
    def get(self, key, default=None):
        if key in self:
            val = super().pop(key)
            super().__setitem__(key, val)  # move to end (recently used)
            return val
        return default
    def set(self, key, value):
        if key in self:
            super().pop(key)
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            self.popitem(last=False)

DATAFRAME_CACHE = TinyLRU(max_items=app.config['MAX_CACHE_ITEMS'])  # key: hashed filename -> DataFrame
NAME_MAP_PATH = os.path.join(UPLOAD_FOLDER, "_name_map.json")  # <— add
app.config['AI_FULL_UPLOAD_MAX_MB'] = 5  # only upload full file if <= 5 MB
AI_FILE_MAP = {}  # key: hashed filename -> genai uploaded file handle
ORIGINAL_NAME_MAP = {}  # ensure default global exists even if name-map file is absent
AI_SUMMARY_CACHE = {}  # key: filename -> HTML snippet for server-side download

# Ensure rotating file handler is attached (avoid duplicates)
if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
    app.logger.addHandler(file_handler)

# also capture werkzeug (request) logs into the same file
werk = logging.getLogger("werkzeug")
werk.setLevel(log_level)
if not any(isinstance(h, RotatingFileHandler) for h in werk.handlers):
    werk.addHandler(file_handler)

def _load_name_map():
    global ORIGINAL_NAME_MAP
    try:
        if os.path.exists(NAME_MAP_PATH):
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                ORIGINAL_NAME_MAP = json.load(f)
    except Exception as e:
        app.logger.warning("Name map load warning: %s", e)

def _save_name_map():
    try:
        with open(NAME_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(ORIGINAL_NAME_MAP, f, ensure_ascii=False, indent=2)
    except Exception as e:
        app.logger.warning("Name map save warning: %s", e)

def _safe_delete(path, retries=3, delay=0.2):
    """Delete a file with small retries to tolerate transient locks (e.g., OneDrive/AV)."""
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception as e:
            app.logger.warning("Delete failed (%s), attempt %d/%d: %s", path, i + 1, retries, e)
            time.sleep(delay)
    return False

# Ensure folders and load map at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)  # ensure uploads subfolder exists
_load_name_map()

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

# Only treat 40-hex digest filenames as app-managed uploads
HASHED_UPLOAD_RE = re.compile(r'^[a-f0-9]{40}\.(txt|csv|xlsx|json)$', re.IGNORECASE)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_ai_html(raw: str) -> str:
    """Coerce Gemini output into a safe, clean HTML snippet.
    - strips Markdown code fences (``` and ```html)
    - unescapes &lt;...&gt; once if needed
    - removes <html>/<body>, <script>/<style>, on* attributes, and 'javascript:' URLs
    - if no HTML structure remains, wraps lines into <p>...</p>
    """
    if raw is None:
        return "<p></p>"
    s = str(raw)
    # strip code fences (opening like ``` or ```html and closing ```)
    s = re.sub(r'^\s*```(?:\w+)?\s*\n?', '', s, flags=re.I | re.M)
    s = re.sub(r'\n?\s*```\s*$', '', s, flags=re.M)
    s = s.replace("```", "")
    # unescape HTML entities if looks escaped
    if re.search(r'&lt;/?[a-zA-Z]', s):
        try:
            s = htmllib.unescape(s)
        except Exception:
            pass
    # drop html/body wrappers
    s = re.sub(r'</?\s*(html|body)[^>]*>', '', s, flags=re.I)
    # drop script/style blocks
    s = re.sub(r'<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>', '', s, flags=re.I | re.S)
    # strip event handler attributes and javascript: URLs
    s = re.sub(r'\s+on\w+\s*=\s*(".*?"|\'.*?\'|\w+)', '', s, flags=re.I)
    s = re.sub(r'javascript\s*:', '', s, flags=re.I)
    s = s.strip()
    # if no recognizable HTML tags, wrap lines into paragraphs
    if not re.search(r'</?(h[1-6]|p|ul|ol|li|strong|em|b|i|br|table|thead|tbody|tr|th|td|a)\b', s, re.I):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = "<p>" + "</p><p>".join(lines) + "</p>" if lines else "<p></p>"
    return s

def html_to_pdf_bytes(html: str) -> bytes:
    """
    Convert an HTML string to PDF bytes using the best-available backend.
    Tries WeasyPrint, then pdfkit (wkhtmltopdf), then xhtml2pdf; raises if none are available.
    """
    # 1) WeasyPrint (pure-Python API; requires cairo/pango libs in environment)
    try:
        import weasyprint  # type: ignore
        return weasyprint.HTML(string=html).write_pdf()
    except Exception:
        pass

    # 2) pdfkit (requires wkhtmltopdf installed on the system PATH)
    try:
        import pdfkit  # type: ignore
        return pdfkit.from_string(html, False)
    except Exception:
        pass

    # 3) xhtml2pdf (pure-Python fallback)
    try:
        from xhtml2pdf import pisa  # type: ignore
        buf = io.BytesIO()
        # pisa expects a text stream for the HTML source
        status = pisa.CreatePDF(io.StringIO(html), dest=buf)
        if getattr(status, "err", 0):
            raise RuntimeError("xhtml2pdf failed to render HTML to PDF")
        return buf.getvalue()
    except Exception:
        pass

    raise RuntimeError("No HTML-to-PDF backend available. Install one of: weasyprint, pdfkit (with wkhtmltopdf), or xhtml2pdf.")

def _extract_text_from_gemini_response(resp) -> str:
    """
    Robustly extract plain text from a Gemini response.
    Falls back to concatenating candidate parts if .text isn't available.
    Returns '' if nothing textual is found.
    """
    # Try the SDK convenience accessor first
    try:
        t = getattr(resp, "text", None)
        if t:
            return str(t)
    except Exception as e:
        app.logger.warning("Gemini response.text accessor failed: %s", e)

    # Try candidates -> content.parts
    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            texts = []
            for part in parts:
                if isinstance(part, dict):
                    if "text" in part and part["text"]:
                        texts.append(str(part["text"]))
                else:
                    pt = getattr(part, "text", None)
                    if pt:
                        texts.append(str(pt))
            if texts:
                return "\n".join(texts)
    except Exception as e:
        app.logger.warning("Gemini candidate parts extraction failed: %s", e)

    return ""

def get_ai_summary(dataframe_description):
    app.logger.debug("Calling get_ai_summary")
    if not AI_ENABLED or model is None:
        app.logger.debug("AI disabled; skipping get_ai_summary")
        return "AI analysis is disabled. Please check your Google API key and terminal for configuration errors."
    try:
        app.logger.debug("Preparing to call Google API for summary")
        prompt = f"""
You are an expert data analyst. Given the following dataset profile, write a clear, rich, and structured HTML summary...
{dataframe_description}
"""
        response = model.generate_content(prompt)
        app.logger.debug("AI summary call successful")

        # Safety blocks (unchanged)
        if hasattr(response, "prompt_feedback"):
            pf = getattr(response, "prompt_feedback", None)
            if pf and getattr(pf, "block_reason", None):
                block_reason = getattr(pf, "block_reason", None)
                app.logger.warning("AI analysis blocked: %s", block_reason)
                return f"AI analysis was blocked by the content filter. Reason: {block_reason}"

        # NEW: safe extraction
        text = _extract_text_from_gemini_response(response).strip()
        if not text:
            # Attempt to log finish_reason for debugging
            fr = None
            try:
                cs = getattr(response, "candidates", None) or []
                if cs:
                    fr = getattr(cs[0], "finish_reason", None)
            except Exception:
                pass
            raise RuntimeError(f"Empty AI response (finish_reason={fr})")

        app.logger.debug("Successfully got AI summary")
        return sanitize_ai_html(text)
    except Exception as e:
        app.logger.exception("AI summary call failed")
        return f"An error occurred during AI analysis. Check the terminal for more details. Error: {e}"

def get_ai_answer(dataframe, question):
    """Generates a specific answer to a user's question about the dataframe."""
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."

    # Give the AI context with structure and a summary (prompt omitted here for brevity)
    try:
        # Build a concise prompt with dataset context and the user's question
        df_desc = describe_for_ai(dataframe)
        prompt = f"""
You are a senior data scientist. Answer the user's question about the dataset clearly and precisely.
Respond strictly in HTML (no Markdown), using tags like <p>, <ul><li>, <table><thead><tbody><tr><th><td>, <strong>, and <em>.
Cite concrete numbers or ranges from the provided context when relevant. If something is uncertain, say so briefly.

Context:
{df_desc}

Question:
{question}
""".strip()

        response = model.generate_content(prompt)
        # NEW: safe extraction
        text = _extract_text_from_gemini_response(response).strip()
        if not text:
            fr = None
            try:
                cs = getattr(response, "candidates", None) or []
                if cs:
                    fr = getattr(cs[0], "finish_reason", None)
            except Exception:
                pass
            raise RuntimeError(f"Empty AI response (finish_reason={fr})")
        return sanitize_ai_html(text)
    except Exception as e:
        return f"An error occurred while generating the AI answer. Error: {e}"

def get_ai_summary_with_file(df, file_asset=None, extra_context: str = ""):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."

    # Build long-form prompt for the model (HTML-only response)
    try:
        df_desc = describe_for_ai(df)
    except Exception:
        df_desc = ""
    prompt = (
        "You are an expert data analyst. Provide a clear, rich, and structured HTML summary of the dataset. "
        "Respond strictly in HTML (no Markdown) using tags like <p>, <ul><li>, <table><thead><tbody><tr><th><td>, "
        "<strong>, and <em>. Keep it concise but informative; mention notable trends, missingness, and caveats.\n\n"
        "Context:\n" + df_desc + ("\n\nAdditional context:\n" + extra_context if extra_context else "")
    )

    gen_cfg = {
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        # Encourage the SDK to produce HTML directly when supported
        "response_mime_type": "text/html",
    }

    try:
        resp = _call_gemini(prompt, file_asset=file_asset, generation_config=gen_cfg)
        text = _extract_text_from_gemini_response(resp).strip()
        if not text:
            # one-shot simplified retry with smaller output to coax a textual reply
            try:
                simple_cfg = {"max_output_tokens": 512, "temperature": 0.2, "response_mime_type": "text/html"}
                simple_prompt = "Provide a concise HTML summary of the dataset using <p> and <ul><li> only."
                simple = _call_gemini(simple_prompt + "\n\n" + extra_context, file_asset=file_asset, generation_config=simple_cfg)
                text2 = _extract_text_from_gemini_response(simple).strip()
                if text2:
                    return sanitize_ai_html(text2)
            except Exception:
                pass
            # If still empty, fall back
            raise RuntimeError("Empty AI response")
        return sanitize_ai_html(text)
    except Exception as e:
        app.logger.warning("AI summary failed, falling back. Error: %s", e)
        return offline_answer(df, "summary", error=e)
    
def get_ai_answer_with_file(df: pd.DataFrame, question: str, file_asset=None) -> str:
    """
    Answer a user's question about the dataset.
    - Uses the uploaded file as context if available (file_asset).
    - Returns sanitized HTML.
    - Falls back to an offline deterministic answer on error or empty AI response.
    """
    try:
        if not AI_ENABLED or model is None:
            return offline_answer(df, question, error="AI disabled.")

        # Build a concise but structured prompt with context
        df_desc = describe_for_ai(df)
        prompt = f"""
You are a senior data scientist. Answer the user's question about the dataset clearly and precisely.
Respond strictly in HTML (no Markdown), using tags like <p>, <ul><li>, <table><thead><tbody><tr><th><td>, <strong>, and <em>.
Cite concrete numbers or ranges from the provided context when relevant. If something is uncertain, say so briefly.

Context:
{df_desc}

Question:
{question}
""".strip()

        # First attempt
        resp = _call_gemini(prompt, file_asset=file_asset, generation_config={
            "max_output_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "response_mime_type": "text/html",
        })
        text = _extract_text_from_gemini_response(resp).strip()

        # If we got no textual parts, try a simplified one-shot retry
        if not text:
            try:
                simple = _call_gemini(
                    f"Answer briefly in HTML (<p>, <ul><li>) only: {question}\n\nContext:\n{df_desc}",
                    file_asset=file_asset,
                    generation_config={"max_output_tokens": 512, "temperature": 0.2, "response_mime_type": "text/html"},
                )
                text2 = _extract_text_from_gemini_response(simple).strip()
                if text2:
                    return sanitize_ai_html(text2)
            except Exception:
                pass
            # If still empty, fall back offline
            return offline_answer(df, question, error="Empty AI response")

        return sanitize_ai_html(text)

    except Exception as e:
        app.logger.warning("AI Q&A failed; falling back. Error: %s", e)
        return offline_answer(df, question, error=e)

def get_or_cache_ai_summary_for(filename: str, df: pd.DataFrame, extra_context: str = "") -> str:
    """
    Return the AI summary for this file, generating it once and caching it.
    Never regenerates on subsequent calls to avoid API limits.
    """
    try:
        cached = AI_SUMMARY_CACHE.get(filename)
        if isinstance(cached, str) and cached.strip():
            return cached
        file_asset = AI_FILE_MAP.get(filename) if 'AI_FILE_MAP' in globals() else None
        ai_html = get_ai_summary_with_file(df, file_asset=file_asset, extra_context=extra_context)
        AI_SUMMARY_CACHE[filename] = ai_html
        return ai_html
    except Exception as e:
        return f"<p>AI summary unavailable: {e}</p>"

def generate_plot(data, title, xlabel, ylabel, anomalies_idx=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax, label='History', color='tab:blue', lw=2)
    if anomalies_idx is not None and len(anomalies_idx):
        aligned = data.loc[data.index.intersection(anomalies_idx)]
        ax.scatter(aligned.index, aligned.values, color='red', s=18, zorder=5, label='Anomaly')
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8'); plt.close(fig); return img

def _ensure_plot_dicts(items):
    """
    Normalize plot items into [{'title': str, 'img': base64}, ...]
    Accepts dicts, tuples, or raw base64 strings; drops invalid entries.
    """
    out = []
    if not items:
        return out
    for p in items:
        try:
            if isinstance(p, dict) and 'img' in p:
                title = p.get('title', '')
                if not isinstance(title, str):
                    title = '' if title is None else str(title)
                out.append({'title': title, 'img': p['img']})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                title, img = p[0], p[1]
                out.append({'title': '' if title is None else str(title), 'img': img})
            elif isinstance(p, str):
                # Assume raw base64 image; synthesize empty title
                out.append({'title': '', 'img': p})
        except Exception:
            # Skip invalid entries quietly
            continue
    return out

# ADD: STL decomposition helper (static PNG like other plots)
def generate_stl_plot(series: pd.Series, title: str, seasonal_period: int):
    try:
        s = normalize_timeseries(series)
        # guard: need enough points to estimate trend/seasonal cleanly
        if s is None or len(s) < max(28, seasonal_period * 2):
            return None
        res = STL(s.astype(float), period=int(seasonal_period), robust=True).fit()

        fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(s.index, s.values, color='tab:blue', lw=1.6); axes[0].set_ylabel("Observed"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(res.trend.index, res.trend.values, color='tab:orange', lw=1.6); axes[1].set_ylabel("Trend"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(res.seasonal.index, res.seasonal.values, color='tab:green', lw=1.6); axes[2].set_ylabel("Seasonal"); axes[2].grid(True, alpha=0.3)
        axes[3].plot(res.resid.index, res.resid.values, color='tab:red', lw=1.6); axes[3].axhline(0, color='gray', ls=':', lw=1)
        axes[3].set_ylabel("Residual"); axes[3].grid(True, alpha=0.3)
        axes[0].set_title(title)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None

# ADD: helpers to build a proper future index and a clearer forecast plot
def _infer_future_index(idx, steps):
    # Datetime: use explicit freq or infer; fallback to median delta
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is not None:
            offset = pd.tseries.frequencies.to_offset(freq)
        else:
            diffs = pd.Series(idx).diff().dropna()
            step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
            offset = pd.tseries.frequencies.to_offset(step)
        start = idx[-1] + offset
        return pd.date_range(start=start, periods=steps, freq=offset)
    # Numeric/other: extend by median step or 1
    try:
        ser_idx = pd.Series(idx.astype('int64') if hasattr(idx, 'astype') else list(idx))
    except Exception:
        ser_idx = pd.Series(range(len(idx)))
    diffs = ser_idx.diff().dropna()
    step = int(diffs.median()) if not diffs.empty else 1
    last = int(ser_idx.iloc[-1])
    return pd.Index([last + step * (i + 1) for i in range(steps)])

# add: simple seasonality inference for DatetimeIndex
def _infer_seasonal_period(idx, min_seasons=2):
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    freq = (idx.freqstr or pd.infer_freq(idx)) or ""
    f = freq.upper()
    # heuristics for common granularities
    if f.startswith("H"):  # hourly
        period = 24
    elif f.startswith("T") or f.startswith("MIN"):  # minutely
        period = 60
    elif f.startswith("S"):  # secondly
        period = 60
    elif f.startswith("D"):  # daily
        period = 7
    elif f.startswith("W"):  # weekly
        period = 52
    elif f.startswith("M"):  # monthly
        period = 12
    elif f.startswith("Q"):  # quarterly
        period = 4
    else:
        period = None
    # ensure enough data to estimate seasonality
    try:
        n = len(idx)
        if period is None or n < period * min_seasons:
            return None
        return period
    except Exception:
        return None

# add/replace: recent-slope forecaster to avoid flat futures
def _recent_slope_forecast(series, steps, window=None, damping=None):
    """
    Forecast using robust recent slope.
    - slope blends linear-regression slope with median step
    - optional damping (None or in (0,1)); default None for clearer trend
    """
    y = series.dropna()
    n = len(y)
    future_idx = _infer_future_index(series.index, steps)

    if n < 3:
        fc_mean = pd.Series([y.iloc[-1]] * steps, index=future_idx)
        ci = pd.concat([fc_mean, fc_mean], axis=1)
        ci.columns = ['lower', 'upper']
        return fc_mean, ci

    w = window or min(max(20, n // 5), n)
    y_win = y.iloc[-w:]

    # Linear trend + robust median step
    x = np.arange(len(y_win), dtype=float)
    slope_lr, intercept = np.polyfit(x, y_win.values, 1)
    diffs = np.diff(y_win.values)
    med_diff = float(np.median(diffs)) if len(diffs) else 0.0

    # Combine and enforce a minimum magnitude relative to recent steps
    slope = 0.5 * float(slope_lr) + 0.5 * med_diff
    baseline = max(abs(med_diff), 1e-12)
    min_mag = 0.25 * baseline
    if abs(slope) < min_mag:
        slope = np.sign(med_diff) * min_mag

    k = np.arange(1, steps + 1, dtype=float)
    if damping is not None and 0 < damping < 1:
        phi = float(damping)
        incr = (1 - np.power(phi, k)) / (1 - phi) * slope
        fc_vals = y.iloc[-1] + incr
    else:
        fc_vals = y.iloc[-1] + slope * k

    fc_mean = pd.Series(fc_vals, index=future_idx)

    # CI from residuals in the window (robust-ish)
    resid = y_win.values - (slope_lr * x + intercept)
    resid_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else float(np.nanstd(y_win.values, ddof=1))
    lower = fc_mean - 1.96 * resid_std
    upper = fc_mean + 1.96 * resid_std
    ci = pd.concat([lower, upper], axis=1)
    ci.columns = ['lower', 'upper']
    return fc_mean, ci

def generate_forecast_plot(history, forecast_series, title, xlabel, ylabel, conf_int=None, history_tail=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Use full history by default; limit only if a positive tail is provided
    history_tail_series = history if not history_tail or history_tail <= 0 else history.tail(history_tail)
    history_tail_series.plot(ax=ax, label='History', color='tab:blue', linewidth=1.8)

    forecast_series.plot(
        ax=ax,
        label='Forecast',
        linestyle='--',
        color='orangered',
        linewidth=3,
        marker='o',
        markersize=3,
        zorder=3
    )

    # Confidence interval shading
    if conf_int is not None:
        try:
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]
            lower.index = forecast_series.index
            upper.index = forecast_series.index
            ax.fill_between(
                forecast_series.index, lower, upper,
                color='orangered', alpha=0.22, label='95% CI', zorder=2
            )
        except Exception:
            pass

    # Clear forecast region separation
    try:
        split_x = history.index[-1]
        ax.axvline(split_x, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
        ax.axvspan(split_x, forecast_series.index[-1], color='orange', alpha=0.08, zorder=0)
    except Exception:
        pass

    # Focus y-limits on history tail + forecast (ignore very wide CI that can flatten visuals)
    try:
        y_stack = pd.concat([history_tail_series, forecast_series]).astype(float)
        y_min = float(np.nanmin(y_stack.values))
        y_max = float(np.nanmax(y_stack.values))
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            ax.set_ylim(y_min - pad, y_max + pad)
    except Exception:
        pass

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def read_csv_fallback(path, **kwargs):
    last_err = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # If it's not a decode error, re-raise
            raise
    # Final lenient attempt for pandas>=2: replace undecodable bytes
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except TypeError:
        pass
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode with common encodings")

def read_json_fallback(path):
    last_err = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return pd.read_json(f, orient="records")
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except ValueError:
            # Try JSON Lines if normal JSON fails
            try:
                with open(path, "r", encoding=enc, errors="strict") as f:
                    return pd.read_json(f, lines=True)
            except Exception:
                continue
    # Final lenient attempt
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return pd.read_json(f, orient="records")
    except ValueError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return pd.read_json(f, lines=True)
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode JSON with common encodings")

def _cleanup_uploads_if_configured():
    days = app.config.get('UPLOAD_RETENTION_DAYS')
    if not days:
        return
    cutoff = datetime.now() - timedelta(days=days)
    uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
    try:
        if not os.path.isdir(uploads_dir):
            return
        for name in os.listdir(uploads_dir):
            path = os.path.join(uploads_dir, name)
            if not os.path.isfile(path):
                continue
            # Only delete app-hashed uploads; leave user files alone
            if not HASHED_UPLOAD_RE.match(name):
                continue
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                try:
                    os.remove(path)
                except Exception as e:
                    app.logger.warning("Cleanup warning: %s", e)
    except Exception as e:
        app.logger.warning("Cleanup scan failed: %s", e)

# Detect if a forecast is too linear (nearly a straight line)
def _is_too_linear(series_like):
    try:
        y = pd.Series(series_like).astype(float).values
        x = np.arange(len(y), dtype=float)
        if len(y) < 3:
            return False
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return r2 > 0.985  # threshold for "too straight"
    except Exception:
        return False

# Bootstrap a "natural-looking" forecast path from recent increments
def _bootstrap_natural_path(series, steps, window=None, base_slope=None, n_samples=200, q_low=0.1, q_high=0.9):
    y = series.dropna()
    n = len(y)
    if n < 5:
        return _recent_slope_forecast(series, steps, window=None, damping=None)

    w = window or min(max(30, n // 4), n)
    y_win = y.iloc[-w:].astype(float)
    diffs = np.diff(y_win.values)
    if len(diffs) < 3 or np.allclose(diffs, 0):
        return _recent_slope_forecast(series, steps, window=None, damping=None)

    # robust center/scale for winsorization
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) + 1e-12
    lo_clip = med - 3.0 * mad
    hi_clip = med + 3.0 * mad

    # mild trend bias so the direction follows recent movement
    bias = base_slope if base_slope is not None else med
    bias_weight = 0.3  # small bias to avoid runaway

    future_idx = _infer_future_index(series.index, steps)
    paths = np.empty((n_samples, steps), dtype=float)

    rng = np.random.default_rng()
    for i in range(n_samples):
        incs = rng.choice(diffs, size=steps, replace=True).astype(float)
        # winsorize outliers
        incs = np.clip(incs, lo_clip, hi_clip)
        # add slight bias
        incs = incs + bias_weight * bias
        # cumulative path from last observed value
        path = y.iloc[-1] + np.cumsum(incs)
        paths[i, :] = path

    # aggregate to median and quantiles
    median_path = np.median(paths, axis=0)
    lower_path = np.quantile(paths, q_low, axis=0)
    upper_path = np.quantile(paths, q_high, axis=0)

    median_series = pd.Series(median_path, index=future_idx)
    lower_series = pd.Series(lower_path, index=future_idx)
    upper_series = pd.Series(upper_path, index=future_idx)
    conf_df = pd.concat([lower_series, upper_series], axis=1)
    conf_df.columns = ['lower', 'upper']
    return median_series, conf_df

def detect_anomalies(series: pd.Series, contamination=0.02):
    """
    IsolationForest-based anomaly detection.
    Returns a tuple: (anomalies_index: pd.Index, anomaly_scores: np.ndarray|None)
    - anomalies_index: index labels corresponding to detected anomalies
    - anomaly_scores: anomaly scores for those points (higher = more anomalous), or None if unavailable
    """
    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
    except Exception:
        return pd.Index([]), None

    if s.empty or s.shape[0] < 20:
        return pd.Index([]), None

    try:
        iso = IsolationForest(n_estimators=200, contamination=float(contamination), random_state=42)
        X = s.values.reshape(-1, 1)
        preds = iso.fit_predict(X)  # -1 = anomaly
        # Decision function: lower = more abnormal; invert so higher is more anomalous
        try:
            scores_all = -iso.decision_function(X)
        except Exception:
            scores_all = None

        mask = preds == -1
        anomalies_idx = s.index[mask]
        if scores_all is not None:
            scores = scores_all[mask]
        else:
            scores = None
        return anomalies_idx, scores
    except Exception:
        return pd.Index([]), None

def normalize_timeseries(series: pd.Series):
    s = series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex) or s.empty:
        return s
    freq = s.index.freq or pd.infer_freq(s.index)
    if freq is None:
        # fallback to median delta
        diffs = pd.Series(s.index).diff().dropna()
        step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
        freq = pd.tseries.frequencies.to_offset(step)
    s = s.asfreq(freq)
    # small-gap interpolation only
    s = s.interpolate(method='time', limit=3, limit_direction='both')
    return s

def _try_parse_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of object-like numeric strings to floats.
    Handles thousands separators, comma-decimals, percents, and stray units."""
    if not isinstance(s, pd.Series):
        return pd.to_numeric(s, errors='coerce')

    # fast path
    out = pd.to_numeric(s, errors='coerce')
    na_ratio = out.isna().mean()

    if na_ratio <= 0.25:
        return out

    # as string for cleanup attempts
    ss = s.astype(str).str.strip()

    # remember percent
    has_pct = ss.str.contains(r'%', regex=True, na=False)

    # remove obvious units/symbols except digits, comma, dot, sign
    cleaned = ss.str.replace(r'[^0-9,.\-+eE]', ' ', regex=True).str.replace(r'\s+', '', regex=True)

    # Heuristic: if more commas than dots overall, treat comma as decimal
    comma_cnt = cleaned.str.count(',').sum()
    dot_cnt = cleaned.str.count(r'\.').sum()
    if comma_cnt > dot_cnt:
        # EU style: dots likely thousands separators; remove dots, change comma to dot
        attempt = cleaned.str.replace(r'\.', '', regex=True).str.replace(',', '.', regex=False)
    else:
        # US style: remove commas as thousands separators
        attempt = cleaned.str.replace(',', '', regex=False)

    out2 = pd.to_numeric(attempt, errors='coerce')

    # Apply percent scaling where appropriate
    if has_pct.any():
        # Only scale entries that originally had %
        out2 = out2.where(~has_pct, out2 / 100.0)

    if out2.notna().sum() >= out.notna().sum():
        return out2
    return out

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply robust numeric parsing to object-like columns; keep native numerics as-is."""
    if df is None or df.empty:
        return pd.DataFrame()
    res = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            res[col] = ser.astype(float)
        else:
            res[col] = _try_parse_numeric_series(ser)
    return pd.DataFrame(res, index=df.index)

def build_ai_context(df: pd.DataFrame, anomalies_found: dict, corr_payload: dict, used_cols: list, is_timeseries: bool, forecast_horizon: int, contamination: float) -> str:
    """Assemble structured stats the AI can leverage for a deeper analysis."""
    try:
        lines = []
        lines.append(f"Shape: {getattr(df, 'shape', None)}")
        # Column dtypes
        try:
            dtypes = {c: str(t) for c, t in df.dtypes.items()}
            lines.append("Dtypes: " + json.dumps(dtypes, ensure_ascii=False))
        except Exception:
            pass
        # Missingness (fraction)
        try:
            mv = df.isna().mean().sort_values(ascending=False)
            top_mv = mv[mv > 0].head(20)
            if not top_mv.empty:
                lines.append("Top missingness (fraction): " + json.dumps({k: float(v) for k, v in top_mv.items()}))
        except Exception:
            pass
        # Numeric summaries
        try:
            nums = df.select_dtypes(include='number')
            if not nums.empty:
                desc = nums.describe().to_dict()
                compact = {}
                for col in nums.columns:
                    stats = {}
                    for k in ("mean", "50%", "std", "min", "max"):
                        if k in desc and col in desc[k]:
                            stats[k] = float(desc[k][col])
                    compact[col] = stats
                lines.append("Numeric summary (mean, median, std, min, max): " + json.dumps(compact))
        except Exception:
            pass
        # Recent trend per used column
        try:
            trend_info = {}
            for col in used_cols[:20]:
                s = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(s) >= 5:
                    w = min(len(s), max(20, len(s)//5))
                    y = s.iloc[-w:]
                    x = np.arange(len(y), dtype=float)
                    slope, intercept = np.polyfit(x, y.values, 1)
                    change = float(y.iloc[-1] - y.iloc[0])
                    pct = float((change / (abs(y.iloc[0]) + 1e-12)) * 100.0)
                    trend_info[col] = {"window": int(w), "slope_per_step": float(slope), "recent_change": float(change), "pct_change": pct, "last": float(y.iloc[-1])}
            if trend_info:
                lines.append("Recent trends: " + json.dumps(trend_info))
        except Exception:
            pass
        # Anomalies
        try:
            if anomalies_found:
                lines.append("Anomalies summary: " + json.dumps(anomalies_found))
        except Exception:
            pass
        # Correlations (top absolute Spearman)
        try:
            if corr_payload and corr_payload.get("z"):
                x = corr_payload["x"]; y = corr_payload["y"]; z = corr_payload["z"]
                pairs = []
                for i, row in enumerate(z):
                    for j, val in enumerate(row):
                        if i >= j:
                            continue
                        if val is None or isinstance(val, str):
                            continue
                        pairs.append((abs(float(val)), float(val), y[i], x[j]))
                pairs.sort(reverse=True)
                top = [{"pair": [a, b], "rho": v, "abs": av} for av, v, a, b in pairs[:15]]
                lines.append("Top correlations (Spearman): " + json.dumps(top))
        except Exception:
            pass
        # Time axis info
        try:
            if is_timeseries and isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.dropna()
                if len(idx):
                    freq = str(idx.freq) if idx.freq is not None else (pd.infer_freq(idx) or "unknown")
                    lines.append(f"Time series detected. Start: {str(idx[0])}, End: {str(idx[-1])}, Freq: {freq}")
        except Exception:
            pass
        # Parameters
        lines.append(f"User settings: forecast_horizon={int(forecast_horizon)}, anomaly_contamination={float(contamination)}")
        return "\n".join(lines)
    except Exception:
        return ""

@app.route('/', methods=['GET', 'POST'])
# Optional rate limit on uploads
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            orig_name = secure_filename(file.filename)
            _, ext = os.path.splitext(orig_name)
            ext = ext.lower()

            # 1) Save to a temp file once (avoid re-reading the stream)
            temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(app.config['UPLOADS_DIR'], temp_name)  # CHANGED
            file.save(temp_path)

            try:
                # 2) Hash the saved temp file
                hasher = hashlib.sha1()
                with open(temp_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
                        hasher.update(chunk)
                digest = hasher.hexdigest()

                # 3) Dedup by content hash
                storage_name = f"{digest}{ext}"
                final_path = os.path.join(app.config['UPLOADS_DIR'], storage_name)  # CHANGED

                if os.path.exists(final_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        app.logger.warning("Could not remove temp file %s: %s", temp_path, e)
                else:
                    os.replace(temp_path, final_path)

                # 4) Optional: upload full CSV to AI if within size cap
                try:
                    size_bytes = os.path.getsize(final_path)
                    if size_bytes <= app.config['AI_FULL_UPLOAD_MAX_MB'] * 1024 * 1024:
                        uploaded = genai.upload_file(path=final_path, mime_type="text/csv", display_name=orig_name)
                        AI_FILE_MAP[storage_name] = uploaded
                except Exception as e:
                    app.logger.info("AI file upload skipped: %s", e)

                # 5) Redirect with the original filename for display only
                # forward user controls for immediate use
                fh = request.form.get('forecast_horizon')
                cont = request.form.get('contamination')
                start_view = request.form.get('view') or 'overview'
                return redirect(url_for(
                    'analyze_file',
                    filename=storage_name,
                    display=orig_name,
                    forecast_horizon=fh,
                    contamination=cont,
                    view=start_view
                ))
            except Exception as e:
                app.logger.exception("Upload failed")
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                flash(f"Upload failed: {e}")
                return redirect(request.url)
    return render_template('index.html')

# Place near other read_* helpers
def read_excel_smart(path: str):
    """Read the first non-empty sheet; try to infer header and index."""
    try:
        # use context manager to avoid file locking on Windows/OneDrive
        with pd.ExcelFile(path) as xls:
            for sheet in xls.sheet_names:
                try:
                    # first attempt: normal header
                    df = pd.read_excel(xls, sheet_name=sheet, header=0)
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    if df is not None and df.shape[1] > 0:
                        # try to set a good index
                        for cand in ['timestamp', 'date', 'time']:
                            if cand in df.columns:
                                with pd.option_context('mode.chained_assignment', None):
                                    try:
                                        df[cand] = pd.to_datetime(df[cand], errors='ignore')
                                    except Exception:
                                        pass
                                try:
                                    df = df.set_index(cand)
                                except Exception:
                                    pass
                                break
                        else:
                            # if first column looks datetime, use it
                            first_col = df.columns[0]
                            try:
                                maybe_dt = pd.to_datetime(df[first_col], errors='coerce')
                                if maybe_dt.notna().sum() >= max(3, int(len(df) * 0.3)):
                                    df = df.set_index(first_col)
                            except Exception:
                                pass
                        return df
                    # second attempt: no header, treat first row as header
                    df2 = pd.read_excel(xls, sheet_name=sheet, header=None)
                    df2 = df2.dropna(how='all').dropna(axis=1, how='all')
                    if df2 is not None and df2.shape[1] > 0:
                        header_row = df2.index[df2.notna().any(axis=1)][0] if not df2.empty else 0
                        df2.columns = df2.iloc[df2.index[df2.notna().any(axis=1)][0]]
                        df2 = df2.drop(df2.index[:header_row + 1])
                        df2 = df2.loc[:, ~df2.columns.isna()]
                        for cand in ['timestamp', 'date', 'time']:
                            if cand in df2.columns:
                                try:
                                    df2[cand] = pd.to_datetime(df2[cand], errors='ignore')
                                    df2 = df2.set_index(cand)
                                except Exception:
                                    pass
                                break
                        return df2
                except Exception:
                    continue
        # nothing usable
        return pd.DataFrame()
    except Exception as e:
        # Fallback to default read_excel if ExcelFile fails
        try:
            return pd.read_excel(path)
        except Exception:
            raise e

# ADD: central DataFrame loader for download/report endpoints (read-only; no deletions)
def get_dataframe_for(filename):
    """
    Best-effort loader for an uploaded dataset by hashed filename.
    - Prefer the in-memory `DATAFRAME_CACHE`.
    - Fallback to reading from `app.config['UPLOADS_DIR']` using robust readers.
    - Infers a datetime index when obvious and sorts time indexes.
    - Drops fully empty columns.
    - Caches the DataFrame for subsequent use.
    Returns: pandas.DataFrame or None if not found/unreadable.
    """
    try:
        # 1) Cache fast-path
        df = DATAFRAME_CACHE.get(filename)
        if df is not None:
            return df

        # 2) Disk path
        uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
        path = os.path.join(uploads_dir, filename)
        if not os.path.exists(path):
            app.logger.info("get_dataframe_for: file not found on disk: %s", path)
            return None

        _, ext = os.path.splitext(filename)
        ext = (ext or "").lower()

        # 3) Read using robust helpers
        if ext == ".csv":
            df = read_csv_fallback(path)
        elif ext == ".xlsx":
            df = read_excel_smart(path)
        elif ext == ".json":
            df = read_json_fallback(path)
        elif ext == ".txt":
            # Try Python engine with sep=None to sniff dialect; fallback to whitespace
            try:
                df = read_csv_fallback(path, sep=None, engine="python")
            except Exception:
                df = read_csv_fallback(path, delim_whitespace=True)
        else:
            # Unknown extension; try CSV as a last resort
            try:
                df = read_csv_fallback(path)
            except Exception as e:
                app.logger.warning("get_dataframe_for: unsupported extension %s; error: %s", ext, e)
                return None

        if not isinstance(df, pd.DataFrame):
            app.logger.info("get_dataframe_for: reader returned non-DataFrame for %s", filename)
            return None

        # 4) Basic cleanup: drop fully empty columns
        try:
            df = df.dropna(axis=1, how='all')
        except Exception:
            pass

        # 5) Infer datetime index (common first-column timestamp)
        try:
            if not isinstance(df.index, pd.DatetimeIndex) and df.shape[1] >= 1:
                candidate_cols = []
                # Prefer obvious datetime-named columns
                for c in df.columns:
                    lc = str(c).strip().lower()
                    if any(tok in lc for tok in ("date", "time", "timestamp", "datetime")):
                        candidate_cols.append(c)
                # Fallback to the first column
                if not candidate_cols:
                    candidate_cols = [df.columns[0]]

                picked = None
                for c in candidate_cols:
                    ts = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
                    if ts.notna().sum() >= max(5, int(0.6 * len(ts))):
                        picked = c
                        break
                if picked is not None:
                    ts = pd.to_datetime(df[picked], errors="coerce", utc=False, infer_datetime_format=True)
                    df = df.drop(columns=[picked])
                    df.index = ts
                    # sort by time for consistency
                    try:
                        df = df[ts.notna()].sort_index()
                    except Exception:
                        pass
        except Exception as e:
            app.logger.debug("get_dataframe_for: datetime inference skipped: %s", e)

        # 6) Cache and return
        DATAFRAME_CACHE.set(filename, df)
        return df
    except Exception as e:
        app.logger.exception("get_dataframe_for failed for %s", filename)
        return None

def safe_df_head_html(df: pd.DataFrame) -> str:
    try:
        if df is None or df.shape[1] == 0:
            return "<p>No columns detected in the uploaded file.</p>"
        return df.head().to_html()
    except Exception:
        return "<p>Could not render head()</p>"

def safe_df_description_html(df: pd.DataFrame) -> str:
    try:
        if df is None or df.shape[1] == 0:
            # show dtypes or placeholder
            return (df.dtypes.to_frame('dtype').to_html()
                    if df is not None and isinstance(df, pd.DataFrame) else "<p>No data.</p>")
        # include='all' is more forgiving; handle pandas versions without datetime_is_numeric
        try:
            desc = df.describe(include='all')
        except Exception:
            desc = df.select_dtypes(include='number').describe()
        return desc.to_html()
    except Exception:
        try:
            return df.dtypes.to_frame('dtype').to_html()
        except Exception:
            return "<p>Could not build description.</p>"

def describe_for_ai(df: pd.DataFrame) -> str:
    """Plain-text summary for AI prompts; never raises."""
    try:
        if df is None or df.shape[1] == 0:
            return f"Empty or headerless table. Shape: {getattr(df, 'shape', None)}"
        try:
            desc = df.describe(include='all')
        except Exception:
            desc = df.select_dtypes(include='number').describe()
        return str(desc)
    except Exception:
        try:
            return "Columns and dtypes:\n" + str(df.dtypes)
        except Exception:
            return "No structured information available."

def offline_answer(df: pd.DataFrame, question: str = "summary", error=None) -> str:
    """
    Lightweight, deterministic HTML answer when AI is unavailable.
    - For 'summary', returns dataset overview, missingness, and recent trends.
    - For other questions, provides basic stats for mentioned columns (best-effort).
    """
    try:
        q = (str(question or "")).strip().lower()
        parts = []
        parts.append("<h3>Offline analysis</h3>")
        # Friendlier banner (hide low-level error details from the UI)
        if error:
            try:
                parts.append("<p><em>AI response unavailable. Showing a quick offline analysis instead.</em></p>")
            except Exception:
                parts.append("<p><em>AI response unavailable. Showing a quick offline analysis instead.</em></p>")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            parts.append("<p>No data available.</p>")
            return "".join(parts)

        # Overview
        try:
            parts.append(f"<p><strong>Shape:</strong> {tuple(df.shape)}</p>")
            dtypes = ", ".join([f"{htmllib.escape(str(c))}: {htmllib.escape(str(t))}" for c, t in df.dtypes.items()])
            parts.append(f"<p><strong>Dtypes:</strong> {dtypes}</p>")
        except Exception:
            pass

        # Missingness
        try:
            mv = df.isna().mean().sort_values(ascending=False)
            mv = mv[mv > 0].head(10)
            if not mv.empty:
                parts.append("<h4>Top missingness</h4><ul>")
                for col, frac in mv.items():
                    parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: {round(float(frac)*100,2)}%</li>")
                parts.append("</ul>")
        except Exception:
            pass

        # Numeric trends
        try:
            df_num = coerce_numeric_df(df)
            sel = df_num.select_dtypes(include='number')
            if not sel.empty:
                parts.append("<h4>Recent trends (last window)</h4><ul>")
                shown = 0
                for col in sel.columns:
                    s = pd.to_numeric(sel[col], errors='coerce').dropna()
                    if len(s) < 5:
                        continue
                    w = min(len(s), max(20, len(s)//5))
                    y = s.iloc[-w:]
                    x = np.arange(len(y), dtype=float)
                    slope, intercept = np.polyfit(x, y.values, 1)
                    change = float(y.iloc[-1] - y.iloc[0])
                    pct = (change / (abs(y.iloc[0]) + 1e-12)) * 100.0
                    parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: slope {slope:.4g}, change {change:.4g} ({pct:.2f}%)</li>")
                    shown += 1
                    if shown >= 8:
                        break
                parts.append("</ul>")
        except Exception:
            pass

        # If it's a specific question (non-summary), show simple stats for mentioned columns
        if q and q != "summary":
            try:
                mentioned = []
                q_low = q.lower()
                for col in df.columns:
                    name = str(col)
                    if name.lower() in q_low:
                        mentioned.append(col)
                # Fallback: top 3 numeric columns if nothing is detected
                if not mentioned:
                    df_num = coerce_numeric_df(df).select_dtypes(include='number')
                    mentioned = list(df_num.columns[:3]) if df_num is not None else []
                if mentioned:
                    parts.append("<h4>Quick stats for relevant columns</h4><ul>")
                    for col in mentioned[:6]:
                        try:
                            s = pd.to_numeric(df[col], errors='coerce').dropna()
                            if s.empty:
                                parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: no numeric data</li>")
                                continue
                            parts.append(
                                f"<li><strong>{htmllib.escape(str(col))}</strong>: "
                                f"mean={float(s.mean()):.4g}, median={float(s.median()):.4g}, "
                                f"std={float(s.std(ddof=1)):.4g}, min={float(s.min()):.4g}, max={float(s.max()):.4g}</li>"
                            )
                        except Exception:
                            parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: unable to compute stats</li>")
                    parts.append("</ul>")
            except Exception:
                pass

        return "".join(parts)
    except Exception:
        return "<p>Offline analysis is unavailable due to an internal error.</p>"

def _get_arg_float(name, default):
    try:
        v = (request.form.get(name) if request.method == 'POST' else None)
        if v in (None, ""):
            v = request.args.get(name)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default

def _get_arg_int(name, default):
    try:
        v = (request.form.get(name) if request.method == 'POST' else None)
        if v in (None, ""):
            v = request.args.get(name)
        return int(float(v)) if v not in (None, "") else default
    except Exception:
        return default

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
# Optional rate limit on analyze
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOADS_DIR'], filename)
    display_name = request.args.get('display') or request.form.get('display') or filename

    # User controls with sensible defaults
    default_steps = int(os.getenv("DEFAULT_FORECAST_STEPS", "40"))
    default_contam = float(os.getenv("DEFAULT_CONTAMINATION", "0.02"))
    user_steps = _get_arg_int("forecast_horizon", default_steps)
    user_contam = _get_arg_float("contamination", default_contam)

    # New: handle missing file up-front
    if not os.path.exists(filepath) and filename not in DATAFRAME_CACHE:
        flash("The uploaded file is no longer available. Please re-upload it.")
        return redirect(url_for('upload_file'))

    try:
        df = DATAFRAME_CACHE.get(filename)
        if df is None:
            # Improved file reading to handle timestamps and non-UTF8 encodings
            if filename.endswith('.csv'):
                df = read_csv_fallback(filepath, index_col=0, parse_dates=True)
            elif filename.endswith('.xlsx'):
                df = read_excel_smart(filepath)
            elif filename.endswith('.json'):
                df = read_json_fallback(filepath)
                for col in ['timestamp', 'date', 'time']:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                        except Exception:
                            pass
                        break
            elif filename.endswith('.txt'):
                df = read_csv_fallback(filepath, sep=',', index_col=0, parse_dates=True)
            else:
                flash('Unsupported file type')
                return redirect(url_for('upload_file'))

            # Cache for follow-up questions
            DATAFRAME_CACHE.set(filename, df)

        # NEW: delete only app-hashed files from uploads dir, never user originals
        if (
            app.config.get('DELETE_UPLOADED_AFTER_PROCESSING', False)
            and HASHED_UPLOAD_RE.match(os.path.basename(filepath))
            and os.path.exists(filepath)
        ):
            _safe_delete(filepath)

        # Optional: opportunistic cleanup for old files if retention is set
        _cleanup_uploads_if_configured()

        # DEFINE file_asset BEFORE any use (fixes possible NameError)
        file_asset = AI_FILE_MAP.get(filename)

        # --- Handle follow-up questions (only when provided) ---
        user_question = None
        ai_answer = None
        if request.method == 'POST':
            user_question = (request.form.get('question') or '').strip()
            ai_answer_html = ""
            if user_question:
                # file_asset assumed available from earlier logic; if not, set to None
                ai_answer_html = get_ai_answer_with_file(df, user_question, file_asset=file_asset)
            ai_answer = ai_answer_html  # <-- ensure the template gets the answer

        analysis = {}
        plots = []
        forecast_plots = []
        anomalies_found = {}  # collect anomalies per column
        is_timeseries = isinstance(df.index, pd.DatetimeIndex)
        used_cols = []  # track columns actually plotted

        # Precompute correlation for advanced view
        corr_payload = None
        try:
            num_df = coerce_numeric_df(df).select_dtypes(include='number')
            # Drop columns that are entirely NaN or nearly constant
            if num_df is not None and not num_df.empty:
                # Require at least 3 non-null values
                valid = [c for c in num_df.columns if num_df[c].notna().sum() >= 3]
                num_df = num_df[valid]
                # Drop near-constant columns
                keep = []
                for c in num_df.columns:
                    s = pd.to_numeric(num_df[c], errors='coerce').dropna()
                    if s.empty:
                        continue
                    if float(s.max()) == float(s.min()):
                        continue
                    keep.append(c)
                num_df = num_df[keep] if keep else num_df
            if num_df is not None and not num_df.empty and len(num_df.columns) >= 2:
                cols = list(num_df.columns)
                payload = {}

                # Spearman (default for z)
                try:
                    spearman = num_df.corr(method='spearman')
                except Exception:
                    spearman = None

                # Pearson (also requested)
                try:
                    pearson = num_df.corr(method='pearson')
                except Exception:
                    pearson = None

                if spearman is not None:
                    payload["z"] = [[float(v) if pd.notna(v) else None for v in spearman.loc[r, cols].tolist()] for r in cols]
                if pearson is not None:
                    payload["pearson"] = {
                        "x": cols,
                        "y": cols,
                        "z": [[float(v) if pd.notna(v) else None for v in pearson.loc[r, cols].tolist()] for r in cols]
                    }
                corr_payload = payload if ("z" in payload or "pearson" in payload) else None
            else:
                corr_payload = None
        except Exception as e:
            app.logger.warning("Correlation computation failed: %s", e)
            corr_payload = None

        interactive = []  # payload for Plotly charts
        # Use full series by default for interactive traces; allow optional tail by env if needed
        # Accept: "all", "", "0", "-1", "none", "false" => full; else parse int
        raw_tail = (os.getenv("PLOTLY_TAIL", "all") or "all").strip().lower()

        for column in df.columns:
            # Coerce series to numeric where possible
            series_raw = df[column]
            try:
                series = pd.to_numeric(series_raw, errors='coerce').dropna()
            except Exception:
                series = pd.Series(dtype=float)
            if series.empty:
                continue
            plots.append(column)  # <-- track all columns for plots
            # Detect anomalies first so we can overlay on both static and interactive
            an_idx, an_score = detect_anomalies(series, contamination=user_contam)
            if len(an_idx):
                try:
                    anomalies_found[str(column)] = [str(i) for i in an_idx]
                except Exception:
                    try:
                        anomalies_found[str(column)] = [str(i) for i in list(an_idx)]
                    except Exception:
                        anomalies_found[str(column)] = []

            # Static trend with anomalies overlay (already full series)
            title_trend = f"Trend for {column}"
            title_trend = f"Trend for {column}"
            plots.append({
                "img": generate_plot(
                    series,
                    title_trend,
                    'Timestamp' if is_timeseries else 'Index',
                    column,
                    anomalies_idx=an_idx
                ),
                "title": title_trend
            })

            if is_timeseries and len(series) >= 10:
                try:
                    steps = max(10, min(240, user_steps))
                    conf_df = None
                    fc_mean = None
                    # Holt-Winters + slope/naturalization (unchanged)
                    try:
                        hw = ExponentialSmoothing(
                            series, trend='add', damped_trend=True, seasonal=None,
                            initialization_method='estimated'
                        ).fit(optimized=True)
                        fc_vals = hw.forecast(steps)
                        future_idx = _infer_future_index(series.index, steps)
                        fc_mean = pd.Series(fc_vals.values, index=future_idx)
                        resid_std = float(np.nanstd(getattr(hw, 'resid', series - hw.fittedvalues), ddof=1))
                        lower = fc_mean - 1.96 * resid_std
                        upper = fc_mean + 1.96 * resid_std
                        conf_df = pd.concat([lower, upper], axis=1)
                        conf_df.columns = ['lower', 'upper']
                    except Exception as e_hw:
                        app.logger.warning("Holt-Winters (damped) failed for %s: %s", column, e_hw)
                    # Need slope fallback?
                    need_slope = False
                    if fc_mean is not None:
                        recent = series.tail(min(len(series), 300)).values
                        diffs = np.diff(recent)
                        recent_step = float(np.median(np.abs(diffs))) if len(diffs) else 0.0
                        slope_fc = float((fc_mean.iloc[-1] - fc_mean.iloc[0]) / max(1, len(fc_mean) - 1))
                        flat_by_range = np.allclose(fc_mean.values, fc_mean.values[0], rtol=1e-3, atol=1e-6)
                        flat_by_slope = (recent_step > 0 and abs(slope_fc) < 0.25 * recent_step)
                    if fc_mean is None or need_slope:
                        fc_mean, conf_df = _recent_slope_forecast(series, steps, window=min(len(series), 200), damping=None)
                    # Naturalize if too straight
                    try:
                        base_slope_est = float((fc_mean.iloc[-1] - fc_mean.iloc[0]) / max(1, len(fc_mean) - 1))
                        if _is_too_linear(fc_mean):
                            fc_mean, conf_df = _bootstrap_natural_path(
                                series, steps, window=min(len(series), 200), base_slope=base_slope_est,
                                n_samples=200, q_low=0.1, q_high=0.9
                            )
                    except Exception as e_nat:
                        app.logger.warning("Naturalization failed for %s: %s", column, e_nat)

                    # Build arrays for interactive forecast
                    # Build arrays for interactive forecast
                    split_x = str(series.index[-1])
                    fc_x = [str(i) for i in fc_mean.index]
                    fc_y = [float(v) for v in fc_mean.values]
                    if conf_df is not None:
                        ci_lower = [float(v) for v in conf_df.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df.iloc[:, 1].values]

                    # Keep static forecast image too — now with full history
                    title_fc = f"Forecast for {column}"
                    forecast_plots.append({
                        "img": generate_forecast_plot(
                            series,
                            fc_mean,
                            title_fc,
                            'Timestamp',
                            column,
                            conf_int=conf_df,
                            history_tail=None  # full history
                        ),
                        "title": title_fc
                    })
                except Exception as e:
                    app.logger.warning("Could not generate forecast for %s: %s", column, e)

            # ----- STL decomposition (static) ----- (unchanged)
            if is_timeseries and len(series) >= 10:
                try:
                    s_norm = normalize_timeseries(series)
                    sp = _infer_seasonal_period(s_norm.index) if isinstance(s_norm.index, pd.DatetimeIndex) else None
                    if sp:
                        stl_img = generate_stl_plot(s_norm, f"STL decomposition for {column}", seasonal_period=sp)
                       
                        if stl_img:
                            forecast_plots.append({"img": stl_img, "title": f"STL decomposition for {column}"})
                except Exception:
                    pass

            # ----- Interactive traces -----
            # Use full series unless PLOTLY_TAIL is set to a positive integer
            if raw_tail in ("all", "", "0", "-1", "none", "false"):
                s_tail = series
            else:
                try:
                    tail_n = int(raw_tail)
                    s_tail = series.tail(max(1, tail_n))
                except Exception:
                    s_tail = series

            x_hist = [str(i) for i in s_tail.index]
            y_hist = [float(v) for v in s_tail.values]
            traces = [{
                "type": "scatter",
                "mode": "lines+markers",  # markers enable box/lasso
                "name": "History",
                "x": x_hist,
                "y": y_hist,
                "line": {"color": "rgb(31,119,180)", "width": 2},
                "marker": {"size": 4, "opacity": 0.6}
            }]

            # anomalies in full or tail (s_tail may be full now)
            if len(an_idx):
                an_tail_idx = [i for i in an_idx if i in s_tail.index]
                if an_tail_idx:
                    traces.append({
                        "type": "scatter",
                        "mode": "markers",
                        "name": "Anomaly",
                        "x": [str(i) for i in an_tail_idx],
                        "y": [float(s_tail.loc[i]) for i in an_tail_idx],
                        "marker": {"color": "#ef4444", "size": 7, "opacity": 0.9},
                        "hovertemplate": "Anomaly<br>%{x}<br>%{y}<extra></extra>"
                    })

            # forecast for interactive (unchanged)
            fc_x = fc_y = ci_lower = ci_upper = split_x = None
            if is_timeseries and len(series) >= 10:
                try:
                    steps = max(10, min(240, user_steps))
                    # normalize spacing, forecast, and naturalize if too straight
                    s_norm = normalize_timeseries(series)
                    fc_mean, conf_df = _recent_slope_forecast(s_norm, steps=steps, window=None, damping=None)
                    try:
                        if _is_too_linear(fc_mean):
                            fc_mean, conf_df = _bootstrap_natural_path(s_norm, steps=steps, base_slope=None, n_samples=200)
                    except Exception:
                        pass

                    split_x = str(series.index[-1])
                    fc_x = [str(i) for i in fc_mean.index]
                    fc_y = [float(v) for v in fc_mean.values]
                    if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                        ci_lower = [float(v) for v in conf_df.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df.iloc[:, 1].values]

                    # 95% CI band (two traces, grouped under one legend item)
                    if fc_x and ci_lower and ci_upper:
                        ci_group = f"ci-{re.sub(r'[^A-Za-z0-9_-]+', '', str(column))}"
                        traces.append({
                            "type": "scatter",
                            "mode": "lines",
                            "name": "95% CI",
                            "x": fc_x, "y": ci_lower,
                            "line": {"width": 0},
                            "hoverinfo": "skip",
                            "showlegend": True,
                            "legendgroup": ci_group
                        })
                        traces.append({
                            "type": "scatter",
                            "mode": "lines",
                            "name": "95% CI",
                            "x": fc_x, "y": ci_upper,
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(255,69,0,0.18)",
                            "hoverinfo": "skip",
                            "showlegend": False,
                            "legendgroup": ci_group,
                            "legendgrouptitle": {"text": "95% CI"}
                        })

                    # Forecast mean line (unchanged)
                    if fc_x and fc_y:
                        traces.append({
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Forecast",
                            "x": fc_x, "y": fc_y,
                            "line": {"color": "orangered", "width": 3, "dash": "dash"},
                            "marker": {"size": 3}
                        })
                except Exception:
                    pass

            # Update layout to include split line and legend group behavior + range UI
            xaxis = {"title": "Timestamp" if is_timeseries else "Index", "showgrid": True}
            if is_timeseries:
                xaxis.update({
                    "rangeslider": {"visible": True},
                    "rangeselector": {
                        "buttons": [
                            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                            {"step": "year", "stepmode": "todate", "label": "YTD"},
                            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                            {"step": "all", "label": "All"}
                        ]
                    }
                })

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
            if is_timeseries:
                layout["xaxis"].update({
                    "rangeslider": {"visible": True},
                    "rangeselector": {
                        "buttons": [
                            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                            {"step": "year", "stepmode": "todate", "label": "YTD"},
                            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                            {"step": "all", "label": "All"}
                        ]
                    }
                })

            # Use full values for the distribution
            dist = {"name": column, "values": [float(v) for v in series.dropna().values]}
            interactive.append({"column": column, "traces": traces, "layout": layout, "distribution": dist})

        # Compute info(), missing values, and AI summary safely
        buf = io.StringIO()
        try:
            df.info(buf=buf)
            info_string = buf.getvalue()
        except Exception:
            info_string = "Unable to render DataFrame info()."

        try:
            mv = df.isnull().sum()
            mvf = mv[mv > 0]
            missing_values_html = mvf.to_frame('missing_count').to_html() if not mvf.empty else None
        except Exception:
            missing_values_html = None

        # Replace this line:
        # ai_summary = get_ai_summary_with_file(df, file_asset)

        # With this cached/skip-on-POST logic:
        # Build richer AI context from current analysis artifacts
        used_cols = list(df.columns)
        ai_context = build_ai_context(
            df=df,
            anomalies_found=anomalies_found,
            corr_payload=corr_payload,
            used_cols=used_cols,
            is_timeseries=is_timeseries,
            forecast_horizon=user_steps,
            contamination=user_contam
        )

        ai_summary = AI_SUMMARY_CACHE.get(filename)
        if ai_summary is None:
            # Only generate on initial GET to avoid rate limits on re-runs/questions
            if request.method == 'GET' and AI_ENABLED and model is not None:
                try:
                    generated = get_ai_summary_with_file(df, file_asset, extra_context=ai_context)
                    ai_summary = generated
                    # cache whatever we got (even an error string) to avoid repeated calls under rate limits
                    AI_SUMMARY_CACHE[filename] = generated
                except Exception as _e:
                    # fallback message if generation fails unexpectedly
                    ai_summary = "<p>AI summary temporarily unavailable.</p>"
            else:
                # POST or AI disabled: reuse cached if any; otherwise neutral message
                ai_summary = "<p>AI summary will appear after initial analysis loads.</p>"

        # Build final analysis dict (use used_cols instead of raw numeric_cols)
        analysis.update({
            'head': safe_df_head_html(df),
            'description': safe_df_description_html(df),
            'info': info_string,
            'missing_values': missing_values_html,
            'plots': plots,
            'forecast_plots': forecast_plots,
            'anomalies': anomalies_found,
            'ai_summary': ai_summary,
            'user_question': user_question,
            'ai_answer': ai_answer,
            # Interactive payloads expected by the template
            'interactive': interactive,
            'columns': used_cols,
            'corr': corr_payload,
            'controls': {
                'forecast_horizon': user_steps,
                'contamination': user_contam
            }
        })

        # Schedule deletion of the hashed upload only if the response is successful
        if (
            app.config.get('DELETE_UPLOADED_AFTER_PROCESSING', False)
            and HASHED_UPLOAD_RE.match(os.path.basename(filepath))
            and os.path.exists(filepath)
        ):
            @after_this_request
            def _delete_hashed_upload(response):
                try:
                    _safe_delete(filepath)
                    app.logger.info("Deferred delete of %s done", filepath)
                except Exception as e:
                    app.logger.warning("Deferred delete of %s failed: %s", filepath, e)
                return response

        return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name)

    except Exception as e:
        flash(f"An error occurred while analyzing the file: {e}")
        return redirect(url_for('upload_file'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/download/<filename>/cleaned.csv', methods=['GET'])
def download_cleaned_csv(filename):
    # Only allow app-managed hashed uploads
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)
    df = DATAFRAME_CACHE.get(filename)
    if df is None:
        uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
        path = os.path.join(uploads_dir, filename)
        if not os.path.exists(path):
            return ("Not found", 404)
        # Re-read using the same heuristics
        df = get_dataframe_for(filename)
        if df is None:
            return ("Not found", 404)

    # Build a “cleaned” version: robust numeric coercion but keep non-numeric columns
    cleaned = df.copy()
    try:
        for col in cleaned.columns:
            ser = cleaned[col]
            if pd.api.types.is_numeric_dtype(ser):
                cleaned[col] = pd.to_numeric(ser, errors='coerce')
            else:
                coerced = _try_parse_numeric_series(ser)
                # Replace only if coercion meaningfully helps

                if coerced.notna().sum() >= pd.to_numeric(ser, errors='coerce').notna().sum():
                    cleaned[col] = coerced
        # Sort by time if datetime index
        if isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned = cleaned.sort_index()
        # Drop fully empty columns after coercion
        cleaned = cleaned.dropna(axis=1, how='all')
    except Exception:
        pass

    csv = cleaned.to_csv(index=True)
    resp = make_response(csv)
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_cleaned.csv")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

# --- AI summary as HTML download ---
@app.route('/download/<filename>/ai_summary.html', methods=['GET'])
def download_ai_summary_html(filename):
    # Only allow app-managed hashed uploads
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    # Try cache first
    ai_html = AI_SUMMARY_CACHE.get(filename)
    if ai_html is None:
        # Best effort to generate now
        df = get_dataframe_for(filename)
        if df is None:
            return ("Not found", 404)
        try:
            file_asset = AI_FILE_MAP.get(filename)
            ai_html = get_ai_summary_with_file(df, file_asset, extra_context=describe_for_ai(df))
        except Exception:
            ai_html = "<p>AI summary temporarily unavailable.</p>"
        AI_SUMMARY_CACHE[filename] = ai_html

    resp = make_response(ai_html if isinstance(ai_html, str) else str(ai_html))
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_ai_summary.html")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

@app.route('/download/<filename>/ai_summary.pdf', methods=['GET'])
def download_ai_summary_pdf(filename):
    # Only allow app-managed hashed uploads
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "message": "Invalid filename."}), 400

    # Reuse cached summary or generate on demand
    ai_html = AI_SUMMARY_CACHE.get(filename)
    if ai_html is None or not isinstance(ai_html, str) or not ai_html.strip():
        df = get_dataframe_for(filename)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return jsonify({"ok": False, "message": "Dataset not found or empty."}), 404
        # Never re-generate more than once per file: use cache-first wrapper
        ai_html = get_or_cache_ai_summary_for(filename, df)

    # Wrap into a minimal printable HTML doc
    doc_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI Summary</title>
<style>
  body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:#0f172a; }}
  h1,h2,h3 {{ color:#0f172a; }}
  @page {{ size: A4; margin: 14mm; }}
</style></head><body>
<h1>AI Summary</h1>
{ai_html}
</body></html>
"""
    try:
        pdf_bytes = html_to_pdf_bytes(doc_html)
    except Exception as e:
        return jsonify({"ok": False, "message": f"PDF rendering failed: {e}"}), 500

    resp = make_response(pdf_bytes)
    resp.headers['Content-Type'] = 'application/pdf'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_ai_summary.pdf")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

@app.route('/download/<filename>/static_plots.zip', methods=['GET'])
def download_static_plots_zip(filename):
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or df.empty:
        return ("Not found", 404)

    is_timeseries = isinstance(df.index, pd.DatetimeIndex)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for col in df.columns:
            # Build a static trend plot for each numeric column
            try:
                s = pd.to_numeric(df[col], errors='coerce').dropna()
            except Exception:
                s = pd.Series(dtype=float)
            if s.empty:
                continue
            title = f"Trend for {col}"
            img_b64 = generate_plot(
                s,
                title,
                'Timestamp' if is_timeseries else 'Index',
                col,
                anomalies_idx=None
            )
            try:
                raw = base64.b64decode(img_b64.encode('utf-8'))
                zf.writestr(f"{secure_filename(str(col))}_trend.png", raw)
            except Exception:
                # Skip this column on any failure
                continue

    bio.seek(0)
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_static_plots.zip")
    resp = make_response(bio.read())
    resp.headers['Content-Type'] = 'application/zip'
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

# --- Full report (single HTML including overview, plots, forecasts, correlation) ---
@app.route('/download/<filename>/report.html', methods=['GET'])
def download_full_report_html(filename):
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ("Not found", 404)

    # Overview
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
    except Exception:
        missing_html = ""

    # AI summary (cached or on-demand)
    ai_html = AI_SUMMARY_CACHE.get(filename)
    if ai_html is None:
        try:
            file_asset = AI_FILE_MAP.get(filename)
            ai_html = get_ai_summary_with_file(df, file_asset, extra_context=describe_for_ai(df))
            AI_SUMMARY_CACHE[filename] = ai_html
        except Exception:
            ai_html = "<p>AI summary temporarily unavailable.</p>"

    # Static plots and forecasts
    is_ts = isinstance(df.index, pd.DatetimeIndex)
    static_sections = []
    forecast_sections = []
    for col in df.columns:
        try:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
        except Exception:
            s = pd.Series(dtype=float)
        img_b64 = generate_plot(s, f"Trend for {col}", 'Timestamp' if is_ts else 'Index', col, anomalies_idx=None)
        static_sections.append(f'<figure><figcaption>Trend for {col}</figcaption><img style="max-width:100%" src="data:image/png;base64,{img_b64}" /></figure>')
        if is_ts and len(s) >= 10:
            try:
                fc_mean, ci = _bootstrap_natural_path(s, steps=app.config.get('DEFAULT_FORECAST_STEPS', 30))
            except Exception:
                fc_mean, ci = _recent_slope_forecast(s, steps=app.config.get('DEFAULT_FORECAST_STEPS', 30))
            fc_b64 = generate_forecast_plot(s, fc_mean, f"Forecast for {col}", 'Timestamp', col, conf_int=ci, history_tail=None)
            forecast_sections.append(f'<figure><figcaption>Forecast for {col}</figcaption><img style="max-width:100%" src="data:image/png;base64,{fc_b64}" /></figure>')

    # Correlation table (robust)
    corr_html = ""
    try:
        df_num = coerce_numeric_df(df)
        sel = df_num.select_dtypes(include='number')
        if not sel.empty:
            # drop constant columns
            nunique = sel.nunique(dropna=True)
            sel = sel.loc[:, nunique > 1]
        if sel.shape[1] >= 2:
            corr = sel.corr(method='spearman').round(3)
            corr_html = corr.to_html()
    except Exception:
        pass

    # Build HTML document
    display = request.args.get('display') or filename
    title = f"Analysis report — {display}"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:#0f172a; background:#ffffff; }}
  h1,h2,h3 {{ color:#0f172a; }}
  .muted {{ color:#475569; }}
  figure {{ margin: 0 0 16px 0; }}
  figcaption {{ margin: 8px 0; font-weight: 600; }}
  pre {{ white-space: pre-wrap; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border:1px solid #e2e8f0; padding:4px 6px; }}
  img {{ max-width: 100%; }}
  @page {{ size: A4; margin: 14mm; }}
</style></head>
<body>
  <h1>{title}</h1>

  <h2>Overview</h2>
  <h3>Preview</h3>{head_html}
  <h3>Description</h3>{desc_html}
  <h3>Info</h3><pre>{info_str}</pre>
  {"<h3>Missing values</h3>" + missing_html if missing_html else ""}

  <h2>AI Summary</h2>
  {ai_html}

  <h2>Static trends</h2>
  {''.join(static_sections) if static_sections else '<div class="muted">No numeric columns to plot.</div>'}

  <h2>Forecasts</h2>
  {''.join(forecast_sections) if forecast_sections else '<div class="muted">No timeseries forecasts available.</div>'}

  <h2>Correlation</h2>
  {corr_html if corr_html else '<div class="muted">Not enough numeric columns to compute correlation.</div>'}
</body></html>
"""
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_report.html")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

@app.route('/download/<filename>/report.pdf', methods=['GET'])
def download_full_report_pdf(filename):
    # Only allow app-managed hashed uploads
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "message": "Invalid filename."}), 400

    # Reuse the HTML builder to avoid duplicating content logic
    html_resp = download_full_report_html(filename)
    try:
        # If the HTML route returned an error Response, propagate it
        if getattr(html_resp, "status_code", 200) != 200:
            return html_resp
        html_doc = html_resp.get_data(as_text=True)
    except Exception:
        return jsonify({"ok": False, "message": "Failed to obtain HTML report"}), 500

    try:
        pdf_bytes = html_to_pdf_bytes(html_doc)
    except Exception as e:
        return jsonify({"ok": False, "message": f"PDF rendering failed: {e}"}), 500

    resp = make_response(pdf_bytes)
    resp.headers['Content-Type'] = 'application/pdf'
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_report.pdf")
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp

@app.route('/full_history_json', methods=['GET'])
def full_history_json():
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
        "columns": [str, ...],  # numeric columns included
        "x": [ ... ],           # ISO timestamps or index values
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

        df = get_dataframe_for(filename)  # uses cache/disk; see get_dataframe_for
        if df is None or df.empty:
            return jsonify({"ok": False, "message": "Dataset not found or empty."}), 404

        # Ensure deterministic order for plotting
        try:
            df = df.sort_index()
        except Exception:
            pass

        is_ts = isinstance(df.index, pd.DatetimeIndex)

        # Build x-axis values
        if is_ts:
            idx = df.index
            # Remove timezone if present for stable JSON
            try:
                idx = idx.tz_convert(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
            try:
                x_all = [ts.isoformat() for ts in idx.to_pydatetime()]
            except Exception:
                # Fallback to string format
                x_all = [str(v) for v in idx.astype('datetime64[ns]').tolist()]
        else:
            # Use index if json-serializable; otherwise fall back to 0..N-1
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
        # Default: full data; thin only if asked and beneficial
        step = 1
        if max_points and max_points > 0 and n > max_points:
            # uniform thinning stride
            step = max(1, n // max_points)

        # Build numeric series payload
        num_df = coerce_numeric_df(df)  # robust numeric parsing from strings
        numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]
        # If nothing numeric detected, try a best-effort pass per column
        if not numeric_cols:
            for c in df.columns:
                try:
                    parsed = _try_parse_numeric_series(df[c])
                    num_df[c] = parsed
                except Exception:
                    continue
            numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]

        # Align X and Y lengths; thin consistently using 'step'
        x_vals = x_all[::step] if step > 1 else x_all
        series = {}
        for c in numeric_cols:
            try:
                y_all = num_df[c].astype(float).tolist()
            except Exception:
                # keep NaNs as None in JSON
                y_all = [float(v) if pd.notna(v) else None for v in num_df[c].tolist()]
            y_vals = y_all[::step] if step > 1 else y_all

            # Ensure alignment (trim to same length as x)
            if len(y_vals) != len(x_vals):
                m = min(len(y_vals), len(x_vals))
                y_vals = y_vals[:m]
                x_vals = x_vals[:m]
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
        # Log and return error
        try:
            app.logger.exception("full_history_json failed: %s", e)
        except Exception:
            pass
        return jsonify({"ok": False, "message": f"Internal error: {e}"}), 500

@app.after_request
def _sanitize_permissions_policy(resp):
    try:
        if 'Permissions-Policy' in resp.headers:
            pol = str(resp.headers.get('Permissions-Policy', ''))
            # Remove a few commonly unsupported/legacy directives
            bad_bits = ['interest-cohort', 'browsing-topics', 'join-ad-interest-group', 'run-ad-auction']
            cleaned = "; ".join(seg for seg in pol.split(';') if seg and not any(b in seg for b in bad_bits)).strip()
            if cleaned:
                resp.headers['Permissions-Policy'] = cleaned
            else:
                # If nothing usable remains, drop the header entirely
                del resp.headers['Permissions-Policy']
    except Exception:
        # Never fail a response due to header cleanup
        pass
    return resp

if __name__ == "__main__":
    pass
    # Read basic server config from env (with safe defaults)
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    # Read basic server config from env (with safe defaults)
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    host = os.getenv("FLASK_HOST", os.getenv("HOST", "0.0.0.0"))
    try:
        port = int(os.getenv("FLASK_PORT", os.getenv("PORT", "5000")))
    except Exception:
        port = 5000

    # Optional: enable security headers only when explicitly requested
    if Talisman and str(os.getenv("USE_TALISMAN", "0")).strip().lower() in ("1", "true", "yes", "on"):
        try:
            default_csp = {
                "default-src": ["'self'", "https:", "http:"],
                "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https:", "http:"],
                "style-src": ["'self'", "'unsafe-inline'", "https:", "http:"],
                "img-src": ["'self'", "data:", "https:", "http:"],
                "connect-src": ["'self'", "https:", "http:"],
            }
            Talisman(app, content_security_policy=default_csp)
            app.logger.info("Talisman enabled.")
        except Exception as e:
            app.logger.warning("Talisman init failed: %s", e)

    # Optional: basic rate limiting if configured
    if Limiter and os.getenv("RATE_LIMIT"):
        try:
            limiter = Limiter(get_remote_address, app=app, default_limits=[os.getenv("RATE_LIMIT")])
            app.logger.info("Rate limiting enabled: %s", os.getenv("RATE_LIMIT"))
        except Exception as e:
            app.logger.warning("Limiter init failed: %s", e)

    app.logger.info("Starting Flask server on %s:%s (debug=%s)", host, port, debug)
    # threaded=True plays nicer on Windows; use_reloader obeys debug
    app.run(host=host, port=port, debug=debug, threaded=True)