import os
import math
import io
import base64
import zipfile
from flask import Flask, request, render_template, redirect, url_for, flash, after_this_request, make_response, jsonify
from fpdf import FPDF

from datetime import datetime, timedelta
import pandas as pd
from werkzeug.utils import secure_filename


try:
    from dotenv import load_dotenv
    load_dotenv(".env.public")   
    load_dotenv(".env")          
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import google.generativeai as genai

# For service account authentication via Vertex AI
try:
    import google.auth
    from google.oauth2 import service_account
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
import hashlib
import uuid
import json  
import numpy as np  
from statsmodels.tsa.holtwinters import ExponentialSmoothing  
from statsmodels.tsa.seasonal import STL  
from collections import OrderedDict  
import re
import html as htmllib  
import math  
from flask import Flask


app = Flask(__name__)

try:
    from flask_limiter import Limiter  
    from flask_limiter.util import get_remote_address  
except Exception:
    Limiter = None
try:
    from flask_talisman import Talisman  
except Exception:
    Talisman = None

UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure an uploads subdirectory exists/configured
app.config.setdefault('UPLOADS_SUBDIR', 'uploaded')
app.config['UPLOADS_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOADS_SUBDIR'])
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY") or "dev-secret-change-me"
app.config['DELETE_UPLOADED_AFTER_PROCESSING'] = os.getenv("DELETE_UPLOADED_AFTER_PROCESSING", "true").strip().lower() in ("1", "true", "yes", "on")
if "UPLOAD_RETENTION_DAYS" in os.environ:
    try:
        app.config['UPLOAD_RETENTION_DAYS'] = int(os.getenv("UPLOAD_RETENTION_DAYS"))
    except Exception:
        app.logger.warning("Invalid UPLOAD_RETENTION_DAYS; ignoring")

app.config.setdefault('MAX_CACHE_ITEMS', int(os.getenv("MAX_CACHE_ITEMS", "6")))
app.config.setdefault('DEFAULT_FORECAST_STEPS', int(os.getenv("DEFAULT_FORECAST_STEPS", "30")))

app.config.setdefault('PLOTLY_TAIL', int(os.getenv("PLOTLY_TAIL", "800")))
app.config.setdefault('AI_TIMEOUT_SECONDS', int(os.getenv("AI_TIMEOUT_SECONDS", "30")))
app.config.setdefault('AI_RETRY_ATTEMPTS', int(os.getenv("AI_RETRY_ATTEMPTS", "2")))
app.config.setdefault('AI_RETRY_BACKOFF_SECONDS', float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "2.0")))
app.config.setdefault('FORECAST_MAX_INPUT_POINTS', int(os.getenv('FORECAST_MAX_INPUT_POINTS', '4000')))
app.config.setdefault('FORECAST_BOOTSTRAP_SAMPLES', int(os.getenv('FORECAST_BOOTSTRAP_SAMPLES', '60')))
import logging
import re
from logging.handlers import RotatingFileHandler
import time 

os.environ.setdefault("NO_COLOR", "1")

class StripAnsiFormatter(logging.Formatter):
    _ansi = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    def format(self, record):
        s = super().format(record)
        return self._ansi.sub('', s)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

file_handler = RotatingFileHandler("app.log", maxBytes=2_000_000, backupCount=3)
file_handler.setLevel(log_level)
file_handler.setFormatter(StripAnsiFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter("%(message)s"))  

for h in (file_handler, console_handler):
    if not any(type(x) is type(h) for x in app.logger.handlers):
        app.logger.addHandler(h)

app.logger.setLevel(log_level)

werk = logging.getLogger("werkzeug")
werk.setLevel(log_level)
for h in (file_handler, console_handler):
    if not any(type(x) is type(h) for x in werk.handlers):
        werk.addHandler(h)

DEFAULT_AI_MODEL = (
    os.getenv("GENAI_MODEL")
    or os.getenv("GOOGLE_MODEL")
    # Prefer a free-tier-friendly default when no env override is provided
    or "models/gemini-3.0-flash"
)
MODEL_CACHE = {}
CURRENT_MODEL_NAME = None
AI_STATUS = {"configured": False, "ready": False, "message": "", "model": None}

def _sanitize_error_message(msg: str) -> str:
    try:
        s = str(msg or "").strip()
        if not s:
            return ""
        s = re.sub(r'(AIza[0-9A-Za-z\-_]{20,})', '***KEY***', s)
        s = re.sub(r'([?&]key=)([^&]+)', r'\1***KEY***', s, flags=re.I)
        if 'api key expired' in s.lower() or 'api_key_invalid' in s.lower():
            return "Google API key invalid or expired."
        if '429' in s or 'rate limit' in s.lower() or 'quota' in s.lower():
            if "doesn't have a free quota" in s.lower() or 'no free quota' in s.lower():
                return "Selected model has no free-tier quota. Switch to a free model (e.g., gemini-3.0-flash)."
            return "Rate limit exceeded. Please retry after a short pause."
        if 'content blocked' in s.lower() or 'block_reason' in s.lower():
            return "Content was blocked by safety filters."
        if 'timeout' in s.lower():
            return "AI request timed out."
        return s
    except Exception:
        return ""

def _set_ai_status(message: str | None = None, *, ready: bool | None = None, model_name: str | None = None, configured: bool | None = None):
    try:
        if configured is not None:
            AI_STATUS["configured"] = bool(configured)
        if ready is not None:
            AI_STATUS["ready"] = bool(ready)
        if model_name is not None:
            AI_STATUS["model"] = model_name
        if message is not None:
            AI_STATUS["message"] = _sanitize_error_message(message)
    except Exception:
        pass

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
    # Only repair the specific 'odels/' typo; do NOT alter correct 'models/'
    if n.startswith("odels/"):
        n = "m" + n

    candidates = [n]

    # Ensure we try both with and without the 'models/' prefix
    if n.startswith("models/"):
        candidates.append(n.replace("models/", "", 1))
    else:
        candidates.append("models/" + n)
    
    # Prioritize free-tier models early in the fallback chain
    preferred_fallbacks = [
        "gemini-3.0-flash", "models/gemini-3.0-flash",
        "gemini-2.5-flash", "models/gemini-2.5-flash",
        "gemini-1.5-flash", "models/gemini-1.5-flash",
        "gemini-2.5-pro", "models/gemini-2.5-pro",
        "gemini-1.5-pro", "models/gemini-1.5-pro",
    ]
    for fb in preferred_fallbacks:
        if fb not in candidates:
            candidates.append(fb)
    
    seen = set(); out = []
    for c in candidates:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

def _extract_text_from_gemini_response(resp) -> str:
    """
    Robustly extract plain text from a Gemini response.
    Falls back to concatenating candidate parts if .text isn't available.
    Returns '' if nothing textual is found.
    """
    try:
        t = getattr(resp, "text", None)
        if t:
            return str(t)
    except Exception as e:
        app.logger.warning("Gemini response.text accessor failed: %s", e)

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

def _make_model(name: str):
    m = genai.GenerativeModel(name)
    
    resp = m.generate_content("OK", request_options={"timeout": 15}, generation_config={"response_mime_type": "text/plain"})
    _ = _extract_text_from_gemini_response(resp)  
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


def configure_ai():
    global model, AI_ENABLED
    try:
        # Try service account JSON authentication first (Vertex AI)
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # If no env var set, look for a JSON credentials file in the app directory
        if not credentials_path:
            # Search for service account JSON files in the app directory
            for f in os.listdir(os.path.dirname(os.path.abspath(__file__)) or '.'):
                if f.endswith('.json') and 'service_account' not in f.lower():
                    try:
                        with open(f, 'r') as jf:
                            content = json.load(jf)
                            if content.get('type') == 'service_account':
                                credentials_path = os.path.abspath(f)
                                app.logger.info("Found service account JSON: %s", f)
                                break
                    except (json.JSONDecodeError, KeyError, IOError):
                        continue
        
        if credentials_path and os.path.exists(credentials_path) and GOOGLE_AUTH_AVAILABLE:
            # Use service account JSON for authentication
            try:
                scopes = ['https://www.googleapis.com/auth/generative-language', 
                          'https://www.googleapis.com/auth/cloud-platform']
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path, scopes=scopes
                )
                genai.configure(credentials=credentials)
                AI_ENABLED = True
                model = None
                app.logger.info("AI configured successfully using service account credentials.")
                try:
                    _set_ai_status("OK (Service Account)", ready=False, configured=True, model_name=None)
                except Exception:
                    pass
                return
            except Exception as e:
                app.logger.warning("Service account auth failed, falling back to API key: %s", e)
        
        # Fall back to API key authentication
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            AI_ENABLED = True
            model = None
            app.logger.info("AI configured successfully using API key.")
            try:
                _set_ai_status("OK", ready=False, configured=True, model_name=None)
            except Exception:
                pass
        else:
            raise ValueError("No valid authentication method found. Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS.")
    except Exception:
        app.logger.exception("AI configuration failed")
        model = None
        AI_ENABLED = False
        try:
            _set_ai_status("Configuration failed", ready=False, configured=False, model_name=None)
        except Exception:
            pass

app.logger.info("Attempting to configure AI...")
configure_ai()

def ensure_ai_ready() -> bool:
    """Ensure AI is enabled and a model is available. Attempt lazy init if needed."""
    global model, AI_ENABLED
    try:
        if not AI_ENABLED:
            try:
                _set_ai_status("AI disabled or not configured.", ready=False)
            except Exception:
                pass
            return False
        if model is None:
            try:
                model = get_or_create_model(DEFAULT_AI_MODEL)
            except Exception as e:
                app.logger.warning("Lazy model init failed: %s", e)
                try:
                    _set_ai_status(str(e), ready=False, model_name=None)
                except Exception:
                    pass
                return False
        try:
            _set_ai_status("OK", ready=True, model_name=CURRENT_MODEL_NAME)
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            app.logger.warning("ensure_ai_ready failed: %s", e)
            try:
                _set_ai_status(str(e), ready=False)
            except Exception:
                pass
        except Exception:
            pass
        return False

def _call_gemini(prompt: str, file_asset=None, *, timeout: int | None = None, retries: int | None = None, generation_config: dict | None = None):
    """
    Thin wrapper around Gemini calls that supports optional file context,
    retry/backoff, a request timeout, and generation config, returning the raw response object.
    """
    global model
    if not ensure_ai_ready():
        raise RuntimeError("AI is disabled or not configured.")
    attempts = int(app.config.get('AI_RETRY_ATTEMPTS', 2)) if retries is None else int(retries)
    backoff = float(app.config.get('AI_RETRY_BACKOFF_SECONDS', 2.0))
    timeout = int(app.config.get('AI_TIMEOUT_SECONDS', 30)) if timeout is None else int(timeout)
    last_err = None
    for i in range(max(1, attempts + 1)):
        try:
            content = [file_asset, prompt] if file_asset is not None else [prompt]
            # Sanitize generation_config to avoid unsupported MIME types
            gc_effective = generation_config or {}
            try:
                if isinstance(gc_effective, dict) and 'response_mime_type' in gc_effective:
                    allowed_mimes = {
                        'text/plain', 'application/json', 'application/xml', 'application/yaml', 'text/x.enum'
                    }
                    if gc_effective['response_mime_type'] not in allowed_mimes:
                        app.logger.debug(
                            "Coercing unsupported response_mime_type %s to text/plain",
                            gc_effective['response_mime_type']
                        )
                        gc_effective = dict(gc_effective)
                        gc_effective['response_mime_type'] = 'text/plain'
            except Exception:
                pass
            resp = model.generate_content(
                content,
                request_options={"timeout": timeout},
                generation_config=gc_effective
            )
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
            # If rate-limited or model lacks free quota, try switching to a free-tier model and/or reduce budget
            msg = str(getattr(e, 'message', None) or e)
            is_rate = ('429' in msg) or ('rate limit' in msg.lower()) or ('quota' in msg.lower())
            if is_rate:
                try:
                    # Attempt to switch to a known free-tier model
                    free_model = 'gemini-1.5-flash'
                    def _strip_models_prefix(x: str) -> str:
                        return x[7:] if isinstance(x, str) and x.startswith('models/') else x
                    current_eq_free = (
                        _strip_models_prefix(CURRENT_MODEL_NAME or '') == _strip_models_prefix(free_model)
                    )
                    if not current_eq_free:
                        app.logger.warning("Rate limit or no free quota; switching model to %s", free_model)
                        free = get_or_create_model(free_model)
                        if free is not None:
                            globals()['model'] = free
                    # Reduce generation budget on retry
                    if generation_config is not None:
                        gc = dict(generation_config)
                        if 'max_output_tokens' in gc:
                            try:
                                gc['max_output_tokens'] = max(256, int(gc['max_output_tokens']) // 2)
                            except Exception:
                                gc['max_output_tokens'] = 512
                        else:
                            gc['max_output_tokens'] = 512
                        gc['temperature'] = min(0.4, float(gc.get('temperature', 0.4)))
                        generation_config = gc
                except Exception as e2:
                    app.logger.info("Model switch on rate limit failed: %s", e2)
            try:
                _set_ai_status(str(last_err), ready=False, model_name=CURRENT_MODEL_NAME)
            except Exception:
                pass

            if i < attempts:
                try:
                    # Exponential backoff with longer wait for rate limits
                    sleep_s = backoff * (i + 1)
                    # If rate-limited, wait much longer before retry
                    if is_rate:
                        sleep_s = max(sleep_s, 10 * (i + 1))  # Minimum 10s, 20s, 30s...
                        app.logger.info("Rate limit detected, waiting %.1f seconds before retry...", sleep_s)
                    time.sleep(sleep_s)
                except Exception:
                    pass
                continue
            # Exhausted attempts
            raise last_err

class TinyLRU(OrderedDict):
    def __init__(self, max_items=6, max_size_mb=None):
        super().__init__()
        self.max_items = max_items
        self.max_size_mb = max_size_mb  # optional size limit in MB
    def get(self, key, default=None):
        if key in self:
            val = super().pop(key)
            super().__setitem__(key, val)  
            return val
        return default
    def set(self, key, value):
        if key in self:
            super().pop(key)
        super().__setitem__(key, value)
        # Evict based on item count
        while len(self) > self.max_items:
            self.popitem(last=False)
        # Evict based on size (for DataFrames)
        if self.max_size_mb:
            try:
                import sys
                total_bytes = sum(sys.getsizeof(v) for v in self.values() if v is not None)
                while total_bytes > (self.max_size_mb * 1024 * 1024) and len(self) > 1:
                    self.popitem(last=False)
                    total_bytes = sum(sys.getsizeof(v) for v in self.values() if v is not None)
            except Exception:
                pass  # If size calculation fails, rely on item count limit only

DATAFRAME_CACHE = TinyLRU(max_items=app.config['MAX_CACHE_ITEMS'], max_size_mb=int(os.getenv('DATAFRAME_CACHE_MAX_MB', '200')))
NAME_MAP_PATH = os.path.join(UPLOAD_FOLDER, "_name_map.json")  
# Allow configuration of file upload size limit to Gemini (smaller = faster, less timeout risk)
app.config['AI_FULL_UPLOAD_MAX_MB'] = int(os.getenv('AI_FULL_UPLOAD_MAX_MB', '5'))  
AI_FILE_MAP = {}  
ORIGINAL_NAME_MAP = {}  
AI_SUMMARY_CACHE = {}
QNA_CACHE = TinyLRU(max_items=50)
FORECAST_CACHE = TinyLRU(max_items=32)

if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
    app.logger.addHandler(file_handler)

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
    """Delete a file with small retries to tolerate transient locks (e.g., OneDrive/AV).
    Returns tuple (success: bool, error_message: str | None)
    """
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True, None
        except PermissionError as e:
            app.logger.warning("Delete failed (permission denied %s), attempt %d/%d: %s", path, i + 1, retries, e)
            if i < retries - 1:
                time.sleep(delay)
            else:
                return False, f"Permission denied (file may be locked by OneDrive or antivirus)"
        except Exception as e:
            app.logger.warning("Delete failed (%s), attempt %d/%d: %s", path, i + 1, retries, e)
            if i < retries - 1:
                time.sleep(delay)
            else:
                return False, str(e)
    return False, "File deletion failed after retries"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)
_load_name_map()

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

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
    s = re.sub(r'^\s*```(?:\w+)?\s*\n?', '', s, flags=re.I | re.M)
    s = re.sub(r'\n?\s*```\s*$', '', s, flags=re.M)
    s = s.replace("```", "")
    if re.search(r'&lt;/?[a-zA-Z]', s):
        try:
            s = htmllib.unescape(s)
        except Exception:
            pass
    s = re.sub(r'</?\s*(html|body)[^>]*>', '', s, flags=re.I)
    s = re.sub(r'<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>', '', s, flags=re.I | re.S)
    s = re.sub(r'\s+on\w+\s*=\s*(".*?"|\'.*?\'|\w+)', '', s, flags=re.I)
    s = re.sub(r'javascript\s*:', '', s, flags=re.I)
    s = s.strip()
    if not re.search(r'</?(h[1-6]|p|ul|ol|li|strong|em|b|i|br|table|thead|tbody|tr|th|td|a)\b', s, re.I):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = "<p>" + "</p><p>".join(lines) + "</p>" if lines else "<p></p>"
    return s

def html_to_plain_text(html: str) -> str:
    """Convert HTML to plain text for PDF output (simple fallback version)."""
    if not html:
        return ""
    s = str(html)
    # Strip all tags
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def convert_html_to_formatted_text(html: str) -> str:
    """Convert HTML to well-formatted plain text for PDF output.
    Preserves structure: headers become UPPERCASE with separators,
    lists become bullet points, paragraphs are separated by blank lines.
    """
    if not html:
        return ""
    
    s = str(html)
    
    # Handle HTML entities first
    s = s.replace('&nbsp;', ' ')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&quot;', '"')
    s = s.replace('&#39;', "'")
    
    # Convert headers to UPPERCASE with separators
    def header_replace(m):
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        separator = '=' * min(len(text), 50)
        return f"\n\n{separator}\n{text.upper()}\n{separator}\n"
    s = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', header_replace, s, flags=re.I | re.S)
    
    # Convert <br> to newlines
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    
    # Convert </p> to double newline (paragraph break)
    s = re.sub(r'</p>', '\n\n', s, flags=re.I)
    
    # Convert list items to bullet points
    s = re.sub(r'<li[^>]*>(.*?)</li>', lambda m: '  - ' + re.sub(r'<[^>]+>', '', m.group(1)).strip() + '\n', s, flags=re.I | re.S)
    
    # Convert <strong> and <b> content - preserve the text with ** markers
    s = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', lambda m: '**' + re.sub(r'<[^>]+>', '', m.group(2)).strip() + '**', s, flags=re.I | re.S)
    
    # Convert <em> and <i> content - preserve the text with * markers
    s = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', lambda m: '*' + re.sub(r'<[^>]+>', '', m.group(2)).strip() + '*', s, flags=re.I | re.S)
    
    # Handle tables - simple text representation
    s = re.sub(r'<table[^>]*>', '\n', s, flags=re.I)
    s = re.sub(r'</table>', '\n', s, flags=re.I)
    s = re.sub(r'<tr[^>]*>', '', s, flags=re.I)
    s = re.sub(r'</tr>', '\n', s, flags=re.I)
    s = re.sub(r'<t[hd][^>]*>(.*?)</t[hd]>', lambda m: re.sub(r'<[^>]+>', '', m.group(1)).strip() + '  |  ', s, flags=re.I | re.S)
    
    # Remove all remaining HTML tags
    s = re.sub(r'<[^>]+>', '', s)
    
    # Clean up whitespace
    s = re.sub(r'[ \t]+', ' ', s)  # Multiple spaces to single
    s = re.sub(r'\n[ \t]+', '\n', s)  # Remove leading whitespace on lines
    s = re.sub(r'[ \t]+\n', '\n', s)  # Remove trailing whitespace on lines
    s = re.sub(r'\n{4,}', '\n\n\n', s)  # Max 3 consecutive newlines
    
    return s.strip()

# Emoji to text replacements for PDF (since most PDF fonts don't support emojis)
EMOJI_REPLACEMENTS = {
    '📊': '[CHART]',
    '🔍': '[SEARCH]',
    '📈': '[TREND UP]',
    '⏱️': '[TIME]',
    '🔗': '[LINK]',
    '⚠️': '[WARNING]',
    '🔮': '[PREDICTION]',
    '💡': '[TIP]',
    '✅': '[OK]',
    '❌': '[X]',
    '📉': '[TREND DOWN]',
    '🎯': '[TARGET]',
    '📋': '[LIST]',
    '🔢': '[NUM]',
    '📝': '[NOTE]',
    '🚀': '[ROCKET]',
    '⭐': '[STAR]',
    '🔥': '[HOT]',
    '💰': '[MONEY]',
    '📅': '[DATE]',
    '🕐': '[CLOCK]',
    '➡️': '->',
    '⬆️': '^',
    '⬇️': 'v',
    '✓': '[OK]',
    '•': '-',
}

def replace_emojis_for_pdf(text: str) -> str:
    """Remove emojis from text for PDF compatibility."""
    if not text:
        return ""
    result = text
    # Remove all known emojis
    for emoji in EMOJI_REPLACEMENTS.keys():
        result = result.replace(emoji, '')
    # Remove any remaining emojis (Unicode ranges for emojis)
    result = re.sub(r'[\U0001F300-\U0001F9FF]', '', result)
    result = re.sub(r'[\u2600-\u26FF]', '', result)
    result = re.sub(r'[\u2700-\u27BF]', '', result)
    # Clean up any double spaces left behind
    result = re.sub(r'  +', ' ', result)
    return result

class PDFStyledText:
    """Helper class to render styled HTML content to PDF with bold/italic support."""
    
    def __init__(self, pdf, font_family="helvetica", base_size=10):
        self.pdf = pdf
        self.font_family = font_family
        self.base_size = base_size
    
    def render_html(self, html: str):
        """Parse HTML and render with proper formatting to PDF."""
        if not html:
            return
        
        # Replace emojis first
        html = replace_emojis_for_pdf(html)
        
        # Handle HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        
        # Process content by sections
        # Split by headers first
        sections = re.split(r'(<h[1-6][^>]*>.*?</h[1-6]>)', html, flags=re.I | re.S)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Check if this is a header
            header_match = re.match(r'<h([1-6])[^>]*>(.*?)</h\1>', section, re.I | re.S)
            if header_match:
                level = int(header_match.group(1))
                header_text = self._strip_tags(header_match.group(2))
                self._render_header(header_text, level)
            else:
                # Process regular content
                self._render_content(section)
    
    def _strip_tags(self, text: str) -> str:
        """Remove all HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text).strip()
    
    def _render_header(self, text: str, level: int):
        """Render a header with appropriate styling."""
        # Font sizes based on header level
        sizes = {1: 16, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10}
        size = sizes.get(level, 12)
        
        self.pdf.ln(4)
        self.pdf.set_font(self.font_family, 'B', size)
        self.pdf.set_fill_color(240, 240, 240)
        
        # Encode for latin-1 if needed
        safe_text = self._safe_encode(text)
        self.pdf.multi_cell(0, 6, safe_text, fill=True)
        self.pdf.ln(2)
        self.pdf.set_font(self.font_family, '', self.base_size)
    
    def _render_content(self, content: str):
        """Render content with inline styling (bold, italic, lists)."""
        if not content or not content.strip():
            return
            
        # Handle lists
        if '<ul' in content.lower() or '<ol' in content.lower():
            self._render_list(content)
            # Also render any content outside the list
            outside_list = re.sub(r'<[uo]l[^>]*>.*?</[uo]l>', '', content, flags=re.I | re.S)
            if outside_list.strip():
                self._render_content(outside_list)
            return
        
        # Handle paragraphs and inline content
        # Split by closing tags and line breaks
        paragraphs = re.split(r'</p>|<br\s*/?>|\n', content, flags=re.I)
        
        rendered_anything = False
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Remove opening <p> tags and any other block-level opening tags
            para = re.sub(r'<p[^>]*>', '', para, flags=re.I)
            para = re.sub(r'<div[^>]*>', '', para, flags=re.I)
            para = re.sub(r'</div>', '', para, flags=re.I)
            
            para = para.strip()
            if not para:
                continue
            
            self._render_styled_paragraph(para)
            self.pdf.ln(3)
            rendered_anything = True
        
        # Fallback: if nothing was rendered, render the whole content as plain text
        if not rendered_anything:
            plain_text = self._strip_tags(content)
            if plain_text:
                self.pdf.set_font(self.font_family, '', self.base_size)
                safe_text = self._safe_encode(plain_text)
                self.pdf.multi_cell(0, 5, safe_text)
                self.pdf.ln(3)
    
    def _render_list(self, content: str):
        """Render a list with bullet points."""
        items = re.findall(r'<li[^>]*>(.*?)</li>', content, re.I | re.S)
        
        for item in items:
            item_text = self._strip_tags(item).strip()
            if not item_text:
                continue
            
            self.pdf.set_font(self.font_family, '', self.base_size)
            safe_text = self._safe_encode(f"  - {item_text}")
            self.pdf.multi_cell(0, 5, safe_text)
        
        self.pdf.ln(2)
    
    def _render_styled_paragraph(self, para: str):
        """Render a paragraph with bold/italic inline styling."""
        # Parse inline styles and render them
        # We'll use a simple approach: find styled segments and render them
        
        # Pattern to find bold/italic segments
        segments = []
        pos = 0
        
        # Combined pattern for bold and italic
        pattern = re.compile(r'<(strong|b|em|i)[^>]*>(.*?)</\1>', re.I | re.S)
        
        for match in pattern.finditer(para):
            # Add text before the match as normal
            if match.start() > pos:
                before_text = self._strip_tags(para[pos:match.start()])
                if before_text:
                    segments.append(('', before_text))
            
            # Add the styled segment
            tag = match.group(1).lower()
            style = 'B' if tag in ('strong', 'b') else 'I'
            styled_text = self._strip_tags(match.group(2))
            if styled_text:
                segments.append((style, styled_text))
            
            pos = match.end()
        
        # Add remaining text
        if pos < len(para):
            remaining = self._strip_tags(para[pos:])
            if remaining:
                segments.append(('', remaining))
        
        # If no segments found, just render plain text
        if not segments:
            plain = self._strip_tags(para)
            if plain:
                self.pdf.set_font(self.font_family, '', self.base_size)
                safe_text = self._safe_encode(plain)
                self.pdf.multi_cell(0, 5, safe_text)
            return
        
        # Render segments - combine into single text and use multi_cell
        # Since fpdf2's write() has issues, we'll render each styled segment separately
        for style, text in segments:
            if not text:
                continue
            self.pdf.set_font(self.font_family, style, self.base_size)
            safe_text = self._safe_encode(text)
            # Use multi_cell with no width limit for safety
            self.pdf.multi_cell(0, 5, safe_text, new_x="LMARGIN", new_y="NEXT")
    
    def _safe_encode(self, text: str) -> str:
        """Safely encode text for PDF output."""
        if not text:
            return ""
        # Replace emojis again just in case
        text = replace_emojis_for_pdf(text)
        # Encode to latin-1, replacing unsupported chars
        try:
            return text.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            return text

def _is_offline_html(s: str) -> bool:
    try:
        t = (s or "").lower()
        return ("<h3>offline analysis</h3>" in t) or ("ai response unavailable" in t)
    except Exception:
        return False

def _diagnose_gemini_response(resp) -> str:
    """Return a compact diagnostic string from a Gemini response object.
    Includes finish_reason, prompt block reason, and any safety ratings if present."""
    try:
        parts = []
        try:
            pf = getattr(resp, 'prompt_feedback', None)
            if pf and getattr(pf, 'block_reason', None):
                br = getattr(pf, 'block_reason', None)
                parts.append(f"prompt_block={getattr(br, 'name', br)}")
        except Exception:
            pass
        try:
            cands = getattr(resp, 'candidates', None) or []
            if cands:
                fr = getattr(cands[0], 'finish_reason', None)
                if fr is not None:
                    parts.append(f"finish_reason={getattr(fr, 'name', fr)}")
                # Try safety ratings
                sr = getattr(cands[0], 'safety_ratings', None)
                if sr:
                    try:
                        labels = []
                        for r in sr:
                            cat = getattr(r, 'category', None)
                            prob = getattr(r, 'probability', None)
                            labels.append(f"{getattr(cat, 'name', cat)}:{getattr(prob, 'name', prob)}")
                        if labels:
                            parts.append("safety=[" + ", ".join(labels) + "]")
                    except Exception:
                        pass
        except Exception:
            pass
        return ("; ".join(parts)) or ""
    except Exception:
        return ""

def _get_finish_reason(resp) -> str | None:
    """Extract the primary candidate's finish_reason as a string if present."""
    try:
        cands = getattr(resp, 'candidates', None) or []
        if not cands:
            return None
        fr = getattr(cands[0], 'finish_reason', None)
        if fr is None:
            return None
        return getattr(fr, 'name', str(fr))
    except Exception:
        return None

def html_to_pdf_bytes(html: str) -> bytes:
    raise RuntimeError("PDF export is disabled because no renderer (weasyprint/pdfkit/xhtml2pdf) is installed.")


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

        if hasattr(response, "prompt_feedback"):
            pf = getattr(response, "prompt_feedback", None)
            if pf and getattr(pf, "block_reason", None):
                block_reason = getattr(pf, "block_reason", None)
                app.logger.warning("AI analysis blocked: %s", block_reason)
                return f"AI analysis was blocked by the content filter. Reason: {block_reason}"

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

        app.logger.debug("Successfully got AI summary")
        return sanitize_ai_html(text)
    except Exception as e:
        app.logger.exception("AI summary call failed")
        return f"An error occurred during AI analysis. Check the terminal for more details. Error: {e}"

def get_ai_answer(dataframe, question):
    """Generates a specific answer to a user's question about the dataframe."""
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."

    try:
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
    if not ensure_ai_ready():
        return "AI analysis is disabled."

    try:
        df_desc = describe_for_ai(df)
    except Exception:
        df_desc = ""
    prompt = (
        "You are a senior data analyst with expertise in statistical analysis and predictive insights. "
        "Produce a comprehensive, insightful HTML analysis of this dataset. "
        "Respond strictly in HTML (no Markdown) using <p>, <ul><li>, <table><thead><tbody><tr><th><td>, <strong>, <em>, <h3> for section headers. "
        "Structure your analysis into clearly separated sections with headers:\n\n"
        
        "<h3>📊 Dataset Overview</h3>\n"
        "Describe what this dataset captures, its size (rows × columns), time span if temporal, and primary purpose/domain.\n\n"
        
        "<h3>🔍 Data Quality Assessment</h3>\n"
        "Analyze: missing values (% per key columns), constant/near-constant columns, data types, suspected quality issues, "
        "duplicate records, outliers. Rate overall quality (Excellent/Good/Fair/Poor) with brief justification.\n\n"
        
        "<h3>📈 Key Statistical Insights</h3>\n"
        "Provide distributions (mean, median, std dev), ranges (min/max), skewness. Identify interesting patterns, "
        "dominant segments, notable ratios. Use concrete numbers. Highlight any surprising findings.\n\n"
        
        "<h3>⏱️ Temporal Patterns & Trends</h3>\n"
        "If time-indexed: identify trends (upward/downward/stable), seasonality (daily/weekly/monthly cycles), "
        "volatility periods, growth rates. If not temporal, state '<p>Not applicable (no datetime index).</p>'\n\n"
        
        "<h3>🔗 Correlations & Relationships</h3>\n"
        "Highlight strongest positive and negative correlations (cite correlation coefficients). "
        "Explain what these relationships might indicate. Identify potential causal links vs. coincidences.\n\n"
        
        "<h3>⚠️ Anomalies & Irregularities</h3>\n"
        "Flag unusual spikes, dips, gaps, or inconsistencies. Estimate their severity and potential causes. "
        "If none found, state clearly.\n\n"
        
        "<h3>🔮 Prognosis & Future Outlook</h3>\n"
        "Based on observed trends and patterns, what are plausible future directions? Identify: "
        "expected continuations, potential risks, growth/decline scenarios. Be specific but qualify uncertainty. "
        "State confidence level (high/medium/low) for each prediction.\n\n"
        
        "<h3>💡 Actionable Observations</h3>\n"
        "Provide 3-5 concrete, prioritized recommendations for: data collection improvements, further analyses, "
        "decision-making insights, or intervention points. Use bullet points.\n\n"
        
        "<h3>⚠️ Limitations & Caveats</h3>\n"
        "Acknowledge: sampling biases, data sparsity, interpretability constraints, assumptions made, "
        "or factors that could invalidate conclusions.\n\n"
        
        "Guidelines:\n"
        "- Be analytical and precise—cite concrete numbers (counts, percentages, ranges, correlation coefficients)\n"
        "- Focus on conclusions and insights, not just descriptions\n"
        "- Explain the 'so what'—why findings matter\n"
        "- Be honest about uncertainty\n"
        "- Avoid jargon; explain technical terms briefly\n"
        "- Do NOT hallucinate fields or values not in the context\n"
        "- Keep professional tone; avoid marketing language\n\n"
        
        "Context:\n" + df_desc + ("\n\nAdditional context:\n" + extra_context if extra_context else "")
    )

    gen_cfg = {
        "max_output_tokens": 16384,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "text/plain",
    }

    try:
        # First try including the file as context (when available)
        resp = _call_gemini(prompt, file_asset=file_asset, generation_config=gen_cfg)
        text = _extract_text_from_gemini_response(resp).strip()
        if not text:
            try:
                diag = _diagnose_gemini_response(resp)
                if diag:
                    app.logger.warning("AI summary empty response: %s", diag)
            except Exception:
                pass
            # Try a simpler prompt with file context
            try:
                simple_cfg = {"max_output_tokens": 384, "temperature": 0.2, "response_mime_type": "text/plain"}
                simple_prompt = "Provide a concise HTML summary of the dataset using <p> and <ul><li> only."
                simple = _call_gemini(simple_prompt + "\n\n" + extra_context, file_asset=file_asset, generation_config=simple_cfg)
                text2 = _extract_text_from_gemini_response(simple).strip()
                if text2:
                    return sanitize_ai_html(text2)
            except Exception:
                pass
            # As a last resort, try text-only (no file attachment)
            try:
                resp2 = _call_gemini(prompt, file_asset=None, generation_config={**gen_cfg, "max_output_tokens": 768})
                text_only = _extract_text_from_gemini_response(resp2).strip()
                if text_only:
                    return sanitize_ai_html(text_only)
            except Exception as e2:
                app.logger.info("Text-only fallback failed for AI summary: %s", e2)
            # Include diagnostics in the raised error so UI shows exact reason
            d = _diagnose_gemini_response(resp)
            raise RuntimeError("Empty AI response" + (f" ({d})" if d else ""))

        # If truncated by tokens, request one short continuation and append
        try:
            fr = _get_finish_reason(resp)
            if isinstance(fr, str) and "MAX_TOKENS" in fr:
                cont_prompt = (
                    "Continue the same HTML summary in the same style. Do not repeat previous text. "
                    "Only output valid HTML fragments (<p>, <ul><li>, <table>).\n\n"
                    f"Previous tail for context (do not repeat):\n{text[-1200:]}"
                )
                cont = _call_gemini(cont_prompt, file_asset=file_asset, generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40,
                    "response_mime_type": "text/plain",
                })
                more = _extract_text_from_gemini_response(cont).strip()
                if more:
                    text = text + "\n" + more
        except Exception as ce:
            app.logger.info("Summary continuation skipped: %s", ce)

        return sanitize_ai_html(text)
    except Exception as e:
        app.logger.warning("AI summary failed, falling back. Error: %s", e)
        return offline_answer(df, "summary", error=e)
    
def get_ai_answer_with_file(df: pd.DataFrame, question: str, file_asset=None, filename: str | None = None) -> str:
    """
    Answer a user's question about the dataset.
    - Uses the uploaded file as context if available (file_asset).
    - Returns sanitized HTML.
    - Falls back to an offline deterministic answer on error or empty AI response.
    """
    try:
        cache_key = None
        try:
            # Build a stable cache key using DataFrame shape and question text
            df_sig = (tuple(df.shape) if isinstance(df, pd.DataFrame) else (None, None))
            q_norm = (question or '').strip().lower()
            cache_key = (df_sig, q_norm)
            cached = QNA_CACHE.get(cache_key)
            if isinstance(cached, str) and cached.strip():
                return cached
        except Exception:
            pass
        if not ensure_ai_ready():
            return offline_answer(df, question, error="AI disabled.")

        # Prefer reusing the previously generated AI summary as compact context to
        # avoid rebuilding a large dataframe description and hitting token limits.
        summary_html = None
        if filename:
            try:
                summary_html = AI_SUMMARY_CACHE.get(filename)
            except Exception:
                summary_html = None

        if summary_html:
            try:
                # Strip HTML tags to reduce token count; keep text content.
                context_text = re.sub(r"<[^>]+>", " ", str(summary_html))
                context_text = re.sub(r"\s+", " ", context_text).strip()
            except Exception:
                context_text = describe_for_ai(df)
        else:
            context_text = describe_for_ai(df)

        prompt = f"""
You are a senior data scientist. Answer the user's question about the dataset clearly and precisely.
Respond strictly in HTML (no Markdown), using tags like <p>, <ul><li>, <table><thead><tbody><tr><th><td>, <strong>, and <em>.
Cite concrete numbers or ranges from the provided context when relevant. If something is uncertain, say so briefly.

Context (from the earlier AI summary; do not regenerate a summary):
{context_text}

Question:
{question}
""".strip()

        resp = _call_gemini(prompt, file_asset=file_asset, generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "response_mime_type": "text/plain",
        })
        text = _extract_text_from_gemini_response(resp).strip()

        if not text:
            try:
                diag = _diagnose_gemini_response(resp)
                if diag:
                    app.logger.warning("AI Q&A empty response: %s", diag)
            except Exception:
                pass
            try:
                simple = _call_gemini(
                    f"Answer briefly in HTML (<p>, <ul><li>) only: {question}\n\nContext:\n{context_text}",
                    file_asset=file_asset,
                    generation_config={"max_output_tokens": 384, "temperature": 0.2, "response_mime_type": "text/plain"},
                )
                text2 = _extract_text_from_gemini_response(simple).strip()
                if text2:
                    html2 = sanitize_ai_html(text2)
                    try:
                        if cache_key:
                            QNA_CACHE.set(cache_key, html2)
                    except Exception:
                        pass
                    return html2
            except Exception:
                pass
            # Try again without attaching the file (some environments reject file references)
            try:
                resp2 = _call_gemini(prompt, file_asset=None, generation_config={
                    "max_output_tokens": 640,
                    "temperature": 0.25,
                    "top_p": 0.95,
                    "top_k": 40,
                    "response_mime_type": "text/plain",
                })
                text3 = _extract_text_from_gemini_response(resp2).strip()
                if text3:
                    html3 = sanitize_ai_html(text3)
                    try:
                        if cache_key:
                            QNA_CACHE.set(cache_key, html3)
                    except Exception:
                        pass
                    return html3
            except Exception as e2:
                app.logger.info("Text-only fallback failed for AI Q&A: %s", e2)
            d = _diagnose_gemini_response(resp)
            return offline_answer(df, question, error=("Empty AI response" + (f" ({d})" if d else "")))

        # If truncated, try a continuation to complete the answer
        try:
            fr = _get_finish_reason(resp)
            if isinstance(fr, str) and "MAX_TOKENS" in fr:
                app.logger.info("Q&A response truncated (MAX_TOKENS), attempting continuation...")
                cont_prompt = (
                    "Continue the same HTML answer to the user's question. Do not repeat previous text. "
                    "Only output valid HTML fragments (<p>, <ul><li>, <table>).\n\n"
                    f"Question: {question}\n"
                    f"Previous tail for context (do not repeat):\n{text[-900:]}"
                )
                try:
                    cont = _call_gemini(cont_prompt, file_asset=file_asset, generation_config={
                        "max_output_tokens": 1536,
                        "temperature": 0.25,
                        "top_p": 0.95,
                        "top_k": 40,
                        "response_mime_type": "text/plain",
                    })
                    more = _extract_text_from_gemini_response(cont).strip()
                    if more:
                        text = text + "\n" + more
                        app.logger.info("Q&A continuation successful")
                    else:
                        app.logger.warning("Q&A continuation returned empty, using truncated response")
                        # Add notice that response was truncated
                        text = text + "\n<p><em>(Response was truncated due to length)</em></p>"
                except Exception as cont_err:
                    app.logger.warning("Q&A continuation failed: %s, using truncated response", cont_err)
                    # Add notice that response was truncated
                    text = text + "\n<p><em>(Response was truncated due to length)</em></p>"
        except Exception as ce:
            app.logger.info("Q&A continuation check skipped: %s", ce)

        html = sanitize_ai_html(text)
        try:
            if cache_key and not _is_offline_html(html):
                QNA_CACHE.set(cache_key, html)
        except Exception:
            pass
        return html

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
        if isinstance(ai_html, str) and not _is_offline_html(ai_html):
            AI_SUMMARY_CACHE[filename] = ai_html
        return ai_html
    except Exception as e:
        return f"<p>AI summary unavailable: {e}</p>"

def generate_plot(data, title, xlabel, ylabel, anomalies_idx=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax, label='History', color='tab:blue', lw=1.2)
    if anomalies_idx is not None and len(anomalies_idx):
        aligned = data.loc[data.index.intersection(anomalies_idx)]
        ax.scatter(aligned.index, aligned.values, color='red', s=18, zorder=5, label='Anomaly')
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8'); plt.close(fig); return img

def generate_correlation_heatmap(df, method='spearman', title='Correlation Heatmap'):
    """Generate a correlation heatmap as base64 image."""
    try:
        import seaborn as sns
        
        # Get numeric columns
        df_num = coerce_numeric_df(df)
        sel = df_num.select_dtypes(include='number')
        if sel.empty:
            return None
        
        # Remove constant columns
        nunique = sel.nunique(dropna=True)
        sel = sel.loc[:, nunique > 1]
        
        if sel.shape[1] < 2:
            return None
        
        # Compute correlation
        corr = sel.corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img
    except Exception:
        try:
            plt.close(fig)
        except:
            pass
        return None

def _thin_series(s: pd.Series, max_points: int) -> pd.Series:
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
                    out = pd.concat([out, s.iloc[[-1]]])
            except Exception:
                pass
            return out
        return s
    except Exception:
        return s

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
                out.append({'title': '', 'img': p})
        except Exception:
            continue
    return out

def normalize_timeseries(series: pd.Series) -> pd.Series:
    """
    Ensure a numeric series with a clean, timezone-naive, sorted DatetimeIndex when possible.
    - Coerces values to numeric and drops NaNs.
    - If index is DatetimeIndex, make it tz-naive, sort it, and drop duplicate index entries (keep last).
    - Otherwise, return the numeric series as-is.
    """
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

def generate_stl_plot(series: pd.Series, title: str, seasonal_period: int):
    try:
        s = normalize_timeseries(series)
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

def _infer_future_index(idx, steps):
    if isinstance(idx, pd.DatetimeIndex):
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
            print(f"[DEBUG] Forecast: {steps} steps, offset={offset}, last_date={idx[-1]}, forecast_end={idx[-1] + offset * steps}")
        except:
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
    if not isinstance(idx, pd.DatetimeIndex):
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

    x = np.arange(len(y_win), dtype=float)
    slope_lr, intercept = np.polyfit(x, y_win.values, 1)
    diffs = np.diff(y_win.values)
    med_diff = float(np.median(diffs)) if len(diffs) else 0.0

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

    resid = y_win.values - (slope_lr * x + intercept)
    resid_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else float(np.nanstd(y_win.values, ddof=1))
    lower = fc_mean - 1.96 * resid_std
    upper = fc_mean + 1.96 * resid_std
    ci = pd.concat([lower, upper], axis=1)
    ci.columns = ['lower', 'upper']
    return fc_mean, ci

def _pattern_replay_forecast(series: pd.Series, steps: int, seasonal_period: int, noise_scale: float = 0.15):
    """Replay the most recent seasonal cycle increments to build a natural forecast.
    - Use last seasonal_period points as a template; compute cyclic diffs; iterate to build future.
    - Add light jitter from recent residual distribution to avoid a perfectly repeated path.
    Returns (forecast_series, conf_df approx) with a simple CI from recent std.
    """
    try:
        y = pd.to_numeric(series, errors='coerce').dropna()
        n = len(y)
        sp = int(seasonal_period)
        if n < max(8, sp + 3) or sp < 2:
            return _recent_slope_forecast(series, steps, window=None, damping=None)
        base = y.iloc[-sp:]
        inc = np.diff(base.values, prepend=base.values[0])  # first diff 0, then increments

        # residuals around a local smoothing to get jitter scale and drift
        w = max(sp, min(200, n//3))
        y_win = y.tail(w)
        x = np.arange(len(y_win), dtype=float)
        slope_lr, intercept = np.polyfit(x, y_win.values, 1)
        fit = slope_lr * x + intercept
        resid = (y_win.values - fit)
        rs = float(np.nanstd(resid, ddof=1))
        jitter = rs * float(noise_scale)
        # light drift to follow recent trend without dominating pattern
        drift = float(slope_lr) * 0.25

        vals = []
        cur = y.iloc[-1]
        for k in range(steps):
            d = inc[(k + 1) % sp]
            if np.isfinite(jitter) and jitter > 0:
                # light jitter from residuals distribution
                noise = np.random.normal(0.0, jitter)
            else:
                noise = 0.0
            cur = cur + d + drift + noise
            vals.append(cur)
        future_idx = _infer_future_index(y.index, steps)
        fc = pd.Series(vals, index=future_idx)
        # Simple CI from residual std
        lower = fc - 1.96 * rs
        upper = fc + 1.96 * rs
        ci = pd.concat([lower, upper], axis=1)
        ci.columns = ['lower', 'upper']
        return fc, ci
    except Exception:
        return _recent_slope_forecast(series, steps, window=None, damping=None)

def _forecast_natural(series: pd.Series, steps: int):
    """Simple, natural forecast that reuses portions of actual data.
    Strategy:
      - Compute recent increments (diffs) from a trailing window.
      - If a seasonal period is detectable (datetime index), replay the last seasonal increments cyclically.
        Otherwise, replay the last K increments cyclically.
      - Add small AR(1) noise proportional to increment volatility to avoid flatlines.
      - Start from the last observed value (hard continuity).
      - Apply a light envelope clip based on recent quantiles to avoid extremes.
      - Build a basic constant-width CI from increment volatility.
    """
    try:
        y = pd.to_numeric(series, errors='coerce').dropna()
        n = len(y)
        future_idx = _infer_future_index(series.index, steps)
        if n < 3:
            fc_mean = pd.Series([y.iloc[-1] if n else 0.0] * steps, index=future_idx)
            ci = pd.DataFrame({"lower": fc_mean.copy(), "upper": fc_mean.copy()})
            return fc_mean, ci

        # Choose a trailing window
        w = min(n, max(30, n // 3))
        tail = y.iloc[-w:]
        diffs = np.diff(tail.values)
        if diffs.size == 0:
            fc_mean = pd.Series([y.iloc[-1]] * steps, index=future_idx)
            ci = pd.DataFrame({"lower": fc_mean.copy(), "upper": fc_mean.copy()})
            return fc_mean, ci

        # Prefer seasonal template when available
        sp = _infer_seasonal_period(y.index) if isinstance(y.index, pd.DatetimeIndex) else None
        if isinstance(sp, int) and sp >= 2 and n >= sp + 3:
            # last sp+1 points to compute sp increments
            recent = y.iloc[-(sp + 1):].values.astype(float)
            template = np.diff(recent)  # length sp
            if template.size < 2:
                template = diffs[-min(len(diffs), max(8, steps)):]  # fallback
        else:
            K = min(steps, max(8, min(60, len(diffs))))
            template = diffs[-K:]

        # Build increments by cycling the template
        incs = np.resize(template, steps).astype(float)

        # AR(1) noise proportional to increment volatility; slightly scale with horizon
        inc_std = float(np.nanstd(diffs, ddof=1)) if diffs.size else 0.0
        hf = float(steps) / max(1.0, float(n))
        sigma = (0.18 + 0.22 * min(1.0, hf)) * inc_std if (np.isfinite(inc_std) and inc_std > 0) else 0.0
        rho = 0.6
        if sigma > 0:
            e = np.zeros(steps, dtype=float)
            for i in range(steps):
                shock = np.random.normal(0.0, sigma)
                e[i] = (rho * (e[i-1] if i > 0 else 0.0)) + shock
            incs = incs + e

        # Integrate increments starting from last value (continuity)
        vals = float(y.iloc[-1]) + np.cumsum(incs)
        fc = pd.Series(vals, index=future_idx)
        if len(fc) > 0:
            fc.iloc[0] = float(y.iloc[-1])

        # Light envelope clip based on recent quantiles + pad
        try:
            lo, hi = np.quantile(tail.values, [0.02, 0.98])
        except Exception:
            lo, hi = float(np.min(tail.values)), float(np.max(tail.values))
        rng = float(max(1e-9, hi - lo))
        pad = 0.2 * rng
        fc_vals = np.minimum(np.maximum(fc.values.astype(float), lo - pad), hi + pad)
        fc = pd.Series(fc_vals, index=fc.index)

        # Basic CI from increment volatility
        sigma_ci = float(np.nanstd(diffs, ddof=1)) if diffs.size else 0.0
        if not np.isfinite(sigma_ci) or sigma_ci <= 0:
            sigma_ci = float(np.nanstd(tail.values, ddof=1)) if tail.size else 0.0
        lower = fc.values.astype(float) - 1.96 * sigma_ci
        upper = fc.values.astype(float) + 1.96 * sigma_ci
        ci = pd.DataFrame({"lower": lower, "upper": upper}, index=fc.index)
        return fc, ci
    except Exception:
        return _recent_slope_forecast(series, steps, window=None, damping=None)
def _seasonality_strength(series: pd.Series, seasonal_period: int | None) -> float:
    """Estimate strength of seasonality (0..1) using STL-like ratio of seasonal var to total var.
    Returns 0 if not enough data or invalid inputs.
    """
    try:
        if not isinstance(seasonal_period, int) or seasonal_period < 2:
            return 0.0
        y = pd.to_numeric(series, errors='coerce').dropna()
        n = len(y)
        if n < seasonal_period * 3:
            return 0.0
        # compute a simple moving average baseline to approximate trend
        w = seasonal_period
        ma = y.rolling(window=w, center=True, min_periods=max(2, w//2)).mean()
        detr = (y - ma).dropna()
        if len(detr) < seasonal_period * 2:
            return 0.0
        # seasonal component via last one-period template repeated and mean-removed
        template = detr.iloc[-seasonal_period:]
        template = template - template.mean()
        seasonal = pd.Series(np.resize(template.values, len(detr)), index=detr.index)
        resid = detr - seasonal
        var_sea = float(np.nanvar(seasonal.values))
        var_tot = float(np.nanvar(detr.values)) + 1e-12
        strength = max(0.0, min(1.0, var_sea / var_tot))
        return float(strength)
    except Exception:
        return 0.0
def generate_forecast_plot(history, forecast_series, title, xlabel, ylabel, conf_int=None, history_tail=None, anomalies_idx=None):
    """Generate a plot showing historical data and forecast with confidence intervals and anomaly markers."""
    # Debug info removed for performance
    # app.logger.debug(f"[PLOT] generate_forecast_plot history={len(history)} forecast={len(forecast_series)}")
    
    fig, ax = plt.subplots(figsize=(10, 4))

    history_tail_series = history if not history_tail or history_tail <= 0 else history.tail(history_tail)
    
    # Use matplotlib ax.plot() for BOTH history and forecast to avoid converter conflicts
    ax.plot(
        history_tail_series.index,
        history_tail_series.values,
        linestyle='-',
        color='tab:blue',
        linewidth=1.2,
        label='History',
        zorder=2
    )

    # Forecast: solid line (no point markers) for clarity on dense plots
    # Prepend the last history point to ensure visual continuity
    try:
        last_x = history_tail_series.index[-1]
        last_y = float(history_tail_series.iloc[-1])
        x_plot = [last_x] + list(forecast_series.index)
        y_plot = [last_y] + list(forecast_series.values.astype(float))
    except Exception:
        x_plot = list(forecast_series.index)
        y_plot = list(forecast_series.values)

    ax.plot(
        x_plot,
        y_plot,
        linestyle='-',
        color='orangered',
        linewidth=1.2,
        alpha=0.9,
        label='Forecast',
        zorder=3
    )

    # Add anomaly markers if provided
    if anomalies_idx is not None and len(anomalies_idx):
        try:
            # Filter anomalies that are within the history_tail_series index
            aligned_anomalies = history_tail_series.loc[history_tail_series.index.intersection(anomalies_idx)]
            if len(aligned_anomalies) > 0:
                ax.scatter(
                    aligned_anomalies.index,
                    aligned_anomalies.values,
                    color='red',
                    s=40,
                    zorder=5,
                    label='Anomaly',
                    marker='o',
                    edgecolors='darkred',
                    linewidths=1.5
                )
        except Exception as e:
            app.logger.warning(f"Could not plot anomalies: {e}")

    if conf_int is not None:
        try:
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]
            # Keep CI bounds aligned with forecast only (do not prepend history point)
            lower.index = forecast_series.index
            upper.index = forecast_series.index
            ax.fill_between(
                forecast_series.index, lower, upper,
                color='orangered', alpha=0.22, label='95% CI', zorder=2
            )
        except Exception:
            pass

    try:
        split_x = history.index[-1]
        ax.axvline(split_x, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
        ax.axvspan(split_x, forecast_series.index[-1], color='orange', alpha=0.08, zorder=0)
    except Exception:
        pass

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
    # Use a sensible x-axis label depending on index type
    try:
        if isinstance(history_tail_series.index, pd.DatetimeIndex):
            ax.set_xlabel('Timestamp')
        else:
            ax.set_xlabel('Index')
    except Exception:
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

def _simple_forecast(series: pd.Series, steps: int):
    """Simple, robust forecast that tends to mimic recent behavior without extremes.
    - Robust trend: median of last-window diffs
    - Optional seasonal template: last seasonal_period values (mean-removed), scaled to residual std
    - Light noise from residuals
    - Mild clamp to recent quantile envelope
    Returns (forecast_series, conf_df) with basic CI from residual std.
    """
    try:
        y = pd.to_numeric(series, errors='coerce').dropna()
        n = len(y)
        future_idx = _infer_future_index(y.index, steps)
        if n < 5:
            fc_mean = pd.Series([y.iloc[-1] if n else 0.0] * steps, index=future_idx)
            ci = pd.concat([fc_mean.copy(), fc_mean.copy()], axis=1)
            ci.columns = ['lower', 'upper']
            return fc_mean, ci

        w = min(max(24, n // 3), n)
        y_win = y.iloc[-w:]
        # Robust slope from diffs
        diffs = np.diff(y_win.values)
        slope = float(np.median(diffs)) if len(diffs) else 0.0
        # Residual std around a simple linear fit (captures volatility)
        x = np.arange(len(y_win), dtype=float)
        try:
            slope_lr, intercept = np.polyfit(x, y_win.values, 1)
            fit = slope_lr * x + intercept
        except Exception:
            fit = np.full_like(y_win.values, y_win.values.mean())
        resid = (y_win.values - fit)
        rs = float(np.nanstd(resid, ddof=1))
        # Also capture increment volatility to keep natural variation
        inc_std = float(np.nanstd(np.diff(y_win.values), ddof=1)) if len(y_win.values) > 2 else rs

        k = np.arange(1, steps + 1, dtype=float)
        baseline = y.iloc[-1] + slope * k

        # Seasonal template from last detected period
        seasonal = np.zeros(steps, dtype=float)
        sp = _infer_seasonal_period(y.index) if isinstance(y.index, pd.DatetimeIndex) else None
        if isinstance(sp, int) and sp >= 2 and n >= sp * 2:
            template = y.iloc[-sp:].values.astype(float)
            template = template - template.mean()
            tstd = float(np.nanstd(template, ddof=1))
            if tstd > 0 and rs > 0:
                scale = float(np.clip(rs / (tstd + 1e-12), 0.5, 2.5))
            else:
                scale = 1.0
            seasonal = np.resize(template, steps) * scale  # only when template exists

        # Add small correlated noise (AR(1)-like) to avoid flatlining
        # Increase slightly for longer horizons to preserve natural variability
        hf = float(steps) / max(1.0, float(n))
        scale_boost = 1.0 + 0.6 * min(1.0, max(0.0, hf))
        eps_scale = (inc_std * 0.35 if np.isfinite(inc_std) and inc_std > 0 else (rs * 0.25 if np.isfinite(rs) and rs > 0 else 0.0)) * scale_boost
        rho = 0.6  # persistence for smooth variation
        eps = np.zeros(steps, dtype=float)
        if eps_scale > 0:
            for i in range(steps):
                shock = np.random.normal(0.0, eps_scale)
                eps[i] = (rho * (eps[i-1] if i > 0 else 0.0)) + shock
        # Optional bootstrap of recent increments to mimic natural variability
        if len(diffs) >= 4:
            boot = np.random.choice(diffs, size=steps, replace=True).astype(float)
            # center and shrink
            boot = (boot - float(np.mean(boot))) * (0.35 * scale_boost)
            boot_walk = np.cumsum(boot)
        else:
            boot_walk = 0.0
        fc_vals = baseline + seasonal + eps + boot_walk

        # Mild clamp to recent envelope to avoid extremes but keep amplitude
        ql, qh = np.quantile(y_win.values, [0.02, 0.98])
        rng = float(max(1e-9, qh - ql))
        lo2 = ql - 0.3 * rng
        hi2 = qh + 0.3 * rng
        fc_vals = np.clip(fc_vals, lo2, hi2)

        fc_mean = pd.Series(fc_vals, index=future_idx)
        # Build CI using combined volatility estimate
        ci_sigma = max(1e-12, float(np.nanstd(resid, ddof=1)))
        ci = pd.concat([fc_mean - 1.96 * ci_sigma, fc_mean + 1.96 * ci_sigma], axis=1)
        ci.columns = ['lower', 'upper']
        return fc_mean, ci
    except Exception:
        # Fallback to simple recent-slope if anything goes wrong
        return _recent_slope_forecast(series, steps, window=None, damping=None)
def _naturalize_forecast(history: pd.Series, forecast_series: pd.Series, conf_df: pd.DataFrame | None = None,
                          q_low: float = 0.02, q_high: float = 0.98, pad: float = 0.2):
    """Constrain forecast to a plausible envelope derived from history.
    - Compute low/high quantiles of recent history (last 30-50% window).
    - Add a proportional pad to allow gentle drift.
    - Clip forecast to [low - pad*range, high + pad*range].
    - If conf_df present, clip it too.
    Returns possibly adjusted (forecast_series, conf_df).
    """
    try:
        y = pd.to_numeric(history, errors='coerce').dropna()
        if len(y) < 5:
            return forecast_series, conf_df
        n = len(y)
        w = max(50, int(n * 0.3))
        recent = y.tail(min(n, w))
        lo = float(np.nanquantile(recent.values, max(0.0, min(q_low, 0.49))))
        hi = float(np.nanquantile(recent.values, min(1.0, max(q_high, 0.51))))

        rng = max(1e-9, hi - lo)
        # widen pad dynamically according to recent volatility to preserve amplitude
        vol = float(np.nanstd(recent.values, ddof=1))
        dyn_pad = pad + 0.5 * (vol / (rng + 1e-12))
        lo2 = lo - dyn_pad * rng
        hi2 = hi + dyn_pad * rng

        fc = forecast_series.copy()
        v = fc.values.astype(float)
        # Only apply soft-bound outside the envelope, keep inside untouched to avoid flattening
        mid = (lo2 + hi2) / 2.0
        half = max(1e-9, (hi2 - lo2) / 2.0)
        z = (v - mid) / (half * 0.9)
        v_adj = v.copy()
        mask_hi = v > hi2
        mask_lo = v < lo2
        v_adj[mask_hi] = mid + half * np.tanh(z[mask_hi])
        v_adj[mask_lo] = mid + half * np.tanh(z[mask_lo])
        # Respect absolute min/max from history with a soft barrier and gentle reflection to avoid harsh saturation
        min_abs = float(np.nanmin(y.values))
        max_abs = float(np.nanmax(y.values))
        rng_abs = max(1e-9, max_abs - min_abs)
        # Iterate with barrier near bounds based on distance from previous value
        out = np.empty_like(v_adj)
        prev = float(y.iloc[-1])
        scale = 0.25 * rng_abs
        # micro-oscillation to avoid long flat segments; use seasonal period if detectable, else a small fixed period
        try:
            osc_period = _infer_seasonal_period(y.index) if isinstance(y.index, pd.DatetimeIndex) else None
        except Exception:
            osc_period = None
        base_period = int(osc_period) if isinstance(osc_period, int) and osc_period >= 3 else 10
        # Scale oscillation amplitude slightly with horizon fraction to avoid late flatlines
        try:
            horizon_frac = float(len(fc)) / max(1.0, float(len(y)))
        except Exception:
            horizon_frac = 0.0
        osc_amp = (0.01 + 0.03 * max(0.0, min(1.0, horizon_frac))) * rng_abs
        for i in range(len(v_adj)):
            target = float(v_adj[i])
            step = target - prev
            # Soft approach to bounds: as prev nears min/max, step is damped but never zero (less damping to avoid flatlines)
            if step >= 0:
                dist = max(0.0, max_abs - prev)
                factor = 0.92 + 0.08 * float(np.tanh(dist / (scale + 1e-12)))  # always >0.92
                step *= factor
            else:
                dist = max(0.0, prev - min_abs)
                factor = 0.92 + 0.08 * float(np.tanh(dist / (scale + 1e-12)))
                step *= factor
            nxt = prev + step
            # Gentle mean reversion toward mid when very close to edges to prevent sticking
            edge_closeness = min(max(0.0, (prev - min_abs) / rng_abs), max(0.0, (max_abs - prev) / rng_abs))
            # edge_closeness is small near edges; use (1 - edge_closeness)
            reversion_strength = 0.08 * (1.0 - edge_closeness)
            nxt += reversion_strength * (mid - nxt)
            # Add micro oscillation
            try:
                phase = (i % max(3, base_period)) / float(max(3, base_period))
                nxt += osc_amp * math.sin(2.0 * math.pi * phase)
            except Exception:
                pass
            # Gentle reflection if outside
            if nxt > max_abs:
                over = nxt - max_abs
                nxt = max_abs - 0.75 * over
            if nxt < min_abs:
                over = min_abs - nxt
                nxt = min_abs + 0.75 * over
            # Final guard: allow a tiny epsilon outside, but never saturate noticeably
            eps = 1e-6 * rng_abs
            if nxt > max_abs + eps:
                nxt = max_abs + eps
            if nxt < min_abs - eps:
                nxt = min_abs - eps
            out[i] = nxt
            prev = nxt
        fc = pd.Series(out, index=fc.index)

        if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
            c = conf_df.copy()
            c0 = c.iloc[:, 0].astype(float).values
            c1 = c.iloc[:, 1].astype(float).values
            # First apply the same soft envelope mapping
            z0 = (c0 - mid) / (half * 0.9)
            z1 = (c1 - mid) / (half * 0.9)
            c0_adj = c0.copy()
            c1_adj = c1.copy()
            c0_adj[c0 > hi2] = mid + half * np.tanh(z0[c0 > hi2])
            c0_adj[c0 < lo2] = mid + half * np.tanh(z0[c0 < lo2])
            c1_adj[c1 > hi2] = mid + half * np.tanh(z1[c1 > hi2])
            c1_adj[c1 < lo2] = mid + half * np.tanh(z1[c1 < lo2])
            # Then ensure final CI stays within absolute min/max and around the adjusted forecast
            c0_adj = np.minimum(np.maximum(c0_adj, min_abs), fc.values)
            c1_adj = np.maximum(np.minimum(c1_adj, max_abs), fc.values)
            c.iloc[:, 0] = c0_adj
            c.iloc[:, 1] = c1_adj
            return fc, c
        return fc, conf_df
    except Exception:
        return forecast_series, conf_df

def _match_amplitude(history: pd.Series, forecast_series: pd.Series, conf_df: pd.DataFrame | None = None,
                     seasonal_period: int | None = None, min_scale: float = 0.85, max_scale: float = 2.5):
    """Scale forecast deviations to better match recent history amplitude.
    - Compute std of recent history increments vs forecast increments.
    - If forecast variance is too low, scale deviations around a linear baseline.
    - Adjust conf intervals by same scale.
    Returns (forecast_series, conf_df).
    """
    try:
        y = pd.to_numeric(history, errors='coerce').dropna()
        fc = pd.to_numeric(forecast_series, errors='coerce')
        if len(y) < 6 or len(fc) < 2:
            return forecast_series, conf_df
        n = len(y)
        w = seasonal_period if (isinstance(seasonal_period, int) and seasonal_period >= 2) else max(12, n // 4)
        y_win = y.tail(min(n, int(w)))
        hist_diffs = np.diff(y_win.values)
        fc_diffs = np.diff(fc.values.astype(float))
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
            slope, intercept = np.polyfit(x, fc.values.astype(float), 1)
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
        slope, intercept = np.polyfit(x, fc.values.astype(float), 1)
        baseline = slope * x + intercept
        deviations = fc.values.astype(float) - baseline
        fc_scaled = baseline + scale * deviations
        fc2 = pd.Series(fc_scaled, index=fc.index)
        c2 = None
        if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
            # Scale CI bounds relative to new center
            lower = conf_df.iloc[:, 0].values.astype(float)
            upper = conf_df.iloc[:, 1].values.astype(float)
            lower_dev = lower - fc.values.astype(float)
            upper_dev = upper - fc.values.astype(float)
            lower2 = fc2.values + scale * lower_dev
            upper2 = fc2.values + scale * upper_dev
            c2 = pd.DataFrame({"lower": lower2, "upper": upper2}, index=fc.index)
        return fc2, (c2 if c2 is not None else conf_df)
    except Exception:
        return forecast_series, conf_df

def _is_too_quiet(history: pd.Series, forecast: pd.Series, frac: float = 0.5):
    """True if forecast dynamics (std of diffs) are much smaller than recent history.
    If std(diff(forecast)) < frac * std(diff(recent history)).
    """
    try:
        h = pd.to_numeric(history, errors='coerce').dropna()
        f = pd.to_numeric(forecast, errors='coerce').dropna()
        if len(h) < 6 or len(f) < 3:
            return False
        n = len(h)
        w = max(12, min(200, n // 2))
        h_win = h.tail(w)
        std_h = float(np.nanstd(np.diff(h_win.values), ddof=1)) if len(h_win) >= 3 else 0.0
        std_f = float(np.nanstd(np.diff(f.values), ddof=1)) if len(f) >= 3 else 0.0
        if not np.isfinite(std_h) or std_h <= 0 or not np.isfinite(std_f):
            return False
        return std_f < (frac * std_h)
    except Exception:
        return False

def _compute_forecast(series: pd.Series, steps: int):
    """Natural-looking forecast that preserves realistic patterns and variance.
    - Uses simple trend continuation
    - Adds realistic noise based on historical volatility
    - Respects historical min/max bounds
    Returns (fc_mean, conf_df).
    """
    def _natural_forecast(s: pd.Series, k: int):
        s = pd.to_numeric(s, errors='coerce').dropna()
        idx = _infer_future_index(s.index if hasattr(s, 'index') else pd.RangeIndex(0, 1), k)
        if s.empty or k <= 0:
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            ci = pd.DataFrame({"lower": zero, "upper": zero})
            return zero, ci
        
        n = len(s)
        values = s.values.astype(float)
        last = float(values[-1])
        
        # Historical bounds - forecast should stay within reasonable range
        data_min = float(np.min(values))
        data_max = float(np.max(values))
        data_range = data_max - data_min
        
        # Use recent window for calculations (adaptive size)
        window_size = max(20, min(240, n // 2 if n >= 40 else n))
        recent_data = values[-window_size:]
        
        # Calculate simple trend from recent data
        trend = 0.0
        if len(recent_data) >= 3:
            # Use weighted average of recent changes (more weight to recent)
            changes = np.diff(recent_data)
            if len(changes) > 0:
                weights = np.exp(np.linspace(-1, 0, len(changes)))
                weights = weights / weights.sum()
                trend = float(np.average(changes, weights=weights))
        
        # Calculate historical volatility for realistic variation
        hist_volatility = 1.0
        if len(recent_data) >= 3:
            changes = np.diff(recent_data)
            hist_volatility = float(np.std(changes, ddof=1))
        
        if hist_volatility < 1e-6:
            hist_volatility = float(np.std(values, ddof=1)) if len(values) > 1 else data_range * 0.1
        
        # Create unique seed based on series data for true uniqueness per column
        # Use multiple aggregates to ensure different columns have different seeds
        # Avoid np.prod to prevent overflow
        series_sum = float(np.sum(values))
        series_mean = float(np.mean(values))
        series_std = float(np.std(values))
        # Use first, middle, and last values for uniqueness
        first_val = float(values[0])
        mid_val = float(values[len(values)//2])
        series_hash = hash((
            series_sum,
            series_mean,
            series_std,
            first_val,
            mid_val,
            last,
            trend,
            len(s),
            data_min,
            data_max
        ))
        seed_val = int(abs(series_hash) % (2**31))
        np.random.seed(seed_val)
        
        # Use actual historical segments to build forecast
        forecast_vals = np.zeros(k, dtype=float)
        
        # Find similar historical patterns to the recent end
        # Use a moving window approach to extract realistic segments
        segment_length = min(k, max(5, len(recent_data) // 10))
        
        if len(values) > segment_length * 2:
            # Find segments in history that are similar to recent data
            recent_pattern = recent_data[-segment_length:]
            recent_mean = np.mean(recent_pattern)
            recent_std = np.std(recent_pattern)
            
            # Search for similar segments in historical data
            similar_segments = []
            for i in range(len(values) - segment_length - k):
                segment = values[i:i+segment_length]
                seg_mean = np.mean(segment)
                seg_std = np.std(segment)
                
                # Check if this segment has similar statistical properties
                if abs(seg_mean - recent_mean) < data_range * 0.3 and abs(seg_std - recent_std) < hist_volatility * 2:
                    # Get what happened after this segment
                    continuation = values[i+segment_length:i+segment_length+k]
                    if len(continuation) == k:
                        similar_segments.append(continuation)
            
            # If we found similar patterns, use them
            if len(similar_segments) > 0:
                # Randomly pick one similar segment's continuation
                chosen_continuation = similar_segments[np.random.randint(0, len(similar_segments))]
                
                # Adjust the continuation to start from our last value
                offset = last - chosen_continuation[0]
                forecast_vals = chosen_continuation + offset
            else:
                # Fallback: use change sampling (no trend addition)
                historical_changes = np.diff(recent_data)
                current_value = last
                for i in range(k):
                    sampled_change = np.random.choice(historical_changes)
                    current_value = current_value + sampled_change
                    forecast_vals[i] = current_value
        else:
            # Not enough data, use simple change sampling (no trend addition)
            historical_changes = np.diff(recent_data)
            current_value = last
            for i in range(k):
                sampled_change = np.random.choice(historical_changes)
                current_value = current_value + sampled_change
                forecast_vals[i] = current_value
        
        # Scale forecast to fit within bounds (no clipping)
        fc_min = float(np.min(forecast_vals))
        fc_max = float(np.max(forecast_vals))
        fc_range = fc_max - fc_min
        
        # Only scale if forecast exceeds bounds
        if fc_min < data_min or fc_max > data_max:
            if fc_range > 1e-9:
                # Scale to fit within data_min and data_max
                scaled_vals = data_min + (forecast_vals - fc_min) * (data_range / fc_range)
                forecast_vals = scaled_vals
        
        fc = pd.Series(forecast_vals, index=idx)
        
        # Build realistic confidence intervals that expand over time
        expanding_uncertainty = hist_volatility * np.sqrt(np.arange(1, k + 1))
        lower = forecast_vals - 1.96 * expanding_uncertainty
        upper = forecast_vals + 1.96 * expanding_uncertainty
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
        
        max_in = int(app.config.get('FORECAST_MAX_INPUT_POINTS', 4000))
        if max_in and len(s) > max_in:
            s = _thin_series(s, max_points=max_in)
        fc, ci = _natural_forecast(s, steps)
        
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
                trend = float(np.mean(np.diff(s.values[-min(20, len(s)):])))
                last = float(s.iloc[-1])
                vals = [last + trend * (i + 1) for i in range(steps)]
            else:
                last = float(s.iloc[-1]) if len(s) else 0.0
                vals = [last] * steps
            fc = pd.Series(vals, index=idx)
            std = float(np.std(s.values, ddof=1)) if len(s) > 1 else 1.0
            ci = pd.DataFrame({"lower": fc.values - 1.96 * std, "upper": fc.values + 1.96 * std}, index=idx)
            return fc, ci
        except Exception:
            idx = _infer_future_index(pd.RangeIndex(0, 1), steps)
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            return zero, pd.DataFrame({"lower": zero, "upper": zero})

def read_csv_fallback(path, **kwargs):
    last_err = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            raise
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
            try:
                with open(path, "r", encoding=enc, errors="strict") as f:
                    return pd.read_json(f, lines=True)
            except Exception:
                continue
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
        return r2 > 0.985  
    except Exception:
        return False

def _bootstrap_natural_path(series, steps, window=None, base_slope=None, n_samples=200, q_low=0.1, q_high=0.9):
    """
    Bootstrap-style natural forecast helper (thin wrapper around _forecast_natural with envelope adjustment).
    """
    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
        if s.empty:
            idx = _infer_future_index(series.index if hasattr(series, 'index') else pd.RangeIndex(0, 1), steps)
            zero = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
            ci = pd.DataFrame({"lower": zero, "upper": zero})
            return zero, ci
        max_in = int(app.config.get('FORECAST_MAX_INPUT_POINTS', 4000))
        if max_in and len(s) > max_in:
            s = _thin_series(s, max_points=max_in)
        fc, ci = _forecast_natural(s, steps)
        try:
            fc, ci = _naturalize_forecast(s, fc, ci)
        except Exception:
            pass
        return fc, ci
    except Exception:
        return _recent_slope_forecast(series, steps, window=None, damping=None)

def _try_parse_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of object-like numeric strings to floats.
    Handles thousands separators, comma-decimals, percents, and stray units."""
    if not isinstance(s, pd.Series):
        return pd.to_numeric(s, errors='coerce')

    
    out = pd.to_numeric(s, errors='coerce')
    na_ratio = out.isna().mean()

    if na_ratio <= 0.25:
        return out

    
    ss = s.astype(str).str.strip()

    
    has_pct = ss.str.contains(r'%', regex=True, na=False)

    
    cleaned = ss.str.replace(r'[^0-9,.\-+eE]', ' ', regex=True).str.replace(r'\s+', '', regex=True)

    
    comma_cnt = cleaned.str.count(',').sum()
    dot_cnt = cleaned.str.count(r'\.').sum()
    if comma_cnt > dot_cnt:
        
        attempt = cleaned.str.replace(r'\.', '', regex=True).str.replace(',', '.', regex=False)
    else:
        
        attempt = cleaned.str.replace(',', '', regex=False)

    out2 = pd.to_numeric(attempt, errors='coerce')

    
    if has_pct.any():
        
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
            coerced = _try_parse_numeric_series(ser)
            if coerced.notna().sum() >= pd.to_numeric(ser, errors='coerce').notna().sum():
                res[col] = coerced
    return pd.DataFrame(res, index=df.index)


def detect_anomalies(series: pd.Series, contamination: float = 0.02):
    """
    Detect anomalies in a numeric series using IsolationForest.
    Returns:
      - an_idx: index of detected anomalies (pd.Index)
      - an_score: anomaly scores as a Series indexed by anomaly index (higher = more anomalous)
    """
    try:
        s = pd.to_numeric(series, errors='coerce').dropna()
    except Exception:
        s = pd.Series(dtype=float)

    if s is None or len(s) < 5:
        return pd.Index([]), pd.Series([], dtype=float)

    try:
        cont = float(contamination) if contamination is not None else 0.02
    except Exception:
        cont = 0.02
    if not (0.0 < cont < 0.5):
        cont = 0.02

    try:
        X = s.values.reshape(-1, 1)
        model = IsolationForest(contamination=cont, random_state=42)
        preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal
        scores = -model.decision_function(X)  # higher means more anomalous
        mask = preds == -1
        an_idx = s.index[mask]
        an_score = pd.Series(scores[mask], index=an_idx)
        return an_idx, an_score
    except Exception:
        return pd.Index([]), pd.Series([], dtype=float)

def build_ai_context(df: pd.DataFrame, anomalies_found: dict, corr_payload: dict, used_cols: list, is_timeseries: bool, forecast_horizon: int, contamination: float) -> str:
    """Assemble structured stats the AI can leverage for a deeper analysis."""
    try:
        lines = []
        lines.append(f"Shape: {getattr(df, 'shape', None)}")
        
        try:
            dtypes = {c: str(t) for c, t in df.dtypes.items()}
            lines.append("Dtypes: " + json.dumps(dtypes, ensure_ascii=False))
        except Exception:
            pass
        
        try:
            mv = df.isna().mean().sort_values(ascending=False)
            top_mv = mv[mv > 0].head(20)
            if not top_mv.empty:
                lines.append("Top missingness (fraction): " + json.dumps({k: float(v) for k, v in top_mv.items()}))
        except Exception:
            pass
        
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
        
        try:
            if anomalies_found:
                lines.append("Anomalies summary: " + json.dumps(anomalies_found))
        except Exception:
            pass
        
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
        
        try:
            if is_timeseries and isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.dropna()
                if len(idx):
                    freq = str(idx.freq) if idx.freq is not None else (pd.infer_freq(idx) or "unknown")
                    lines.append(f"Time series detected. Start: {str(idx[0])}, End: {str(idx[-1])}, Freq: {freq}")
        except Exception:
            pass
        lines.append(f"User settings: forecast_horizon={int(forecast_horizon)}, anomaly_contamination={float(contamination)}")
        return "\n".join(lines)
    except Exception:
        return ""

@app.route('/', methods=['GET', 'POST'])
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

            
            temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(app.config['UPLOADS_DIR'], temp_name)  
            file.save(temp_path)

            try:
                
                hasher = hashlib.sha1()
                with open(temp_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):  
                        hasher.update(chunk)
                digest = hasher.hexdigest()

                
                storage_name = f"{digest}{ext}"
                final_path = os.path.join(app.config['UPLOADS_DIR'], storage_name)  

                if os.path.exists(final_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        app.logger.warning("Could not remove temp file %s: %s", temp_path, e)
                else:
                    os.replace(temp_path, final_path)

                
                try:
                   # Configure basic logging
                    logging.basicConfig(level=logging.INFO)

                    # Add file handler for debugging persistence
                    fh = logging.FileHandler('app_debug.log')
                    fh.setLevel(logging.INFO)
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    fh.setFormatter(formatter)
                    app.logger.addHandler(fh)
                    app.logger.info("Application starting - Log file initialized")

                    # Increase upload size limit to 64MB
                    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
                    size_bytes = os.path.getsize(final_path)
                    if size_bytes <= app.config['AI_FULL_UPLOAD_MAX_MB'] * 1024 * 1024:
                        uploaded = genai.upload_file(path=final_path, mime_type="text/csv", display_name=orig_name)
                        AI_FILE_MAP[storage_name] = uploaded
                except Exception as e:
                    app.logger.info("AI file upload skipped: %s", e)

                
                
                fh = request.form.get('forecast_horizon')  # legacy hidden
                fpct = request.form.get('forecast_pct')
                cont = request.form.get('contamination')
                start_view = request.form.get('view') or 'overview'
                return redirect(url_for(
                    'analyze_file',
                    filename=storage_name,
                    display=orig_name,
                    forecast_horizon=fh,
                    forecast_pct=fpct,
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

def read_excel_smart(path: str):
    """Read the first non-empty sheet; try to infer header and index."""
    try:
        with pd.ExcelFile(path) as xls:
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet, header=0)
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    if df is not None and df.shape[1] > 0:
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
                            first_col = df.columns[0]
                            try:
                                maybe_dt = pd.to_datetime(df[first_col], errors='coerce')
                                if maybe_dt.notna().sum() >= max(3, int(len(df) * 0.3)):
                                    df = df.set_index(first_col)
                            except Exception:
                                pass
                        return df
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
        return pd.DataFrame()
    except Exception as e:
        try:
            return pd.read_excel(path)
        except Exception:
            raise e

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
        df = DATAFRAME_CACHE.get(filename)
        if df is not None:
            return df

        uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
        path = os.path.join(uploads_dir, filename)
        if not os.path.exists(path):
            app.logger.info("get_dataframe_for: file not found on disk: %s", path)
            return None

        _, ext = os.path.splitext(filename)
        ext = (ext or "").lower()

        if ext == ".csv":
            df = read_csv_fallback(path, index_col=0, parse_dates=True)
        elif ext == ".xlsx":
            df = read_excel_smart(path)
        elif ext == ".json":
            df = read_json_fallback(path)
            for col in ['timestamp', 'date', 'time']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                    except Exception:
                        pass
                    break
        elif ext == ".txt":
            df = read_csv_fallback(path, sep=',', index_col=0, parse_dates=True)
        else:
            app.logger.warning("get_dataframe_for: unsupported extension %s", ext)
            return None

        if not isinstance(df, pd.DataFrame):
            app.logger.info("get_dataframe_for: reader returned non-DataFrame for %s", filename)
            return None

        try:
            df = df.dropna(axis=1, how='all')
        except Exception:
            pass

        try:
            if not isinstance(df.index, pd.DatetimeIndex) and df.shape[1] >= 1:
                candidate_cols = []
                for c in df.columns:
                    lc = str(c).strip().lower()
                    if any(tok in lc for tok in ("date", "time", "timestamp", "datetime")):
                        candidate_cols.append(c)
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
                    try:
                        df = df[ts.notna()].sort_index()
                    except Exception:
                        pass
        except Exception as e:
            app.logger.debug("get_dataframe_for: datetime inference skipped: %s", e)

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
            return (df.dtypes.to_frame('dtype').to_html()
                    if df is not None and isinstance(df, pd.DataFrame) else "<p>No data.</p>")
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
        if error:
            try:
                reason_raw = getattr(error, 'message', None) or str(error)
            except Exception:
                reason_raw = None
            try:
                reason = _sanitize_error_message(reason_raw) or (AI_STATUS.get('message') or '')
            except Exception:
                reason = ''
            detail = f" Reason: {htmllib.escape(str(reason))}" if reason else ""
            parts.append(f"<p><em>AI response unavailable. Showing a quick offline analysis instead.</em></p><p class=\"muted\"><small>{detail}</small></p>")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            parts.append("<p>No data available.</p>")
            return "".join(parts)

        
        try:
            parts.append(f"<p><strong>Shape:</strong> {tuple(df.shape)}</p>")
            dtypes = ", ".join([f"{htmllib.escape(str(c))}: {htmllib.escape(str(t))}" for c, t in df.dtypes.items()])
            parts.append(f"<p><strong>Dtypes:</strong> {dtypes}</p>")
        except Exception:
            pass

        
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

        
        if q and q != "summary":
            try:
                mentioned = []
                q_low = q.lower()
                for col in df.columns:
                    name = str(col)
                    if name.lower() in q_low:
                        mentioned.append(col)
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
def analyze_file(filename):
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
            # Validate percentage is in reasonable range
            if pct < 0.01 or pct > 0.5:
                app.logger.warning("forecast_pct out of range (%.4f), clamping to [0.01, 0.5]", pct)
                pct = max(0.01, min(0.5, pct))
    except Exception:
        pct = None
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
        return redirect(url_for('upload_file'))

    df = DATAFRAME_CACHE.get(filename)
    if df is None:
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

        DATAFRAME_CACHE.set(filename, df)

    if (
        app.config.get('DELETE_UPLOADED_AFTER_PROCESSING', False)
        and HASHED_UPLOAD_RE.match(os.path.basename(filepath))
        and os.path.exists(filepath)
    ):
        success, error_msg = _safe_delete(filepath)
        if not success and error_msg:
            app.logger.warning("Could not delete uploaded file %s: %s", filepath, error_msg)

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
    plots = []
    forecast_plots = []
    anomalies_found = {}
    is_timeseries = isinstance(df.index, pd.DatetimeIndex)
    used_cols = []

    # Correlation/precompute metrics for heatmap
    corr_payload = None
    try:
        num_df = coerce_numeric_df(df).select_dtypes(include='number')
        if num_df is not None and not num_df.empty:
            valid = [c for c in num_df.columns if num_df[c].notna().sum() >= 3]
            num_df = num_df[valid]
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
            try:
                spearman = num_df.corr(method='spearman')
            except Exception:
                spearman = None
            try:
                pearson = num_df.corr(method='pearson')
            except Exception:
                pearson = None
            if spearman is not None:
                payload["x"] = cols
                payload["y"] = cols
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

    interactive = []

    raw_tail = (os.getenv("PLOTLY_TAIL", "all") or "all").strip().lower()

    # Apply percentage-based forecast horizon early (if provided) so all downstream forecast logic uses updated user_steps.
    total_rows = int(getattr(df, 'shape', (0,))[0]) if hasattr(df, 'shape') else 0
    try:
        if 'pct' in locals() and pct and pct > 0 and total_rows > 0:
            user_steps = max(2, int(math.ceil(total_rows * float(pct))))
    except Exception:
        pass
    # Compute an effective steps value (even for legacy numeric) without upper clamping.
    try:
        effective_steps = max(2, int(user_steps))
    except Exception:
        effective_steps = 2

    # Determine whether to build static/forecast/interactive content based on active_view.
    # IMPORTANT: To keep upload/overview fast, run heavy forecasting only in the explicit Forecast view.
    build_static = False  # Static view removed from application
    build_forecast = active_view == "forecast"
    build_interactive = active_view == "interactive"

    # Per-request timing and budgets to prevent long hangs
    request_start = time.perf_counter()
    overview_budget_s = float(os.getenv("OVERVIEW_TIME_BUDGET_SEC", "6.0"))
    # Slightly higher default budget for forecast view to avoid missing plots under normal loads
    forecast_budget_s = float(os.getenv("FORECAST_TIME_BUDGET_SEC", "36.0"))
    budget_s = forecast_budget_s if build_forecast or build_interactive else overview_budget_s
    # Column limits for forecasting to avoid O(N cols) explosion
    overview_cols_max = int(os.getenv("OVERVIEW_FORECAST_COLS_MAX", "2"))
    # Allow more columns by default on Forecast view
    forecast_cols_max = int(os.getenv("FORECAST_COLS_MAX", "20"))
    cols_limit = forecast_cols_max if build_forecast else overview_cols_max
    forecast_done = 0
    skip_forecasts = False

    for column in df.columns:
            
            series_raw = df[column]
            try:
                series = pd.to_numeric(series_raw, errors='coerce').dropna()
            except Exception:
                series = pd.Series(dtype=float)
            if series.empty:
                continue
            used_cols.append(column)
            
            # Optimization: Only run anomaly detection if we are in a view that needs it
            # or if we are reasonably sure it won't kill performance (e.g. small number of columns)
            # 'overview' doesn't show anomalies in the UI directly, only in AI context.
            # We skip it for overview on large datasets to speed up transitions.
            an_idx = []
            if build_forecast or build_interactive or len(df.columns) < 10:
                 an_idx, an_score = detect_anomalies(series, contamination=user_contam)
                 if len(an_idx):
                     try:
                         anomalies_found[str(column)] = [str(i) for i in an_idx]
                     except Exception:
                        pass

            if build_static:
                title_trend = f"Trend for {column}"
                s_plot = _thin_series(series, max_points=400)
                plots.append({
                    "img": generate_plot(
                        s_plot,
                        title_trend,
                        'Timestamp' if is_timeseries else 'Index',
                        column,
                        anomalies_idx=an_idx
                    ),
                    "title": title_trend
                })

            # Stop forecasting if time/column limits are exceeded
            # Generate forecasts for any numeric series with sufficient length
            if build_forecast and not skip_forecasts and len(series) >= 5:
                try:
                    t0 = time.perf_counter()
                    steps = effective_steps
                    app.logger.info("Forecast start col=%s steps=%s rows=%s pct=%s", column, steps, len(series), pct)

                    # Unified pipeline
                    fc_mean, conf_df = _compute_forecast(series, steps)

                    title_fc = f"Forecast for {column} (with anomalies)"
                    s_hist = _thin_series(series, max_points=600)
                    xlab = 'Timestamp' if isinstance(series.index, pd.DatetimeIndex) else 'Index'
                    try:
                        # Pass anomaly indices to the forecast plot generation
                        img_fc = generate_forecast_plot(
                            s_hist,
                            fc_mean,
                            title_fc,
                            xlab,
                            column,
                            conf_int=conf_df,
                            history_tail=None,
                            anomalies_idx=an_idx  # Add anomaly markers to forecast plot
                        )
                        forecast_plots.append({"img": img_fc, "title": title_fc})
                    except Exception as _e:
                        app.logger.warning("Could not render forecast image for %s: %s", column, _e)
                    try:
                        app.logger.info("Forecast plot ready col=%s forecast_points=%d", column, len(fc_mean) if hasattr(fc_mean, '__len__') else -1)
                    except Exception:
                        pass

                    dt = time.perf_counter() - t0
                    forecast_done += 1
                    app.logger.info("Forecast done col=%s took=%.2fs steps=%s points=%s", column, dt, steps, len(series))
                    # Enforce budgets
                    if (time.perf_counter() - request_start) > budget_s or forecast_done >= cols_limit:
                        skip_forecasts = True
                        app.logger.info(
                            "Forecast budget reached: elapsed=%.2fs limit=%.2fs cols=%d/%d",
                            time.perf_counter() - request_start, budget_s, forecast_done, cols_limit
                        )
                except Exception as e:
                    app.logger.warning("Could not generate forecast for %s: %s", column, e)

            if build_forecast and not skip_forecasts and len(series) >= 5:
                try:
                    if isinstance(series.index, pd.DatetimeIndex):
                        s_norm = normalize_timeseries(series)
                        sp = _infer_seasonal_period(s_norm.index)
                        if sp:
                            try:
                                stl_img = generate_stl_plot(s_norm, f"STL decomposition for {column}", seasonal_period=sp)
                                if stl_img:
                                    forecast_plots.append({"img": stl_img, "title": f"STL decomposition for {column}"})
                            except Exception as _e:
                                app.logger.warning("STL plot failed for %s: %s", column, _e)
                except Exception as e:
                    app.logger.warning("Could not generate forecast for %s: %s", column, e)

            if build_interactive and not skip_forecasts and len(series) >= 5:
                try:
                    s_norm = normalize_timeseries(series)
                    sp = _infer_seasonal_period(s_norm.index) if isinstance(s_norm.index, pd.DatetimeIndex) else None
                    if sp:
                        stl_img = generate_stl_plot(s_norm, f"STL decomposition for {column}", seasonal_period=sp)
                       
                        if stl_img:
                            forecast_plots.append({"img": stl_img, "title": f"STL decomposition for {column}"})
                except Exception:
                    pass

            if build_interactive:
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
                    "mode": "lines+markers",
                    "name": "History",
                    "x": x_hist,
                    "y": y_hist,
                    "line": {"color": "rgb(31,119,180)", "width": 2},
                    "marker": {"size": 4, "opacity": 0.6}
                }]

            
            if build_interactive and len(an_idx):
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

            
            fc_x = fc_y = ci_lower = ci_upper = split_x = None
            if build_interactive and not skip_forecasts and is_timeseries and len(series) >= 5:
                try:
                    steps = effective_steps
                    fc_mean, conf_df = _compute_forecast(series, steps)
                    split_x = str(series.index[-1])
                    fc_x = [str(i) for i in fc_mean.index]
                    fc_y = [float(v) for v in fc_mean.values]
                    if isinstance(conf_df, pd.DataFrame) and conf_df.shape[1] >= 2:
                        ci_lower = [float(v) for v in conf_df.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df.iloc[:, 1].values]

                    # Add interactive traces using the forecast
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

                    if fc_x and fc_y:
                        # Prepend last history point for visual continuity
                        try:
                            last_x_hist = str(series.index[-1])
                            last_y_hist = float(series.iloc[-1])
                            x_plot = [last_x_hist] + list(fc_x)
                            y_plot = [last_y_hist] + list(fc_y)
                        except Exception:
                            x_plot, y_plot = fc_x, fc_y
                        traces.append({
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Forecast",
                            "x": x_plot, "y": y_plot,
                            "line": {"color": "orangered", "width": 3},
                            "marker": {"size": 3}
                        })
                except Exception as e:
                    app.logger.warning("Interactive forecast build failed for %s: %s", column, e)

                xaxis = {"title": ("Timestamp" if is_timeseries else "Index"), "showgrid": True}
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

                dist = {"name": column, "values": [float(v) for v in series.dropna().values]}
                interactive.append({"column": column, "traces": traces, "layout": layout, "distribution": dist})

    # If in forecast view and no forecast plots were generated (due to errors or strict budgets),
    # render a fallback forecast for the first eligible numeric column to avoid an empty page.
    try:
        if build_forecast and not forecast_plots:
            for column in df.columns:
                try:
                    series = pd.to_numeric(df[column], errors='coerce').dropna()
                except Exception:
                    continue
                if len(series) >= 5:
                    # Detect anomalies for fallback forecast
                    an_idx_fb, _ = detect_anomalies(series, contamination=user_contam)
                    steps = effective_steps
                    fc_mean, conf_df = _compute_forecast(series, steps)
                    title_fc = f"Forecast for {column}"
                    s_hist = _thin_series(series, max_points=600)
                    forecast_plots.append({
                        "img": generate_forecast_plot(
                            s_hist,
                            fc_mean,
                            title_fc,
                            'Timestamp' if isinstance(series.index, pd.DatetimeIndex) else 'Index',
                            column,
                            conf_int=conf_df,
                            history_tail=None,
                            anomalies_idx=an_idx_fb
                        ),
                        "title": title_fc
                    })
                    break
    except Exception as e:
        app.logger.warning("Fallback static forecast failed: %s", e)

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

    ai_summary = AI_SUMMARY_CACHE.get(filename)
    if ai_summary is None:
        # Defer AI summary on overview GET requests - will load via AJAX for faster initial page render
        # Only block on AI summary for POST requests (Q&A needs it) or non-overview views
        defer_ai_on_overview = (active_view == 'overview' and request.method == 'GET')
        
        if not defer_ai_on_overview and ensure_ai_ready():
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
        elif defer_ai_on_overview:
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
    except Exception:
        pass

    analysis.update({
        'head': safe_df_head_html(df),
        'description': safe_df_description_html(df),
        'info': info_string,
        'missing_values': missing_values_html,
        'plots': _ensure_plot_dicts(plots) if build_static else [],
        'forecast_plots': _ensure_plot_dicts(forecast_plots) if build_forecast else [],
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
            'contamination': user_contam
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
    return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name)

    # If an unexpected error occurs, Flask's error handler will handle it.
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/api/ai-summary/<filename>', methods=['GET'])
def api_ai_summary(filename):
    """Async endpoint for fetching AI summary via AJAX for faster page loads."""
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "html": "<p>Invalid file.</p>"}), 400
    
    # Check cache first for instant response
    cached = AI_SUMMARY_CACHE.get(filename)
    if cached:
        return jsonify({"ok": True, "html": cached, "cached": True})
    
    # Generate new summary
    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return jsonify({"ok": False, "html": "<p>Dataset not found.</p>"}), 404
    
    file_asset = AI_FILE_MAP.get(filename) if 'AI_FILE_MAP' in globals() else None
    ai_context = describe_for_ai(df)
    
    try:
        if ensure_ai_ready():
            summary = get_ai_summary_with_file(df, file_asset, extra_context=ai_context)
            if isinstance(summary, str) and not _is_offline_html(summary):
                AI_SUMMARY_CACHE[filename] = summary
            return jsonify({"ok": True, "html": summary, "cached": False})
        else:
            fallback = offline_answer(df, "summary", error="AI not ready")
            return jsonify({"ok": True, "html": fallback, "cached": False})
    except Exception as e:
        app.logger.warning("API AI summary failed: %s", e)
        fallback = offline_answer(df, "summary", error=e)
        return jsonify({"ok": True, "html": fallback, "cached": False})

@app.route('/download/<filename>/cleaned.csv', methods=['GET'])
def download_cleaned_csv(filename):
    
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


@app.route('/download/<filename>/ai_summary.html', methods=['GET'])
def download_ai_summary_html(filename):
    
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    
    ai_html = AI_SUMMARY_CACHE.get(filename)
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

@app.route('/download/<filename>/ai_summary.pdf', methods=['GET'])
def download_ai_summary_pdf(filename):
    if not HASHED_UPLOAD_RE.match(filename):
        return jsonify({"ok": False, "message": "Invalid filename."}), 400
    return jsonify({"ok": False, "message": "PDF export is disabled."}), 501

@app.route('/download/<filename>/static_plots.zip', methods=['GET'])
def download_static_plots_zip(filename):
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or df.empty:
        return ("Not found", 404)

    is_timeseries = isinstance(df.index, pd.DatetimeIndex)
    bio = io.BytesIO()
    
    # Calculate forecast steps as 10% of dataset size
    total_rows = len(df)
    forecast_steps = max(10, int(total_rows * 0.1))
    
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Generate correlation heatmaps
        try:
            spearman_heatmap = generate_correlation_heatmap(df, method='spearman', title='Spearman Correlation Heatmap')
            if spearman_heatmap:
                raw = base64.b64decode(spearman_heatmap.encode('utf-8'))
                zf.writestr("correlation_spearman.png", raw)
        except Exception:
            pass
        
        try:
            pearson_heatmap = generate_correlation_heatmap(df, method='pearson', title='Pearson Correlation Heatmap')
            if pearson_heatmap:
                raw = base64.b64decode(pearson_heatmap.encode('utf-8'))
                zf.writestr("correlation_pearson.png", raw)
        except Exception:
            pass
        
        # Generate plots for each numeric column
        for col in df.columns:
            try:
                s = pd.to_numeric(df[col], errors='coerce').dropna()
            except Exception:
                s = pd.Series(dtype=float)
            if s.empty or len(s) < 3:
                continue
            
            # Detect anomalies for this column
            an_idx, an_score = detect_anomalies(s, contamination=0.02)
            
            # Trend plot with anomalies
            try:
                title = f"Trend for {col}"
                img_b64 = generate_plot(
                    s,
                    title,
                    'Timestamp' if is_timeseries else 'Index',
                    col,
                    anomalies_idx=an_idx
                )
                raw = base64.b64decode(img_b64.encode('utf-8'))
                zf.writestr(f"{secure_filename(str(col))}_trend.png", raw)
            except Exception:
                pass
            
            # Distribution histogram
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(s.values, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
                ax.set_title(f"Distribution: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                buf.seek(0)
                zf.writestr(f"{secure_filename(str(col))}_distribution.png", buf.read())
            except Exception:
                pass
            
            # STL decomposition (for timeseries with sufficient data)
            if is_timeseries and len(s) >= 28:
                try:
                    s_norm = normalize_timeseries(s)
                    sp = _infer_seasonal_period(s_norm.index)
                    if sp and isinstance(sp, int) and sp >= 2 and len(s_norm) >= sp * 2:
                        stl_img = generate_stl_plot(s_norm, f"STL Decomposition: {col}", seasonal_period=sp)
                        if stl_img:
                            raw = base64.b64decode(stl_img.encode('utf-8'))
                            zf.writestr(f"{secure_filename(str(col))}_stl.png", raw)
                except Exception:
                    pass
            
            # Forecast (for timeseries)
            if is_timeseries and len(s) >= 10:
                try:
                    fc_mean, ci = _compute_forecast(s, steps=forecast_steps)
                except Exception:
                    try:
                        fc_mean, ci = _recent_slope_forecast(s, steps=forecast_steps)
                    except Exception:
                        fc_mean, ci = None, None
                
                if fc_mean is not None and len(fc_mean) > 0:
                    try:
                        fc_b64 = generate_forecast_plot(s, fc_mean, f"Forecast: {col} ({forecast_steps} steps)", 'Timestamp', col, conf_int=ci, history_tail=None, anomalies_idx=an_idx)
                        raw = base64.b64decode(fc_b64.encode('utf-8'))
                        zf.writestr(f"{secure_filename(str(col))}_forecast.png", raw)
                    except Exception:
                        pass

    bio.seek(0)
    display = request.args.get('display') or filename
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_all_plots.zip")
    resp = make_response(bio.read())
    resp.headers['Content-Type'] = 'application/zip'
    resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
    return resp



class PDFReport(FPDF):
    def __init__(self, title_str, display_name):
        super().__init__()
        self.report_title = title_str
        self.display_name = display_name
    
    def header(self):
        # Only show header on first page
        if self.page_no() != 1:
            return
        try:
            # Determine strictness based on available fonts
            font_family = "Arial" if "Arial" in self.fonts else "helvetica"
            
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
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%H:%M %d.%m.%Y")}', align='R')

@app.route('/download/<filename>/report.pdf', methods=['GET'])
def download_full_report_pdf(filename):
    if not HASHED_UPLOAD_RE.match(filename):
        return ("Not found", 404)

    df = get_dataframe_for(filename)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ("Not found", 404)

    display = request.args.get('display') or filename
    
    # Ensure AI summary is generated if not already cached
    if AI_SUMMARY_CACHE.get(filename) is None:
        try:
            if ensure_ai_ready():
                # Build context for AI
                try:
                    ai_context = describe_for_ai(df)
                except Exception:
                    ai_context = ""
                
                # Get file asset if available
                file_asset = AI_FILE_MAP.get(filename)
                
                # Generate AI summary
                generated = get_ai_summary_with_file(df, file_asset, extra_context=ai_context)
                if isinstance(generated, str) and not _is_offline_html(generated):
                    AI_SUMMARY_CACHE[filename] = generated
                    app.logger.info(f"Generated AI summary for PDF: {filename}")
        except Exception as e:
            app.logger.warning(f"Could not generate AI summary for PDF: {e}")
    
    try:
        app.logger.info(f"Starting PDF generation for {filename}, display={display}")
        pdf = PDFReport("Data Analysis Report", display)
        pdf.alias_nb_pages()
        
        # Try to load a unicode font (Arial) from Windows fonts to support emojis/UTF-8
        # Load BEFORE add_page because header() needs it
        try:
            # Common paths for Arial on Windows
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            if os.path.exists(font_path):
                # fpdf2 add_font(family, style, fname, uni=True) 
                # Note: 'unique' arg is not supported on all versions
                pdf.add_font("Arial", "", font_path)
                app.logger.info("Loaded Arial font successfully")
            else:
                pass
        except Exception as e:
            app.logger.warning(f"Could not load custom font: {e}")

        app.logger.info("Adding page...")
        pdf.add_page()
        app.logger.info(f"Page added. Page No: {pdf.page_no()}")
        
        # Set default font
        if "Arial" in pdf.fonts:
             pdf.set_font("Arial", size=12)
        else:
             pdf.set_font("helvetica", size=12)

        # Helper for adding sections
        def add_section_title(title, new_page=True):
            if new_page:
                pdf.add_page()
            else:
                pdf.ln(5)
            
            # Use Arial if available, else helvetica
            font_family = "Arial" if "Arial" in pdf.fonts else "helvetica"
            pdf.set_font(font_family, style="B", size=13)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)
            pdf.set_font(font_family, size=10)

        def add_text_block(text, courier=False, is_html=False):
            font_family = "Arial" if "Arial" in pdf.fonts else "helvetica"
            
            # Ensure we have a page
            if pdf.page_no() == 0:
                app.logger.warning("No page open, adding one.")
                pdf.add_page()
            
            # Use fpdf2's write_html for HTML content with actual formatting
            if is_html:
                try:
                    # Replace emojis with text equivalents (PDF fonts don't support most emojis)
                    html_text = replace_emojis_for_pdf(text)
                    
                    # Sanitize HTML for fpdf2 compatibility
                    # fpdf2 write_html supports: b, i, u, h1-h6, p, br, ul, ol, li, font, a, img, table
                    html_text = re.sub(r'<strong([^>]*)>', r'<b\1>', html_text, flags=re.I)
                    html_text = re.sub(r'</strong>', '</b>', html_text, flags=re.I)
                    html_text = re.sub(r'<em([^>]*)>', r'<i\1>', html_text, flags=re.I)
                    html_text = re.sub(r'</em>', '</i>', html_text, flags=re.I)
                    
                    # Remove problematic tags but keep their content
                    html_text = re.sub(r'</?div[^>]*>', '', html_text, flags=re.I)
                    html_text = re.sub(r'</?span[^>]*>', '', html_text, flags=re.I)
                    html_text = re.sub(r'</?section[^>]*>', '', html_text, flags=re.I)
                    
                    # Set base font before rendering HTML
                    pdf.set_font(font_family, size=10)
                    
                    # Use write_html for rich formatting (bold, italic, headers, lists)
                    pdf.write_html(html_text)
                    pdf.ln(5)
                    pdf.set_font(font_family, size=10)
                    return
                except Exception as e:
                    app.logger.warning(f"write_html failed: {e}, falling back to plain text")
                    # Fall back to plain text conversion
                    text = convert_html_to_formatted_text(text)
                    text = replace_emojis_for_pdf(text)
            
            # Standard text rendering
            if courier:
                pdf.set_font("Courier", size=9)
            elif "Arial" in pdf.fonts:
                pdf.set_font("Arial", size=10)
            else:
                pdf.set_font("helvetica", size=10)
            
            # Replace emojis for plain text too
            text = replace_emojis_for_pdf(text)
            
            # Render text
            if "Arial" in pdf.fonts:
                pdf.multi_cell(0, 5, text)
            else:
                safe_text = text.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, safe_text)
            pdf.ln(5)
            
            # Reset font
            pdf.set_font(font_family, size=10)

        def add_df_table(df_table, title=None):
            """Renders a pandas DataFrame as a table in the PDF."""
            # Use fpdf2's built-in table context if available, otherwise manual
            try:
                # Basic data prep: Convert all to string, handle truncation
                max_cols = 10
                if len(df_table.columns) > max_cols:
                    df_display = df_table.iloc[:, :max_cols].copy()
                    df_display["..."] = "..."
                else:
                    df_display = df_table.copy()
                
                # Convert all to string
                df_display = df_display.astype(str)
                
                # Include header
                headers = [str(c) for c in df_display.columns]
                # If we want the index to be the first column
                data = []
                for idx, row in df_display.iterrows():
                    row_data = [str(idx)] + [str(x) for x in row.values]
                    data.append(row_data)
                
                headers = ["Index"] + headers
                
                # Estimate table height: Header + Rows
                row_height_est = 8
                # If title is present, add its height
                title_height = 8 if title else 0
                
                total_height_est = (len(data) + 1) * row_height_est + title_height
                
                # If table won't fit on current page (with some margin), add value
                space_left = pdf.h - pdf.b_margin - pdf.get_y()
                
                # Add page if needed
                if total_height_est > space_left and total_height_est < (pdf.h - pdf.t_margin - pdf.b_margin):
                     app.logger.info("Table won't fit (needed %s, left %s), adding page", total_height_est, space_left)
                     pdf.add_page()

                # Print Title NOW, after potential page break
                if title:
                    pdf.set_font("Arial" if "Arial" in pdf.fonts else "helvetica", 'B', 10)
                    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

                # Print Table
                pdf.set_font("Arial" if "Arial" in pdf.fonts else "helvetica", size=8)
                with pdf.table() as table:
                    row = table.row()
                    for h in headers:
                        row.cell(h)
                    for data_row in data:
                        row = table.row()
                        for item in data_row:
                            row.cell(item)
                
                pdf.ln(2)
            except Exception as e:
                app.logger.warning(f"Table rendering failed, falling back to text: {e}")
                # Fallback
                add_text_block(df_table.to_string(), courier=True)

        # Basic Info
        # First section doesn't need a new page (already on p1)
        add_section_title("1. Dataset Overview", new_page=False)
        
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        
        font_family = "Arial" if "Arial" in pdf.fonts else "helvetica"
        
        add_text_block(info_str, courier=True)

        
        # Use new table function for head
        add_df_table(df.head(), title="First 5 Rows:")

        # Use new table function for describe
        add_df_table(df.describe(), title="Statistical Description:")

        # Missing Values
        try:
            mv = df.isnull().sum()
            mvf = mv[mv > 0]
            if not mvf.empty:
                pdf.set_font(font_family, 'B', 10)
                pdf.cell(0, 6, "Missing Values:", new_x="LMARGIN", new_y="NEXT")
                add_text_block(mvf.to_string(), courier=True)
        except:
            pass

        # AI Summary
        ai_html = AI_SUMMARY_CACHE.get(filename)
        if ai_html:
            add_section_title("2. AI Analysis Summary")
            # Use write_html to preserve formatting
            add_text_block(ai_html, is_html=True)

        # Correlation Heatmaps
        try:
            corr_header_added = False
            
            # Helper to check/add header
            def ensure_corr_header():
                nonlocal corr_header_added
                if not corr_header_added:
                    add_section_title("3. Correlation Analysis")
                    corr_header_added = True

            corr_heatmap_spearman = generate_correlation_heatmap(df, method='spearman', title='Spearman Correlation')
            if corr_heatmap_spearman:
                ensure_corr_header()
                # Keep-with-next logic: If near bottom, page break
                if pdf.get_y() > 200: # Approx 297mm height, safety margin
                    pdf.add_page()
                    
                pdf.set_font(font_family, 'B', 10)
                pdf.cell(0, 8, "Spearman Correlation:", new_x="LMARGIN", new_y="NEXT")
                img_data = base64.b64decode(corr_heatmap_spearman)
                # Keep image within page width
                pdf.image(io.BytesIO(img_data), w=150, x=30)
                pdf.ln(5)

            corr_heatmap_pearson = generate_correlation_heatmap(df, method='pearson', title='Pearson Correlation')
            if corr_heatmap_pearson:
                ensure_corr_header()
                # Keep label and image together - add page if not enough space for both
                # Image height is approximately 100-120mm, so break earlier
                if pdf.get_y() > 120:
                    pdf.add_page()
                    
                pdf.set_font(font_family, 'B', 10)
                pdf.cell(0, 8, "Pearson Correlation:", new_x="LMARGIN", new_y="NEXT")
                img_data = base64.b64decode(corr_heatmap_pearson)
                pdf.image(io.BytesIO(img_data), w=150, x=30)
                pdf.ln(5)
        except Exception as e:
            app.logger.error(f"Error adding correlation heatmaps to PDF: {e}")

        # Plots
        # Plots
        add_section_title("4. Column Analysis", new_page=True)
        
        is_ts = isinstance(df.index, pd.DatetimeIndex)
        first_col = True
        
        for col in df.columns:
            try:
                s = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(s) < 3:
                    continue
            except:
                continue
                
            if not isinstance(s.index, type(df.index)):
                 try:
                     s_temp = df[col].copy()
                     s = pd.to_numeric(s_temp, errors='coerce').dropna()
                 except:
                     pass

            # Force new page for EACH column to ensure clean layout
            # We don't use add_section_title here because we want a specific format
            # FIX: Only add page if it's NOT the first column (title page covers it)
            if first_col:
                first_col = False
            else:
                pdf.add_page()
            
            # Use Arial if available
            font_family = "Arial" if "Arial" in pdf.fonts else "helvetica"
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
            
            pdf.set_font(font_family, style="B", size=11)
            # Safe column name
            if "Arial" in pdf.fonts:
                pdf.cell(0, 8, f"Column: {col}", new_x="LMARGIN", new_y="NEXT")
            else:
                safe_col = col.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(0, 8, f"Column: {safe_col}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_family, size=10)

            # Distribution
            try:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.hist(s.values, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
                ax.set_title(f"Distribution: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                buf.seek(0)
                pdf.image(buf, w=140, x=35)
                pdf.ln(2)
            except Exception:
                pass
                
            # STL
            if is_ts and len(s) >= 28:
                try:
                    s_norm = normalize_timeseries(s)
                    sp = _infer_seasonal_period(s_norm.index)
                    if sp and isinstance(sp, int) and sp >= 2 and len(s_norm) >= sp * 2:
                        stl_b64 = generate_stl_plot(s_norm, f"STL Decomposition: {col}", seasonal_period=sp)
                        if stl_b64:
                            pdf.image(io.BytesIO(base64.b64decode(stl_b64)), w=140, x=35)
                            pdf.ln(2)
                except Exception:
                    pass
            
            # Forecast (for timeseries)
            if is_ts and len(s) >= 10:
                try:
                    # Forecast logic from download_full_report_html
                    total_rows = len(df)
                    forecast_steps = max(10, int(total_rows * 0.1))
                    
                    fc_mean, ci = None, None
                    try:
                         fc_mean, ci = _compute_forecast(s, steps=forecast_steps)
                    except Exception as e:
                        try:
                            fc_mean, ci = _recent_slope_forecast(s, steps=forecast_steps)
                        except Exception:
                            fc_mean, ci = None, None
                    
                    if fc_mean is not None and len(fc_mean) > 0:
                        # Find anomaly index if available
                        an_idx, _ = detect_anomalies(s, contamination=0.02)
                        
                        fc_b64 = generate_forecast_plot(
                            s, 
                            fc_mean, 
                            f"Forecast: {col} ({forecast_steps} steps)", 
                            'Timestamp', 
                            col, 
                            conf_int=ci, 
                            history_tail=None, 
                            anomalies_idx=an_idx
                        )
                        if fc_b64:
                            pdf.image(io.BytesIO(base64.b64decode(fc_b64)), w=140, x=35)
                            pdf.ln(2)
                except Exception as e:
                    app.logger.error(f"Error adding forecast to PDF for {col}: {e}")
                    pass
    except Exception as e:
        app.logger.error(f"Error generating PDF: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"ok": False, "message": f"PDF generation failed: {str(e)}"}), 500

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    
    base = os.path.splitext(display)[0]
    out_name = secure_filename(f"{base}_report.pdf")
    
    return make_response(out.read(), 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'attachment; filename="{out_name}"'
    })

@app.route('/download/<filename>/report.html', methods=['GET'])
def download_full_report_html(filename):
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
    except Exception:
        missing_html = ""

    # AI summary
    ai_html = AI_SUMMARY_CACHE.get(filename)
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
        df_num = coerce_numeric_df(df)
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
    except Exception:
        pass

    # Generate plots for each numeric column
    is_ts = isinstance(df.index, pd.DatetimeIndex)
    distribution_sections = []
    stl_sections = []
    forecast_sections = []
    
    # Calculate forecast steps as 10% of dataset size
    total_rows = len(df)
    forecast_steps = max(10, int(total_rows * 0.1))
    
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
            except:
                pass
        
        # Detect anomalies for forecast plots
        an_idx, an_score = detect_anomalies(s, contamination=0.02)
        
        # Generate distribution histogram for this column
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(s.values, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
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
            except:
                pass
        
        # STL decomposition (for timeseries with sufficient data)
        if is_ts and len(s) >= 28:
            try:
                s_norm = normalize_timeseries(s)
                sp = _infer_seasonal_period(s_norm.index)
                if sp and isinstance(sp, int) and sp >= 2 and len(s_norm) >= sp * 2:
                    stl_img = generate_stl_plot(s_norm, f"STL Decomposition: {col}", seasonal_period=sp)
                    if stl_img:
                        stl_sections.append(f'<figure><figcaption><strong>STL Decomposition: {col}</strong></figcaption><img style="max-width:100%" src="data:image/png;base64,{stl_img}" /></figure>')
            except Exception:
                pass
        
        # Forecast (for timeseries) - use 10% of data as forecast horizon
        if is_ts and len(s) >= 10:
            try:
                print(f"[DEBUG] Generating forecast for {col}: is_ts={is_ts}, len={len(s)}, steps={forecast_steps}, index_type={type(s.index)}")
                fc_mean, ci = _compute_forecast(s, steps=forecast_steps)
                print(f"[DEBUG] Forecast result for {col}: fc_mean is {'None' if fc_mean is None else f'{len(fc_mean)} points'}")
            except Exception as e:
                print(f"Forecast error for {col} (_compute_forecast): {e}")
                try:
                    fc_mean, ci = _recent_slope_forecast(s, steps=forecast_steps)
                except Exception as e2:
                    print(f"Forecast error for {col} (_recent_slope_forecast): {e2}")
                    fc_mean, ci = None, None
            
            if fc_mean is not None and len(fc_mean) > 0:
                try:
                    print(f"[DEBUG] Creating forecast plot for {col}")
                    fc_b64 = generate_forecast_plot(s, fc_mean, f"Forecast: {col} ({forecast_steps} steps = 10%)", 'Timestamp', col, conf_int=ci, history_tail=None, anomalies_idx=an_idx)
                    forecast_sections.append(f'<figure><figcaption><strong>Forecast: {col}</strong> ({forecast_steps} steps, 10% of data)</figcaption><img style="max-width:100%" src="data:image/png;base64,{fc_b64}" /></figure>')
                    print(f"[DEBUG] Successfully created forecast plot for {col}")
                except Exception as e:
                    print(f"Forecast plot error for {col}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Forecast is None or empty for {col}")

    # Build HTML report
    print(f"[DEBUG] Report generation complete:")
    print(f"  - Distribution sections: {len(distribution_sections)}")
    print(f"  - STL sections: {len(stl_sections)}")
    print(f"  - Forecast sections: {len(forecast_sections)}")
    if len(forecast_sections) == 0:
        print(f"  - WARNING: No forecast sections were generated!")
    
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
        except Exception:
            pass

        is_ts = isinstance(df.index, pd.DatetimeIndex)

        
        if is_ts:
            idx = df.index
            
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

        
        num_df = coerce_numeric_df(df)  
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
        
        try:
            app.logger.exception("full_history_json failed: %s", e)
        except Exception:
            pass
        return jsonify({"ok": False, "message": f"Internal error: {e}"}), 500

@app.after_request
def _add_cache_and_security_headers(resp):
    """Add caching headers for performance and sanitize security policies."""
    try:
        # Cache headers for API endpoints
        if request.path.startswith('/api/ai-summary/'):
            # Cache AI summary responses for 5 minutes if successful
            if resp.status_code == 200:
                resp.headers['Cache-Control'] = 'private, max-age=300'
            else:
                resp.headers['Cache-Control'] = 'no-cache'
        elif request.path.startswith('/api/') or request.path == '/full_history_json':
            # JSON API responses should be cached briefly
            resp.headers['Cache-Control'] = 'private, max-age=60'
        elif request.path.startswith('/static/'):
            # Static files can be cached for 1 week
            resp.headers['Cache-Control'] = 'public, max-age=604800'
        
        # Sanitize Permissions-Policy header
        if 'Permissions-Policy' in resp.headers:
            pol = str(resp.headers.get('Permissions-Policy', ''))
            bad_bits = ['interest-cohort', 'browsing-topics', 'join-ad-interest-group', 'run-ad-auction']
            cleaned = "; ".join(seg for seg in pol.split(';') if seg and not any(b in seg for b in bad_bits)).strip()
            if cleaned:
                resp.headers['Permissions-Policy'] = cleaned
            else:
                del resp.headers['Permissions-Policy']
    except Exception:
        pass
    return resp

if __name__ == "__main__":
    pass
    
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    host = os.getenv("FLASK_HOST", os.getenv("HOST", "0.0.0.0"))
    try:
        port = int(os.getenv("FLASK_PORT", os.getenv("PORT", "5000")))
    except Exception:
        port = 5000

    
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

    
    if Limiter and os.getenv("RATE_LIMIT"):
        try:
            limiter = Limiter(get_remote_address, app=app, default_limits=[os.getenv("RATE_LIMIT")])
            app.logger.info("Rate limiting enabled: %s", os.getenv("RATE_LIMIT"))
        except Exception as e:
            app.logger.warning("Limiter init failed: %s", e)

    app.logger.info("Starting Flask server on %s:%s (debug=%s)", host, port, debug)
    
    app.run(host=host, port=port, debug=debug, threaded=True)