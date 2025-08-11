import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash, after_this_request

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

# Also forward Werkzeug logs to both handlers
werk = logging.getLogger("werkzeug")
werk.setLevel(log_level)
for h in (file_handler, console_handler):
    if not any(type(x) is type(h) for x in werk.handlers):
        werk.addHandler(h)

# --- AI Configuration with Debugging ---  (moved below logging)
app.logger.info("Attempting to configure AI...")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    AI_ENABLED = True
    app.logger.info("AI configured successfully.")
except Exception:
    app.logger.exception("AI configuration failed")
    model = None
    AI_ENABLED = False

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

def get_ai_summary(dataframe_description):
    app.logger.debug("Calling get_ai_summary")
    if not AI_ENABLED or model is None:
        app.logger.debug("AI disabled; skipping get_ai_summary")
        return "AI analysis is disabled. Please check your Google API key and terminal for configuration errors."
    
    try:
        app.logger.debug("Preparing to call Google API for summary")
        prompt = f"""
        You are a data analyst. Based on the following statistical description of a dataset, 
        provide a brief summary and highlight potential insights, trends, or anomalies.
        The user is interested in conclusions, prognostics, and anomalies.
        Keep your analysis concise and easy to understand.

        Strict formatting instructions:
        - Output strictly as an HTML snippet (no <html> or <body> tags).
        - Use <h3>, <p>, <ul><li>, and <strong> for structure.
        - Do NOT use Markdown symbols like #, *, or ```; no code fences.

        Data Description:
        {dataframe_description}
        """
        response = model.generate_content(prompt)
        app.logger.debug("AI summary call successful")

        if not response.parts:
            block_reason = response.prompt_feedback.block_reason.name
            app.logger.warning("AI analysis blocked: %s", block_reason)
            return f"AI analysis was blocked by the content filter. Reason: {block_reason}"

        app.logger.debug("Successfully got AI summary")
        return sanitize_ai_html(response.text)
        
    except Exception as e:
        app.logger.exception("AI summary call failed")
        return f"An error occurred during AI analysis. Check the terminal for more details. Error: {e}"

def get_ai_answer(dataframe, question):
    """Generates a specific answer to a user's question about the dataframe."""
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    
    # Give the AI context with the dataframe's structure and a summary
    df_head = dataframe.head().to_string()
    df_description = dataframe.describe().to_string()

    try:
        prompt = f"""
        You are a helpful data analyst. A user has a specific question about a dataset.
        Use the following information to answer the question.

        Data Summary:
        {df_description}

        First 5 rows of the dataset:
        {df_head}

        Answer the user's question strictly as an HTML snippet (no <html> or <body> tags).
        Use <h4>, <p>, <ul><li>, and <strong>. Do NOT use Markdown symbols.

        User's Question: "{question}"
        """
        
        response = model.generate_content(prompt)
        return sanitize_ai_html(response.text)
    except Exception as e:
        return f"<p>Error while answering the question: {e}</p>"

def get_ai_summary_with_file(df, file_asset=None):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    # concise stats context + prefer attaching the full file if available
    df_description = describe_for_ai(df)
    prompt = f"""
    You are a data analyst. Provide a concise HTML snippet with insights, trends, and anomalies.
    Use <h3>, <p>, <ul><li>, <strong>. No Markdown.

    Data Description:
    {df_description}
    """
    try:
        if file_asset:
            resp = model.generate_content([file_asset, prompt])
        else:
            resp = model.generate_content(prompt)
        return sanitize_ai_html(resp.text)
    except Exception as e:
        return f"<p>Error during AI summary: {e}</p>"

def get_ai_answer_with_file(df, question, file_asset=None):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    df_head = df.head().to_string()
    df_description = df.describe().to_string()
    prompt = f"""
    You are a helpful data analyst. Answer as an HTML snippet (no <html>/<body>).
    Use <h4>, <p>, <ul><li>, <strong>. No Markdown.

    Context:
    Data Summary:
    {df_description}

    First 5 rows:
    {df_head}

    Question: "{question}"
    """
    try:
        if file_asset:
            resp = model.generate_content([file_asset, prompt])
        else:
            resp = model.generate_content(prompt)
        return sanitize_ai_html(resp.text)
    except Exception as e:
        return f"<p>Error while answering the question: {e}</p>"

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

def generate_forecast_plot(history, forecast_series, title, xlabel, ylabel, conf_int=None, history_tail=200):
    fig, ax = plt.subplots(figsize=(10, 4))

    history_tail_series = history.tail(history_tail)
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
    y = series.dropna().astype(float).values.reshape(-1, 1)
    if len(y) < 20:
        return pd.Index([]), pd.Series(dtype=float)
    try:
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        scores = iso.fit_predict(y)  # -1 = anomaly
        anomalies = series.dropna().index[np.where(scores == -1)[0]]
        return anomalies, pd.Series(iso.decision_function(y), index=series.dropna().index)
    except Exception:
        return pd.Index([]), pd.Series(dtype=float)

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
                return redirect(url_for('analyze_file', filename=storage_name, display=orig_name,
                                        forecast_horizon=fh, contamination=cont))
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
                        df2.columns = df2.iloc[header_row]
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
            user_question = request.form.get('question')
            if user_question:
                ai_answer = get_ai_answer_with_file(df, user_question, file_asset)

        # --- Data Analysis & Plotting ---
        analysis = {}
        plots = []
        forecast_plots = []
        anomalies_found = {}
        is_timeseries = isinstance(df.index, pd.DatetimeIndex)
        used_cols = []  # columns that actually yielded numeric data

        # Precompute correlation for advanced view
        corr_payload = None
        try:
            # robust numeric coercion (handles comma-decimals, percents, units)
            df_num = coerce_numeric_df(df)

            # filter to numeric columns with enough data and non-constant
            sel = df_num.select_dtypes(include='number')
            if not sel.empty:
                # drop columns with too few non-nulls
                sel = sel.loc[:, sel.count() >= 3]
                # drop near-constant columns
                try:
                    sel = sel.loc[:, sel.std(skipna=True) > 0]
                except Exception:
                    # fallback if std fails
                    nunique = sel.nunique(dropna=True)
                    sel = sel.loc[:, nunique > 1]

            if sel.shape[1] >= 2:
                corr_df = sel.corr(method='spearman', min_periods=3)
                # drop rows/cols that are all NaN
                corr_df = corr_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                # ensure at least 2x2 left
                if corr_df.shape[0] >= 2 and corr_df.shape[1] >= 2:
                    corr_payload = {
                        "z": corr_df.values.tolist(),
                        "x": corr_df.columns.tolist(),
                        "y": corr_df.index.tolist(),
                    }
                else:
                    corr_payload = None
            else:
                corr_payload = None
        except Exception as e:
            app.logger.info("Correlation build skipped: %s", e)
            corr_payload = None

        interactive = []  # payload for Plotly charts
        PLOTLY_TAIL = int(os.getenv("PLOTLY_TAIL", "800"))
        for column in df.columns:
            # Coerce series to numeric where possible
            series_raw = df[column]
            try:
                series = pd.to_numeric(series_raw, errors='coerce').dropna()
            except Exception:
                series = pd.Series(dtype=float)
            if series.empty:
                continue
            used_cols.append(column)

            # Detect anomalies first so we can overlay on both static and interactive
            an_idx, an_score = detect_anomalies(series, contamination=user_contam)

            # Static trend with anomalies overlay
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
                    # Holt-Winters damped
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
                        need_slope = flat_by_range or flat_by_slope
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
                    split_x = str(series.index[-1])
                    fc_x = [str(i) for i in fc_mean.index]
                    fc_y = [float(v) for v in fc_mean.values]
                    if conf_df is not None:
                        ci_lower = [float(v) for v in conf_df.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df.iloc[:, 1].values]

                    # Keep static forecast image too
                    title_fc = f"Forecast for {column}"
                    forecast_plots.append({
                        "img": generate_forecast_plot(
                            series,
                            fc_mean,
                            title_fc,
                            'Timestamp',
                            column,
                            conf_int=conf_df,
                            history_tail=300
                        ),
                        "title": title_fc
                    })
                except Exception as e:
                    app.logger.warning("Could not generate forecast for %s: %s", column, e)

            # Record anomaly summary (already computed)
            if len(an_idx):
                try:
                    aligned_idx = series.index.intersection(an_idx)
                    an_values = series.loc[aligned_idx].astype(float)
                    min_v = float(np.nanmin(an_values.values))
                    max_v = float(np.nanmax(an_values.values))
                    anomalies_found[column] = {
                        "count": int(len(aligned_idx)),
                        "min_value": min_v,
                        "max_value": max_v,
                        "indices": [str(i) for i in aligned_idx[:50]]
                    }
                except Exception:
                    anomalies_found[column] = {
                        "count": int(len(an_idx)),
                        "indices": [str(i) for i in an_idx[:50]]
                    }

            # ----- Interactive traces -----
            s_tail = series.tail(PLOTLY_TAIL)
            x_hist = [str(i) for i in s_tail.index]
            y_hist = [float(v) for v in s_tail.values]
            traces = [{
                "type": "scatter", "mode": "lines",
                "name": "History", "x": x_hist, "y": y_hist,
                "line": {"color": "rgb(31,119,180)", "width": 2}
            }]
            # anomalies in tail
            if len(an_idx):
                an_tail = [i for i in an_idx if i in s_tail.index]
                if an_tail:
                    traces.append({
                        "type": "scatter", "mode": "markers",
                        "name": "Anomaly",
                        "x": [str(i) for i in an_tail],
                        "y": [float(series.loc[i]) for i in an_tail],
                        "marker": {"color": "red", "size": 7, "symbol": "x"}
                    })
            # forecast for interactive
            fc_x = fc_y = ci_lower = ci_upper = split_x = None
            if is_timeseries and len(series) >= 10:
                try:
                    steps_i = max(10, min(240, user_steps))
                    fc_mean_i, conf_df_i = _recent_slope_forecast(series, steps_i, window=min(len(series), 200), damping=None)
                    try:
                        base_slope_est = float((fc_mean_i.iloc[-1] - fc_mean_i.iloc[0]) / max(1, len(fc_mean_i) - 1))
                        if _is_too_linear(fc_mean_i):
                            fc_mean_i, conf_df_i = _bootstrap_natural_path(
                                series, steps_i, window=min(len(series), 200), base_slope=base_slope_est,
                                n_samples=200, q_low=0.1, q_high=0.9
                            )
                    except Exception:
                        pass
                    split_x = str(series.index[-1])
                    fc_x = [str(i) for i in fc_mean_i.index]
                    fc_y = [float(v) for v in fc_mean_i.values]
                    if conf_df_i is not None and not conf_df_i.empty:
                        ci_lower = [float(v) for v in conf_df_i.iloc[:, 0].values]
                        ci_upper = [float(v) for v in conf_df_i.iloc[:, 1].values]
                    traces.append({
                        "type": "scatter", "mode": "lines+markers",
                        "name": "Forecast", "x": fc_x, "y": fc_y,
                        "line": {"color": "orangered", "width": 3, "dash": "dash"},
                        "marker": {"size": 4}
                    })
                    if ci_lower and ci_upper:
                        traces.append({
                            "type": "scatter", "name": "95% CI",
                            "x": fc_x + fc_x[::-1],
                            "y": ci_upper + ci_lower[::-1],
                            "fill": "toself",
                            "fillcolor": "rgba(255,69,0,0.2)",
                            "line": {"color": "rgba(255,69,0,0)"},
                            "hoverinfo": "skip",
                            "showlegend": True
                        })
                except Exception:
                    pass
            layout = {
                "title": {"text": f"{column} (interactive)", "x": 0.02},
                "xaxis": {"title": "Timestamp" if is_timeseries else "Index", "showgrid": True},
                "yaxis": {"title": column, "showgrid": True},
                "shapes": [] if not split_x else [{
                    "type": "line", "xref": "x", "yref": "paper",
                    "x0": split_x, "x1": split_x, "y0": 0, "y1": 1,
                    "line": {"color": "gray", "width": 1, "dash": "dot"}
                }],
                "legend": {"orientation": "h"},
                "margin": {"l": 40, "r": 10, "t": 40, "b": 40}
            }
            dist = {"name": column, "values": [float(v) for v in series.dropna().tail(5000).values]}
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
        ai_summary = AI_SUMMARY_CACHE.get(filename)
        if ai_summary is None:
            # Only generate on initial GET to avoid rate limits on re-runs/questions
            if request.method == 'GET' and AI_ENABLED and model is not None:
                try:
                    generated = get_ai_summary_with_file(df, file_asset)
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
            def _delete_hashed_file(response):
                try:
                    # delete only on success to avoid "lost file on failure"
                    if 200 <= response.status_code < 300:
                        ok = _safe_delete(filepath)
                        if not ok:
                            app.logger.info("Deferred deletion skipped (could not delete): %s", filepath)
                    else:
                        app.logger.info("Deferred deletion skipped due to non-2xx response: %s", response.status_code)
                except Exception as e:
                    app.logger.warning("Deferred deletion error: %s", e)
                return response

        return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name)

    except Exception as e:
        flash(f"An error occurred while analyzing the file: {e}")
        return redirect(url_for('upload_file'))

@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

if __name__ == '__main__':
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.logger.info(f"Starting server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)