# mypy: ignore-errors
# ruff: noqa: F401

import base64
import hashlib
import html as htmllib
import io
import json
import logging
import math
import os
import re
import sys
import time
import traceback
import uuid
import warnings
import zipfile
from collections import OrderedDict
from datetime import datetime, timedelta
from html.parser import HTMLParser
from logging.handlers import RotatingFileHandler
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    after_this_request,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)
from matplotlib.container import BarContainer
from matplotlib.transforms import blended_transform_factory
from statsmodels.tsa.seasonal import STL  # type: ignore[import-untyped]
from werkzeug.utils import secure_filename

from data_analysis import middleware as app_middleware
from data_analysis.ai import engine as ai_engine
from data_analysis.ai import html_format as ai_html_format
from data_analysis.ai import service as ai_service
from data_analysis.analysis import anomaly as analysis_anomaly
from data_analysis.analysis import context as analysis_context
from data_analysis.analysis import dataframe_ops as analysis_dataframe_ops
from data_analysis.analysis import forecast as analysis_forecast
from data_analysis.analysis import plot as analysis_plots
from data_analysis.core.lazy_imports import get_genai

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"google\.generativeai.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*google\.generativeai.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You have both PyFPDF & fpdf2 installed\..*",
        category=UserWarning,
        module=r"fpdf",
    )
    from fpdf import FPDF

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"app",
)

# Optional / Feature Flag Imports
try:
    import google.auth  # noqa: F401
    from google.oauth2 import service_account
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(".env.public")   
    load_dotenv(".env")          
except Exception:
    pass

try:
    from flask_compress import Compress  # type: ignore[import-untyped]
except ImportError:
    Compress = None

try:
    from flask_limiter import Limiter  # type: ignore[import-untyped]
    from flask_limiter.util import get_remote_address  # type: ignore[import-untyped]
except ImportError:
    Limiter = None

try:
    from flask_talisman import Talisman  # type: ignore[import-untyped]
except ImportError:
    Talisman = None

matplotlib.use('Agg')
# HIGH QUALITY: Increased DPI for sharp images in PDF, ZIP, and web views
matplotlib.rcParams['savefig.dpi'] = 150  
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.3 
matplotlib.rcParams['agg.path.chunksize'] = 10000 
import matplotlib.pyplot as plt  # noqa: E402

# Globally disable pandas column truncation so ALL columns are always shown
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
app = Flask(__name__, template_folder=os.path.join(_PROJECT_ROOT, "templates"))

# PERFORMANCE: Flask-Compress for automatic gzip/brotli compression
if Compress:
    Compress(app)
    app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'text/javascript', 'application/json', 'application/javascript']
    app.config['COMPRESS_LEVEL'] = 6  # Balance between speed and compression ratio
    app.config['COMPRESS_MIN_SIZE'] = 500  # Only compress responses > 500 bytes


UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

from data_analysis.core.config import apply_default_config  # noqa: E402
from data_analysis.core.logging_setup import (  # noqa: E402
    StripAnsiFormatter,
    configure_logging,
)

apply_default_config(app)
os.environ.setdefault("NO_COLOR", "1")
log_level = configure_logging(app)

# AI state variables - synchronized from ai_engine at runtime
DEFAULT_AI_MODEL: str = cast(str, None)
MODEL_CACHE: dict = cast(dict, None)
CURRENT_MODEL_NAME: str = cast(str, None)
AI_STATUS: dict = cast(dict, None)
AI_ENABLED: bool = cast(bool, None)
model: Any = cast(Any, None)
AI_CONFIG_ATTEMPTED: bool = cast(bool, None)
RATE_LIMITED_MODELS: set = cast(set, None)
RATE_LIMIT_COOLDOWN_SECONDS: int = cast(int, None)


def _sync_ai_engine_state() -> None:
    """Mirror mutable AI engine state into runtime_app globals."""
    global DEFAULT_AI_MODEL, MODEL_CACHE, CURRENT_MODEL_NAME, AI_STATUS
    global AI_ENABLED, model, AI_CONFIG_ATTEMPTED
    global RATE_LIMITED_MODELS, RATE_LIMIT_COOLDOWN_SECONDS

    DEFAULT_AI_MODEL = ai_engine.DEFAULT_AI_MODEL
    MODEL_CACHE = ai_engine.MODEL_CACHE
    CURRENT_MODEL_NAME = ai_engine.CURRENT_MODEL_NAME
    AI_STATUS = ai_engine.AI_STATUS
    AI_ENABLED = ai_engine.AI_ENABLED
    model = ai_engine.model
    AI_CONFIG_ATTEMPTED = ai_engine.AI_CONFIG_ATTEMPTED
    RATE_LIMITED_MODELS = ai_engine.RATE_LIMITED_MODELS
    RATE_LIMIT_COOLDOWN_SECONDS = ai_engine.RATE_LIMIT_COOLDOWN_SECONDS


_sync_ai_engine_state()


def _get_genai():
    return get_genai()

def _sanitize_error_message(msg):
    return ai_engine._sanitize_error_message(msg)

def _set_ai_status(message=None, *, ready=None, model_name=None, configured=None):
    return ai_engine._set_ai_status(message, ready=ready, model_name=model_name, configured=configured)

def _normalize_model_aliases(name):
    return ai_engine._normalize_model_aliases(name)

def _extract_text_from_gemini_response(resp):
    return ai_engine._extract_text_from_gemini_response(resp, logger=app.logger)

def _make_model(name):
    return ai_engine._make_model(name, logger=app.logger)

def get_or_create_model(preferred=None):
    try:
        return ai_engine.get_or_create_model(preferred, logger=app.logger)
    finally:
        _sync_ai_engine_state()

def configure_ai():
    try:
        return ai_engine.configure_ai(logger=app.logger)
    finally:
        _sync_ai_engine_state()

def ensure_ai_ready():
    # Preserve compatibility with tests/integrations that monkeypatch app.ensure_ai_ready.
    app_module = sys.modules.get("app")
    override = getattr(app_module, "ensure_ai_ready", None) if app_module is not None else None
    if callable(override) and override is not ensure_ai_ready:
        return override()
    try:
        return ai_engine.ensure_ai_ready(logger=app.logger)
    finally:
        _sync_ai_engine_state()

def _call_gemini(prompt, file_asset=None, *, timeout=None, retries=None, generation_config=None):
    try:
        return ai_engine._call_gemini(
            prompt, file_asset, timeout=timeout, retries=retries,
            generation_config=generation_config,
        )
    finally:
        _sync_ai_engine_state()


from data_analysis.core.cache import TinyLRU  # noqa: E402

DATAFRAME_CACHE = TinyLRU(max_items=app.config['MAX_CACHE_ITEMS'], max_size_mb=int(os.getenv('DATAFRAME_CACHE_MAX_MB', '200')))
NAME_MAP_PATH = os.path.join(UPLOAD_FOLDER, "_name_map.json")  
# Allow configuration of file upload size limit to Gemini (smaller = faster, less timeout risk)
app.config['AI_FULL_UPLOAD_MAX_MB'] = int(os.getenv('AI_FULL_UPLOAD_MAX_MB', '5'))  
AI_FILE_MAP: dict[str, Any] = {}
ORIGINAL_NAME_MAP: dict[str, Any] = {}
AI_SUMMARY_CACHE: dict[str, Any] = {}
QNA_CACHE = TinyLRU(max_items=50)
FORECAST_CACHE = TinyLRU(max_items=32)
# Performance optimization caches - avoid recomputing expensive operations
CORRELATION_CACHE = TinyLRU(max_items=20)  # Cache correlation matrices per dataset
DESCRIPTION_CACHE = TinyLRU(max_items=20)  # Cache describe() and info() per dataset
INTERACTIVE_DATA_CACHE = TinyLRU(
    max_items=10,
    max_size_mb=int(app.config.get('INTERACTIVE_CACHE_MAX_MB', 80)),
)  # Cache interactive chart JSON data for AJAX loading
ANOMALY_CACHE = TinyLRU(max_items=20)  # Cache anomaly detection results per column
# PERFORMANCE: Per-column forecast cache with (filename, column, steps) key
# This ensures forecasts are computed once and reused across Forecast/Interactive/PDF views
COLUMN_FORECAST_CACHE = TinyLRU(max_items=200)
# PERFORMANCE: STL decomposition cache - expensive operation, cache the base64 images
STL_CACHE = TinyLRU(max_items=100)
# PERFORMANCE: Correlation heatmap cache - (filename, method) -> base64_img
HEATMAP_CACHE = TinyLRU(max_items=20)
# PERFORMANCE: Numeric DataFrame cache - avoids repeated coerce_numeric_df calls
NUMERIC_DF_CACHE = TinyLRU(max_items=10)
# PERFORMANCE: Cache describe_for_ai output - expensive string building with sorts
AI_DESCRIBE_CACHE = TinyLRU(max_items=20)

_CACHE_LOG_COUNTER = 0

def _log_cache_stats_if_needed(context: str):
    """Periodically log cache hit/miss stats for request-time observability."""
    global _CACHE_LOG_COUNTER
    try:
        every_n = int(app.config.get('CACHE_STATS_LOG_EVERY', 0) or 0)
    except Exception:
        every_n = 0
    if every_n <= 0:
        return

    _CACHE_LOG_COUNTER += 1
    if _CACHE_LOG_COUNTER % every_n != 0:
        return

    try:
        watched = {
            "DF": DATAFRAME_CACHE,
            "NUM": NUMERIC_DF_CACHE,
            "ANOM": ANOMALY_CACHE,
            "FC": COLUMN_FORECAST_CACHE,
            "INT": INTERACTIVE_DATA_CACHE,
            "CORR": CORRELATION_CACHE,
        }
        parts = []
        for name, cache in watched.items():
            st = cache.stats()
            size_bytes = int(st.get('size_bytes', 0))
            parts.append(
                f"{name}({st['size']}/{st['max_items']} h={st['hits']} m={st['misses']} e={st['evictions']} b={size_bytes})"
            )
        app.logger.info("CacheStats[%s]: %s", context, " | ".join(parts))
    except Exception as e:
        app.logger.debug("Cache stats logging skipped: %s", e)

def _build_interactive_cache_key(filename: str, forecast_pct: float, contamination: float) -> tuple[str, float, float]:
    try:
        pct_key = round(float(forecast_pct), 6)
    except Exception:
        pct_key = 0.05
    try:
        contam_key = round(float(contamination), 6)
    except Exception:
        contam_key = 0.02
    return (str(filename), pct_key, contam_key)

def _load_name_map():
    global ORIGINAL_NAME_MAP
    loaded = analysis_dataframe_ops._load_name_map(NAME_MAP_PATH, logger=app.logger)
    if loaded is not None:
        ORIGINAL_NAME_MAP = loaded

def _save_name_map():
    analysis_dataframe_ops._save_name_map(
        NAME_MAP_PATH,
        ORIGINAL_NAME_MAP,
        logger=app.logger,
    )

def _safe_delete(path, retries=3, delay=0.2):
    return analysis_dataframe_ops._safe_delete(
        path,
        retries=retries,
        delay=delay,
        logger=app.logger,
    )

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)
_load_name_map()

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

HASHED_UPLOAD_RE = re.compile(r'^[a-f0-9]{40,64}\.(txt|csv|xlsx|json)$', re.IGNORECASE)

def allowed_file(filename):
    return analysis_dataframe_ops.allowed_file(filename, ALLOWED_EXTENSIONS)


# ---------------------------------------------------------------------------
# Text / HTML processing  (delegates to ai.html_format)
# ---------------------------------------------------------------------------
CODE_FENCE_START_RE = ai_html_format.CODE_FENCE_START_RE
CODE_FENCE_END_RE = ai_html_format.CODE_FENCE_END_RE
HTML_ENTITY_TAG_RE = ai_html_format.HTML_ENTITY_TAG_RE
HTML_STRUCTURE_RE = ai_html_format.HTML_STRUCTURE_RE
HTML_BODY_TAG_RE = ai_html_format.HTML_BODY_TAG_RE
SCRIPT_STYLE_RE = ai_html_format.SCRIPT_STYLE_RE
EVENT_ATTR_RE = ai_html_format.EVENT_ATTR_RE
JS_PROTOCOL_RE = ai_html_format.JS_PROTOCOL_RE
QUOTED_TERM_RE = ai_html_format.QUOTED_TERM_RE
PLACEHOLDER_TO_EMOJI = ai_html_format.PLACEHOLDER_TO_EMOJI
EMOJI_REPLACEMENTS = ai_html_format.EMOJI_REPLACEMENTS


def _is_reliable_timeseries_index(idx) -> bool:
    """Return True only for clean, monotonic DatetimeIndex suited for time-series logic."""
    if not isinstance(idx, pd.DatetimeIndex):
        return False
    if len(idx) < 2:
        return False
    # Reject if index has NaT values
    if bool(pd.isna(idx).any()):
        return False
    # Reject non-monotonic (increasing) indices
    if not idx.is_monotonic_increasing:
        return False
    # Reject if all timestamps are the same
    if idx[0] == idx[-1]:
        return False
    # Reject if the frequency looks like epoch nanoseconds stored as dates
    # (e.g., 0, 1, 2, ... mapped to 1970-01-01)
    try:
        diffs = idx.to_series().diff().dropna()
        if len(diffs) == 0:
            return False
        median_diff = diffs.median()
        # If the median difference is < 1 microsecond, it's likely epoch nanoseconds
        if median_diff < pd.Timedelta(microseconds=1):
            return False
        # If all timestamps are in the epoch year (1970), likely a mis-parse
        if idx.year.nunique() == 1 and idx.year[0] == 1970:
            return False
    except Exception:
        return False
    return True




def _restore_emoji_placeholders(text: str) -> str:
    return ai_html_format._restore_emoji_placeholders(text)

def _trim_ai_garbage_tail(html_text: str) -> str:
    return ai_html_format._trim_ai_garbage_tail(html_text)

def _apply_text_segment_emphasis(html_text: str) -> str:
    return ai_html_format._apply_text_segment_emphasis(html_text)

def sanitize_ai_html(raw: str) -> str:
    return ai_html_format.sanitize_ai_html(raw)

def convert_html_to_formatted_text(html: str) -> str:
    return ai_html_format.convert_html_to_formatted_text(html)

def replace_emojis_for_pdf(text: str) -> str:
    return ai_html_format.replace_emojis_for_pdf(text)


# ---------------------------------------------------------------------------
# AI service helpers  (delegates to ai.service)
# ---------------------------------------------------------------------------
def _resolve_app_override(name: str):
    """Return a monkeypatched callable from app.py when present."""
    app_module = sys.modules.get("app")
    if app_module is None:
        return None
    override = getattr(app_module, name, None)
    current = globals().get(name)
    if callable(override) and override is not current:
        return override
    return None


def _is_offline_html(html: str) -> bool:
    return ai_engine._is_offline_html(html)


def _build_qna_cache_key(*args, **kwargs):
    return ai_service._build_qna_cache_key(*args, **kwargs)


def get_ai_summary_with_file(*args, **kwargs):
    override = _resolve_app_override("get_ai_summary_with_file")
    if override is not None:
        return override(*args, **kwargs)
    return ai_service.get_ai_summary_with_file(*args, **kwargs)


def get_ai_answer_with_file(*args, **kwargs):
    override = _resolve_app_override("get_ai_answer_with_file")
    if override is not None:
        return override(*args, **kwargs)
    return ai_service.get_ai_answer_with_file(*args, **kwargs)


def get_or_cache_ai_summary_for(*args, **kwargs):
    kwargs.setdefault("ai_summary_cache", AI_SUMMARY_CACHE)
    kwargs.setdefault("ai_file_map", AI_FILE_MAP)
    kwargs.setdefault("sanitize_ai_html_fn", sanitize_ai_html)
    kwargs.setdefault("app_config", app.config)
    kwargs.setdefault("logger", app.logger)
    return ai_service.get_or_cache_ai_summary_for(*args, **kwargs)


def _get_clean_ai_summary_from_cache(
    filename: str,
    *,
    ai_summary_cache: dict[str, Any] | None = None,
    sanitize_ai_html_fn=None,
    app_config: dict[str, Any] | None = None,
):
    return ai_service._get_clean_ai_summary_from_cache(
        filename,
        ai_summary_cache=AI_SUMMARY_CACHE if ai_summary_cache is None else ai_summary_cache,
        sanitize_ai_html_fn=sanitize_ai_html if sanitize_ai_html_fn is None else sanitize_ai_html_fn,
        app_config=app.config if app_config is None else app_config,
    )



def _cap_anomalies_for_display(
    anomalies_idx: pd.Index | None,
    anomalies_score: pd.Series | None = None,
    max_points: int | None = None,
) -> pd.Index:
    if max_points is None:
        try:
            max_points = int(app.config.get('ANOMALY_MARKER_CAP', 20))
        except Exception:
            max_points = 20
    return analysis_plots._cap_anomalies_for_display(
        anomalies_idx,
        anomalies_score=anomalies_score,
        max_points=max_points,
    )


def _anomaly_positions_for_index(data_index: pd.Index, anomalies_idx: pd.Index | None) -> list[int]:
    return analysis_plots._anomaly_positions_for_index(data_index, anomalies_idx)

def generate_plot(data, title, xlabel, ylabel, anomalies_idx=None, use_webp=False):
    return analysis_plots.generate_plot(data, title, xlabel, ylabel, anomalies_idx=anomalies_idx, use_webp=use_webp)

def generate_correlation_heatmap(df, method='spearman', title='Correlation Heatmap'):
    """Generate a correlation heatmap as base64 image."""
    return analysis_plots.generate_correlation_heatmap(df, method=method, title=title)

def get_cached_heatmap(filename: str, df: pd.DataFrame, method: str = 'spearman'):
    """Get correlation heatmap from cache or generate and cache it."""
    cache_key = (filename, method)
    cached = HEATMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    img = generate_correlation_heatmap(df, method=method, title=f'{method.capitalize()} Correlation')
    if img:
        HEATMAP_CACHE.set(cache_key, img)
    return img

def _thin_series(s: pd.Series, max_points: int) -> pd.Series:
    return analysis_forecast._thin_series(s, max_points)

def _thin_series_keep_extrema(s: pd.Series, max_points: int, keep_idx: pd.Index | None = None) -> pd.Series:
    return analysis_forecast._thin_series_keep_extrema(s, max_points, keep_idx=keep_idx)

def _ensure_plot_dicts(items):
    return analysis_context._ensure_plot_dicts(items)

def normalize_timeseries(series: pd.Series) -> pd.Series:
    return analysis_forecast.normalize_timeseries(series)

def generate_stl_plot(series: pd.Series, title: str, seasonal_period: int):
    return analysis_plots.generate_stl_plot(series, title, seasonal_period)

def _infer_future_index(idx, steps):
    return analysis_forecast._infer_future_index(idx, steps)

def _infer_seasonal_period(idx, min_seasons=2):
    return analysis_forecast._infer_seasonal_period(idx, min_seasons)

def _compute_basic_stats(series: pd.Series) -> dict[str, float]:
    return analysis_forecast._compute_basic_stats(series)


def _build_category_plotly_chart(s_cat: pd.Series, col: str) -> dict[str, object] | None:
    """Build Plotly traces/layout for a categorical bar chart with Avg/Med annotations."""
    return analysis_plots._build_category_plotly_chart(s_cat, col)


def generate_forecast_plot(
    history,
    forecast_series,
    title,
    xlabel,
    ylabel,
    conf_int=None,
    history_tail=None,
    anomalies_idx=None,
    anomalies_score=None,
    stats=None,
    legend_y=None,
    xlabel_labelpad=None,
):
    """Generate a plot showing historical data and forecast with confidence intervals."""
    return analysis_plots.generate_forecast_plot(
        history, forecast_series, title, xlabel, ylabel,
        conf_int=conf_int, history_tail=history_tail,
        anomalies_idx=anomalies_idx, anomalies_score=anomalies_score,
        stats=stats, legend_y=legend_y, xlabel_labelpad=xlabel_labelpad,
    )



def _match_amplitude(
    history: pd.Series,
    forecast_series: pd.Series,
    conf_df: pd.DataFrame | None = None,
    seasonal_period: int | None = None,
    min_scale: float = 0.85,
    max_scale: float = 2.5,
):
    return analysis_forecast._match_amplitude(
        history,
        forecast_series,
        conf_df=conf_df,
        seasonal_period=seasonal_period,
        min_scale=min_scale,
        max_scale=max_scale,
    )

def _compute_forecast(series: pd.Series, steps: int):
    return analysis_forecast._compute_forecast(series, steps)

def _recent_slope_forecast(series: pd.Series, steps: int = 10, lookback: int = 20):
    return analysis_forecast._recent_slope_forecast(series, steps, lookback)

def _forecast_with_fallback(series: pd.Series, steps: int, filename: str | None = None, col: str | None = None):
    return analysis_forecast._forecast_with_fallback(series, steps, filename, col)

def get_cached_column_forecast(filename: str, column: str, series: pd.Series, steps: int):
    return analysis_forecast.get_cached_column_forecast(filename, column, series, steps)

def get_cached_stl_plot(filename: str, column: str, series: pd.Series, seasonal_period: int):
    """Get STL decomposition plot from cache or generate and cache it.
    
    STL decomposition is computationally expensive. This ensures each unique
    (filename, column, seasonal_period) is only computed once.
    """
    if seasonal_period is None or seasonal_period < 2:
        return None
    cache_key = (filename, str(column), int(seasonal_period))
    cached = STL_CACHE.get(cache_key)
    if cached is not None:
        app.logger.debug("STL cache HIT: %s/%s", filename[:8], column)
        return cached
    app.logger.debug("STL cache MISS: %s/%s - generating", filename[:8], column)
    s_norm = normalize_timeseries(series)
    if s_norm is None or len(s_norm) < max(28, seasonal_period * 2):
        return None
    stl_img = generate_stl_plot(s_norm, f"STL decomposition for {column}", seasonal_period=seasonal_period)
    if stl_img:
        STL_CACHE.set(cache_key, stl_img)
    return stl_img

def read_csv_fallback(path, **kwargs):
    return analysis_dataframe_ops.read_csv_fallback(
        path,
        supported_encodings=SUPPORTED_ENCODINGS,
        **kwargs,
    )

def read_json_fallback(path):
    return analysis_dataframe_ops.read_json_fallback(
        path,
        supported_encodings=SUPPORTED_ENCODINGS,
    )

def _cleanup_uploads_if_configured():
    uploads_dir = app.config.get('UPLOADS_DIR', UPLOAD_FOLDER)
    analysis_dataframe_ops._cleanup_uploads_if_configured(
        upload_retention_days=app.config.get('UPLOAD_RETENTION_DAYS'),
        uploads_dir=uploads_dir,
        hashed_upload_re=HASHED_UPLOAD_RE,
        logger=app.logger,
    )
    try:
        stale_keys = [
            key
            for key in list(AI_FILE_MAP.keys())
            if not os.path.isfile(os.path.join(uploads_dir, str(key)))
        ]
        for key in stale_keys:
            AI_FILE_MAP.pop(key, None)
        if stale_keys:
            app.logger.debug("Pruned %d stale AI_FILE_MAP entries", len(stale_keys))
    except Exception as e:
        app.logger.debug("AI_FILE_MAP prune skipped: %s", e)

def _try_parse_numeric_series(s: pd.Series) -> pd.Series:
    return analysis_dataframe_ops._try_parse_numeric_series(s)

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return analysis_dataframe_ops.coerce_numeric_df(
        df,
        parse_numeric_series_fn=_try_parse_numeric_series,
    )

def get_cached_numeric_df(filename: str, df: pd.DataFrame) -> pd.DataFrame:
    return analysis_dataframe_ops.get_cached_numeric_df(
        filename,
        df,
        numeric_df_cache=NUMERIC_DF_CACHE,
        coerce_numeric_df_fn=coerce_numeric_df,
    )

def detect_anomalies(series: pd.Series, contamination: float = 0.02):
    return analysis_anomaly.detect_anomalies(
        series,
        contamination=contamination,
        is_reliable_timeseries_index=_is_reliable_timeseries_index,
        infer_seasonal_period=_infer_seasonal_period,
        logger=app.logger,
    )

def _anomaly_series_signature(series: pd.Series) -> tuple[Any, ...]:
    return analysis_anomaly._anomaly_series_signature(series)


def get_cached_anomalies(filename: str, column: str, series: pd.Series, contamination: float = 0.02):
    return analysis_anomaly.get_cached_anomalies(
        filename,
        column,
        series,
        contamination=contamination,
        cache=ANOMALY_CACHE,
        logger=app.logger,
        is_reliable_timeseries_index=_is_reliable_timeseries_index,
        infer_seasonal_period=_infer_seasonal_period,
    )

def build_ai_context(df: pd.DataFrame, anomalies_found: dict, corr_payload: dict | None, used_cols: list, is_timeseries: bool, forecast_horizon: int, contamination: float) -> str:
    """Assemble structured stats the AI can leverage for a deeper analysis."""
    return analysis_context.build_ai_context(
        df=df,
        anomalies_found=anomalies_found,
        corr_payload=corr_payload,
        used_cols=used_cols,
        is_timeseries=is_timeseries,
        forecast_horizon=forecast_horizon,
        contamination=contamination,
        is_reliable_timeseries_index=_is_reliable_timeseries_index,
    )


def read_excel_smart(path: str):
    return analysis_dataframe_ops.read_excel_smart(path)

def get_dataframe_for(filename):
    return analysis_dataframe_ops.get_dataframe_for(
        filename,
        dataframe_cache=DATAFRAME_CACHE,
        uploads_dir=app.config.get('UPLOADS_DIR', UPLOAD_FOLDER),
        upload_folder=UPLOAD_FOLDER,
        logger=app.logger,
        read_csv_fallback_fn=read_csv_fallback,
        read_json_fallback_fn=read_json_fallback,
        read_excel_smart_fn=read_excel_smart,
        is_reliable_timeseries_index=_is_reliable_timeseries_index,
    )

def _should_show_index_as_column(df: pd.DataFrame) -> bool:
    """Return True when the DataFrame index carries meaningful dataset information."""
    try:
        idx = df.index
        if isinstance(idx, pd.RangeIndex):
            if idx.name in (None, "") and int(idx.start) == 0 and int(idx.step) == 1:
                return False
        return True
    except Exception as e:
        app.logger.debug("Index visibility check failed: %s", e)
        return False


def _display_df_with_index(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize the index as the first display column when it is meaningful."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    try:
        if not _should_show_index_as_column(df):
            return df.copy()

        idx_name = str(df.index.name).strip() if df.index.name not in (None, "") else "Index"
        display_df = df.reset_index(drop=False)

        # Keep first column name stable and avoid collisions with existing data columns.
        first_col = display_df.columns[0]
        if str(first_col) != idx_name:
            safe_name = idx_name
            if safe_name in display_df.columns[1:]:
                suffix = 1
                while f"{idx_name}_{suffix}" in display_df.columns[1:]:
                    suffix += 1
                safe_name = f"{idx_name}_{suffix}"
            display_df = display_df.rename(columns={first_col: safe_name})
        return display_df
    except Exception as e:
        app.logger.debug("Failed to materialize DataFrame index column: %s", e)
        return df.copy()


def safe_df_head_html(df: pd.DataFrame) -> str:
    return analysis_context.safe_df_head_html(df, logger=app.logger)

def safe_df_description_html(df: pd.DataFrame) -> str:
    return analysis_context.safe_df_description_html(df, logger=app.logger)


def get_dataset_overview_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return analysis_context.get_dataset_overview_tables(df)


def safe_dataset_overview_html(df: pd.DataFrame) -> str:
    return analysis_context.safe_dataset_overview_html(df, logger=app.logger)

def get_cached_df_info(filename: str, df: pd.DataFrame) -> dict:
    return analysis_context.get_cached_df_info(
        filename,
        df,
        cache=DESCRIPTION_CACHE,
        logger=app.logger,
    )

def describe_for_ai(df: pd.DataFrame, filename: str | None = None) -> str:
    return analysis_context.describe_for_ai(
        df,
        filename=filename,
        describe_cache=AI_DESCRIBE_CACHE,
    )

def offline_answer(df: pd.DataFrame, question: str = "summary", error=None, filename: str | None = None) -> str:
    return analysis_context.offline_answer(
        df,
        question=question,
        error=error,
        filename=filename,
        sanitize_error_message=_sanitize_error_message,
        ai_status=AI_STATUS,
        get_cached_numeric_df_fn=get_cached_numeric_df,
        coerce_numeric_df_fn=coerce_numeric_df,
    )

def _get_arg_float(name, default):
    return analysis_context._get_arg_float(name, default, request)

def _get_arg_int(name, default):
    return analysis_context._get_arg_int(name, default, request)


class PDFReport:
    """Compatibility proxy for relocated PDFReport implementation."""

    def __new__(cls, *args, **kwargs):
        from data_analysis.reports.pdf_report import PDFReport as _PDFReport

        return _PDFReport(*args, **kwargs)



def _add_cache_and_security_headers(resp):
    return app_middleware.add_cache_and_security_headers(
        resp,
        request_path=request.path,
        logger=app.logger,
    )


app_middleware.register_after_request_middleware(app, handler=_add_cache_and_security_headers)

from data_analysis.routes.api_routes import api_bp  # noqa: E402
from data_analysis.routes.pages import pages_bp  # noqa: E402

app.register_blueprint(pages_bp)
app.register_blueprint(api_bp)

if __name__ == "__main__":
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
                "img-src": ["'self'", "data:", "blob:", "https:", "http:"],
                "connect-src": ["'self'", "https:", "http:"],
            }
            Talisman(app, content_security_policy=default_csp)
            app.logger.info("Talisman enabled.")
        except Exception as e:
            app.logger.warning("Talisman init failed: %s", e)

    
    rate_limit = os.getenv("RATE_LIMIT")
    if Limiter and rate_limit:
        try:
            limiter = Limiter(get_remote_address, app=app, default_limits=[rate_limit])
            app.logger.info("Rate limiting enabled: %s", rate_limit)
        except Exception as e:
            app.logger.warning("Limiter init failed: %s", e)

    app.logger.info("Starting Flask server on %s:%s (debug=%s)", host, port, debug)
    
    app.run(host=host, port=port, debug=debug, threaded=True)

