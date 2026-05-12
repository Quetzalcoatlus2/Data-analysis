# mypy: ignore-errors
"""Standalone AI engine: model management, configuration, and Gemini API calls.

All functions accept explicit parameters (app config, logger, etc.) instead of
accessing module-level globals, following the dependency-injection pattern used
throughout the ``data_analysis.analysis`` sub-package.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import contextlib
from typing import Any

from flask import current_app


def _get_logger(logger: Any | None = None) -> Any:
    if logger:
        return logger
    try:
        return current_app.logger
    except RuntimeError:
        return logging.getLogger(__name__)

from data_analysis.core.lazy_imports import get_genai  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level mutable state (owned here, not in runtime_app)
# ---------------------------------------------------------------------------

DEFAULT_AI_MODEL: str = (
    os.getenv("GENAI_MODEL")
    or os.getenv("GOOGLE_MODEL")
    or "gemini-3-flash-preview"
)

MODEL_CACHE: dict[str, Any] = {}
CURRENT_MODEL_NAME: str | None = None
AI_STATUS: dict[str, Any] = {
    "configured": False,
    "ready": False,
    "message": "",
    "model": None,
}
AI_ENABLED: bool = False
model: Any | None = None
AI_CONFIG_ATTEMPTED: bool = False

RATE_LIMITED_MODELS: dict[str, float] = {}
RATE_LIMIT_COOLDOWN_SECONDS: int = 60
_AI_STATE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sanitize_error_message(msg: str | None) -> str:
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
                return "Selected model has no free-tier quota. Switch to a free model (e.g., gemini-3-flash-preview)."
            return "Rate limit exceeded. Please retry after a short pause."
        if 'content blocked' in s.lower() or 'block_reason' in s.lower():
            return "Content was blocked by safety filters."
        if 'timeout' in s.lower():
            return "AI request timed out."
        return s
    except Exception:
        return ""


def _set_ai_status(
    message: str | None = None,
    *,
    ready: bool | None = None,
    model_name: str | None = None,
    configured: bool | None = None,
) -> None:
    try:
        with _AI_STATE_LOCK:
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
    """Produce a robust list of candidate identifiers for a requested model name."""
    if not name:
        return []
    n = name.strip()
    if n.startswith("odels/"):
        n = "m" + n

    candidates = [n]

    if n.startswith("models/"):
        candidates.append(n.replace("models/", "", 1))
    else:
        candidates.append("models/" + n)

    base_name = n.replace("models/", "", 1) if n.startswith("models/") else n
    preview_aliases = {
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    }
    stable_name = preview_aliases.get(base_name)
    if stable_name:
        stable_variants = [stable_name, "models/" + stable_name]
        for variant in reversed(stable_variants):
            if variant not in candidates:
                candidates.insert(0, variant)

    preferred_fallbacks = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest",
    ]

    for fb in preferred_fallbacks:
        if fb not in candidates:
            candidates.append(fb)
        prefixed = "models/" + fb
        if prefixed not in candidates:
            candidates.append(prefixed)

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _extract_text_from_gemini_response(resp: Any, *, logger: Any | None = None) -> str:
    """Robustly extract plain text from a Gemini response."""
    try:
        t = getattr(resp, "text", None)
        if t:
            return str(t)
    except Exception as e:
        logger = _get_logger(logger)
        if logger:
            logger.warning("Gemini response.text accessor failed: %s", e)

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
        logger = _get_logger(logger)
        if logger:
            logger.warning("Gemini candidate parts extraction failed: %s", e)

    return ""


def _make_model(name: str, *, logger: Any | None = None) -> Any:
    genai = get_genai()
    model_ctor = genai.GenerativeModel
    m = model_ctor(name)

    resp = m.generate_content(
        "OK",
        request_options={"timeout": 15},
        generation_config={"response_mime_type": "text/plain"},
    )
    _ = _extract_text_from_gemini_response(resp, logger=logger)
    return m


def get_or_create_model(
    preferred: str | None = None,
    *,
    logger: Any | None = None,
) -> Any:
    """Return a working GenerativeModel, trying aliases and fallbacks. Caches instances."""
    global CURRENT_MODEL_NAME, RATE_LIMITED_MODELS
    with _AI_STATE_LOCK:
        if not AI_ENABLED:
            raise RuntimeError("AI is disabled or not configured.")

        now = time.time()

        expired = [m for m, t in RATE_LIMITED_MODELS.items() if t <= now]
        for m in expired:
            del RATE_LIMITED_MODELS[m]
            logger = _get_logger(logger)
            if logger:
                logger.debug("Rate limit cooldown expired for model: %s", m)

        candidates = _normalize_model_aliases(preferred or DEFAULT_AI_MODEL)
        last_error = None
        models_tried = 0

        for nm in candidates:
            if nm in RATE_LIMITED_MODELS:
                cooldown_remaining = int(RATE_LIMITED_MODELS[nm] - now)
                logger = _get_logger(logger)
                if logger:
                    logger.debug("Skipping rate-limited model %s (cooldown: %ds)", nm, cooldown_remaining)
                continue

            base_nm = nm.replace("models/", "", 1) if nm.startswith("models/") else nm
            prefixed_nm = "models/" + base_nm
            if base_nm in RATE_LIMITED_MODELS or prefixed_nm in RATE_LIMITED_MODELS:
                logger = _get_logger(logger)
                if logger:
                    logger.debug("Skipping rate-limited model variant: %s", nm)
                continue

            models_tried += 1

            try:
                if nm in MODEL_CACHE:
                    CURRENT_MODEL_NAME = nm
                    return MODEL_CACHE[nm]
                m = _make_model(nm, logger=logger)
                MODEL_CACHE[nm] = m
                CURRENT_MODEL_NAME = nm
                logger = _get_logger(logger)
                if logger:
                    logger.info("Using Gemini model: %s", nm)
                return m
            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                if '429' in str(e) or 'quota' in err_str or 'rate limit' in err_str:
                    cooldown_until = now + RATE_LIMIT_COOLDOWN_SECONDS
                    RATE_LIMITED_MODELS[nm] = cooldown_until
                    RATE_LIMITED_MODELS[base_nm] = cooldown_until
                    RATE_LIMITED_MODELS[prefixed_nm] = cooldown_until
                    logger = _get_logger(logger)
                    if logger:
                        logger.warning(
                            "Model %s rate-limited, cooldown %ds: %s",
                            nm, RATE_LIMIT_COOLDOWN_SECONDS, str(e)[:100],
                        )
                elif '404' in str(e) or 'not found' in err_str:
                    logger = _get_logger(logger)
                    if logger:
                        logger.debug("Model %s not found (404), trying next", nm)
                else:
                    logger = _get_logger(logger)
                    if logger:
                        logger.warning("Model candidate failed (%s): %s", nm, str(e)[:150])
                continue

        if models_tried == 0:
            raise RuntimeError("All models are rate-limited. Please wait ~60 seconds and try again.")
        raise RuntimeError(
            f"No working Gemini model available after trying {models_tried} candidates. Last error: {last_error}"
        )


def configure_ai(*, logger: Any | None = None) -> None:
    global model, AI_ENABLED, AI_CONFIG_ATTEMPTED
    with _AI_STATE_LOCK:
        AI_CONFIG_ATTEMPTED = True
        try:
            genai = get_genai()

            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                except Exception as e:
                    logger = _get_logger(logger)
                    if logger:
                        logger.warning("First configure attempt failed, retrying: %s", e)
                    genai.configure(api_key=api_key)

                AI_ENABLED = True
                model = None
                logger = _get_logger(logger)
                if logger:
                    logger.info("AI configured successfully using API key.")
                with contextlib.suppress(Exception):
                    _set_ai_status("OK", ready=False, configured=True, model_name=None)
                return

            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if not credentials_path:
                scan_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
                for f in os.listdir(scan_dir):
                    if f.endswith('.json'):
                        json_path = os.path.join(scan_dir, f)
                        try:
                            with open(json_path, encoding='utf-8') as jf:
                                content = json.load(jf)
                                if content.get('type') == 'service_account':
                                    credentials_path = json_path
                                    logger = _get_logger(logger)
                                    if logger:
                                        logger.info("Found service account JSON: %s", json_path)
                                    break
                        except (OSError, json.JSONDecodeError, KeyError):
                            continue

            google_auth_available = False
            try:
                import google.auth  # noqa: F401
                from google.oauth2 import service_account as _sa
                google_auth_available = True
            except ImportError:
                _sa = None  # type: ignore[assignment]

            if credentials_path and os.path.exists(credentials_path) and google_auth_available and _sa is not None:
                try:
                    scopes = [
                        'https://www.googleapis.com/auth/generative-language',
                        'https://www.googleapis.com/auth/cloud-platform',
                    ]
                    credentials = _sa.Credentials.from_service_account_file(
                        credentials_path, scopes=scopes
                    )
                    genai.configure(credentials=credentials)
                    AI_ENABLED = True
                    model = None
                    logger = _get_logger(logger)
                    if logger:
                        logger.info("AI configured successfully using service account credentials.")
                    with contextlib.suppress(Exception):
                        _set_ai_status("OK (Service Account)", ready=False, configured=True, model_name=None)
                    return
                except Exception as e:
                    logger = _get_logger(logger)
                    if logger:
                        logger.warning("Service account auth failed: %s", e)

            raise ValueError(
                "No valid authentication method found. Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
            )
        except Exception:
            logger = _get_logger(logger)
            if logger:
                logger.exception("AI configuration failed")
            model = None
            AI_ENABLED = False
            with contextlib.suppress(Exception):
                _set_ai_status("Configuration failed", ready=False, configured=False, model_name=None)


def ensure_ai_ready(*, logger: Any | None = None) -> bool:
    """Ensure AI is enabled and a model is available. Attempt lazy init if needed."""
    global model, AI_ENABLED, AI_CONFIG_ATTEMPTED
    try:
        with _AI_STATE_LOCK:
            if not AI_CONFIG_ATTEMPTED:
                logger = _get_logger(logger)
                if logger:
                    logger.info("Attempting to configure AI...")
                configure_ai(logger=logger)

            if not AI_ENABLED:
                with contextlib.suppress(Exception):
                    _set_ai_status("AI disabled or not configured.", ready=False)
                return False
            if model is None:
                try:
                    model = get_or_create_model(DEFAULT_AI_MODEL, logger=logger)
                except Exception as e:
                    logger = _get_logger(logger)
                    if logger:
                        logger.warning("Lazy model init failed: %s", e)
                    with contextlib.suppress(Exception):
                        _set_ai_status(str(e), ready=False, model_name=None)
                    return False
            with contextlib.suppress(Exception):
                _set_ai_status("OK", ready=True, model_name=CURRENT_MODEL_NAME)
            return True
    except Exception as e:
        try:
            logger = _get_logger(logger)
            if logger:
                logger.warning("ensure_ai_ready failed: %s", e)
            with contextlib.suppress(Exception):
                _set_ai_status(str(e), ready=False)
        except Exception:
            pass
        return False


def _call_gemini(
    prompt: str,
    file_asset: Any = None,
    *,
    timeout: int | None = None,
    retries: int | None = None,
    generation_config: dict | None = None,
    app_config: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> Any:
    """Thin wrapper around Gemini calls with retry/backoff, timeout, and generation config."""
    global model
    if not ensure_ai_ready(logger=logger):
        raise RuntimeError("AI is disabled or not configured.")

    cfg = app_config or {}
    attempts = int(cfg.get('AI_RETRY_ATTEMPTS', 2)) if retries is None else int(retries)
    backoff = float(cfg.get('AI_RETRY_BACKOFF_SECONDS', 2.0))
    timeout = int(cfg.get('AI_TIMEOUT_SECONDS', 30)) if timeout is None else int(timeout)
    last_err = None

    for i in range(max(1, attempts + 1)):
        try:
            content = [file_asset, prompt] if file_asset is not None else [prompt]
            gc_effective = generation_config or {}
            if isinstance(gc_effective, dict) and 'response_mime_type' in gc_effective:
                allowed_mimes = {
                    'text/plain', 'application/json', 'application/xml',
                    'application/yaml', 'text/x.enum',
                }
                if gc_effective['response_mime_type'] not in allowed_mimes:
                    logger = _get_logger(logger)
                    if logger:
                        logger.debug(
                            "Coercing unsupported response_mime_type %s to text/plain",
                            gc_effective['response_mime_type'],
                        )
                    gc_effective = dict(gc_effective)
                    gc_effective['response_mime_type'] = 'text/plain'
            active_model = model
            if active_model is None:
                raise RuntimeError("AI model is not initialized")
            resp = active_model.generate_content(
                content,
                request_options={"timeout": timeout},
                generation_config=gc_effective,
            )
            block_reason = None
            try:
                pf = getattr(resp, "prompt_feedback", None)
                block_reason = getattr(pf, "block_reason", None) if pf else None
            except Exception:
                block_reason = None
            if block_reason:
                raise RuntimeError(f"Content blocked: {getattr(block_reason, 'name', block_reason)}")
            return resp
        except Exception as e:
            last_err = e
            msg = str(getattr(e, 'message', None) or e)
            msg_l = msg.lower()
            is_rate = ('429' in msg) or ('rate limit' in msg_l) or ('quota' in msg_l)
            if is_rate:
                try:
                    free_model = DEFAULT_AI_MODEL

                    def _strip_models_prefix(x: str) -> str:
                        return x[7:] if isinstance(x, str) and x.startswith('models/') else x

                    current_eq_free = (
                        _strip_models_prefix(CURRENT_MODEL_NAME or '') == _strip_models_prefix(free_model)
                    )
                    if not current_eq_free:
                        logger = _get_logger(logger)
                        if logger:
                            logger.warning("Rate limit or no free quota; switching model to %s", free_model)
                        free = get_or_create_model(free_model, logger=logger)
                        if free is not None:
                            with _AI_STATE_LOCK:
                                globals()['model'] = free
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
                    logger = _get_logger(logger)
                    if logger:
                        logger.info("Model switch on rate limit failed: %s", e2)
            try:
                with _AI_STATE_LOCK:
                    _set_ai_status(str(last_err), ready=False, model_name=CURRENT_MODEL_NAME)
            except Exception:
                pass

            if i < attempts:
                try:
                    sleep_s = backoff * (i + 1)
                    if is_rate:
                        sleep_s = max(sleep_s, 10 * (i + 1))
                        logger = _get_logger(logger)
                        if logger:
                            logger.info("Rate limit detected, waiting %.1f seconds before retry...", sleep_s)
                    time.sleep(sleep_s)
                except Exception:
                    pass
                continue
            raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Response diagnostics
# ---------------------------------------------------------------------------

def _is_offline_html(s: str) -> bool:
    try:
        t = (s or "").lower()
        return (
            "<h3>offline analysis</h3>" in t
            or "ai response unavailable" in t
            or "ai not ready" in t
            or "ai is disabled" in t
            or "rate limit exceeded" in t
            or "quota exceeded" in t
            or "exceeded your current quota" in t
            or "current quota" in t
            or "plan and billing details" in t
            or "no free-tier quota" in t
        )
    except Exception:
        return False


def _diagnose_gemini_response(resp: Any) -> str:
    """Return a compact diagnostic string from a Gemini response object."""
    try:
        parts: list[str] = []
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


def _get_finish_reason(resp: Any) -> str | None:
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


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # State
    "DEFAULT_AI_MODEL",
    "MODEL_CACHE",
    "CURRENT_MODEL_NAME",
    "AI_STATUS",
    "AI_ENABLED",
    "model",
    "AI_CONFIG_ATTEMPTED",
    "RATE_LIMITED_MODELS",
    "RATE_LIMIT_COOLDOWN_SECONDS",
    # Functions
    "_sanitize_error_message",
    "_set_ai_status",
    "_normalize_model_aliases",
    "_extract_text_from_gemini_response",
    "_make_model",
    "get_or_create_model",
    "configure_ai",
    "ensure_ai_ready",
    "_call_gemini",
    "_is_offline_html",
    "_diagnose_gemini_response",
    "_get_finish_reason",
]
