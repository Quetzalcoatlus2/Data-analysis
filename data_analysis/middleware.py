from __future__ import annotations

import os
from typing import Any, Callable


_HEADERS_MIDDLEWARE_KEY = "data_analysis_headers_middleware_registered"


def add_cache_and_security_headers(resp, *, request_path: str, logger: Any | None = None):
    """Apply cache headers and sanitize Permissions-Policy response header."""
    try:
        if request_path.startswith("/api/ai-summary/"):
            if resp.status_code == 200:
                resp.headers["Cache-Control"] = "private, max-age=300"
            else:
                resp.headers["Cache-Control"] = "no-cache"
        elif request_path.startswith("/api/") or request_path == "/full_history_json":
            resp.headers["Cache-Control"] = "private, max-age=60"
        elif request_path.startswith("/static/"):
            resp.headers["Cache-Control"] = "public, max-age=604800"

        if "Permissions-Policy" in resp.headers:
            policy = str(resp.headers.get("Permissions-Policy", ""))
            bad_bits = ["interest-cohort", "browsing-topics", "join-ad-interest-group", "run-ad-auction"]
            cleaned = "; ".join(
                segment for segment in policy.split(";") if segment and not any(bit in segment for bit in bad_bits)
            ).strip()
            if cleaned:
                resp.headers["Permissions-Policy"] = cleaned
            else:
                try:
                    del resp.headers["Permissions-Policy"]
                except Exception as exc:
                    if logger is not None:
                        logger.debug("Could not delete Permissions-Policy header: %s", exc)
    except Exception as exc:
        if logger is not None:
            logger.warning("Header middleware failed for %s: %s", request_path, exc)
    return resp


def register_after_request_middleware(
    app: Any,
    *,
    handler: Callable[[Any], Any] | None = None,
) -> None:
    """Register after_request middleware once per app instance."""
    extensions = getattr(app, "extensions", None)
    if not isinstance(extensions, dict):
        extensions = {}
        app.extensions = extensions
    if extensions.get(_HEADERS_MIDDLEWARE_KEY):
        return

    if handler is None:
        from flask import request

        def handler(resp):
            return add_cache_and_security_headers(resp, request_path=request.path, logger=app.logger)

    app.after_request(handler)
    extensions[_HEADERS_MIDDLEWARE_KEY] = True


def init_optional_security(app: Any) -> None:
    """Initialize optional Talisman/Limiter integrations from env flags."""
    if str(os.getenv("USE_TALISMAN", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        try:
            from flask_talisman import Talisman
            default_csp = {
                "default-src": ["'self'", "https:", "http:"],
                "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https:", "http:"],
                "style-src": ["'self'", "'unsafe-inline'", "https:", "http:"],
                "img-src": ["'self'", "data:", "blob:", "https:", "http:"],
                "connect-src": ["'self'", "https:", "http:"],
            }
            Talisman(app, content_security_policy=default_csp)
            app.logger.info("Talisman enabled.")
        except ImportError:
            app.logger.warning("Flask-Talisman requested but not installed.")
        except Exception as exc:
            app.logger.warning("Talisman init failed: %s", exc)

    rate_limit = os.getenv("RATE_LIMIT")
    if rate_limit:
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            Limiter(get_remote_address, app=app, default_limits=[rate_limit])
            app.logger.info("Rate limiting enabled: %s", rate_limit)
        except ImportError:
            app.logger.warning("Flask-Limiter requested but not installed.")
        except Exception as exc:
            app.logger.warning("Limiter init failed: %s", exc)


__all__ = [
    "add_cache_and_security_headers",
    "register_after_request_middleware",
    "init_optional_security",
]
