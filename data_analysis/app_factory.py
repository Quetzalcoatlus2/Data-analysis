from __future__ import annotations

import os
from typing import Any

from data_analysis.runtime_app import app
from data_analysis.middleware import init_optional_security, register_after_request_middleware

def create_app() -> Any:
    """Create and return the Flask app instance from the modularized runtime."""
    register_after_request_middleware(app)
    return app


def run_from_env(app: Any) -> None:
    """Run Flask app using environment-driven runtime options."""
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
    host = os.getenv("FLASK_HOST", os.getenv("HOST", "0.0.0.0"))
    try:
        port = int(os.getenv("FLASK_PORT", os.getenv("PORT", "5000")))
    except Exception:
        port = 5000

    register_after_request_middleware(app)
    init_optional_security(app)

    app.logger.info("Starting Flask server on %s:%s (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug, threaded=True)


__all__ = ["create_app", "run_from_env"]
