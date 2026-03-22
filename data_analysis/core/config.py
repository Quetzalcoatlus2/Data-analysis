from __future__ import annotations

import os
from typing import Any


def load_env_files() -> None:
    """Load public and private env files when python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(".env.public")
        load_dotenv(".env")
    except Exception:
        return


def apply_default_config(app: Any) -> None:
    """Apply baseline Flask configuration defaults."""
    upload_folder = "datasets"
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config.setdefault("UPLOADS_SUBDIR", "uploaded")
    app.config["UPLOADS_DIR"] = os.path.join(upload_folder, app.config["UPLOADS_SUBDIR"])

    env_secret = os.getenv("SECRET_KEY")
    if env_secret:
        app.config["SECRET_KEY"] = env_secret
    else:
        import secrets

        app.config["SECRET_KEY"] = secrets.token_hex(32)
        app.logger.warning(
            "SECRET_KEY not set — using randomly generated key. Sessions will not persist across restarts."
        )

    app.config["DELETE_UPLOADED_AFTER_PROCESSING"] = (
        os.getenv("DELETE_UPLOADED_AFTER_PROCESSING", "true").strip().lower()
        in ("1", "true", "yes", "on")
    )

    if "UPLOAD_RETENTION_DAYS" in os.environ:
        try:
            retention_val = os.getenv("UPLOAD_RETENTION_DAYS")
            if retention_val is not None:
                app.config["UPLOAD_RETENTION_DAYS"] = int(retention_val)
        except Exception:
            app.logger.warning("Invalid UPLOAD_RETENTION_DAYS; ignoring")

    app.config.setdefault("MAX_CACHE_ITEMS", int(os.getenv("MAX_CACHE_ITEMS", "6")))
    app.config.setdefault("LAB_CACHE_MAX_ITEMS", int(os.getenv("LAB_CACHE_MAX_ITEMS", "32")))
    app.config.setdefault("LAB_CACHE_MAX_MB", int(os.getenv("LAB_CACHE_MAX_MB", "64")))
    app.config.setdefault("DEFAULT_FORECAST_STEPS", int(os.getenv("DEFAULT_FORECAST_STEPS", "30")))
    app.config.setdefault("INTERACTIVE_CACHE_MAX_MB", int(os.getenv("INTERACTIVE_CACHE_MAX_MB", "80")))
    app.config.setdefault("AI_TIMEOUT_SECONDS", int(os.getenv("AI_TIMEOUT_SECONDS", "30")))
    app.config.setdefault("AI_RETRY_ATTEMPTS", int(os.getenv("AI_RETRY_ATTEMPTS", "2")))
    app.config.setdefault("AI_RETRY_BACKOFF_SECONDS", float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "2.0")))
    app.config.setdefault("FORECAST_MAX_INPUT_POINTS", int(os.getenv("FORECAST_MAX_INPUT_POINTS", "4000")))
    app.config.setdefault("FORECAST_BOOTSTRAP_SAMPLES", int(os.getenv("FORECAST_BOOTSTRAP_SAMPLES", "60")))
    app.config.setdefault("CACHE_STATS_LOG_EVERY", int(os.getenv("CACHE_STATS_LOG_EVERY", "0")))
    app.config.setdefault("ANOMALY_MARKER_CAP", int(os.getenv("ANOMALY_MARKER_CAP", "20")))
    app.config.setdefault("SEND_FILE_MAX_AGE_DEFAULT", int(os.getenv("SEND_FILE_MAX_AGE", "3600")))
    app.config["AI_FULL_UPLOAD_MAX_MB"] = int(os.getenv("AI_FULL_UPLOAD_MAX_MB", "5"))


__all__ = ["load_env_files", "apply_default_config"]
