from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import Any


class StripAnsiFormatter(logging.Formatter):
    """Formatter that strips ANSI control sequences from log file output."""

    _ansi = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return self._ansi.sub("", rendered)


def configure_logging(app: Any) -> str:
    """Configure app and werkzeug loggers with shared handlers."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    file_handler = RotatingFileHandler("app.log", maxBytes=2_000_000, backupCount=3)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        StripAnsiFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    for handler in (file_handler, console_handler):
        if not any(type(existing) is type(handler) for existing in app.logger.handlers):
            app.logger.addHandler(handler)

    app.logger.setLevel(log_level)

    werk_logger = logging.getLogger("werkzeug")
    werk_logger.setLevel(log_level)
    for handler in (file_handler, console_handler):
        if not any(type(existing) is type(handler) for existing in werk_logger.handlers):
            werk_logger.addHandler(handler)

    return log_level


__all__ = ["StripAnsiFormatter", "configure_logging"]
