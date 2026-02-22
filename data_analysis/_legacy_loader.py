from __future__ import annotations

from importlib import import_module
from threading import Lock
from types import ModuleType

_LOCK = Lock()
_LEGACY_MODULE: ModuleType | None = None


def get_legacy() -> ModuleType:
    """Load and cache the runtime module on first access."""
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        with _LOCK:
            if _LEGACY_MODULE is None:
                _LEGACY_MODULE = import_module("data_analysis.runtime_app")
    return _LEGACY_MODULE


def get_legacy_if_loaded() -> ModuleType | None:
    """Return the cached legacy module if it has already been imported."""
    return _LEGACY_MODULE
