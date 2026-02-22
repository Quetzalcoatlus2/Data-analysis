from __future__ import annotations

import importlib
import warnings
from threading import Lock
from types import ModuleType
from typing import Any

_NO_COLOR_SET = False
_MATPLOTLIB_READY = False

_GENAI_LOCK = Lock()
_GENAI_MODULE: ModuleType | None = None

_MPL_LOCK = Lock()
_MPL_MODULE: ModuleType | None = None
_PYPLOT_MODULE: ModuleType | None = None

_SKLEARN_LOCK = Lock()
_ISOLATION_FOREST: Any | None = None

_STL_LOCK = Lock()
_STL_CLASS: Any | None = None


def get_genai() -> ModuleType:
    """Thread-safe lazy loader for google.generativeai."""
    global _GENAI_MODULE
    if _GENAI_MODULE is None:
        with _GENAI_LOCK:
            if _GENAI_MODULE is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        module=r"google\\.generativeai.*",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=r".*google\\.generativeai.*",
                    )
                    _GENAI_MODULE = importlib.import_module("google.generativeai")
    return _GENAI_MODULE


def _ensure_matplotlib_setup() -> None:
    global _NO_COLOR_SET, _MATPLOTLIB_READY
    if _MATPLOTLIB_READY:
        return
    if not _NO_COLOR_SET:
        import os

        os.environ.setdefault("NO_COLOR", "1")
        _NO_COLOR_SET = True

    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = 0.3
    mpl.rcParams["agg.path.chunksize"] = 10000
    _MATPLOTLIB_READY = True


def get_matplotlib() -> ModuleType:
    """Thread-safe lazy loader for matplotlib."""
    global _MPL_MODULE
    if _MPL_MODULE is None:
        with _MPL_LOCK:
            if _MPL_MODULE is None:
                _ensure_matplotlib_setup()
                _MPL_MODULE = importlib.import_module("matplotlib")
    return _MPL_MODULE


def get_pyplot() -> ModuleType:
    """Thread-safe lazy loader for matplotlib.pyplot."""
    global _PYPLOT_MODULE
    if _PYPLOT_MODULE is None:
        with _MPL_LOCK:
            if _PYPLOT_MODULE is None:
                _ensure_matplotlib_setup()
                _PYPLOT_MODULE = importlib.import_module("matplotlib.pyplot")
    return _PYPLOT_MODULE


def get_isolation_forest() -> Any:
    """Thread-safe lazy loader for sklearn IsolationForest."""
    global _ISOLATION_FOREST
    if _ISOLATION_FOREST is None:
        with _SKLEARN_LOCK:
            if _ISOLATION_FOREST is None:
                module = importlib.import_module("sklearn.ensemble")
                _ISOLATION_FOREST = getattr(module, "IsolationForest")
    return _ISOLATION_FOREST


def get_stl() -> Any:
    """Thread-safe lazy loader for statsmodels STL."""
    global _STL_CLASS
    if _STL_CLASS is None:
        with _STL_LOCK:
            if _STL_CLASS is None:
                module = importlib.import_module("statsmodels.tsa.seasonal")
                _STL_CLASS = getattr(module, "STL")
    return _STL_CLASS


__all__ = [
    "get_genai",
    "get_matplotlib",
    "get_pyplot",
    "get_isolation_forest",
    "get_stl",
]
