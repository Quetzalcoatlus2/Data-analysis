from __future__ import annotations

import sys
import types
from types import ModuleType
from typing import Any

from data_analysis._legacy_loader import get_legacy, get_legacy_if_loaded
from data_analysis.app_factory import create_app, run_from_env


def _load_legacy() -> ModuleType:
    return get_legacy()


def __getattr__(name: str) -> Any:
    if name in {"create_app", "run_from_env"}:
        return globals()[name]
    legacy = _load_legacy()
    try:
        return getattr(legacy, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'app' has no attribute '{name}'") from exc


def __dir__() -> list[str]:
    base = set(globals().keys())
    loaded = get_legacy_if_loaded()
    if loaded is not None:
        base.update(dir(loaded))
    return sorted(base)


class _ProxyModule(types.ModuleType):
    """Mirror writes onto the legacy module for monkeypatch compatibility."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name.startswith("__"):
            return

        legacy = get_legacy_if_loaded()
        if legacy is None:
            try:
                legacy = _load_legacy()
            except Exception:
                return

        if hasattr(legacy, name):
            setattr(legacy, name, value)


current_module = sys.modules[__name__]
if not isinstance(current_module, _ProxyModule):
    current_module.__class__ = _ProxyModule


if __name__ == "__main__":
    run_from_env(create_app())
