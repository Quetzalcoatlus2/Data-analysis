from __future__ import annotations

from typing import Any


def bind_runtime_globals(
    module_globals: dict[str, Any],
    local_symbols: set[str] | frozenset[str],
) -> Any:
    """Sync and mirror ``runtime_app`` symbols into a module's globals namespace."""
    import data_analysis.runtime_app as rt

    sync = getattr(rt, "_sync_ai_engine_state", None)
    if callable(sync):
        sync()

    for key, value in rt.__dict__.items():
        if key.startswith("__") or key in local_symbols:
            continue
        module_globals[key] = value
    return rt


__all__ = ["bind_runtime_globals"]
