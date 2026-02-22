from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_analysis.core.cache import TinyLRU


@dataclass(slots=True)
class RuntimeState:
    """Container for mutable runtime state and caches."""

    model_cache: dict[str, Any] = field(default_factory=dict)
    current_model_name: str | None = None
    ai_status: dict[str, Any] = field(
        default_factory=lambda: {"configured": False, "ready": False, "message": "", "model": None}
    )
    ai_enabled: bool = False
    model: Any | None = None

    dataframe_cache: TinyLRU = field(default_factory=lambda: TinyLRU(max_items=6))
    interactive_cache: TinyLRU = field(default_factory=lambda: TinyLRU(max_items=10))
    numeric_df_cache: TinyLRU = field(default_factory=lambda: TinyLRU(max_items=10))
    ai_describe_cache: TinyLRU = field(default_factory=lambda: TinyLRU(max_items=20))

    ai_file_map: dict[str, Any] = field(default_factory=dict)
    original_name_map: dict[str, Any] = field(default_factory=dict)
    ai_summary_cache: dict[str, Any] = field(default_factory=dict)


__all__ = ["RuntimeState"]
