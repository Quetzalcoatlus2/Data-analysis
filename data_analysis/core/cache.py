from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd


_MISSING = object()


class TinyLRU(OrderedDict[Any, Any]):
    """Small LRU cache with optional memory-size based eviction."""

    def __init__(self, max_items: int = 6, max_size_mb: int | None = None):
        super().__init__()
        self.max_items = max_items
        self.max_size_mb = max_size_mb
        self._entry_sizes: dict[Any, int] = {}
        self._total_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

    def _evict_oldest(self) -> None:
        """Evict the oldest item while keeping size bookkeeping in sync."""
        self.popitem(last=False)
        self.evictions += 1

    def clear(self) -> None:
        super().clear()
        self._entry_sizes.clear()
        self._total_size_bytes = 0

    def __delitem__(self, key: Any) -> None:
        value = super().__getitem__(key)
        super().__delitem__(key)
        size = self._entry_sizes.pop(key, self._estimate_size_bytes(value))
        self._total_size_bytes = max(0, self._total_size_bytes - int(size))

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if key in self:
            value = super().pop(key)
            size = self._entry_sizes.pop(key, self._estimate_size_bytes(value))
            self._total_size_bytes = max(0, self._total_size_bytes - int(size))
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default

    def popitem(self, last: bool = True) -> tuple[Any, Any]:
        key, value = super().popitem(last=last)
        size = self._entry_sizes.pop(key, self._estimate_size_bytes(value))
        self._total_size_bytes = max(0, self._total_size_bytes - int(size))
        return key, value

    def _estimate_size_bytes(self, value: Any) -> int:
        try:
            if value is None:
                return 0
            if isinstance(value, pd.DataFrame):
                return int(value.memory_usage(deep=True).sum())
            if isinstance(value, pd.Series):
                return int(value.memory_usage(deep=True))
            if isinstance(value, np.ndarray):
                return int(value.nbytes)
            if isinstance(value, (bytes, bytearray)):
                return int(len(value))
            if isinstance(value, str):
                return int(len(value.encode("utf-8", errors="ignore")))
            if isinstance(value, dict):
                total = int(sys.getsizeof(value))
                for key, item in value.items():
                    total += self._estimate_size_bytes(key)
                    total += self._estimate_size_bytes(item)
                return total
            if isinstance(value, (list, tuple, set)):
                total = int(sys.getsizeof(value))
                for item in value:
                    total += self._estimate_size_bytes(item)
                return total
            return int(sys.getsizeof(value))
        except Exception:
            try:
                return int(sys.getsizeof(value))
            except Exception:
                return 0

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self:
            self.hits += 1
            value = super().pop(key)
            super().__setitem__(key, value)
            return value
        self.misses += 1
        return default

    def set(self, key: Any, value: Any) -> None:
        self.sets += 1
        if key in self:
            old_value = super().pop(key)
            old_size = self._entry_sizes.pop(key, self._estimate_size_bytes(old_value))
            self._total_size_bytes = max(0, self._total_size_bytes - int(old_size))

        super().__setitem__(key, value)
        entry_size = int(self._estimate_size_bytes(value))
        self._entry_sizes[key] = entry_size
        self._total_size_bytes += entry_size

        while len(self) > self.max_items:
            self._evict_oldest()

        if self.max_size_mb:
            try:
                max_bytes = self.max_size_mb * 1024 * 1024
                while self._total_size_bytes > max_bytes and len(self) > 1:
                    self._evict_oldest()
            except Exception:
                return

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self),
            "max_items": int(self.max_items),
            "size_bytes": int(self._total_size_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "sets": int(self.sets),
            "evictions": int(self.evictions),
        }


__all__ = ["TinyLRU"]
