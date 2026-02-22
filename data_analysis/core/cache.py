from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd


class TinyLRU(OrderedDict[Any, Any]):
    """Small LRU cache with optional memory-size based eviction."""

    def __init__(self, max_items: int = 6, max_size_mb: int | None = None):
        super().__init__()
        self.max_items = max_items
        self.max_size_mb = max_size_mb
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

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
            super().pop(key)
        super().__setitem__(key, value)

        while len(self) > self.max_items:
            self.evictions += 1
            self.popitem(last=False)

        if self.max_size_mb:
            try:
                total_bytes = sum(self._estimate_size_bytes(v) for v in self.values())
                max_bytes = self.max_size_mb * 1024 * 1024
                while total_bytes > max_bytes and len(self) > 1:
                    self.evictions += 1
                    self.popitem(last=False)
                    total_bytes = sum(self._estimate_size_bytes(v) for v in self.values())
            except Exception:
                return

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self),
            "max_items": int(self.max_items),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "sets": int(self.sets),
            "evictions": int(self.evictions),
        }


__all__ = ["TinyLRU"]
