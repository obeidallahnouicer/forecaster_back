"""Lightweight in-memory cache for responses and embeddings.

This is intentionally simple; it can be swapped for Redis or disk-backed
storage in production.
"""
import time
from typing import Any, Optional


class SimpleCache:
    def __init__(self):
        self._store = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = 3600):
        expiry = time.time() + ttl if ttl else None
        self._store[key] = (value, expiry)

    def get(self, key: str) -> Optional[Any]:
        v = self._store.get(key)
        if not v:
            return None
        value, expiry = v
        if expiry and time.time() > expiry:
            del self._store[key]
            return None
        return value

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

    def clear(self):
        self._store.clear()


_global_cache = SimpleCache()


def get_cache():
    return _global_cache

