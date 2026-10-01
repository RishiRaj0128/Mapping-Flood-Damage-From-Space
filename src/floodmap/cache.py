"""Deterministic disk cache manager for spatial and STAC queries."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from floodmap.config import settings
from floodmap.logger import get_logger

logger = get_logger("floodmap.cache")


class DiskCache:
    """Manages disk-based caching of API calls, STAC metadata, and processed spatial data."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, namespace: str, params: dict[str, Any]) -> str:
        """Computes a SHA256 fingerprint for a query parameter set."""
        serialized = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{namespace}_{digest}"

    def get_path(self, namespace: str, params: dict[str, Any], ext: str = "json") -> Path:
        """Returns the file path for a given namespace and parameter dict."""
        key = self._generate_key(namespace, params)
        sub_dir = self.cache_dir / namespace
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir / f"{key}.{ext}"

    def get_json(self, namespace: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Retrieves cached JSON data if present."""
        path = self.get_path(namespace, params, ext="json")
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    logger.debug(f"Cache hit: {path.name}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache at {path}: {e}")
                return None
        return None

    def put_json(self, namespace: str, params: dict[str, Any], data: dict[str, Any]) -> Path:
        """Stores JSON data to the cache."""
        path = self.get_path(namespace, params, ext="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Cached data to: {path.name}")
        return path

    def cached_call(
        self,
        namespace: str,
        params: dict[str, Any],
        fetcher: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Fetches from cache or executes fetcher and populates cache."""
        cached = self.get_json(namespace, params)
        if cached is not None:
            return cached
        result = fetcher()
        self.put_json(namespace, params, result)
        return result


default_cache = DiskCache()
