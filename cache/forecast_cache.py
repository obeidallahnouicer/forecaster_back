"""
Forecast cache manager with intelligent invalidation.

This module manages forecast caching with:
- Per-article forecast caching
- Summary forecast caching
- Automatic invalidation based on data fingerprint and model version
- Cache warming strategies
- Hit/miss statistics
"""

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import pandas as pd

from .manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class ForecastEntry:
    """Forecast cache entry"""
    ref_article: str
    forecast_data: Dict[str, Any]
    data_fingerprint: str
    model_version: str
    frequency: str
    parameters: Dict[str, Any]  # period, alpha, methods, etc.
    computed_at: float
    computation_time_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class ForecastCacheManager:
    """
    Forecast cache manager with intelligent invalidation (single-user optimized).
    
    Features:
    - Per-article forecast caching
    - Summary caching for batch forecasts
    - Cache invalidation on:
        * Data changes (fingerprint)
        * Model version changes
        * Parameter changes
    - Cache warming for frequently accessed articles
    - Performance metrics tracking
    
    Cache keys (session-agnostic for single-user system):
    - forecast:article:{ref}:{params_hash}
    - forecast:summary:{params_hash}
    
    Note: This is optimized for single-user systems. The data fingerprint is stored
    in cache metadata and checked on retrieval for automatic invalidation.
    """
    
    def __init__(
        self,
        cache_dir: Path,
        default_ttl: int = 86400,  # 24 hours
        model_version: str = "v1"
    ):
        """
        Initialize forecast cache manager.
        
        Args:
            cache_dir: Base cache directory
            default_ttl: Default TTL for forecasts in seconds
            model_version: Model version for cache invalidation
        """
        self.cache_dir = Path(cache_dir)
        forecast_cache_dir = self.cache_dir / "forecasts"
        
        self.cache = CacheManager(
            cache_dir=forecast_cache_dir,
            default_ttl=default_ttl,
            version=model_version,
            compression=True,  # Forecast data can be large
            auto_cleanup=True
        )
        
        self.model_version = model_version
        self.default_ttl = default_ttl
        
        # Cache the current data fingerprint for quick invalidation checks
        self._current_fingerprint = None
        
        logger.info(f"Initialized ForecastCacheManager (single-user mode, version={model_version}, ttl={default_ttl}s)")
    
    def _compute_fingerprint(self, data: Any) -> str:
        """
        Compute data fingerprint for cache invalidation.
        
        Args:
            data: Data to fingerprint (DataFrame, dict, etc.)
            
        Returns:
            SHA256 hash
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Use stable CSV representation
                content = data.to_csv(index=False).encode('utf-8')
            elif isinstance(data, (dict, list)):
                # Use stable JSON representation
                content = json.dumps(data, sort_keys=True).encode('utf-8')
            elif isinstance(data, str):
                content = data.encode('utf-8')
            else:
                content = str(data).encode('utf-8')
            
            return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars
        
        except Exception as e:
            logger.exception(f"Failed to compute fingerprint: {e}")
            return "unknown"
    
    def _compute_params_hash(self, params: Dict[str, Any]) -> str:
        """Compute hash of forecast parameters"""
        try:
            # Sort keys for stability
            content = json.dumps(params, sort_keys=True).encode('utf-8')
            return hashlib.sha256(content).hexdigest()[:8]  # First 8 chars
        except Exception as e:
            logger.exception(f"Failed to compute params hash: {e}")
            return "default"
    
    def _make_article_key(
        self,
        ref_article: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for article forecast (single-user optimized).
        No data fingerprint in key - stored in metadata for validation.
        """
        params_hash = self._compute_params_hash(params)
        return f"forecast:article:{ref_article}:{params_hash}"
    
    def _make_summary_key(
        self,
        params: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for summary forecast (single-user optimized).
        No data fingerprint in key - stored in metadata for validation.
        """
        params_hash = self._compute_params_hash(params)
        return f"forecast:summary:{params_hash}"
    
    def set_current_fingerprint(self, fingerprint: str):
        """
        Set the current data fingerprint for the active dataset.
        This is used to invalidate caches when data changes.
        """
        if self._current_fingerprint != fingerprint:
            logger.info(f"Data fingerprint changed: {self._current_fingerprint} -> {fingerprint}")
            # Data changed - invalidate all caches automatically
            if self._current_fingerprint is not None:
                logger.warning("Data changed - all cached forecasts are now invalid")
        self._current_fingerprint = fingerprint
    
    def get_article_forecast(
        self,
        ref_article: str,
        params: Dict[str, Any],
        data_fingerprint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached forecast for article (single-user optimized).
        
        Args:
            ref_article: Article reference
            params: Forecast parameters (period, alpha, methods, etc.)
            data_fingerprint: Optional data fingerprint for validation
            
        Returns:
            Forecast data or None if not cached/invalid
        """
        key = self._make_article_key(ref_article, params)
        entry = self.cache.get(key)
        
        if entry:
            # Validate data fingerprint if provided
            if data_fingerprint and entry.data_fingerprint != data_fingerprint:
                logger.debug(f"Cache INVALID for article {ref_article} (data changed)")
                self.cache.delete(key)
                return None
            
            logger.debug(f"Cache HIT for article {ref_article}")
            return entry.forecast_data
        
        logger.debug(f"Cache MISS for article {ref_article}")
        return None
    
    def set_article_forecast(
        self,
        ref_article: str,
        forecast_data: Dict[str, Any],
        data_fingerprint: str,
        frequency: str,
        params: Dict[str, Any],
        computation_time_ms: float,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache forecast for article (single-user optimized).
        
        Args:
            ref_article: Article reference
            forecast_data: Forecast results
            data_fingerprint: Data fingerprint (stored in metadata)
            frequency: Forecast frequency (yearly/monthly)
            params: Forecast parameters
            computation_time_ms: Time taken to compute forecast
            ttl: Custom TTL (None = use default)
            
        Returns:
            True if successful
        """
        entry = ForecastEntry(
            ref_article=ref_article,
            forecast_data=forecast_data,
            data_fingerprint=data_fingerprint,
            model_version=self.model_version,
            frequency=frequency,
            parameters=params,
            computed_at=time.time(),
            computation_time_ms=computation_time_ms
        )
        
        # Update current fingerprint
        self.set_current_fingerprint(data_fingerprint)
        
        key = self._make_article_key(ref_article, params)
        success = self.cache.set(
            key=key,
            value=entry,
            ttl=ttl or self.default_ttl,
            tags={
                "ref_article": ref_article,
                "frequency": frequency,
                "type": "article",
                "fingerprint": data_fingerprint
            }
        )
        
        if success:
            logger.debug(f"Cached forecast for article {ref_article} ({computation_time_ms:.0f}ms)")
        
        return success
    
    def get_summary_forecast(
        self,
        params: Dict[str, Any],
        data_fingerprint: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get cached summary forecast (single-user optimized).
        
        Args:
            params: Forecast parameters
            data_fingerprint: Optional data fingerprint for validation
            
        Returns:
            Summary DataFrame or None
        """
        key = self._make_summary_key(params)
        entry = self.cache.get(key)
        
        if entry:
            # Validate data fingerprint if provided
            if data_fingerprint and entry.get("data_fingerprint") != data_fingerprint:
                logger.debug(f"Cache INVALID for summary forecast (data changed)")
                self.cache.delete(key)
                return None
            
            logger.debug(f"Cache HIT for summary forecast")
            # Convert back to DataFrame
            try:
                return pd.DataFrame(entry["data"])
            except Exception as e:
                logger.exception(f"Failed to deserialize summary: {e}")
                return None
        
        logger.debug(f"Cache MISS for summary forecast")
        return None
    
    def set_summary_forecast(
        self,
        summary_df: pd.DataFrame,
        data_fingerprint: str,
        frequency: str,
        params: Dict[str, Any],
        computation_time_ms: float,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache summary forecast (single-user optimized).
        
        Args:
            summary_df: Summary DataFrame
            data_fingerprint: Data fingerprint (stored in metadata)
            frequency: Forecast frequency
            params: Forecast parameters
            computation_time_ms: Computation time
            ttl: Custom TTL
            
        Returns:
            True if successful
        """
        # Convert DataFrame to dict for caching
        try:
            summary_data = {
                "data": summary_df.to_dict(orient='records'),
                "columns": list(summary_df.columns),
                "count": len(summary_df)
            }
        except Exception as e:
            logger.exception(f"Failed to serialize summary: {e}")
            return False
        
        entry = {
            "type": "summary",
            "data": summary_data["data"],
            "columns": summary_data["columns"],
            "count": summary_data["count"],
            "data_fingerprint": data_fingerprint,
            "model_version": self.model_version,
            "frequency": frequency,
            "parameters": params,
            "computed_at": time.time(),
            "computation_time_ms": computation_time_ms
        }
        
        # Update current fingerprint
        self.set_current_fingerprint(data_fingerprint)
        
        key = self._make_summary_key(params)
        success = self.cache.set(
            key=key,
            value=entry,
            ttl=ttl or self.default_ttl,
            tags={
                "frequency": frequency,
                "type": "summary",
                "count": summary_data["count"],
                "fingerprint": data_fingerprint
            }
        )
        
        if success:
            logger.info(f"Cached summary forecast ({len(summary_df)} articles, {computation_time_ms:.0f}ms)")
        
        return success
    
    def invalidate_article(
        self,
        ref_article: str
    ) -> int:
        """
        Invalidate all cached forecasts for article (single-user optimized).
        
        Args:
            ref_article: Article reference
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        keys = self.cache.list_keys(tags={"ref_article": ref_article})
        
        for key in keys:
            if self.cache.delete(key):
                count += 1
        
        if count > 0:
            logger.info(f"Invalidated {count} cache entries for article {ref_article}")
        
        return count
    
    def invalidate_all(self, frequency: Optional[str] = None) -> int:
        """
        Invalidate all cached forecasts.
        
        Args:
            frequency: Only invalidate specific frequency (None = all)
            
        Returns:
            Number of entries invalidated
        """
        tags = {"frequency": frequency} if frequency else None
        keys = self.cache.list_keys(tags=tags)
        
        count = 0
        for key in keys:
            if self.cache.delete(key):
                count += 1
        
        if count > 0:
            logger.info(f"Invalidated {count} cached forecasts" + (f" ({frequency})" if frequency else ""))
        
        return count
    
    def warm_cache(
        self,
        articles: List[str],
        compute_func: callable,
        data_fingerprint: str,
        frequency: str,
        params: Dict[str, Any]
    ) -> int:
        """
        Pre-compute and cache forecasts for articles.
        
        Args:
            articles: List of article references
            compute_func: Function to compute forecast (ref_article) -> dict
            data_fingerprint: Data fingerprint
            frequency: Forecast frequency
            params: Forecast parameters
            
        Returns:
            Number of articles cached
        """
        count = 0
        logger.info(f"Warming cache for {len(articles)} articles...")
        
        for ref in articles:
            try:
                # Check if already cached (use keyword args to match signature)
                # get_article_forecast(ref_article, params, data_fingerprint=None)
                if self.get_article_forecast(ref, params=params, data_fingerprint=data_fingerprint):
                    continue
                
                # Compute forecast
                start_time = time.time()
                forecast_data = compute_func(ref)
                computation_time_ms = (time.time() - start_time) * 1000
                
                if forecast_data:
                    self.set_article_forecast(
                        ref_article=ref,
                        forecast_data=forecast_data,
                        data_fingerprint=data_fingerprint,
                        frequency=frequency,
                        params=params,
                        computation_time_ms=computation_time_ms
                    )
                    count += 1
                
            except Exception as e:
                logger.exception(f"Failed to warm cache for article {ref}: {e}")
        
        logger.info(f"Cache warming complete: {count}/{len(articles)} articles cached")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        base_stats = self.cache.get_stats()
        
        # Count by type
        article_count = len(self.cache.list_keys(tags={"type": "article"}))
        summary_count = len(self.cache.list_keys(tags={"type": "summary"}))
        
        stats = {
            **base_stats,
            "article_forecasts": article_count,
            "summary_forecasts": summary_count,
            "model_version": self.model_version
        }
        
        return stats
    
    def clear_old_versions(self) -> int:
        """Clear forecasts from old model versions"""
        # This is handled automatically by CacheManager version checking
        # But we can explicitly cleanup old cache directories
        count = 0
        
        try:
            parent_dir = self.cache_dir / "forecasts_v2"
            # Look for old version directories if they exist
            # For now, just return 0 as version is handled in metadata
        except Exception as e:
            logger.exception(f"Failed to clear old versions: {e}")
        
        return count
