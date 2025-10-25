"""
Production-grade cache management system.

This module provides a robust caching layer with:
- TTL (time-to-live) support
- Versioning for cache invalidation
- Metadata tracking
- Compression support for large entries
- Thread-safe operations
- Automatic cleanup of expired entries
"""

import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip

logger = logging.getLogger(__name__)


class CacheEntryStatus(Enum):
    """Status of cache entry"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class CacheMetadata:
    """Metadata for cache entries"""
    key: str
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    version: str
    status: str
    size_bytes: int
    hit_count: int = 0
    last_accessed: Optional[float] = None
    tags: Dict[str, Any] = None
    checksum: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)


class CacheManager:
    """
    Base cache manager with persistent storage and TTL support.
    
    Features:
    - File-based persistent cache
    - TTL (time-to-live) support
    - Automatic expiration cleanup
    - Versioning for cache invalidation
    - Thread-safe operations
    - Compression for large entries
    - Hit tracking and statistics
    
    Storage structure:
    cache_dir/
        data/
            <key_hash>.cache    # Compressed/pickled data
        metadata/
            <key_hash>.json     # Metadata as JSON
        locks/
            <key_hash>.lock     # Lock files for thread safety
    """
    
    def __init__(
        self,
        cache_dir: Path,
        default_ttl: int = 3600,  # 1 hour default
        version: str = "v1",
        compression: bool = True,
        auto_cleanup: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            default_ttl: Default TTL in seconds (None = no expiration)
            version: Cache version for invalidation
            compression: Enable gzip compression for large entries
            auto_cleanup: Automatically cleanup expired entries
        """
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / "data"
        self.metadata_dir = self.cache_dir / "metadata"
        self.locks_dir = self.cache_dir / "locks"
        
        # Create directories
        for dir_path in [self.data_dir, self.metadata_dir, self.locks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.version = version
        self.compression = compression
        self.auto_cleanup = auto_cleanup
        
        # Thread locks for cache operations
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        
        # Start cleanup thread if enabled
        if self.auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(f"Initialized CacheManager at {cache_dir} (version={version}, ttl={default_ttl}s)")
    
    def _get_key_hash(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create lock for key"""
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]
    
    def _get_paths(self, key: str) -> tuple:
        """Get file paths for key"""
        key_hash = self._get_key_hash(key)
        data_path = self.data_dir / f"{key_hash}.cache"
        metadata_path = self.metadata_dir / f"{key_hash}.json"
        lock_path = self.locks_dir / f"{key_hash}.lock"
        return data_path, metadata_path, lock_path
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute checksum for data"""
        return hashlib.md5(data).hexdigest()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes"""
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compression and len(data) > 1024:  # Compress if > 1KB
            data = gzip.compress(data)
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value"""
        try:
            # Try decompression first
            data = gzip.decompress(data)
        except gzip.BadGzipFile:
            # Not compressed
            pass
        return pickle.loads(data)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
        status: str = CacheEntryStatus.COMPLETED.value
    ) -> bool:
        """
        Set cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            tags: Additional metadata tags
            status: Entry status
            
        Returns:
            True if successful
        """
        lock = self._get_lock(key)
        with lock:
            try:
                data_path, metadata_path, _ = self._get_paths(key)
                
                # Serialize and write data
                data = self._serialize(value)
                data_path.write_bytes(data)
                
                # Create metadata
                now = time.time()
                ttl_seconds = ttl if ttl is not None else self.default_ttl
                expires_at = now + ttl_seconds if ttl_seconds else None
                
                metadata = CacheMetadata(
                    key=key,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                    version=self.version,
                    status=status,
                    size_bytes=len(data),
                    checksum=self._compute_checksum(data),
                    tags=tags or {}
                )
                
                # Write metadata
                metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))
                
                logger.debug(f"Cache set: {key} ({len(data)} bytes, ttl={ttl_seconds}s)")
                return True
                
            except Exception as e:
                logger.exception(f"Failed to set cache entry {key}: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get cache entry.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        lock = self._get_lock(key)
        with lock:
            try:
                data_path, metadata_path, _ = self._get_paths(key)
                
                # Check if entry exists
                if not data_path.exists() or not metadata_path.exists():
                    return default
                
                # Load metadata
                metadata = CacheMetadata.from_dict(json.loads(metadata_path.read_text()))
                
                # Check version
                if metadata.version != self.version:
                    logger.debug(f"Cache version mismatch for {key}: {metadata.version} != {self.version}")
                    self.delete(key)
                    return default
                
                # Check expiration
                if metadata.is_expired():
                    logger.debug(f"Cache expired for {key}")
                    self.delete(key)
                    return default
                
                # Read and deserialize data
                data = data_path.read_bytes()
                
                # Verify checksum
                if metadata.checksum and self._compute_checksum(data) != metadata.checksum:
                    logger.warning(f"Checksum mismatch for {key}, cache corrupted")
                    self.delete(key)
                    return default
                
                value = self._deserialize(data)
                
                # Update hit stats
                metadata.hit_count += 1
                metadata.last_accessed = time.time()
                metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))
                
                logger.debug(f"Cache hit: {key}")
                return value
                
            except Exception as e:
                logger.exception(f"Failed to get cache entry {key}: {e}")
                return default
    
    def delete(self, key: str) -> bool:
        """
        Delete cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        lock = self._get_lock(key)
        with lock:
            try:
                data_path, metadata_path, lock_path = self._get_paths(key)
                
                deleted = False
                for path in [data_path, metadata_path, lock_path]:
                    if path.exists():
                        path.unlink()
                        deleted = True
                
                if deleted:
                    logger.debug(f"Cache deleted: {key}")
                
                return deleted
                
            except Exception as e:
                logger.exception(f"Failed to delete cache entry {key}: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is valid"""
        return self.get(key) is not None
    
    def get_metadata(self, key: str) -> Optional[CacheMetadata]:
        """Get metadata for key"""
        try:
            _, metadata_path, _ = self._get_paths(key)
            if not metadata_path.exists():
                return None
            return CacheMetadata.from_dict(json.loads(metadata_path.read_text()))
        except Exception as e:
            logger.exception(f"Failed to get metadata for {key}: {e}")
            return None
    
    def update_status(self, key: str, status: str) -> bool:
        """Update entry status"""
        lock = self._get_lock(key)
        with lock:
            try:
                metadata = self.get_metadata(key)
                if not metadata:
                    return False
                
                metadata.status = status
                metadata.updated_at = time.time()
                
                _, metadata_path, _ = self._get_paths(key)
                metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))
                
                logger.debug(f"Updated status for {key}: {status}")
                return True
                
            except Exception as e:
                logger.exception(f"Failed to update status for {key}: {e}")
                return False
    
    def list_keys(self, status: Optional[str] = None, tags: Optional[Dict] = None) -> List[str]:
        """
        List all cache keys.
        
        Args:
            status: Filter by status
            tags: Filter by tags (all must match)
            
        Returns:
            List of matching keys
        """
        keys = []
        try:
            for metadata_path in self.metadata_dir.glob("*.json"):
                try:
                    metadata = CacheMetadata.from_dict(json.loads(metadata_path.read_text()))
                    
                    # Apply filters
                    if status and metadata.status != status:
                        continue
                    
                    if tags:
                        if not metadata.tags:
                            continue
                        if not all(metadata.tags.get(k) == v for k, v in tags.items()):
                            continue
                    
                    keys.append(metadata.key)
                    
                except Exception:
                    continue
            
        except Exception as e:
            logger.exception(f"Failed to list keys: {e}")
        
        return keys
    
    def clear(self, status: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            status: Only clear entries with this status (None = all)
            
        Returns:
            Number of entries cleared
        """
        count = 0
        keys = self.list_keys(status=status)
        for key in keys:
            if self.delete(key):
                count += 1
        
        logger.info(f"Cleared {count} cache entries" + (f" with status={status}" if status else ""))
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        count = 0
        try:
            for metadata_path in self.metadata_dir.glob("*.json"):
                try:
                    metadata = CacheMetadata.from_dict(json.loads(metadata_path.read_text()))
                    if metadata.is_expired():
                        if self.delete(metadata.key):
                            count += 1
                except Exception:
                    continue
        except Exception as e:
            logger.exception(f"Failed to cleanup expired entries: {e}")
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "by_status": {},
            "version": self.version,
            "cache_dir": str(self.cache_dir)
        }
        
        try:
            for metadata_path in self.metadata_dir.glob("*.json"):
                try:
                    metadata = CacheMetadata.from_dict(json.loads(metadata_path.read_text()))
                    stats["total_entries"] += 1
                    stats["total_size_bytes"] += metadata.size_bytes
                    
                    status = metadata.status
                    if status not in stats["by_status"]:
                        stats["by_status"][status] = 0
                    stats["by_status"][status] += 1
                    
                except Exception:
                    continue
        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
        
        return stats
    
    def _start_cleanup_thread(self):
        """Start background thread for automatic cleanup"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # Every 5 minutes
                    self.cleanup_expired()
                except Exception as e:
                    logger.exception(f"Error in cleanup thread: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True, name="CacheCleanup")
        thread.start()
        logger.info("Started cache cleanup thread")
