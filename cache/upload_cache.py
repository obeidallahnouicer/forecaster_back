"""
Upload cache manager with pending/retry/completed state tracking.

This module implements an "upload-first" caching strategy:
1. Cache pending uploads locally until confirmed by backend
2. Track upload state (pending, processing, completed, failed)
3. Automatic retry for failed uploads
4. Cleanup of stale uploads
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .manager import CacheManager, CacheEntryStatus

logger = logging.getLogger(__name__)


class UploadStatus(Enum):
    """Upload status states"""
    PENDING = "pending"          # Uploaded to server, not yet processed
    PROCESSING = "processing"    # Currently being processed
    COMPLETED = "completed"      # Successfully processed
    FAILED = "failed"           # Processing failed
    RETRYING = "retrying"       # Retry in progress


@dataclass
class UploadEntry:
    """Upload cache entry data"""
    upload_id: str
    file_path: str
    file_name: str
    file_size: int
    frequency: str
    session_id: Optional[str] = None
    status: str = UploadStatus.PENDING.value
    error_message: Optional[str] = None
    retry_count: int = 0
    uploaded_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class UploadCacheManager:
    """
    Upload cache manager with state tracking and retry logic.
    
    Features:
    - Track upload state through lifecycle
    - Automatic retry for failed uploads
    - Cleanup of stale uploads
    - Query uploads by status
    - Upload statistics
    
    Upload lifecycle:
    1. PENDING: File uploaded, waiting for processing
    2. PROCESSING: Backend is processing the upload
    3. COMPLETED: Successfully processed, session created
    4. FAILED: Processing failed, eligible for retry
    5. RETRYING: Retry attempt in progress
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_retries: int = 3,
        retry_delay: int = 60,  # seconds
        stale_threshold: int = 86400,  # 24 hours
        ttl: int = 604800  # 7 days
    ):
        """
        Initialize upload cache manager.
        
        Args:
            cache_dir: Base cache directory
            max_retries: Maximum retry attempts for failed uploads
            retry_delay: Delay between retries in seconds
            stale_threshold: Time in seconds before upload is considered stale
            ttl: Time-to-live for completed uploads in seconds
        """
        self.cache_dir = Path(cache_dir)
        upload_cache_dir = self.cache_dir / "uploads"
        
        self.cache = CacheManager(
            cache_dir=upload_cache_dir,
            default_ttl=ttl,
            version="v1",
            compression=False,  # Upload metadata is small
            auto_cleanup=True
        )
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stale_threshold = stale_threshold
        
        logger.info(f"Initialized UploadCacheManager (max_retries={max_retries}, ttl={ttl}s)")
    
    def _make_upload_key(self, upload_id: str) -> str:
        """Generate cache key for upload"""
        return f"upload:{upload_id}"
    
    def create_upload(
        self,
        upload_id: str,
        file_path: str,
        file_name: str,
        file_size: int,
        frequency: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadEntry:
        """
        Create new upload entry.
        
        Args:
            upload_id: Unique upload identifier
            file_path: Path to uploaded file
            file_name: Original filename
            file_size: File size in bytes
            frequency: Forecast frequency (yearly/monthly)
            metadata: Additional metadata
            
        Returns:
            Created upload entry
        """
        entry = UploadEntry(
            upload_id=upload_id,
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            frequency=frequency,
            status=UploadStatus.PENDING.value,
            uploaded_at=time.time(),
            metadata=metadata or {}
        )
        
        key = self._make_upload_key(upload_id)
        self.cache.set(
            key=key,
            value=entry,
            tags={"frequency": frequency, "file_name": file_name},
            status=UploadStatus.PENDING.value
        )
        
        logger.info(f"Created upload entry: {upload_id} ({file_name}, {frequency})")
        return entry
    
    def get_upload(self, upload_id: str) -> Optional[UploadEntry]:
        """Get upload entry by ID"""
        key = self._make_upload_key(upload_id)
        entry = self.cache.get(key)
        return entry
    
    def update_status(
        self,
        upload_id: str,
        status: str,
        session_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update upload status.
        
        Args:
            upload_id: Upload identifier
            status: New status
            session_id: Session ID if completed
            error_message: Error message if failed
            
        Returns:
            True if successful
        """
        entry = self.get_upload(upload_id)
        if not entry:
            logger.warning(f"Upload not found: {upload_id}")
            return False
        
        entry.status = status
        
        if session_id:
            entry.session_id = session_id
        
        if error_message:
            entry.error_message = error_message
        
        if status == UploadStatus.COMPLETED.value:
            entry.completed_at = time.time()
        
        if status == UploadStatus.FAILED.value:
            entry.retry_count += 1
        
        key = self._make_upload_key(upload_id)
        self.cache.set(
            key=key,
            value=entry,
            status=status
        )
        
        logger.info(f"Updated upload {upload_id}: {status}" + (f" (session={session_id})" if session_id else ""))
        return True
    
    def mark_processing(self, upload_id: str) -> bool:
        """Mark upload as processing"""
        return self.update_status(upload_id, UploadStatus.PROCESSING.value)
    
    def mark_completed(self, upload_id: str, session_id: str) -> bool:
        """Mark upload as completed"""
        return self.update_status(upload_id, UploadStatus.COMPLETED.value, session_id=session_id)
    
    def mark_failed(self, upload_id: str, error_message: str) -> bool:
        """Mark upload as failed"""
        return self.update_status(upload_id, UploadStatus.FAILED.value, error_message=error_message)
    
    def can_retry(self, upload_id: str) -> bool:
        """Check if upload can be retried"""
        entry = self.get_upload(upload_id)
        if not entry:
            return False
        
        if entry.status != UploadStatus.FAILED.value:
            return False
        
        if entry.retry_count >= self.max_retries:
            logger.debug(f"Upload {upload_id} exceeded max retries ({entry.retry_count}/{self.max_retries})")
            return False
        
        # Check retry delay
        if entry.completed_at:
            elapsed = time.time() - entry.completed_at
            if elapsed < self.retry_delay:
                logger.debug(f"Upload {upload_id} retry delayed ({elapsed:.0f}s < {self.retry_delay}s)")
                return False
        
        return True
    
    def retry_upload(self, upload_id: str) -> bool:
        """
        Retry failed upload.
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            True if retry initiated
        """
        if not self.can_retry(upload_id):
            return False
        
        entry = self.get_upload(upload_id)
        entry.status = UploadStatus.RETRYING.value
        entry.error_message = None
        
        key = self._make_upload_key(upload_id)
        self.cache.set(key=key, value=entry, status=UploadStatus.RETRYING.value)
        
        logger.info(f"Retrying upload {upload_id} (attempt {entry.retry_count + 1}/{self.max_retries})")
        return True
    
    def list_uploads(
        self,
        status: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> List[UploadEntry]:
        """
        List uploads with optional filters.
        
        Args:
            status: Filter by status
            frequency: Filter by frequency
            
        Returns:
            List of matching upload entries
        """
        tags = {}
        if frequency:
            tags["frequency"] = frequency
        
        keys = self.cache.list_keys(status=status, tags=tags if tags else None)
        
        uploads = []
        for key in keys:
            entry = self.cache.get(key)
            if entry:
                uploads.append(entry)
        
        # Sort by upload time (newest first)
        uploads.sort(key=lambda x: x.uploaded_at or 0, reverse=True)
        return uploads
    
    def get_pending_uploads(self) -> List[UploadEntry]:
        """Get all pending uploads"""
        return self.list_uploads(status=UploadStatus.PENDING.value)
    
    def get_failed_uploads(self) -> List[UploadEntry]:
        """Get all failed uploads eligible for retry"""
        failed = self.list_uploads(status=UploadStatus.FAILED.value)
        return [u for u in failed if self.can_retry(u.upload_id)]
    
    def cleanup_stale(self) -> int:
        """
        Cleanup stale uploads that are stuck in processing.
        
        Returns:
            Number of uploads cleaned up
        """
        count = 0
        now = time.time()
        
        # Check processing uploads
        processing = self.list_uploads(status=UploadStatus.PROCESSING.value)
        for entry in processing:
            if entry.uploaded_at and (now - entry.uploaded_at) > self.stale_threshold:
                logger.warning(f"Marking stale upload as failed: {entry.upload_id}")
                self.mark_failed(
                    entry.upload_id,
                    f"Upload stale after {self.stale_threshold}s"
                )
                count += 1
        
        # Check pending uploads
        pending = self.list_uploads(status=UploadStatus.PENDING.value)
        for entry in pending:
            if entry.uploaded_at and (now - entry.uploaded_at) > self.stale_threshold:
                logger.warning(f"Marking stale pending upload as failed: {entry.upload_id}")
                self.mark_failed(
                    entry.upload_id,
                    f"Pending upload stale after {self.stale_threshold}s"
                )
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} stale uploads")
        
        return count
    
    def delete_upload(self, upload_id: str) -> bool:
        """Delete upload entry"""
        key = self._make_upload_key(upload_id)
        deleted = self.cache.delete(key)
        if deleted:
            logger.info(f"Deleted upload entry: {upload_id}")
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics"""
        stats = {
            "total_uploads": 0,
            "by_status": {},
            "by_frequency": {},
            "retryable_failed": 0
        }
        
        all_uploads = self.list_uploads()
        stats["total_uploads"] = len(all_uploads)
        
        for entry in all_uploads:
            # Count by status
            status = entry.status
            if status not in stats["by_status"]:
                stats["by_status"][status] = 0
            stats["by_status"][status] += 1
            
            # Count by frequency
            freq = entry.frequency
            if freq not in stats["by_frequency"]:
                stats["by_frequency"][freq] = 0
            stats["by_frequency"][freq] += 1
            
            # Count retryable
            if entry.status == UploadStatus.FAILED.value and self.can_retry(entry.upload_id):
                stats["retryable_failed"] += 1
        
        return stats
