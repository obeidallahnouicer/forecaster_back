"""
Forecast Session Registry - Manages forecast sessions and metadata.

Simple registry for tracking forecast upload sessions and their metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    session_id: str
    file_path: str
    frequency: str
    created_at: str
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    status: str = "ready"
    forecaster: Any = None


class ForecastRegistry:
    """Registry for managing forecast sessions."""

    def __init__(self, cache_dir: Path = None):
        """Initialize registry with cache directory."""
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.forecasts_dir = self.cache_dir / "forecasts"
        self.summary_file = self.cache_dir / "summary" / "summary.parquet"
        
        # Ensure directories exist
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "summary").mkdir(parents=True, exist_ok=True)
        # In-memory session metadata (map to SessionInfo)
        self.sessions: Dict[str, SessionInfo] = {}
        # Thread lock for concurrent session creation
        self._lock = threading.Lock()
        # on-disk sessions file for simple persistence across restarts
        self._sessions_file = self.cache_dir / "sessions.json"
        # attempt to load persisted sessions
        try:
            self._load_sessions()
        except Exception:
            # if loading fails, start with empty sessions but continue
            logger.exception("Failed to load persisted sessions - starting fresh")
    
    def create_session_from_file(
        self,
        file_path: str,
        frequency: str = "monthly"
    ) -> SessionInfo:
        """
        Create a forecast session from uploaded file.
        Thread-safe with deduplication to prevent multiple sessions for same file.
        
        Args:
            file_path: Path to forecast file
            frequency: Forecast frequency (monthly, weekly, etc.)
            
        Returns:
            Session info dictionary
        """
        from datetime import datetime
        import pandas as pd
        
        with self._lock:
            # Check if session already exists for this file
            # Use file content hash for deduplication
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    with open(file_path_obj, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()[:16]
                    
                    # Check if we already have a session with this file hash
                    for session_id, session_info in self.sessions.items():
                        if (session_info.file_path == file_path and 
                            session_info.frequency == frequency):
                            logger.info(f"Reusing existing session {session_id} for file {file_path}")
                            
                            # Ensure forecaster is initialized (might be None after restart)
                            if session_info.forecaster is None:
                                logger.info(f"Re-initializing forecaster for session {session_id}")
                                try:
                                    import pandas as pd
                                    from sales_forecaster import SalesForecaster
                                    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                                    session_info.forecaster = SalesForecaster(df, cache_dir=str(self.cache_dir), frequency=frequency)
                                    session_info.rows = int(len(df))
                                    session_info.columns = list(df.columns)
                                except Exception as e:
                                    logger.exception(f"Failed to re-initialize forecaster for session {session_id}: {e}")
                            
                            return session_info
            except Exception as e:
                logger.warning(f"Could not check for duplicate session: {e}")
                # Continue with new session creation
            
            # Generate unique session_id with millisecond precision
            # Use timestamp + random component to avoid collisions
            timestamp = datetime.now()
            session_id = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 10000:04d}"
            
            try:
                # Read file to get metadata
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

                session_info = SessionInfo(
                    session_id=session_id,
                    file_path=file_path,
                    frequency=frequency,
                    created_at=datetime.now().isoformat(),
                    rows=int(len(df)),
                    columns=list(df.columns),
                    status="ready"
                )

                # Attempt to instantiate a SalesForecaster for convenience so
                # API callers can immediately use info.forecaster methods.
                try:
                    # Import locally to avoid top-level import cycles
                    from sales_forecaster import SalesForecaster
                    forecaster = SalesForecaster(df, cache_dir=str(self.cache_dir), frequency=frequency)
                    session_info.forecaster = forecaster
                except Exception:
                    # Don't fail session creation if the forecaster can't be instantiated;
                    # keep forecaster as None but log the problem so callers get a clearer
                    # error later and session metadata is still available.
                    logger.exception("Failed to instantiate SalesForecaster for session %s", session_id)

                self.sessions[session_id] = session_info
                logger.info(f"Created forecast session {session_id} (total sessions: {len(self.sessions)})")

                # Persist session metadata (exclude non-serializable objects)
                try:
                    self._save_sessions()
                except Exception:
                    logger.exception("Failed to persist session metadata for %s", session_id)

                return session_info

            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise
    
    def get(self, session_id: str) -> Optional[SessionInfo]:
        """Get session info by ID. Reinitializes forecaster if None."""
        entry = self.sessions.get(session_id)
        # If legacy dict stored, convert to SessionInfo
        if entry is None:
            return None
        if isinstance(entry, dict):
            try:
                si = SessionInfo(
                    session_id=entry.get('session_id', session_id),
                    file_path=entry.get('file_path', ''),
                    frequency=entry.get('frequency', 'monthly'),
                    created_at=entry.get('created_at', ''),
                    rows=int(entry.get('rows', 0)),
                    columns=entry.get('columns', []),
                    status=entry.get('status', 'ready')
                )
                # replace stored dict with SessionInfo
                self.sessions[session_id] = si
                entry = si
            except Exception:
                return None
        
        # Ensure forecaster is initialized (lazy initialization)
        if entry and entry.forecaster is None and entry.file_path:
            logger.info(f"Lazy-loading forecaster for session {session_id}")
            try:
                import pandas as pd
                from sales_forecaster import SalesForecaster
                file_path = entry.file_path
                if Path(file_path).exists():
                    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                    entry.forecaster = SalesForecaster(df, cache_dir=str(self.cache_dir), frequency=entry.frequency)
                    entry.rows = int(len(df))
                    entry.columns = list(df.columns)
                    logger.info(f"Forecaster loaded for session {session_id}: {entry.rows} rows")
                else:
                    logger.warning(f"File not found for session {session_id}: {file_path}")
            except Exception as e:
                logger.exception(f"Failed to load forecaster for session {session_id}: {e}")
        
        return entry
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            try:
                self._save_sessions()
            except Exception:
                logger.exception("Failed to persist sessions after deleting %s", session_id)
            return True
        return False


    def _save_sessions(self) -> None:
        """Save session metadata to disk (simple JSON). Forecaster object is omitted."""
        out = {}
        for sid, si in self.sessions.items():
            try:
                out[sid] = {
                    "session_id": si.session_id,
                    "file_path": si.file_path,
                    "frequency": si.frequency,
                    "created_at": si.created_at,
                    "rows": si.rows,
                    "columns": si.columns,
                    "status": si.status,
                }
            except Exception:
                # skip non-serializable entries
                logger.exception("Failed to serialize session %s", sid)

        try:
            with open(self._sessions_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info("Persisted %d sessions to %s", len(out), self._sessions_file)
        except Exception:
            logger.exception("Failed to write sessions file %s", self._sessions_file)

    def _load_sessions(self) -> None:
        """Load persisted session metadata from disk."""
        if not self._sessions_file.exists():
            return
        try:
            with open(self._sessions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, entry in data.items():
                try:
                    si = SessionInfo(
                        session_id=entry.get("session_id", sid),
                        file_path=entry.get("file_path", ""),
                        frequency=entry.get("frequency", "monthly"),
                        created_at=entry.get("created_at", ""),
                        rows=int(entry.get("rows", 0)),
                        columns=entry.get("columns", []),
                        status=entry.get("status", "ready"),
                        forecaster=None,
                    )
                    self.sessions[sid] = si
                except Exception:
                    logger.exception("Failed to reconstruct session %s from persisted data", sid)
            logger.info("Loaded %d persisted sessions from %s", len(self.sessions), self._sessions_file)
        except Exception:
            logger.exception("Failed to read sessions file %s", self._sessions_file)


# Global registry instance
REGISTRY = ForecastRegistry()
