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
    
    def create_session_from_file(
        self,
        file_path: str,
        frequency: str = "monthly"
    ) -> SessionInfo:
        """
        Create a forecast session from uploaded file.
        
        Args:
            file_path: Path to forecast file
            frequency: Forecast frequency (monthly, weekly, etc.)
            
        Returns:
            Session info dictionary
        """
        from datetime import datetime
        import pandas as pd
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
            logger.info(f"Created forecast session {session_id}")

            return session_info

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get(self, session_id: str) -> Optional[SessionInfo]:
        """Get session info by ID."""
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
                return si
            except Exception:
                return None
        return entry
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False


# Global registry instance
REGISTRY = ForecastRegistry()
