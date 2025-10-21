"""
Advanced Logging System - Logs agent reasoning steps for transparency.

This module provides comprehensive logging capabilities for the multi-agent
system, enabling full audit trails and debugging of agent reasoning processes.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from enum import Enum

logger = logging.getLogger("core.logger")


class LogLevel(Enum):
    """Log levels for agent reasoning."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ReasoningStep:
    """Individual reasoning step from an agent."""
    step_id: str
    agent_name: str
    timestamp: datetime
    level: LogLevel
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error
        }


@dataclass
class ReasoningSession:
    """Complete reasoning session with all steps."""
    session_id: str
    query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[ReasoningStep] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the session."""
        self.steps.append(step)
    
    def get_duration(self) -> Optional[float]:
        """Get total session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_success_rate(self) -> float:
        """Get success rate of reasoning steps."""
        if not self.steps:
            return 0.0
        
        successful_steps = sum(1 for step in self.steps if step.success)
        return successful_steps / len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "success_rate": self.get_success_rate(),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata
        }


class AgentLogger:
    """
    Specialized logger for agent reasoning steps.
    
    Features:
    - Structured logging with reasoning steps
    - Session-based logging for complete audit trails
    - Performance monitoring and timing
    - Error tracking and analysis
    - Export capabilities for debugging
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize agent logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory session storage
        self._sessions: Dict[str, ReasoningSession] = {}
        self._lock = threading.RLock()
        
        # Setup file logging
        self._setup_file_logging()
        
        self.logger = logging.getLogger("core.agent_logger")
    
    def _setup_file_logging(self) -> None:
        """Setup file-based logging."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup main log file
        main_handler = logging.FileHandler(self.log_dir / "agent_reasoning.log")
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.DEBUG)
        
        # Setup error log file
        error_handler = logging.FileHandler(self.log_dir / "agent_errors.log")
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Setup logger
        agent_logger = logging.getLogger("agent_reasoning")
        agent_logger.setLevel(logging.DEBUG)
        agent_logger.addHandler(main_handler)
        agent_logger.addHandler(error_handler)
    
    def start_session(
        self, 
        session_id: str, 
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningSession:
        """
        Start a new reasoning session.
        
        Args:
            session_id: Unique session identifier
            query: User query being processed
            metadata: Optional session metadata
            
        Returns:
            New ReasoningSession instance
        """
        with self._lock:
            session = ReasoningSession(
                session_id=session_id,
                query=query,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            
            self._sessions[session_id] = session
            self.logger.info(f"Started reasoning session {session_id}")
            
            return session
    
    def end_session(self, session_id: str) -> Optional[ReasoningSession]:
        """
        End a reasoning session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Completed ReasoningSession if found, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.end_time = datetime.now()
                self.logger.info(f"Ended reasoning session {session_id} (duration: {session.get_duration():.2f}s)")
                
                # Persist session
                self._persist_session(session)
                
                return session
            
            return None
    
    def log_step(
        self,
        session_id: str,
        agent_name: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Log a reasoning step.
        
        Args:
            session_id: Session identifier
            agent_name: Name of the agent
            message: Log message
            level: Log level
            data: Optional structured data
            duration_ms: Step duration in milliseconds
            success: Whether the step was successful
            error: Error message if step failed
        """
        import uuid
        
        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            agent_name=agent_name,
            timestamp=datetime.now(),
            level=level,
            message=message,
            data=data,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
        
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.add_step(step)
            else:
                self.logger.warning(f"Session {session_id} not found for step logging")
        
        # Also log to standard logger
        log_message = f"[{agent_name}] {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        if error:
            log_message += f" | Error: {error}"
        
        if duration_ms:
            log_message += f" | Duration: {duration_ms:.2f}ms"
        
        # Use appropriate log level
        agent_logger = logging.getLogger("agent_reasoning")
        if level == LogLevel.DEBUG:
            agent_logger.debug(log_message)
        elif level == LogLevel.INFO:
            agent_logger.info(log_message)
        elif level == LogLevel.WARNING:
            agent_logger.warning(log_message)
        elif level == LogLevel.ERROR:
            agent_logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            agent_logger.critical(log_message)
    
    def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Get a reasoning session by ID."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return [
                session_id for session_id, session in self._sessions.items()
                if session.end_time is None
            ]
    
    def _persist_session(self, session: ReasoningSession) -> None:
        """Persist session to disk."""
        try:
            file_path = self.log_dir / f"session_{session.session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Persisted session {session.session_id}")
            
        except Exception as e:
            self.logger.exception(f"Failed to persist session {session.session_id}: {e}")
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export session data for analysis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary if found, None otherwise
        """
        session = self.get_session(session_id)
        if session:
            return session.to_dict()
        
        # Try to load from disk
        try:
            file_path = self.log_dir / f"session_{session_id}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.exception(f"Failed to export session {session_id}: {e}")
        
        return None
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a reasoning session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "query": session.query,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "duration_seconds": session.get_duration(),
            "total_steps": len(session.steps),
            "success_rate": session.get_success_rate(),
            "agents_involved": list(set(step.agent_name for step in session.steps)),
            "error_count": sum(1 for step in session.steps if not step.success)
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed sessions.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            sessions_to_remove = []
            
            for session_id, session in self._sessions.items():
                if session.end_time and session.end_time < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old sessions")
        return cleaned_count


# Global agent logger instance
_agent_logger: Optional[AgentLogger] = None


def get_agent_logger() -> AgentLogger:
    """
    Get or create the global agent logger instance.
    
    Returns:
        AgentLogger instance
    """
    global _agent_logger
    
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    
    return _agent_logger


def log_agent_step(
    session_id: str,
    agent_name: str,
    message: str,
    level: LogLevel = LogLevel.INFO,
    data: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """
    Convenience function to log an agent step.
    
    Args:
        session_id: Session identifier
        agent_name: Name of the agent
        message: Log message
        level: Log level
        data: Optional structured data
        duration_ms: Step duration in milliseconds
        success: Whether the step was successful
        error: Error message if step failed
    """
    logger = get_agent_logger()
    logger.log_step(
        session_id=session_id,
        agent_name=agent_name,
        message=message,
        level=level,
        data=data,
        duration_ms=duration_ms,
        success=success,
        error=error
    )
