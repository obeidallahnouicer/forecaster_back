"""
Context Manager - Shared memory and reasoning state across agents.

This module manages the shared context that flows between agents during
multi-agent reasoning, ensuring data consistency and enabling complex
analytical workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from pathlib import Path

logger = logging.getLogger("core.context_manager")


@dataclass
class AgentContext:
    """Context data shared between agents during reasoning."""
    query: str
    session_id: str
    timestamp: datetime
    query_history: List[str] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_agent_output(self, agent_name: str, output: Dict[str, Any]) -> None:
        """Add output from an agent to the context."""
        self.agent_outputs[agent_name] = {
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name
        }
        self.reasoning_chain.append({
            "step": len(self.reasoning_chain) + 1,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "summary": self._summarize_output(output)
        })
    
    def _summarize_output(self, output: Dict[str, Any]) -> str:
        """Create a brief summary of agent output for the reasoning chain."""
        if "success" in output:
            status = "✓" if output["success"] else "✗"
            return f"{status} {output.get('agent_name', 'Unknown')}"
        
        # Try to extract key insights
        if "data" in output:
            data = output["data"]
            if "products" in data:
                count = len(data["products"]) if isinstance(data["products"], dict) else 0
                return f"Processed {count} products"
            elif "insights" in data:
                count = len(data["insights"]) if isinstance(data["insights"], list) else 0
                return f"Generated {count} insights"
            elif "recommendations" in data:
                count = len(data["recommendations"]) if isinstance(data["recommendations"], list) else 0
                return f"Created {count} recommendations"
        
        return "Completed processing"
    
    def get_agent_output(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a specific agent."""
        return self.agent_outputs.get(agent_name)

    def add_query(self, query: str) -> None:
        """Append a new user query to the context's history and update timestamp."""
        self.query_history.append(query)
        self.query = query
        self.timestamp = datetime.now()
    
    def get_reasoning_summary(self) -> str:
        """Get a human-readable summary of the reasoning chain."""
        if not self.reasoning_chain:
            return "No reasoning steps recorded"
        
        steps = []
        for step in self.reasoning_chain:
            steps.append(f"{step['step']}. {step['summary']}")
        
        return "\n".join(steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "retrieved_documents": self.retrieved_documents,
            "agent_outputs": self.agent_outputs,
            "reasoning_chain": self.reasoning_chain,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """Create context from dictionary."""
        context = cls(
            query=data["query"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            retrieved_documents=data.get("retrieved_documents", []),
            agent_outputs=data.get("agent_outputs", {}),
            reasoning_chain=data.get("reasoning_chain", []),
            metadata=data.get("metadata", {})
        )
        return context


class ContextManager:
    """
    Manages shared context across agents during multi-agent reasoning.
    
    Features:
    - Thread-safe context management
    - Persistent context storage
    - Context validation and cleanup
    - Performance monitoring
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        """
        Initialize context manager.
        
        Args:
            persistence_dir: Directory for persistent context storage
        """
        self.persistence_dir = persistence_dir or Path("cache/contexts")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory context storage
        self._contexts: Dict[str, AgentContext] = {}
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger("core.context_manager")
    
    def create_context(
        self, 
        query: str, 
        session_id: str,
        retrieved_documents: Optional[List[Dict[str, Any]]] = None
    ) -> AgentContext:
        """
        Create a new context for multi-agent reasoning.
        
        Args:
            query: User query
            session_id: Unique session identifier
            retrieved_documents: Retrieved documents from vector store
            
        Returns:
            New AgentContext instance
        """
        with self._lock:
            context = AgentContext(
                query=query,
                session_id=session_id,
                timestamp=datetime.now(),
                retrieved_documents=retrieved_documents or []
            )
            
            self._contexts[session_id] = context
            self.logger.info(f"Created context for session {session_id}")
            
            return context
    
    def get_context(self, session_id: str) -> Optional[AgentContext]:
        """
        Get context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AgentContext if found, None otherwise
        """
        with self._lock:
            return self._contexts.get(session_id)

    def get_or_create_context(self, session_id: str, query: str = "") -> AgentContext:
        """
        Return existing context for session_id or create a new one.

        If a persisted context exists on disk it will be loaded. Otherwise a
        new AgentContext is created with an empty query (or provided query).
        """
        with self._lock:
            ctx = self._contexts.get(session_id)
            if ctx:
                # If a new query is provided, append it to the context history
                if query:
                    try:
                        ctx.add_query(query)
                    except Exception:
                        pass
                return ctx

            # Try to load persisted context from disk
            loaded = self.load_context(session_id)
            if loaded:
                return loaded

            # Create a fresh context
            new_ctx = AgentContext(query=query, session_id=session_id, timestamp=datetime.now())
            self._contexts[session_id] = new_ctx
            self.logger.info(f"Created new context for session {session_id} via get_or_create_context")
            return new_ctx
    
    def update_context(
        self, 
        session_id: str, 
        agent_name: str, 
        output: Dict[str, Any]
    ) -> bool:
        """
        Update context with agent output.
        
        Args:
            session_id: Session identifier
            agent_name: Name of the agent
            output: Agent output data
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            context = self._contexts.get(session_id)
            if not context:
                self.logger.warning(f"Context not found for session {session_id}")
                return False
            
            context.add_agent_output(agent_name, output)
            self.logger.debug(f"Updated context for session {session_id} with {agent_name}")
            return True
    
    def persist_context(self, session_id: str) -> bool:
        """
        Persist context to disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            context = self._contexts.get(session_id)
            if not context:
                return False
            
            try:
                file_path = self.persistence_dir / f"{session_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(context.to_dict(), f, indent=2, ensure_ascii=False)
                
                self.logger.debug(f"Persisted context for session {session_id}")
                return True
                
            except Exception as e:
                self.logger.exception(f"Failed to persist context {session_id}: {e}")
                return False
    
    def load_context(self, session_id: str) -> Optional[AgentContext]:
        """
        Load context from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AgentContext if found, None otherwise
        """
        try:
            file_path = self.persistence_dir / f"{session_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            context = AgentContext.from_dict(data)
            
            with self._lock:
                self._contexts[session_id] = context
            
            self.logger.debug(f"Loaded context for session {session_id}")
            return context
            
        except Exception as e:
            self.logger.exception(f"Failed to load context {session_id}: {e}")
            return None
    
    def cleanup_context(self, session_id: str) -> bool:
        """
        Clean up context from memory and disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Remove from memory
            if session_id in self._contexts:
                del self._contexts[session_id]
            
            # Remove from disk
            try:
                file_path = self.persistence_dir / f"{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
                
                self.logger.debug(f"Cleaned up context for session {session_id}")
                return True
                
            except Exception as e:
                self.logger.exception(f"Failed to cleanup context {session_id}: {e}")
                return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self._contexts.keys())
    
    def get_context_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context summary dictionary
        """
        context = self.get_context(session_id)
        if not context:
            return None
        
        return {
            "session_id": session_id,
            "query": context.query,
            "timestamp": context.timestamp.isoformat(),
            "documents_count": len(context.retrieved_documents),
            "agents_executed": len(context.agent_outputs),
            "reasoning_steps": len(context.reasoning_chain),
            "reasoning_summary": context.get_reasoning_summary()
        }


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """
    Get or create the global context manager instance.
    
    Returns:
        ContextManager instance
    """
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextManager()
    
    return _context_manager


