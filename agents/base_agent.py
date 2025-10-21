"""
Base Agent - Abstract base class for all reasoning agents.

This module provides the foundation for all agents in the multi-agent
analytical reasoning system, ensuring consistent interfaces and behavior.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import uuid

from core.context_manager import AgentContext
from core.logger import AgentLogger, LogLevel, log_agent_step
from core.message_bus import MessageBus, Message, MessageType


@dataclass
class AgentInput:
    """Input data for an agent."""
    query: str
    context: AgentContext
    retrieved_documents: List[Dict[str, Any]]
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentOutput:
    """Output from an agent."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    reasoning_steps: List[str]
    confidence: float  # 0.0 to 1.0
    execution_time_ms: float
    error: Optional[str] = None
    validation_passed: bool = True


class BaseAgent(ABC):
    """
    Abstract base class for all reasoning agents.
    
    Provides:
    - Consistent interface for all agents
    - Logging and monitoring capabilities
    - Error handling and validation
    - Performance tracking
    - Message bus integration
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agents.{name.lower()}")
        self.message_bus = MessageBus()
        self.agent_logger = AgentLogger()
        
        # Register message handlers
        self._register_handlers()
    
    @abstractmethod
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """
        Perform reasoning and return structured output.
        
        Args:
            agent_input: Input data for reasoning
            
        Returns:
            AgentOutput with results and reasoning trail
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def _register_handlers(self) -> None:
        """Register message handlers for this agent."""
        # Override in subclasses to register specific handlers
        pass
    
    async def _log_reasoning_step(
        self, 
        session_id: str, 
        step: str, 
        data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log a reasoning step."""
        log_agent_step(
            session_id=session_id,
            agent_name=self.name,
            message=step,
            level=LogLevel.INFO,
            data=data,
            success=success,
            error=error
        )
    
    async def _validate_input(self, agent_input: AgentInput) -> bool:
        """
        Validate agent input.
        
        Args:
            agent_input: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not agent_input.query:
            await self._log_reasoning_step(
                agent_input.session_id,
                "Input validation failed: empty query",
                success=False,
                error="Empty query provided"
            )
            return False
        
        if not agent_input.retrieved_documents:
            await self._log_reasoning_step(
                agent_input.session_id,
                "Input validation failed: no documents retrieved",
                success=False,
                error="No documents provided for analysis"
            )
            return False
        
        return True
    
    async def _create_output(
        self,
        success: bool,
        data: Dict[str, Any],
        reasoning_steps: List[str],
        confidence: float,
        execution_time_ms: float,
        error: Optional[str] = None
    ) -> AgentOutput:
        """
        Create agent output with validation.
        
        Args:
            success: Whether reasoning was successful
            data: Output data
            reasoning_steps: List of reasoning steps
            confidence: Confidence level (0.0 to 1.0)
            execution_time_ms: Execution time in milliseconds
            error: Error message if failed
            
        Returns:
            AgentOutput instance
        """
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return AgentOutput(
            agent_name=self.name,
            success=success,
            data=data,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            error=error,
            validation_passed=success and confidence > 0.0
        )
    
    async def _send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        session_id: str
    ) -> Optional[Any]:
        """
        Send a message to another agent.
        
        Args:
            recipient: Recipient agent name
            message_type: Type of message
            content: Message content
            session_id: Session identifier
            
        Returns:
            Response from recipient (if any)
        """
        try:
            message = self.message_bus.create_message(
                message_type=message_type,
                sender=self.name,
                recipient=recipient,
                content=content
            )
            
            response = await self.message_bus.send_message(message, wait_for_response=True)
            
            await self._log_reasoning_step(
                session_id,
                f"Sent message to {recipient}",
                data={"message_type": message_type.value, "recipient": recipient}
            )
            
            return response
            
        except Exception as e:
            await self._log_reasoning_step(
                session_id,
                f"Failed to send message to {recipient}",
                success=False,
                error=str(e)
            )
            return None
    
    def _extract_products_from_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extract product codes from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of unique product codes
        """
        products = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            if "ref_article" in metadata:
                products.add(metadata["ref_article"])
        
        return list(products)
    
    def _calculate_confidence(
        self, 
        data_quality: str, 
        data_points: int, 
        validation_passed: bool
    ) -> float:
        """
        Calculate confidence score based on data quality and validation.
        
        Args:
            data_quality: Quality assessment ("high", "medium", "low")
            data_points: Number of data points available
            validation_passed: Whether validation passed
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not validation_passed:
            return 0.0
        
        # Base confidence from data quality
        quality_scores = {"high": 0.9, "medium": 0.6, "low": 0.3}
        base_confidence = quality_scores.get(data_quality, 0.3)
        
        # Adjust based on data points
        if data_points >= 10:
            data_factor = 1.0
        elif data_points >= 5:
            data_factor = 0.8
        elif data_points >= 3:
            data_factor = 0.6
        else:
            data_factor = 0.4
        
        return min(1.0, base_confidence * data_factor)
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute agent reasoning with timing and error handling.
        
        Args:
            agent_input: Input data for reasoning
            
        Returns:
            AgentOutput with results
        """
        start_time = datetime.now()
        reasoning_steps = []
        
        try:
            # Log start
            await self._log_reasoning_step(
                agent_input.session_id,
                f"Starting {self.name} reasoning",
                data={"query": agent_input.query[:100] + "..." if len(agent_input.query) > 100 else agent_input.query}
            )
            
            # Validate input
            if not await self._validate_input(agent_input):
                return await self._create_output(
                    success=False,
                    data={},
                    reasoning_steps=["Input validation failed"],
                    confidence=0.0,
                    execution_time_ms=0.0,
                    error="Invalid input"
                )
            
            # Perform reasoning
            output = await self.reason(agent_input)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log completion
            await self._log_reasoning_step(
                agent_input.session_id,
                f"Completed {self.name} reasoning",
                data={
                    "success": output.success,
                    "confidence": output.confidence,
                    "execution_time_ms": execution_time
                }
            )
            
            return output
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            await self._log_reasoning_step(
                agent_input.session_id,
                f"Error in {self.name} reasoning",
                success=False,
                error=str(e)
            )
            
            return await self._create_output(
                success=False,
                data={},
                reasoning_steps=[f"Error: {str(e)}"],
                confidence=0.0,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Dictionary with agent status
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.get_capabilities(),
            "status": "active"
        }
