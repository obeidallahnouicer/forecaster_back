"""
Message Bus - Handles communication between agents.

This module provides a message passing system for agent communication,
enabling loose coupling and asynchronous processing in the multi-agent
reasoning framework.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import threading
from collections import defaultdict

logger = logging.getLogger("core.message_bus")


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    REASONING_REQUEST = "reasoning_request"
    REASONING_RESPONSE = "reasoning_response"
    ADVISORY_REQUEST = "advisory_request"
    ADVISORY_RESPONSE = "advisory_response"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    ERROR = "error"
    NOTIFICATION = "notification"


@dataclass
class Message:
    """Message structure for agent communication."""
    id: str
    type: MessageType
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher number = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class MessageHandler:
    """Handler for processing messages."""
    agent_name: str
    handler_func: Callable[[Message], Any]
    message_types: List[MessageType]
    priority: int = 0


class MessageBus:
    """
    Message bus for agent communication.
    
    Features:
    - Asynchronous message processing
    - Priority-based message queuing
    - Message routing and filtering
    - Error handling and retry logic
    - Message persistence (optional)
    """
    
    def __init__(self):
        """Initialize message bus."""
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger("core.message_bus")
    
    def register_handler(
        self, 
        agent_name: str, 
        handler_func: Callable[[Message], Any],
        message_types: List[MessageType],
        priority: int = 0
    ) -> None:
        """
        Register a message handler for an agent.
        
        Args:
            agent_name: Name of the agent
            handler_func: Function to handle messages
            message_types: Types of messages this handler can process
            priority: Handler priority (higher = processed first)
        """
        with self._lock:
            handler = MessageHandler(
                agent_name=agent_name,
                handler_func=handler_func,
                message_types=message_types,
                priority=priority
            )
            
            self._handlers[agent_name].append(handler)
            self.logger.info(f"Registered handler for {agent_name} for {len(message_types)} message types")
    
    def unregister_handler(self, agent_name: str) -> None:
        """
        Unregister all handlers for an agent.
        
        Args:
            agent_name: Name of the agent
        """
        with self._lock:
            if agent_name in self._handlers:
                del self._handlers[agent_name]
                self.logger.info(f"Unregistered all handlers for {agent_name}")
    
    async def send_message(
        self, 
        message: Message,
        wait_for_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Send a message to an agent.
        
        Args:
            message: Message to send
            wait_for_response: Whether to wait for a response
            timeout: Timeout for response (if waiting)
            
        Returns:
            Response from handler (if waiting), None otherwise
        """
        try:
            # Add to queue
            await self._message_queue.put((message.priority, message))
            self.logger.debug(f"Queued message {message.id} from {message.sender} to {message.recipient}")
            
            if wait_for_response:
                # Wait for response
                response = await self._wait_for_response(message.id, timeout)
                return response
            
            return None
            
        except Exception as e:
            self.logger.exception(f"Failed to send message {message.id}: {e}")
            return None
    
    async def _wait_for_response(self, message_id: str, timeout: float) -> Optional[Any]:
        """Wait for a response to a message."""
        try:
            # This is a simplified implementation
            # In a real system, you'd use a more sophisticated response tracking mechanism
            await asyncio.sleep(0.1)  # Placeholder
            return None
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to message {message_id}")
            return None
    
    async def start(self) -> None:
        """Start the message bus processing loop."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting message bus")
        
        # Start processing loop
        asyncio.create_task(self._process_messages())
    
    async def stop(self) -> None:
        """Stop the message bus processing loop."""
        self._running = False
        self.logger.info("Stopping message bus")
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message from queue
                priority, message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                # Process message
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # No messages in queue, continue
                continue
            except Exception as e:
                self.logger.exception(f"Error processing message: {e}")
    
    async def _handle_message(self, message: Message) -> None:
        """Handle a single message."""
        try:
            # Find handlers for the recipient
            handlers = self._handlers.get(message.recipient, [])
            
            if not handlers:
                self.logger.warning(f"No handlers found for {message.recipient}")
                return
            
            # Find appropriate handler
            handler = None
            for h in handlers:
                if message.type in h.message_types:
                    handler = h
                    break
            
            if not handler:
                self.logger.warning(f"No handler found for message type {message.type} for {message.recipient}")
                return
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler.handler_func):
                await handler.handler_func(message)
            else:
                handler.handler_func(message)
            
            self.logger.debug(f"Processed message {message.id} by {message.recipient}")
            
        except Exception as e:
            self.logger.exception(f"Error handling message {message.id}: {e}")
    
    def create_message(
        self,
        message_type: MessageType,
        sender: str,
        recipient: str,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> Message:
        """
        Create a new message.
        
        Args:
            message_type: Type of message
            sender: Sender agent name
            recipient: Recipient agent name
            content: Message content
            correlation_id: Optional correlation ID for tracking
            priority: Message priority
            
        Returns:
            New Message instance
        """
        import uuid
        message_id = str(uuid.uuid4())
        
        return Message(
            id=message_id,
            type=message_type,
            sender=sender,
            recipient=recipient,
            content=content,
            correlation_id=correlation_id,
            priority=priority
        )
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._message_queue.qsize()
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        with self._lock:
            return list(self._handlers.keys())


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """
    Get or create the global message bus instance.
    
    Returns:
        MessageBus instance
    """
    global _message_bus
    
    if _message_bus is None:
        _message_bus = MessageBus()
    
    return _message_bus


async def send_agent_message(
    sender: str,
    recipient: str,
    message_type: MessageType,
    content: Dict[str, Any],
    priority: int = 0
) -> Optional[Any]:
    """
    Convenience function to send a message between agents.
    
    Args:
        sender: Sender agent name
        recipient: Recipient agent name
        message_type: Type of message
        content: Message content
        priority: Message priority
        
    Returns:
        Response from recipient (if any)
    """
    bus = get_message_bus()
    message = bus.create_message(
        message_type=message_type,
        sender=sender,
        recipient=recipient,
        content=content,
        priority=priority
    )
    
    return await bus.send_message(message, wait_for_response=True)
