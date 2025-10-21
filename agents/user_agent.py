"""
User Agent - Handles user input/output and conversation context.

This agent serves as the interface between users and the multi-agent system,
managing conversation context and coordinating the overall reasoning process.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.context_manager import AgentContext, get_context_manager
from core.logger import LogLevel


class UserAgent(BaseAgent):
    """
    User interface agent for the multi-agent reasoning system.
    
    Responsibilities:
    - Process user queries and extract intent
    - Manage conversation context and history
    - Coordinate with other agents for analysis
    - Format and present results to users
    - Handle user feedback and follow-up questions
    """
    
    def __init__(self):
        """Initialize user agent."""
        super().__init__(
            name="UserAgent",
            description="Handles user input/output and conversation context"
        )
        self.context_manager = get_context_manager()
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """
        Process user query and coordinate multi-agent analysis.
        
        Args:
            agent_input: Input containing user query and context
            
        Returns:
            AgentOutput with user interface response
        """
        reasoning_steps = []
        session_id = agent_input.session_id
        query = agent_input.query
        
        try:
            # Step 1: Extract user intent
            await self._log_reasoning_step(session_id, "Extracting user intent from query")
            intent = await self._extract_user_intent(query)
            reasoning_steps.append(f"Extracted intent: {intent['type']}")
            
            # Step 2: Analyze query complexity
            await self._log_reasoning_step(session_id, "Analyzing query complexity")
            complexity = await self._analyze_query_complexity(query, agent_input.retrieved_documents)
            reasoning_steps.append(f"Query complexity: {complexity['level']}")
            
            # Step 3: Determine required agents
            await self._log_reasoning_step(session_id, "Determining required agents for analysis")
            required_agents = await self._determine_required_agents(intent, complexity)
            reasoning_steps.append(f"Required agents: {', '.join(required_agents)}")
            
            # Step 4: Prepare analysis context
            await self._log_reasoning_step(session_id, "Preparing analysis context")
            analysis_context = await self._prepare_analysis_context(
                query, intent, complexity, agent_input.retrieved_documents
            )
            reasoning_steps.append("Analysis context prepared")
            
            # Step 5: Generate user-friendly response
            await self._log_reasoning_step(session_id, "Generating user response")
            user_response = await self._generate_user_response(
                query, intent, complexity, analysis_context
            )
            reasoning_steps.append("User response generated")
            
            # Calculate confidence based on intent clarity and data availability
            confidence = self._calculate_user_confidence(intent, complexity, analysis_context)
            
            return await self._create_output(
                success=True,
                data={
                    "user_response": user_response,
                    "intent": intent,
                    "complexity": complexity,
                    "required_agents": required_agents,
                    "analysis_context": analysis_context,
                    "conversation_ready": True
                },
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                execution_time_ms=0.0  # Will be set by base class
            )
            
        except Exception as e:
            await self._log_reasoning_step(
                session_id,
                f"Error processing user query: {str(e)}",
                success=False,
                error=str(e)
            )
            
            return await self._create_output(
                success=False,
                data={"user_response": "I apologize, but I encountered an error processing your request."},
                reasoning_steps=[f"Error: {str(e)}"],
                confidence=0.0,
                execution_time_ms=0.0,
                error=str(e)
            )
    
    async def _extract_user_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract user intent from query.
        
        Args:
            query: User query
            
        Returns:
            Intent analysis dictionary
        """
        query_lower = query.lower()
        
        # Intent patterns
        if any(word in query_lower for word in ["stable", "stability", "volatile", "least volatile"]):
            return {
                "type": "stability_analysis",
                "focus": "product_stability",
                "keywords": ["stability", "volatility"],
                "priority": "high"
            }
        elif any(word in query_lower for word in ["best", "worst", "top", "performance", "highest", "lowest"]):
            return {
                "type": "performance_analysis",
                "focus": "product_performance",
                "keywords": ["performance", "ranking"],
                "priority": "high"
            }
        elif any(word in query_lower for word in ["trend", "uptrend", "downtrend", "growth", "decline"]):
            return {
                "type": "trend_analysis",
                "focus": "trend_direction",
                "keywords": ["trend", "growth"],
                "priority": "medium"
            }
        elif any(word in query_lower for word in ["recommend", "advice", "should", "suggest"]):
            return {
                "type": "advisory_request",
                "focus": "business_recommendations",
                "keywords": ["recommendation", "advice"],
                "priority": "high"
            }
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return {
                "type": "comparative_analysis",
                "focus": "product_comparison",
                "keywords": ["comparison", "versus"],
                "priority": "medium"
            }
        else:
            return {
                "type": "general_inquiry",
                "focus": "general_analysis",
                "keywords": [],
                "priority": "low"
            }
    
    async def _analyze_query_complexity(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze query complexity.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Complexity analysis dictionary
        """
        # Simple complexity metrics
        word_count = len(query.split())
        document_count = len(documents)
        
        # Determine complexity level
        if word_count > 20 or document_count > 10:
            level = "high"
        elif word_count > 10 or document_count > 5:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "word_count": word_count,
            "document_count": document_count,
            "requires_deep_analysis": level in ["high", "medium"],
            "estimated_processing_time": "high" if level == "high" else "medium" if level == "medium" else "low"
        }
    
    async def _determine_required_agents(
        self, 
        intent: Dict[str, Any], 
        complexity: Dict[str, Any]
    ) -> List[str]:
        """
        Determine which agents are required for the analysis.
        
        Args:
            intent: User intent analysis
            complexity: Query complexity analysis
            
        Returns:
            List of required agent names
        """
        required = ["RetrieverAgent", "AnalysisAgent", "ReasoningAgent"]
        
        # Add agents based on intent
        if intent["type"] == "advisory_request":
            required.append("AdvisorAgent")
        
        if intent["type"] in ["stability_analysis", "performance_analysis", "trend_analysis"]:
            required.append("AnalysisAgent")
        
        # Add validator for complex queries
        if complexity["level"] == "high":
            required.append("ValidatorAgent")
        
        return required
    
    async def _prepare_analysis_context(
        self,
        query: str,
        intent: Dict[str, Any],
        complexity: Dict[str, Any],
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare context for multi-agent analysis.
        
        Args:
            query: User query
            intent: Intent analysis
            complexity: Complexity analysis
            documents: Retrieved documents
            
        Returns:
            Analysis context dictionary
        """
        return {
            "query": query,
            "intent": intent,
            "complexity": complexity,
            "document_count": len(documents),
            "analysis_scope": "comprehensive" if complexity["level"] == "high" else "focused",
            "timestamp": datetime.now().isoformat(),
            "session_metadata": {
                "user_type": "business_analyst",
                "domain": "sales_forecasting",
                "urgency": intent.get("priority", "medium")
            }
        }
    
    async def _generate_user_response(
        self,
        query: str,
        intent: Dict[str, Any],
        complexity: Dict[str, Any],
        analysis_context: Dict[str, Any]
    ) -> str:
        """
        Generate user-friendly response.
        
        Args:
            query: User query
            intent: Intent analysis
            complexity: Complexity analysis
            analysis_context: Analysis context
            
        Returns:
            User response string
        """
        # Generate contextual response based on intent
        if intent["type"] == "stability_analysis":
            return (
                "I'll analyze the stability of products in your forecast data. "
                "This will help identify which products have the most consistent "
                "and reliable forecasts across different models."
            )
        elif intent["type"] == "performance_analysis":
            return (
                "I'll perform a comprehensive performance analysis of your products. "
                "This will include identifying top performers, underperformers, "
                "and opportunities for optimization."
            )
        elif intent["type"] == "advisory_request":
            return (
                "I'll provide business recommendations based on your forecast data. "
                "This will include actionable insights for inventory management, "
                "marketing strategies, and risk mitigation."
            )
        else:
            return (
                "I'll analyze your forecast data to provide insights and recommendations. "
                "Let me process the information and generate a comprehensive response."
            )
    
    def _calculate_user_confidence(
        self,
        intent: Dict[str, Any],
        complexity: Dict[str, Any],
        analysis_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence for user agent output.
        
        Args:
            intent: Intent analysis
            complexity: Complexity analysis
            analysis_context: Analysis context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from intent clarity
        intent_confidence = 0.9 if intent["type"] != "general_inquiry" else 0.6
        
        # Adjust for complexity
        complexity_factor = 1.0 if complexity["level"] == "low" else 0.8 if complexity["level"] == "medium" else 0.6
        
        # Adjust for data availability
        data_factor = 0.9 if analysis_context["document_count"] > 5 else 0.7
        
        return min(1.0, intent_confidence * complexity_factor * data_factor)
    
    def get_capabilities(self) -> List[str]:
        """Get list of user agent capabilities."""
        return [
            "User query processing and intent extraction",
            "Conversation context management",
            "Multi-agent coordination",
            "Response formatting and presentation",
            "Query complexity analysis",
            "User feedback handling"
        ]
    
    async def handle_follow_up(self, session_id: str, follow_up_query: str) -> Dict[str, Any]:
        """
        Handle follow-up questions from users.
        
        Args:
            session_id: Session identifier
            follow_up_query: Follow-up question
            
        Returns:
            Response to follow-up question
        """
        # Get existing context
        context = self.context_manager.get_context(session_id)
        if not context:
            return {"error": "No existing context found for follow-up"}
        
        # Process follow-up with existing context
        agent_input = AgentInput(
            query=follow_up_query,
            context=context,
            retrieved_documents=context.retrieved_documents,
            session_id=session_id
        )
        
        output = await self.execute(agent_input)
        return output.data
