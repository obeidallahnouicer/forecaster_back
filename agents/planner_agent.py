"""
Planner Agent - Interprets user query and orchestrates agent collaboration.

This agent serves as the central coordinator, analyzing user queries and
determining the optimal sequence of agents to achieve the desired analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.logger import LogLevel


class PlannerAgent(BaseAgent):
    """
    Central planning agent for multi-agent reasoning coordination.
    
    Responsibilities:
    - Parse user intent and determine analysis requirements
    - Create execution plans for agent collaboration
    - Manage agent dependencies and execution order
    - Monitor progress and handle failures
    - Optimize resource allocation and timing
    """
    
    def __init__(self):
        """Initialize planner agent."""
        super().__init__(
            name="PlannerAgent",
            description="Interprets user query and orchestrates agent collaboration"
        )
        
        # Define agent capabilities and dependencies
        self.agent_capabilities = {
            "RetrieverAgent": ["data_retrieval", "vector_search", "document_ranking"],
            "AnalysisAgent": ["statistical_analysis", "trend_detection", "anomaly_detection"],
            "ReasoningAgent": ["insight_synthesis", "pattern_recognition", "logical_reasoning"],
            "AdvisorAgent": ["business_recommendations", "risk_assessment", "strategy_planning"],
            "ValidatorAgent": ["data_validation", "consistency_checking", "quality_assurance"]
        }
        
        self.agent_dependencies = {
            "RetrieverAgent": [],
            "AnalysisAgent": ["RetrieverAgent"],
            "ReasoningAgent": ["AnalysisAgent"],
            "AdvisorAgent": ["ReasoningAgent"],
            "ValidatorAgent": ["AdvisorAgent", "ReasoningAgent"]
        }
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """
        Create execution plan for multi-agent analysis.
        
        Args:
            agent_input: Input containing user query and context
            
        Returns:
            AgentOutput with execution plan
        """
        reasoning_steps = []
        session_id = agent_input.session_id
        query = agent_input.query
        
        try:
            # Step 1: Analyze query requirements
            await self._log_reasoning_step(session_id, "Analyzing query requirements")
            requirements = await self._analyze_query_requirements(query, agent_input.retrieved_documents)
            reasoning_steps.append(f"Identified {len(requirements['required_capabilities'])} required capabilities")
            
            # Step 2: Select appropriate agents
            await self._log_reasoning_step(session_id, "Selecting appropriate agents")
            selected_agents = await self._select_agents(requirements)
            reasoning_steps.append(f"Selected {len(selected_agents)} agents: {', '.join(selected_agents)}")
            
            # Step 3: Create execution plan
            await self._log_reasoning_step(session_id, "Creating execution plan")
            execution_plan = await self._create_execution_plan(selected_agents, requirements)
            reasoning_steps.append(f"Created execution plan with {len(execution_plan['phases'])} phases")
            
            # Step 4: Optimize plan
            await self._log_reasoning_step(session_id, "Optimizing execution plan")
            optimized_plan = await self._optimize_plan(execution_plan, requirements)
            reasoning_steps.append("Execution plan optimized")
            
            # Step 5: Validate plan
            await self._log_reasoning_step(session_id, "Validating execution plan")
            validation_result = await self._validate_plan(optimized_plan)
            reasoning_steps.append(f"Plan validation: {'passed' if validation_result['valid'] else 'failed'}")
            
            # Calculate confidence based on plan completeness and validation
            confidence = self._calculate_plan_confidence(requirements, optimized_plan, validation_result)
            
            return await self._create_output(
                success=True,
                data={
                    "execution_plan": optimized_plan,
                    "requirements": requirements,
                    "selected_agents": selected_agents,
                    "validation_result": validation_result,
                    "estimated_duration": optimized_plan.get("estimated_duration", 0),
                    "resource_requirements": optimized_plan.get("resource_requirements", {})
                },
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                execution_time_ms=0.0
            )
            
        except Exception as e:
            await self._log_reasoning_step(
                session_id,
                f"Error creating execution plan: {str(e)}",
                success=False,
                error=str(e)
            )
            
            return await self._create_output(
                success=False,
                data={"execution_plan": None},
                reasoning_steps=[f"Error: {str(e)}"],
                confidence=0.0,
                execution_time_ms=0.0,
                error=str(e)
            )
    
    async def _analyze_query_requirements(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze query to determine required capabilities.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Requirements analysis dictionary
        """
        query_lower = query.lower()
        
        # Determine required capabilities based on query content
        required_capabilities = []
        
        # Data retrieval requirements
        if any(word in query_lower for word in ["find", "search", "retrieve", "get"]):
            required_capabilities.append("data_retrieval")
        
        # Analysis requirements
        if any(word in query_lower for word in ["analyze", "compare", "calculate", "compute"]):
            required_capabilities.append("statistical_analysis")
        
        # Trend analysis requirements
        if any(word in query_lower for word in ["trend", "pattern", "change", "over time"]):
            required_capabilities.append("trend_detection")
        
        # Reasoning requirements
        if any(word in query_lower for word in ["why", "how", "explain", "insight"]):
            required_capabilities.append("insight_synthesis")
        
        # Advisory requirements
        if any(word in query_lower for word in ["recommend", "advise", "suggest", "should"]):
            required_capabilities.append("business_recommendations")
        
        # Validation requirements
        if any(word in query_lower for word in ["verify", "validate", "check", "confirm"]):
            required_capabilities.append("data_validation")
        
        # Determine complexity and urgency
        complexity = "high" if len(query.split()) > 15 else "medium" if len(query.split()) > 8 else "low"
        urgency = "high" if any(word in query_lower for word in ["urgent", "asap", "immediately"]) else "medium"
        
        return {
            "required_capabilities": required_capabilities,
            "complexity": complexity,
            "urgency": urgency,
            "document_count": len(documents),
            "estimated_processing_time": self._estimate_processing_time(complexity, len(documents))
        }
    
    async def _select_agents(self, requirements: Dict[str, Any]) -> List[str]:
        """
        Select appropriate agents based on requirements.
        
        Args:
            requirements: Requirements analysis
            
        Returns:
            List of selected agent names
        """
        selected = []
        required_capabilities = requirements["required_capabilities"]
        
        # Always include RetrieverAgent for data access
        selected.append("RetrieverAgent")
        
        # Select agents based on required capabilities
        for agent, capabilities in self.agent_capabilities.items():
            if agent == "RetrieverAgent":
                continue  # Already added
            
            # Check if agent has any required capabilities
            if any(cap in required_capabilities for cap in capabilities):
                selected.append(agent)
        
        # Ensure we have at least AnalysisAgent for basic analysis
        if "AnalysisAgent" not in selected and any(cap in required_capabilities for cap in ["statistical_analysis", "trend_detection"]):
            selected.append("AnalysisAgent")
        
        return selected
    
    async def _create_execution_plan(
        self, 
        selected_agents: List[str], 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create execution plan for selected agents.
        
        Args:
            selected_agents: List of selected agents
            requirements: Requirements analysis
            
        Returns:
            Execution plan dictionary
        """
        # Create phases based on dependencies
        phases = []
        remaining_agents = selected_agents.copy()
        
        while remaining_agents:
            phase_agents = []
            
            # Find agents that can run in this phase (no unmet dependencies)
            for agent in remaining_agents:
                dependencies = self.agent_dependencies.get(agent, [])
                if all(dep in [a for phase in phases for a in phase.get("agents", [])] for dep in dependencies):
                    phase_agents.append(agent)
            
            if not phase_agents:
                # If no agents can run, force the first remaining agent
                phase_agents = [remaining_agents[0]]
            
            phases.append({
                "phase_id": len(phases) + 1,
                "agents": phase_agents,
                "parallel_execution": len(phase_agents) > 1,
                "estimated_duration": self._estimate_phase_duration(phase_agents, requirements)
            })
            
            # Remove processed agents
            for agent in phase_agents:
                remaining_agents.remove(agent)
        
        return {
            "phases": phases,
            "total_phases": len(phases),
            "estimated_duration": sum(phase["estimated_duration"] for phase in phases),
            "parallel_execution_possible": any(phase["parallel_execution"] for phase in phases),
            "resource_requirements": self._calculate_resource_requirements(selected_agents, requirements)
        }
    
    async def _optimize_plan(
        self, 
        execution_plan: Dict[str, Any], 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize execution plan for efficiency.
        
        Args:
            execution_plan: Original execution plan
            requirements: Requirements analysis
            
        Returns:
            Optimized execution plan
        """
        optimized_plan = execution_plan.copy()
        
        # Optimize for parallel execution where possible
        for phase in optimized_plan["phases"]:
            if phase["parallel_execution"] and requirements["urgency"] == "high":
                phase["priority"] = "high"
                phase["timeout"] = 30  # seconds
            else:
                phase["priority"] = "normal"
                phase["timeout"] = 60  # seconds
        
        # Add retry logic for critical phases
        for phase in optimized_plan["phases"]:
            if "RetrieverAgent" in phase["agents"] or "AnalysisAgent" in phase["agents"]:
                phase["retry_count"] = 2
                phase["retry_delay"] = 5  # seconds
        
        return optimized_plan
    
    async def _validate_plan(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate execution plan for correctness.
        
        Args:
            execution_plan: Execution plan to validate
            
        Returns:
            Validation result dictionary
        """
        issues = []
        
        # Check if all phases have agents
        for i, phase in enumerate(execution_plan["phases"]):
            if not phase["agents"]:
                issues.append(f"Phase {i+1} has no agents")
        
        # Check for circular dependencies
        all_agents = [agent for phase in execution_plan["phases"] for agent in phase["agents"]]
        for agent in all_agents:
            dependencies = self.agent_dependencies.get(agent, [])
            for dep in dependencies:
                if dep not in all_agents:
                    issues.append(f"Agent {agent} depends on {dep} which is not in the plan")
        
        # Check resource requirements
        resource_reqs = execution_plan.get("resource_requirements", {})
        if resource_reqs.get("memory_mb", 0) > 1000:  # 1GB limit
            issues.append("Memory requirements exceed limit")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": [],
            "recommendations": []
        }
    
    def _estimate_processing_time(self, complexity: str, document_count: int) -> int:
        """Estimate processing time in seconds."""
        base_time = {"low": 5, "medium": 15, "high": 30}[complexity]
        document_factor = min(2.0, document_count / 10)
        return int(base_time * (1 + document_factor))
    
    def _estimate_phase_duration(self, agents: List[str], requirements: Dict[str, Any]) -> int:
        """Estimate duration for a phase in seconds."""
        base_duration = {"low": 3, "medium": 8, "high": 15}[requirements["complexity"]]
        agent_factor = len(agents) * 2
        return base_duration + agent_factor
    
    def _calculate_resource_requirements(
        self, 
        agents: List[str], 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate resource requirements for the plan."""
        memory_mb = 100  # Base memory
        cpu_cores = 1  # Base CPU
        
        # Add requirements for each agent
        for agent in agents:
            if agent == "RetrieverAgent":
                memory_mb += 200
            elif agent == "AnalysisAgent":
                memory_mb += 300
                cpu_cores += 1
            elif agent == "ReasoningAgent":
                memory_mb += 150
            elif agent == "AdvisorAgent":
                memory_mb += 100
            elif agent == "ValidatorAgent":
                memory_mb += 50
        
        return {
            "memory_mb": memory_mb,
            "cpu_cores": cpu_cores,
            "disk_space_mb": 50,
            "network_bandwidth": "low"
        }
    
    def _calculate_plan_confidence(
        self,
        requirements: Dict[str, Any],
        execution_plan: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the execution plan."""
        if not validation_result["valid"]:
            return 0.0
        
        # Base confidence from plan completeness
        required_caps = len(requirements["required_capabilities"])
        covered_caps = sum(
            1 for cap in requirements["required_capabilities"]
            for phase in execution_plan["phases"]
            for agent in phase["agents"]
            if cap in self.agent_capabilities.get(agent, [])
        )
        
        completeness = covered_caps / required_caps if required_caps > 0 else 1.0
        
        # Adjust for complexity
        complexity_factor = {"low": 1.0, "medium": 0.9, "high": 0.8}[requirements["complexity"]]
        
        # Adjust for parallel execution capability
        parallel_factor = 1.1 if execution_plan["parallel_execution_possible"] else 1.0
        
        return min(1.0, completeness * complexity_factor * parallel_factor)
    
    def get_capabilities(self) -> List[str]:
        """Get list of planner agent capabilities."""
        return [
            "Query requirement analysis",
            "Agent selection and coordination",
            "Execution plan creation",
            "Dependency management",
            "Resource optimization",
            "Plan validation and monitoring"
        ]
