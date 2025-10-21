"""
Multi-Agent Analytical Reasoning System

This package contains specialized agents for business and financial analysis
of sales forecast data, providing structured reasoning and actionable insights.
"""

# Export commonly used agents
from .base_agent import BaseAgent, AgentInput, AgentOutput
from .user_agent import UserAgent

# Optional agents - import when available to avoid heavy dependency loading
def _import_optional_agents():
	try:
		from .analysis_agent import AnalysisAgent  # type: ignore
		from .reasoning_agent import ReasoningAgent  # type: ignore
		from .advisor_agent import AdvisorAgent  # type: ignore
		from .validator_agent import ValidatorAgent  # type: ignore
		return AnalysisAgent, ReasoningAgent, AdvisorAgent, ValidatorAgent
	except Exception:
		return None, None, None, None

AnalysisAgent, ReasoningAgent, AdvisorAgent, ValidatorAgent = _import_optional_agents()
