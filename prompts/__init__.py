"""
Prompt Templates Module

Centralized prompt templates for the SQL chatbot workflow.
All prompts are modular and reusable using LangChain's PromptTemplate.
"""

from .sql_generation import SQL_GENERATION_PROMPT, SQL_SYSTEM_PROMPT
from .insight_generation import INSIGHT_GENERATION_PROMPT, INSIGHT_SYSTEM_PROMPT
from .validation import VALIDATION_PROMPT

__all__ = [
    'SQL_GENERATION_PROMPT',
    'SQL_SYSTEM_PROMPT',
    'INSIGHT_GENERATION_PROMPT',
    'INSIGHT_SYSTEM_PROMPT',
    'VALIDATION_PROMPT'
]
