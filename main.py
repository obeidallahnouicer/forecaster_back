"""
Orchestrator entrypoint for the multi-agent analytical reasoning system.

Provides a simple FastAPI endpoint /analyze that runs the Planner -> Retriever ->
Analysis -> Reasoning -> Advisor -> Validator chain and returns structured JSON
with insights, recommendations, and a reasoning audit trail.
"""

import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from core.context_manager import get_context_manager
from core.logger import get_agent_logger, LogLevel
from core.chroma_client import get_chroma_client

from agents import (
    UserAgent,
    PlannerAgent,
    RetrieverAgent,
    AnalysisAgent,
    ReasoningAgent,
    AdvisorAgent,
    ValidatorAgent
)

app = FastAPI(title="Forecast Multi-Agent Analyzer")


class AnalyzeRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    session_id = str(uuid.uuid4())
    context_manager = get_context_manager()
    agent_logger = get_agent_logger()

    # Start a reasoning session
    agent_logger.start_session(session_id, req.query)

    try:
        # 1) Retriever
        retriever = RetrieverAgent()
        retriever_input = RetrieverAgent.__module__  # keep typing checker happy
        from .agents import base_agent as _ba

        from agents.base_agent import AgentInput as AI  # noqa: E402

        agent_input = AI(query=req.query, context=None, retrieved_documents=[], session_id=session_id)

        retriever_output = await retriever.execute(agent_input)
        context_manager.create_context(req.query, session_id, retrieved_documents=retriever_output.data.get("retrieved_documents", []))
        context = context_manager.get_context(session_id)
        context.add_agent_output("RetrieverAgent", retriever_output.__dict__)

        # 2) Analysis
        analysis = AnalysisAgent()
        analysis_input = AI(query=req.query, context=context, retrieved_documents=retriever_output.data.get("retrieved_documents", []), session_id=session_id)
        analysis_output = await analysis.execute(analysis_input)
        context.add_agent_output("AnalysisAgent", analysis_output.__dict__)

        # 3) Reasoning
        reasoning = ReasoningAgent()
        reasoning_input = AI(query=req.query, context=context, retrieved_documents=retriever_output.data.get("retrieved_documents", []), session_id=session_id)
        reasoning_output = await reasoning.execute(reasoning_input)
        context.add_agent_output("ReasoningAgent", reasoning_output.__dict__)

        # 4) Advisor
        advisor = AdvisorAgent()
        advisor_input = AI(query=req.query, context=context, retrieved_documents=retriever_output.data.get("retrieved_documents", []), session_id=session_id)
        advisor_output = await advisor.execute(advisor_input)
        context.add_agent_output("AdvisorAgent", advisor_output.__dict__)

        # 5) Validator
        validator = ValidatorAgent()
        validator_input = AI(query=req.query, context=context, retrieved_documents=retriever_output.data.get("retrieved_documents", []), session_id=session_id)
        validator_output = await validator.execute(validator_input)
        context.add_agent_output("ValidatorAgent", validator_output.__dict__)

        # End session
        agent_logger.end_session(session_id)

        result = {
            "session_id": session_id,
            "query": req.query,
            "retriever": retriever_output.data,
            "analysis": analysis_output.data,
            "reasoning": reasoning_output.data,
            "advisor": advisor_output.data,
            "validator": validator_output.data,
            "reasoning_chain": context.get_reasoning_summary(),
            "confidence": min(1.0, analysis_output.confidence * reasoning_output.confidence * advisor_output.confidence * validator_output.confidence)
        }

        return result

    except Exception as e:
        agent_logger.end_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))
