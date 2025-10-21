import json
import uuid
import pytest

from agents.analysis_agent import AnalysisAgent
from core.context_manager import get_context_manager
from agents.base_agent import AgentInput


class DummyLLM:
    def __init__(self, response: str, available: bool = True):
        self._resp = response
        self._available = available

    def is_available(self):
        return self._available

    def invoke_with_prompts(self, system_prompt: str, human_prompt: str) -> str:
        return self._resp


@pytest.fixture(autouse=True)
def clear_context_dir(tmp_path, monkeypatch):
    # Use a fresh ContextManager for each test
    cm = get_context_manager()
    # Cleanup any existing contexts
    for s in cm.get_active_sessions():
        cm.cleanup_context(s)
    yield


def make_agent_input(retrieved_docs):
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    cm = get_context_manager()
    ctx = cm.create_context("test query", session_id, retrieved_documents=[d.get("metadata", {}) for d in retrieved_docs])
    return AgentInput(query="test query", context=ctx, retrieved_documents=retrieved_docs, session_id=session_id)


def test_analysis_agent_parses_llm_json(monkeypatch):
    # LLM returns well-formed JSON
    llm_response = json.dumps({
        "product_commentary": {
            "P1": {"alternatives": [{"explanation": "A", "likelihood": 0.8, "impact": 0.6}], "critical_metrics": {"stability": 0.9}}
        },
        "summary_insights": ["insight1"],
        "overall_confidence": 0.85
    })

    monkeypatch.setattr('rag_chatbot.llm_reasoner.LLMReasoner', lambda *a, **k: DummyLLM(llm_response))

    docs = [{"id": "d1", "page_content": "", "full_content": "", "metadata": {"ref_article": "P1", "avg_forecast": 100, "trend_pct": 5, "trend_label": "Uptrend", "data_points": 10}}]
    ai = make_agent_input(docs)

    agent = AnalysisAgent()
    out = pytest.MonkeyPatch().context()  # placeholder to ensure pytest doesn't complain about async

    # Run the agent (synchronously via execute wrapper)
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(agent.execute(ai))

    assert result.success
    # Ensure llm_chain and summary_insights persisted
    cm = get_context_manager()
    ctx = cm.get_context(ai.session_id)
    assert ctx is not None
    aa = ctx.get_agent_output('AnalysisAgent')
    assert aa is not None
    data = aa['output']['data'] if isinstance(aa, dict) and 'output' in aa else aa.get('data', {})
    assert 'llm_chain' in data or 'llm_insights' in data or 'summary_insights' in data


def test_analysis_agent_llm_raw_fallback(monkeypatch):
    # LLM returns non-JSON raw text
    llm_response = "I think product P1 shows growth because models agree. Chain: ..."
    monkeypatch.setattr('rag_chatbot.llm_reasoner.LLMReasoner', lambda *a, **k: DummyLLM(llm_response))

    docs = [{"id": "d1", "page_content": "", "full_content": "", "metadata": {"ref_article": "P1", "avg_forecast": 100, "trend_pct": 5, "trend_label": "Uptrend", "data_points": 10}}]
    ai = make_agent_input(docs)

    agent = AnalysisAgent()
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(agent.execute(ai))

    assert result.success
    cm = get_context_manager()
    ctx = cm.get_context(ai.session_id)
    aa = ctx.get_agent_output('AnalysisAgent')
    data = aa['output']['data'] if isinstance(aa, dict) and 'output' in aa else aa.get('data', {})
    # Raw response should be stored under audit
    assert 'audit' in data and ('llm_raw_insights' in data['audit'] or 'llm_raw_chain' in data['audit'])
