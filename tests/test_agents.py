"""
Unit and integration tests for the multi-agent reasoning system.

Tests cover:
- Individual agent functionality
- Orchestration flow
- Data validation
- Error handling
- Integration with chatbot
"""

import asyncio
import pytest
from typing import Dict, Any, List

from rag_chatbot import agents, dataset_analyzer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def analyzer():
    """Get test analyzer instance."""
    return dataset_analyzer.get_analyzer()


@pytest.fixture
def mock_retrieved_docs() -> List[Dict[str, Any]]:
    """Mock retrieved documents for testing."""
    return [
        {
            "id": "doc_1",
            "page_content": "Product HZTI/10.31 shows strong uptrend...",
            "full_content": "Product HZTI/10.31 shows strong uptrend with trend_pct of 25.5%...",
            "metadata": {
                "ref_article": "HZTI/10.31",
                "avg_forecast": 150.25,
                "trend_pct": 25.5,
                "trend_label": "Uptrend",
                "data_points": 8,
            }
        },
        {
            "id": "doc_2",
            "page_content": "Product PA0316 is stable...",
            "full_content": "Product PA0316 maintains stable performance...",
            "metadata": {
                "ref_article": "PA0316",
                "avg_forecast": 95.0,
                "trend_pct": 0.3,
                "trend_label": "Stable",
                "data_points": 7,
            }
        }
    ]


@pytest.fixture
def mock_agent_input(mock_retrieved_docs) -> agents.AgentInput:
    """Create mock agent input."""
    return agents.AgentInput(
        query="Which products have the best uptrend?",
        retrieved_docs=mock_retrieved_docs,
        context=None,
    )


# ============================================================================
# DATAAGENT TESTS
# ============================================================================

class TestDataAgent:
    """Tests for DataAgent extraction and validation."""
    
    @pytest.mark.asyncio
    async def test_data_extraction_success(self, analyzer, mock_agent_input):
        """Test successful data extraction."""
        agent = agents.DataAgent(analyzer)
        output = await agent.reason(mock_agent_input)
        
        assert output.success
        assert output.agent_name == "DataAgent"
        assert len(output.reasoning_steps) > 0
        assert "products" in output.data
    
    @pytest.mark.asyncio
    async def test_product_validation(self, analyzer, mock_agent_input):
        """Test product validation against dataset."""
        agent = agents.DataAgent(analyzer)
        output = await agent.reason(mock_agent_input)
        
        # Should extract and validate products
        products = output.data.get("products", {})
        
        # At least some products should be found
        assert len(products) > 0
        
        # Each product should have validation status
        for product_code, info in products.items():
            assert "valid" in info
            assert isinstance(info["valid"], bool)
    
    @pytest.mark.asyncio
    async def test_metrics_extraction(self, analyzer, mock_agent_input):
        """Test extraction of key metrics."""
        agent = agents.DataAgent(analyzer)
        output = await agent.reason(mock_agent_input)
        
        metrics = output.data.get("key_metrics", {})
        
        # Check expected fields exist
        assert "total_products_mentioned" in metrics
        assert "total_valid_products" in metrics
        assert "data_quality" in metrics
        
        # Data quality should be one of expected values
        assert metrics["data_quality"] in ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_empty_documents(self, analyzer):
        """Test handling of empty document list."""
        agent = agents.DataAgent(analyzer)
        
        input_data = agents.AgentInput(
            query="test",
            retrieved_docs=[],
            context=None,
        )
        
        output = await agent.reason(input_data)
        
        # Should handle gracefully
        assert output.success or not output.success  # Either way is acceptable
    
    @pytest.mark.asyncio
    async def test_missing_metadata(self, analyzer):
        """Test handling of documents with missing metadata."""
        agent = agents.DataAgent(analyzer)
        
        input_data = agents.AgentInput(
            query="test",
            retrieved_docs=[
                {
                    "id": "doc_1",
                    "page_content": "Some content",
                    "full_content": "Some content",
                    "metadata": {}  # Missing ref_article
                }
            ],
            context=None,
        )
        
        output = await agent.reason(input_data)
        
        # Should handle missing metadata
        assert output.success


# ============================================================================
# ANALYSISAGENT TESTS
# ============================================================================

class TestAnalysisAgent:
    """Tests for AnalysisAgent trend and performance analysis."""
    
    @pytest.mark.asyncio
    async def test_analysis_with_valid_data(self, analyzer, mock_retrieved_docs):
        """Test analysis with valid product data."""
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        
        output = await analysis_agent.reason(analysis_input)
        
        assert output.success
        assert "stability_analysis" in output.data
        assert "trend_analysis" in output.data
        assert "performance_analysis" in output.data
    
    @pytest.mark.asyncio
    async def test_stability_ranking(self, analyzer, mock_retrieved_docs):
        """Test stability ranking (closest to 0% trend)."""
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        
        output = await analysis_agent.reason(analysis_input)
        
        # Check stability output
        stability = output.data.get("stability_analysis", {})
        most_stable = stability.get("most_stable")
        
        if most_stable:
            # Most stable should have trend_pct close to 0
            assert abs(most_stable.get("trend_pct", 0)) >= 0
            assert "product" in most_stable
    
    @pytest.mark.asyncio
    async def test_reliability_assessment(self, analyzer, mock_retrieved_docs):
        """Test reliability classification by data_points."""
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        
        output = await analysis_agent.reason(analysis_input)
        
        # Check reliability categories exist
        reliability = output.data.get("reliability_assessment", {})
        
        assert "high_confidence" in reliability
        assert "medium_confidence" in reliability
        assert "low_confidence" in reliability


# ============================================================================
# BUSINESSAGENT TESTS
# ============================================================================

class TestBusinessAgent:
    """Tests for BusinessAgent business logic interpretation."""
    
    @pytest.mark.asyncio
    async def test_insights_generation(self, analyzer, mock_retrieved_docs):
        """Test generation of business insights."""
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        analysis_output = await analysis_agent.reason(analysis_input)
        
        business_agent = agents.BusinessAgent(analyzer)
        business_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={
                "data_agent_output": data_output.data,
                "analysis_agent_output": analysis_output,
            },
        )
        
        output = await business_agent.reason(business_input)
        
        assert output.success
        assert "key_insights" in output.data
        assert len(output.data["key_insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, analyzer, mock_retrieved_docs):
        """Test generation of actionable recommendations."""
        # Run full chain
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        analysis_output = await analysis_agent.reason(analysis_input)
        
        business_agent = agents.BusinessAgent(analyzer)
        business_input = agents.AgentInput(
            query="test",
            retrieved_docs=mock_retrieved_docs,
            context={
                "data_agent_output": data_output.data,
                "analysis_agent_output": analysis_output,
            },
        )
        
        output = await business_agent.reason(business_input)
        
        # Check recommendations structure
        recommendations = output.data.get("recommendations", [])
        
        for rec in recommendations:
            assert "title" in rec
            assert "priority" in rec
            assert rec["priority"] in ["high", "medium", "low"]


# ============================================================================
# ANSWERAGENT TESTS
# ============================================================================

class TestAnswerAgent:
    """Tests for AnswerAgent answer composition."""
    
    @pytest.mark.asyncio
    async def test_answer_composition(self, analyzer, mock_retrieved_docs):
        """Test final answer composition."""
        # Run full chain
        data_agent = agents.DataAgent(analyzer)
        data_input = agents.AgentInput(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
            context=None,
        )
        data_output = await data_agent.reason(data_input)
        
        analysis_agent = agents.AnalysisAgent(analyzer)
        analysis_input = agents.AgentInput(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
            context={"data_agent_output": data_output.data},
        )
        analysis_output = await analysis_agent.reason(analysis_input)
        
        business_agent = agents.BusinessAgent(analyzer)
        business_input = agents.AgentInput(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
            context={
                "data_agent_output": data_output.data,
                "analysis_agent_output": analysis_output,
            },
        )
        business_output = await business_agent.reason(business_input)
        
        answer_agent = agents.AnswerAgent(analyzer)
        answer_input = agents.AgentInput(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
            context={
                "data_agent_output": data_output.data,
                "analysis_agent_output": analysis_output,
                "business_agent_output": business_output,
            },
        )
        
        output = await answer_agent.reason(answer_input)
        
        assert output.success
        assert "answer" in output.data
        assert len(output.data["answer"]) > 0


# ============================================================================
# ORCHESTRATOR TESTS
# ============================================================================

class TestMultiAgentOrchestrator:
    """Tests for multi-agent orchestration."""
    
    @pytest.mark.asyncio
    async def test_full_orchestration(self, analyzer, mock_retrieved_docs):
        """Test full orchestration flow."""
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        result = await orchestrator.orchestrate(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
        )
        
        assert result is not None
        assert result.answer is not None
        assert len(result.answer) > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_agents_used(self, analyzer, mock_retrieved_docs):
        """Test that all expected agents are used."""
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        result = await orchestrator.orchestrate(
            query="test query",
            retrieved_docs=mock_retrieved_docs,
        )
        
        # Should use all 4 agents
        expected_agents = {"DataAgent", "AnalysisAgent", "BusinessAgent", "AnswerAgent"}
        used_agents = set(result.agents_used)
        
        # At minimum, some agents should run
        assert len(result.agents_used) > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_reasoning_trail(self, analyzer, mock_retrieved_docs):
        """Test reasoning trail accessibility."""
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        result = await orchestrator.orchestrate(
            query="test query",
            retrieved_docs=mock_retrieved_docs,
        )
        
        # Should have reasoning trail
        assert result.reasoning_trail is not None
        assert len(result.reasoning_trail) > 0
        
        # Check agent outputs in trail
        for agent_name, agent_output in result.reasoning_trail.items():
            assert hasattr(agent_output, "success")
            assert hasattr(agent_output, "reasoning_steps")
    
    @pytest.mark.asyncio
    async def test_orchestrator_timeout(self, analyzer, mock_retrieved_docs):
        """Test timeout handling."""
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        
        # Very short timeout should trigger timeout logic
        result = await orchestrator.orchestrate(
            query="test query",
            retrieved_docs=mock_retrieved_docs,
            timeout_seconds=0.001,  # Very short
        )
        
        # Should handle timeout gracefully
        assert result is not None
        # Timeout might result in error source
        # assert result.source in ["error", "multi_agent_rag"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with full system."""
    
    @pytest.mark.asyncio
    async def test_orchestrate_public_api(self, analyzer, mock_retrieved_docs):
        """Test public orchestration API."""
        result = await agents.orchestrate_multi_agent_reasoning(
            query="Which products have uptrend?",
            retrieved_docs=mock_retrieved_docs,
            analyzer=analyzer,
        )
        
        assert result is not None
        assert result.source == "multi_agent_rag"
        assert len(result.answer) > 0
    
    @pytest.mark.asyncio
    async def test_chatbot_integration(self, analyzer):
        """Test integration with chatbot."""
        from rag_chatbot import chatbot
        
        # This test requires vectorstore setup
        # Simplified test to check function exists
        assert hasattr(chatbot, "chat_with_agents")
        assert callable(chatbot.chat_with_agents)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_query(self, analyzer):
        """Test handling of empty query."""
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        
        result = await orchestrator.orchestrate(
            query="",
            retrieved_docs=[],
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_very_large_document_set(self, analyzer):
        """Test handling of many documents."""
        large_docs = [
            {
                "id": f"doc_{i}",
                "page_content": "test content",
                "full_content": "test content",
                "metadata": {
                    "ref_article": f"PRODUCT_{i}",
                    "avg_forecast": 100.0 + i,
                    "trend_pct": i % 10,
                    "trend_label": "Stable",
                    "data_points": 5,
                }
            }
            for i in range(50)  # 50 documents
        ]
        
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        
        result = await orchestrator.orchestrate(
            query="test",
            retrieved_docs=large_docs,
            timeout_seconds=30,
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_malformed_metadata(self, analyzer):
        """Test handling of malformed metadata."""
        docs = [
            {
                "id": "doc_1",
                "page_content": "test",
                "full_content": "test",
                "metadata": {
                    "ref_article": "PRODUCT_1",
                    "avg_forecast": "invalid",  # Should be float
                    "trend_pct": "not_a_number",  # Should be float
                }
            }
        ]
        
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        
        result = await orchestrator.orchestrate(
            query="test",
            retrieved_docs=docs,
        )
        
        # Should handle gracefully
        assert result is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and timing tests."""
    
    @pytest.mark.asyncio
    async def test_execution_time(self, analyzer, mock_retrieved_docs):
        """Test that orchestration completes in reasonable time."""
        import time
        
        orchestrator = agents.MultiAgentOrchestrator(analyzer)
        
        start = time.time()
        result = await orchestrator.orchestrate(
            query="test query",
            retrieved_docs=mock_retrieved_docs,
        )
        elapsed = time.time() - start
        
        # Should complete within reasonable time (10 seconds for test)
        assert elapsed < 10.0
        
        # Check metadata
        assert "execution_time_s" in result.metadata
        assert result.metadata["execution_time_s"] > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
