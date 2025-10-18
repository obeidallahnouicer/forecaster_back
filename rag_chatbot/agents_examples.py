"""
Examples and tests for the multi-agent reasoning system.

This module demonstrates:
1. How to use the multi-agent orchestrator directly
2. How to integrate with the chatbot
3. Expected output formats
4. Debugging and monitoring reasoning trails
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_direct_orchestrator_usage():
    """
    Example 1: Using the orchestrator directly.
    
    This demonstrates:
    - Creating an orchestrator
    - Running multi-agent reasoning on a query
    - Inspecting the reasoning trail
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Direct Orchestrator Usage")
    print("="*80)
    
    from rag_chatbot import agents, dataset_analyzer, retriever
    
    # Initialize analyzer
    analyzer = dataset_analyzer.get_analyzer()
    print(f"Loaded analyzer with {analyzer.get_product_count()} products")
    
    # Simulate retrieved documents
    query = "Which products have the best uptrend?"
    
    # Get real retrieved docs
    try:
        retr = retriever.get_retriever(k=5)
        docs = retr.get_relevant_documents(query)
        
        retrieved_docs = [
            {
                "id": doc.metadata.get("source_id", f"doc_{i}"),
                "content": doc.page_content[:300],
                "page_content": doc.page_content,
                "full_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for i, doc in enumerate(docs)
        ]
        print(f"Retrieved {len(retrieved_docs)} documents from vectorstore")
    except Exception as e:
        print(f"Warning: Could not retrieve docs from vectorstore: {e}")
        print("Using mock documents instead")
        
        # Create mock documents for demonstration
        retrieved_docs = [
            {
                "id": "doc_1",
                "content": "Product HZTI/10.31 shows strong uptrend...",
                "page_content": "Product HZTI/10.31 shows strong uptrend with trend_pct of 25.5%...",
                "full_content": "Product HZTI/10.31 shows strong uptrend with trend_pct of 25.5% and avg_forecast of 150.25",
                "metadata": {
                    "ref_article": "HZTI/10.31",
                    "avg_forecast": 150.25,
                    "trend_pct": 25.5,
                    "trend_label": "Uptrend",
                    "data_points": 8,
                }
            }
        ]
    
    # Create orchestrator
    orchestrator = agents.MultiAgentOrchestrator(analyzer)
    
    # Run orchestration
    print(f"\nOrchestrating agents for query: '{query}'")
    result = await orchestrator.orchestrate(query, retrieved_docs)
    
    print(f"\n--- RESULT ---")
    print(f"Source: {result.source}")
    print(f"Validation: {result.validation}")
    print(f"Agents used: {', '.join(result.agents_used)}")
    print(f"\n--- ANSWER ---")
    print(result.answer)
    
    print(f"\n--- REASONING TRAIL ---")
    for agent_name, agent_output in result.reasoning_trail.items():
        print(f"\n{agent_name}:")
        print(f"  Success: {agent_output.success}")
        print(f"  Validation: {agent_output.validation_passed}")
        print(f"  Steps: {len(agent_output.reasoning_steps)}")
        for step in agent_output.reasoning_steps[:3]:
            print(f"    - {step}")
        if len(agent_output.reasoning_steps) > 3:
            print(f"    ... and {len(agent_output.reasoning_steps) - 3} more steps")
    
    print(f"\n--- METADATA ---")
    print(json.dumps(result.metadata, indent=2))
    
    return result


async def example_chatbot_integration():
    """
    Example 2: Integrating multi-agent reasoning with chatbot.
    
    This demonstrates:
    - Using chat_with_agents() from chatbot module
    - Conversational flow with multiple messages
    - Accessing reasoning trail from chatbot interface
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Chatbot Integration with Multi-Agent Reasoning")
    print("="*80)
    
    from rag_chatbot import chatbot
    
    thread_id = "example_thread_001"
    
    # Message 1: Initial query
    message1 = "What are the most stable products in our portfolio?"
    print(f"\nUser: {message1}")
    
    response1 = chatbot.chat_with_agents(message1, thread_id)
    
    print(f"\nChatbot (source={response1['source']}):")
    print(response1["answer"])
    print(f"Validation: {response1['validation']}")
    print(f"Agents used: {response1.get('agents_used', [])}")
    
    # Message 2: Follow-up query
    message2 = "Which products are showing strong uptrend?"
    print(f"\nUser: {message2}")
    
    response2 = chatbot.chat_with_agents(message2, thread_id)
    
    print(f"\nChatbot (source={response2['source']}):")
    print(response2["answer"])
    print(f"Validation: {response2['validation']}")
    
    # Inspect conversation memory
    memory = chatbot.get_memory(thread_id)
    print(f"\n--- CONVERSATION MEMORY ---")
    print(f"Messages in memory: {len(memory.chat_memory.messages)}")
    for i, msg in enumerate(memory.chat_memory.messages):
        msg_type = "User" if msg.type == "human" else "Assistant"
        print(f"{i+1}. [{msg_type}] {msg.content[:80]}...")


def example_agent_outputs():
    """
    Example 3: Understanding agent output structures.
    
    This demonstrates:
    - The structure of each agent's output
    - How to extract and interpret agent data
    - Expected fields in each agent's output
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Agent Output Structures")
    print("="*80)
    
    # DataAgent output structure
    print("\n--- DATA AGENT OUTPUT ---")
    data_agent_example = {
        "products": {
            "HZTI/10.31": {
                "mentions": 2,
                "sources": ["doc_1", "doc_2"],
                "avg_forecast": 150.25,
                "trend_pct": 25.5,
                "trend_label": "Uptrend",
                "data_points": 8,
                "valid": True,
            },
            "RBKTFC": {
                "mentions": 1,
                "sources": ["doc_3"],
                "avg_forecast": 85.0,
                "trend_pct": -5.2,
                "trend_label": "Downtrend",
                "data_points": 4,
                "valid": True,
            }
        },
        "key_metrics": {
            "total_products_mentioned": 2,
            "total_valid_products": 2,
            "total_invalid_products": 0,
            "metrics_found": ["avg_forecast", "trend_pct", "trend_label", "data_points"],
            "data_quality": "high",
        }
    }
    
    print("Extracted products:")
    for product, info in data_agent_example["products"].items():
        print(f"  {product}: {info['trend_label']} ({info['trend_pct']:+.1f}%)")
    
    # AnalysisAgent output structure
    print("\n--- ANALYSIS AGENT OUTPUT ---")
    analysis_agent_example = {
        "stability_analysis": {
            "most_stable": {
                "product": "PA0316",
                "trend_pct": 0.3,
                "avg_forecast": 95.0,
                "data_points": 7,
            },
            "stability_ranking": [
                {"product": "PA0316", "trend_pct": 0.3, "abs_trend": 0.3},
                {"product": "SH_RB_ANTIC_1L", "trend_pct": -1.2, "abs_trend": 1.2},
            ]
        },
        "trend_analysis": {
            "uptrending": [
                {"product": "HZTI/10.31", "trend_pct": 25.5, "avg_forecast": 150.25},
            ],
            "downtrending": [
                {"product": "RBKTFC", "trend_pct": -5.2, "avg_forecast": 85.0},
            ],
            "stable": [],
        },
        "performance_analysis": {
            "best_performing": {"product": "HZTI/10.31", "avg_forecast": 150.25},
            "low_performing": {"product": "RBKTFC", "avg_forecast": 85.0},
        },
        "reliability_assessment": {
            "high_confidence": ["PA0316", "HZTI/10.31"],
            "medium_confidence": ["RBKTFC"],
            "low_confidence": [],
        },
        "anomalies": []
    }
    
    print(f"Most stable product: {analysis_agent_example['stability_analysis']['most_stable']['product']}")
    print(f"Uptrending products: {len(analysis_agent_example['trend_analysis']['uptrending'])}")
    print(f"Downtrending products: {len(analysis_agent_example['trend_analysis']['downtrending'])}")
    
    # BusinessAgent output structure
    print("\n--- BUSINESS AGENT OUTPUT ---")
    business_agent_example = {
        "key_insights": [
            "Product HZTI/10.31 offers strong uptrend opportunity at 25.5% growth",
            "2 products are on downtrend; monitor for further deterioration",
            "Portfolio has good stability baseline with PA0316",
        ],
        "recommendations": [
            {
                "title": "Growth Investment",
                "priority": "high",
                "description": "Increase investment in uptrending products, particularly HZTI/10.31",
            }
        ],
        "risks": [
            {
                "type": "market_risk",
                "severity": "medium",
                "description": "1 products experiencing downtrend",
                "products": ["RBKTFC"],
            }
        ],
        "opportunities": [
            {
                "type": "market_expansion",
                "strength": "high",
                "description": "Product HZTI/10.31 showing strong uptrend",
            }
        ],
        "actions_required": [
            {
                "action": "Weekly Review",
                "priority": "high",
                "owner": "Product Manager",
                "description": "Review performance of downtrending products",
            }
        ],
        "business_impact": {
            "portfolio_health": "positive",
            "risk_level": "medium",
            "growth_momentum": "positive",
        }
    }
    
    print(f"Key insights: {len(business_agent_example['key_insights'])}")
    for insight in business_agent_example['key_insights']:
        print(f"  - {insight}")
    
    print(f"\nRecommendations: {len(business_agent_example['recommendations'])}")
    for rec in business_agent_example['recommendations']:
        print(f"  - [{rec['priority'].upper()}] {rec['title']}")
    
    print(f"\nPortfolio health: {business_agent_example['business_impact']['portfolio_health']}")
    
    # AnswerAgent output
    print("\n--- ANSWER AGENT OUTPUT ---")
    answer_agent_example = {
        "answer": (
            "**Key Insights:**\n"
            "1. Product HZTI/10.31 offers the most stable forecast (trend: 0.3%), making it a reliable baseline for planning\n"
            "2. 1 products are on uptrend, led by HZTI/10.31; consider allocating resources to capitalize on growth\n"
            "3. 1 products show downtrend; monitor for further deterioration\n\n"
            "**Recommendations:**\n"
            "• [HIGH] Growth Investment: Increase investment in uptrending products, particularly HZTI/10.31\n"
            "• [MEDIUM] Risk Mitigation: Review and optimize downtrending product portfolio\n\n"
            "**Data Quality:** HIGH\n"
            "Valid products: 2 | Data issues: 0"
        ),
        "sections": 3,
        "includes_insights": True,
        "includes_recommendations": True,
    }
    
    print("Generated answer:")
    print(answer_agent_example["answer"])


async def example_monitoring_and_debugging():
    """
    Example 4: Monitoring agent execution and debugging.
    
    This demonstrates:
    - Accessing reasoning trails
    - Monitoring execution time
    - Debugging agent failures
    - Extracting intermediate results
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Monitoring and Debugging")
    print("="*80)
    
    from rag_chatbot import agents, dataset_analyzer
    
    analyzer = dataset_analyzer.get_analyzer()
    
    # Create mock input
    query = "What are the best products?"
    retrieved_docs = [
        {
            "id": "doc_1",
            "page_content": "Mock product data",
            "full_content": "Mock product data",
            "metadata": {
                "ref_article": "PRODUCT_001",
                "avg_forecast": 100.0,
                "trend_pct": 5.0,
                "trend_label": "Uptrend",
                "data_points": 5,
            }
        }
    ]
    
    # Run orchestrator
    orchestrator = agents.MultiAgentOrchestrator(analyzer)
    result = await orchestrator.orchestrate(query, retrieved_docs, timeout_seconds=30)
    
    print("\n--- EXECUTION METRICS ---")
    print(f"Total execution time: {result.metadata.get('execution_time_s', 0):.2f}s")
    print(f"Agents executed: {len(result.agents_used)}")
    print(f"Source: {result.source}")
    print(f"Validation: {result.validation}")
    
    print("\n--- AGENT REASONING TRAILS ---")
    for agent_name, agent_output in result.reasoning_trail.items():
        print(f"\n{agent_name}:")
        print(f"  Status: {'SUCCESS' if agent_output.success else 'FAILED'}")
        print(f"  Validation: {'PASSED' if agent_output.validation_passed else 'FAILED'}")
        print(f"  Reasoning steps: {len(agent_output.reasoning_steps)}")
        
        # Show first 3 steps
        for i, step in enumerate(agent_output.reasoning_steps[:3], 1):
            print(f"    {i}. {step}")
        
        if len(agent_output.reasoning_steps) > 3:
            print(f"    ... and {len(agent_output.reasoning_steps) - 3} more steps")
        
        if agent_output.error:
            print(f"  Error: {agent_output.error}")
    
    print("\n--- FINAL ANSWER ---")
    print(result.answer)


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("MULTI-AGENT REASONING SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    # Synchronous examples
    print("\n[Example 3] Understanding Agent Output Structures:")
    example_agent_outputs()
    
    # Asynchronous examples
    print("\n[Example 1] Direct Orchestrator Usage:")
    try:
        asyncio.run(example_direct_orchestrator_usage())
    except Exception as e:
        print(f"Note: Example 1 requires vectorstore initialization: {e}")
    
    print("\n[Example 2] Chatbot Integration:")
    try:
        asyncio.run(example_chatbot_integration())
    except Exception as e:
        print(f"Note: Example 2 requires chatbot setup: {e}")
    
    print("\n[Example 4] Monitoring and Debugging:")
    try:
        asyncio.run(example_monitoring_and_debugging())
    except Exception as e:
        print(f"Note: Example 4 error: {e}")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_all_examples()
