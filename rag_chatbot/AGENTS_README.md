Multi-Agent Reasoning System
=============================

Overview
--------

This is a business-intelligent multi-agent reasoning layer built on top of the existing RAG chatbot. It orchestrates four specialized agents that work together to analyze product data and provide actionable business insights.

Architecture
------------

```
User Query
    ↓
Retrieve Documents (get_answer)
    ↓
┌─────────────────────────────────────────┐
│   Multi-Agent Orchestrator              │
├─────────────────────────────────────────┤
│ DataAgent                               │
│ ├─ Extract product codes & metrics    │
│ ├─ Validate against dataset            │
│ └─ Assess data quality                │
│                                         │
│ AnalysisAgent                          │
│ ├─ Compute stability (trend_pct)     │
│ ├─ Categorize by trend direction     │
│ ├─ Identify best/worst performers    │
│ ├─ Assess reliability (data_points)  │
│ └─ Detect anomalies                   │
│                                         │
│ BusinessAgent                          │
│ ├─ Generate key insights              │
│ ├─ Formulate recommendations          │
│ ├─ Assess risks                       │
│ ├─ Identify opportunities             │
│ └─ Define action items                │
│                                         │
│ AnswerAgent                            │
│ ├─ Answer user query directly         │
│ ├─ Format insights                    │
│ ├─ Include recommendations            │
│ └─ Compose human-readable answer      │
└─────────────────────────────────────────┘
    ↓
Final Answer with:
├─ Business insights
├─ Data-backed recommendations
├─ Risk assessment
└─ Reasoning trail (for audit)
```

Core Agents
-----------

### 1. DataAgent
**Responsibility:** Extract and validate structured data

**Inputs:**
- User query
- Retrieved documents from vectorstore

**Process:**
1. Extract product codes and metrics from document metadata
2. Validate each product exists in the dataset
3. Enrich with full product information (forecast, trend, etc.)
4. Extract products mentioned in query text
5. Assess overall data quality

**Output Structure:**
```python
{
    "products": {
        "PRODUCT_CODE": {
            "mentions": int,
            "sources": [list of doc IDs],
            "avg_forecast": float,
            "trend_pct": float,
            "trend_label": str,
            "data_points": int,
            "valid": bool,
        },
        ...
    },
    "key_metrics": {
        "total_products_mentioned": int,
        "total_valid_products": int,
        "total_invalid_products": int,
        "metrics_found": [list of metric names],
        "data_quality": "high" | "medium" | "low",
    },
    "raw_extracts": {
        "product_codes": [list],
        "query_products": [list],
    }
}
```

**Key Features:**
- Strict validation against dataset
- Automatic product enrichment
- Multi-source extraction (metadata + text)
- Data quality scoring

### 2. AnalysisAgent
**Responsibility:** Perform trend and performance analysis

**Inputs:**
- DataAgent output (validated products)
- Retrieved documents

**Process:**
1. Compute stability metrics (absolute trend_pct)
2. Categorize by trend direction (Uptrend/Downtrend/Stable)
3. Identify best and worst performers by forecast
4. Assess data reliability based on data_points count
5. Detect anomalies (e.g., high forecast with strong downtrend)

**Output Structure:**
```python
{
    "stability_analysis": {
        "most_stable": {product_dict},
        "stability_ranking": [list of products sorted by trend]
    },
    "trend_analysis": {
        "uptrending": [list],
        "downtrending": [list],
        "stable": [list]
    },
    "performance_analysis": {
        "best_performing": {product_dict},
        "low_performing": {product_dict}
    },
    "reliability_assessment": {
        "high_confidence": [products with 5+ data_points],
        "medium_confidence": [products with 3-4 data_points],
        "low_confidence": [products with <3 data_points]
    },
    "anomalies": [
        {
            "type": str,
            "product": str,
            "description": str,
            ...
        }
    ]
}
```

**Key Features:**
- Stability ranking (closest to 0% trend)
- Trend categorization with sorting
- Reliability scoring based on data quantity
- Anomaly detection for data quality issues

### 3. BusinessAgent
**Responsibility:** Interpret data in business context

**Inputs:**
- AnalysisAgent output
- Retrieved documents

**Process:**
1. Generate key business insights
2. Formulate actionable recommendations
3. Assess business risks
4. Identify market opportunities
5. Define immediate action items
6. Compute overall portfolio health

**Output Structure:**
```python
{
    "key_insights": [
        "Insight 1: ...",
        "Insight 2: ...",
        ...
    ],
    "recommendations": [
        {
            "title": str,
            "priority": "high" | "medium" | "low",
            "description": str,
            "rationale": str
        },
        ...
    ],
    "risks": [
        {
            "type": str,
            "severity": "high" | "medium" | "low",
            "description": str,
            "products": [list],
            ...
        },
        ...
    ],
    "opportunities": [
        {
            "type": str,
            "strength": "high" | "medium" | "low",
            "description": str,
            "potential": str,
        },
        ...
    ],
    "actions_required": [
        {
            "action": str,
            "priority": str,
            "owner": str,
            "description": str,
            "target_date": str,
        },
        ...
    ],
    "business_impact": {
        "portfolio_health": "positive" | "neutral" | "negative",
        "risk_level": "high" | "medium" | "low",
        "growth_momentum": str,
        "forecast_confidence": str,
    }
}
```

**Key Features:**
- Business-focused language (not technical jargon)
- Risk assessment with severity levels
- Opportunity identification
- Actionable recommendations with owners and deadlines
- Portfolio-level health assessment

### 4. AnswerAgent
**Responsibility:** Compose final human-readable answer

**Inputs:**
- All previous agent outputs
- Original user query

**Process:**
1. Generate direct answer to user query
2. Format insights section
3. Format recommendations section
4. Add data quality assessment
5. Compose final answer integrating all sections

**Output Structure:**
```python
{
    "answer": str,  # Final formatted answer
    "sections": int,  # Number of sections included
    "includes_insights": bool,
    "includes_recommendations": bool,
}
```

**Key Features:**
- Query-aware answer generation
- Multiple formatting styles (headers, bullets, metrics)
- Data quality transparency
- Balanced detail level

Orchestration Flow
------------------

```python
from rag_chatbot import agents, dataset_analyzer, retriever

# 1. Get analyzer
analyzer = dataset_analyzer.get_analyzer()

# 2. Retrieve documents
docs = retriever.get_retriever(k=5).get_relevant_documents(query)

# 3. Orchestrate agents (async)
orchestrator = agents.MultiAgentOrchestrator(analyzer)
result = await orchestrator.orchestrate(query, docs)

# 4. Access result
print(result.answer)
print(f"Validation: {result.validation}")
print(f"Agents used: {result.agents_used}")
print(f"Reasoning trail: {result.reasoning_trail}")
```

Integration with Chatbot
------------------------

### Using Multi-Agent Chat

```python
from rag_chatbot import chatbot

# Simple call with multi-agent reasoning
response = chatbot.chat_with_agents(
    message="Which products have the highest avg_forecast?",
    thread_id="user_123"
)

# Response structure
{
    "answer": "...",
    "source": "multi_agent_rag",
    "agents_used": ["DataAgent", "AnalysisAgent", "BusinessAgent", "AnswerAgent"],
    "validation": "passed",
    "thread_id": "user_123",
    "source_documents": [...],
    "metadata": {
        "retrieval_time_s": 0.15,
        "agent_orchestration_time_s": 2.3,
        "total_time_s": 2.45,
        "sources": ["doc_1", "doc_2", "doc_3"],
    },
    "reasoning_trail": {
        "data_agent": {...},
        "analysis_agent": {...},
        "business_agent": {...},
        "answer_agent": {...},
    }
}
```

### Fallback to Simple Chat

```python
# If multi-agent reasoning fails, automatically falls back to simple RAG
response = chatbot.chat(message="...", thread_id="user_123")
```

Output Formats
--------------

### Multi-Agent Result

```python
@dataclass
class MultiAgentResult:
    answer: str                              # Final answer
    source: str                              # "multi_agent_rag", "fallback", "error"
    agents_used: List[str]                   # List of agents that executed
    validation: str                          # "passed", "partial", "fallback"
    source_documents: List[Dict[str, Any]]   # Retrieved documents
    metadata: Dict[str, Any]                 # Execution stats
    reasoning_trail: Dict[str, AgentOutput]  # All agent outputs (for debugging)
```

### Agent Output

```python
@dataclass
class AgentOutput:
    agent_name: str                  # "DataAgent", "AnalysisAgent", etc.
    success: bool                    # Whether agent executed successfully
    data: Dict[str, Any]             # Structured output from agent
    reasoning_steps: List[str]       # Audit trail of reasoning
    validation_passed: bool          # Whether validation checks passed
    error: Optional[str]             # Error message if failed
```

Debugging and Monitoring
------------------------

### Access Reasoning Trail

```python
# Get detailed reasoning from each agent
for agent_name, agent_output in result.reasoning_trail.items():
    print(f"{agent_name}: {agent_output.success}")
    for step in agent_output.reasoning_steps:
        print(f"  - {step}")
```

### Monitor Execution Time

```python
metadata = result.metadata
print(f"Retrieval: {metadata['retrieval_time_s']:.2f}s")
print(f"Agent orchestration: {metadata['agent_orchestration_time_s']:.2f}s")
print(f"Total: {metadata['total_time_s']:.2f}s")
```

### Validate Data Quality

```python
for agent_name, agent_output in result.reasoning_trail.items():
    if not agent_output.validation_passed:
        print(f"⚠️ {agent_name} validation failed")
    if agent_output.error:
        print(f"❌ {agent_name} error: {agent_output.error}")
```

### Test Orchestrator Directly

```python
# Run orchestrator with detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

result = await orchestrator.orchestrate(query, docs)
```

Extensibility
-------------

### Adding Custom Agents

To add a new agent:

```python
from rag_chatbot.agents import Agent, AgentInput, AgentOutput

class CustomAgent(Agent):
    """Custom business logic agent."""
    
    def __init__(self, analyzer):
        super().__init__("CustomAgent", analyzer)
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        reasoning_steps = []
        
        try:
            # Your custom logic
            reasoning_steps.append(
                self._log_reasoning_step("Custom processing step")
            )
            
            output_data = {...}  # Your output structure
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=True,
            )
        except Exception as e:
            self.logger.exception(f"Custom agent failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
```

### Modifying Prompts

All prompts are in `prompt_templates.py`. Customize system prompts and reasoning templates:

```python
from rag_chatbot import prompt_templates

# Modify prompts for your LLM
prompt_templates.RETRIEVAL_SYSTEM_PROMPT = "Your custom system prompt..."
prompt_templates.RETRIEVAL_PROMPT_TEMPLATE = "Your custom template..."
```

### Plugging in Different LLMs

The answer composition uses LLM-agnostic prompts. Plug in any LLM:

```python
from langchain_openai import ChatOpenAI  # or your LLM provider
from rag_chatbot import prompt_templates

llm = ChatOpenAI(model="gpt-4", temperature=0.0)
response = llm.invoke([
    HumanMessage(content=prompt_templates.compose_prompt_for_llm(...))
])
```

Performance Characteristics
---------------------------

### Typical Execution Times

| Component | Time |
|-----------|------|
| Document Retrieval (k=5) | 100-300ms |
| DataAgent | 50-200ms |
| AnalysisAgent | 100-300ms |
| BusinessAgent | 150-400ms |
| AnswerAgent | 50-150ms |
| **Total** | **1-2 seconds** |

### Optimization Tips

1. **Reduce retrieval k**: Fewer documents = faster analysis
   ```python
   result = await orchestrator.orchestrate(query, docs, timeout_seconds=10)
   ```

2. **Cache analyzer**: Reuse analyzer instance
   ```python
   analyzer = dataset_analyzer.get_analyzer()  # Cached globally
   ```

3. **Parallel agent execution**: (Future enhancement)
   - DataAgent can run independently
   - AnalysisAgent and BusinessAgent could run in parallel once DataAgent completes

Examples
--------

See `agents_examples.py` for comprehensive examples:

```bash
python -m rag_chatbot.agents_examples
```

Includes:
1. Direct orchestrator usage
2. Chatbot integration
3. Understanding agent outputs
4. Monitoring and debugging

Error Handling
--------------

### Agent Failures

If an agent fails:
1. Error is logged with full traceback
2. AgentOutput.success = False
3. AgentOutput.error contains error message
4. Orchestration continues to next agent (if possible)
5. Final answer may use fallback or partial results

### Data Validation Failures

If DataAgent finds invalid products:
1. Validation_passed = False
2. Products still included in output
3. Business/Answer agents handle gracefully
4. Final validation status = "partial"

### Timeout Handling

If orchestration exceeds timeout:
1. Returns MultiAgentResult with source = "error"
2. Answer = "Request processing timed out..."
3. Reasoning trail contains partially completed agents

Testing
-------

### Unit Tests

```python
# Test individual agents
async def test_data_agent():
    analyzer = dataset_analyzer.get_analyzer()
    agent = agents.DataAgent(analyzer)
    
    output = await agent.reason(AgentInput(
        query="test query",
        retrieved_docs=[{...}],
    ))
    
    assert output.success
    assert len(output.data["products"]) > 0
```

### Integration Tests

```python
# Test full orchestration
async def test_orchestration():
    result = await agents.orchestrate_multi_agent_reasoning(
        query="Which products have uptrend?",
        retrieved_docs=[{...}],
    )
    
    assert result.source == "multi_agent_rag"
    assert len(result.agents_used) == 4
    assert result.validation in ["passed", "partial"]
```

Contributing
------------

To contribute to the multi-agent system:

1. **Add new agent analysis**: Modify existing agents' reasoning methods
2. **Add new agent type**: Subclass `Agent` and implement `reason()`
3. **Improve prompts**: Update `prompt_templates.py`
4. **Add metrics**: Track execution time, validation rates, etc.
5. **Extend output**: Add new fields to agent outputs

API Reference
-------------

### Main Functions

```python
# Async orchestration (recommended)
result = await agents.orchestrate_multi_agent_reasoning(
    query: str,
    retrieved_docs: List[Dict],
    analyzer: Optional[DatasetAnalyzer] = None
) -> MultiAgentResult

# Sync chatbot integration
response = chatbot.chat_with_agents(
    message: str,
    thread_id: str
) -> Dict[str, Any]
```

### Classes

```python
Agent                          # Base class for all agents
DataAgent                      # Extraction and validation
AnalysisAgent                  # Trend and performance analysis
BusinessAgent                  # Business interpretation
AnswerAgent                    # Final answer composition
MultiAgentOrchestrator         # Orchestration coordinator
```

### Data Classes

```python
AgentInput                     # Input to agent.reason()
AgentOutput                    # Output from agent.reason()
MultiAgentResult              # Final orchestration result
```

Troubleshooting
---------------

### "Orchestration timed out"
- Reduce number of retrieved documents
- Increase timeout_seconds parameter
- Check dataset size (analyzer.get_product_count())

### "Multi-agent orchestration failed"
- Check logs: `logger = logging.getLogger("rag.agents")`
- Verify vectorstore is initialized
- Check DatasetAnalyzer has valid data

### "Products mentioned are invalid"
- Check product codes in query
- Verify vectorstore metadata includes ref_article
- Manually verify product exists with: `analyzer.verify_product_exists("CODE")`

### "Answer quality is poor"
- Check DataAgent is extracting products correctly
- Verify AnalysisAgent is finding correct trends
- Review BusinessAgent insights in reasoning_trail

Performance Tuning
------------------

### For Production:

1. **Caching**:
   ```python
   analyzer = dataset_analyzer.get_analyzer()  # Reuse instance
   orchestrator = agents.MultiAgentOrchestrator(analyzer)  # Reuse
   ```

2. **Batch Processing**:
   ```python
   for query in queries:
       result = await orchestrator.orchestrate(query, docs)
   ```

3. **Async Processing**:
   ```python
   results = await asyncio.gather(
       orchestrator.orchestrate(q1, docs1),
       orchestrator.orchestrate(q2, docs2),
       ...
   )
   ```

4. **Monitoring**:
   - Track execution time per agent
   - Monitor validation pass rates
   - Log reasoning trail for audit
   - Alert on timeout/failure conditions

License
-------

Same as parent project (forecaster_back)

Contact
-------

For questions or issues, refer to the main project README.md
