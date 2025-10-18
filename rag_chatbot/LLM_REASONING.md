# LLM Reasoning Layer - Deep Strategic Analysis

## Overview

The LLM reasoning layer applies advanced chain-of-thought reasoning to the multi-agent system output. It enhances the business analysis with:

- **Deep Strategic Thinking**: LLM reasons through patterns and implications
- **Risk Pattern Recognition**: Identifies hidden risks from data anomalies
- **Opportunity Discovery**: Finds strategic opportunities beyond surface analysis
- **Executive Insights**: Provides C-level recommendations
- **Chain-of-Thought**: Shows reasoning process explicitly

## Architecture

```
Multi-Agent System Output
    ↓
LLM Reasoner (Optional Enhancement)
├─ Analyzes agent data summaries
├─ Identifies patterns and correlations
├─ Reasons through business implications
├─ Detects hidden opportunities and risks
└─ Generates strategic recommendations
    ↓
Enhanced Final Answer with:
├─ Strategic analysis section
├─ Key insights from LLM reasoning
├─ Risk warnings with justification
├─ Growth opportunities identified
├─ Strategic recommendations
└─ Original agent analysis for reference
```

## How It Works

### 1. Deep Chain-of-Thought Reasoning

The LLM receives structured data from all agents and reasons through:

```
REASONING FRAMEWORK:
1. UNDERSTAND THE DATA
   - What products are we analyzing?
   - What are the key metrics?
   - How reliable is the data?

2. IDENTIFY PATTERNS
   - What trends are visible?
   - Which products move together?
   - What anomalies suggest hidden issues?

3. ANALYZE BUSINESS CONTEXT
   - What do these patterns mean?
   - Which decisions are critical?
   - Where are vulnerabilities?

4. THINK STRATEGICALLY
   - Why are these trends happening?
   - What could change them?
   - How do we capitalize?

5. RECOMMEND ACTIONS
   - What should we do immediately?
   - What should we monitor?
   - What long-term strategies?
```

### 2. Enhanced Answer Composition

The LLM-enhanced answer includes:

```
**STRATEGIC ANALYSIS:** (LLM reasoning process)
"Examining the data, I notice that products P1 and P2 are 
negatively correlated. While P1 shows downtrend, P2's 
uptrend could indicate market shift. This suggests..."

**KEY INSIGHTS:** (From LLM analysis)
- Pattern 1: Market segmentation shift
- Pattern 2: Demand migration between categories
- Pattern 3: Leading indicator of Q3 performance

**⚠️ WARNINGS & RISKS:** (LLM-identified risks)
- Risk: Market saturation in P1 category
- Alert: Inventory mismatch could lead to stockouts in P2

**💡 OPPORTUNITIES:** (LLM-discovered opportunities)
- Cross-sell P1+P2 bundle to capture both segments
- Expand P2 production before competitors respond

**STRATEGIC RECOMMENDATIONS:**
→ Pivot inventory allocation from P1 to P2 within 2 weeks
→ Launch co-marketing campaign to drive awareness
→ Develop contingency for P1 category decline

**DETAILED ANALYSIS:** (Original agent analysis for reference)
...
```

## System Prompt

The LLM operates under an expert system prompt that positions it as a business strategist:

```
You are an exceptionally intelligent business strategist with deep expertise in:
- Data analysis and trend forecasting
- Business strategy and competitive positioning
- Risk management and mitigation
- Market dynamics and consumer behavior
- Supply chain optimization
- Portfolio optimization and resource allocation

Your task is to perform DEEP CHAIN-OF-THOUGHT reasoning...
```

This ensures the LLM thinks strategically, not just summarizes data.

## Configuration

### Environment Setup

```bash
# Required: Groq API key
export GROQ_API_KEY=gsk_your_key_here

# Optional: Model selection
export GROQ_MODEL=llama-3.3-70b-versatile

# Optional: Enable debug logging
export RAG_AGENTS_DEBUG=1
```

### Python Configuration

```python
from rag_chatbot import config

# Check if LLM is available
config.GROQ_API_KEY  # Must be set
config.GROQ_MODEL    # Default: llama-3.3-70b-versatile

# Customize if needed (before calling LLM)
config.GROQ_API_KEY = "gsk_..."
config.GROQ_MODEL = "llama-3.1-70b-versatile"
```

## Usage

### Automatic (Integrated)

```python
from rag_chatbot import chatbot

# LLM reasoning happens automatically in multi-agent system
response = chatbot.chat_with_agents(
    message="Which products should we focus on?",
    thread_id="user_123"
)

print(response["answer"])
# Response now includes LLM strategic analysis!
```

### Direct Usage

```python
import asyncio
from rag_chatbot import llm_reasoner, agents, dataset_analyzer

async def deep_analysis():
    # Get agent outputs first
    analyzer = dataset_analyzer.get_analyzer()
    result = await agents.orchestrate_multi_agent_reasoning(
        query="What's our market position?",
        retrieved_docs=docs,
        analyzer=analyzer,
    )
    
    # Extract agent outputs
    all_agent_outputs = {
        "data_agent": result.reasoning_trail["data_agent"],
        "analysis_agent": result.reasoning_trail["analysis_agent"],
        "business_agent": result.reasoning_trail["business_agent"],
    }
    
    # Apply LLM reasoning
    enhanced_answer = await llm_reasoner.apply_llm_reasoning(
        query="What's our market position?",
        agent_outputs=all_agent_outputs,
        retrieved_docs=result.source_documents,
        answer_agent_answer=result.answer,
    )
    
    print(enhanced_answer)

asyncio.run(deep_analysis())
```

## Output Structure

### Enhanced Answer Sections

```python
{
    "full_response": str,          # Complete LLM output
    "reasoning": str,              # Chain-of-thought process
    "insights": [str, ...],        # Key insights (up to 5)
    "recommendations": [str, ...], # Strategic recommendations
    "warnings": [str, ...],        # Risk warnings
    "opportunities": [str, ...],   # Growth opportunities
}
```

### Example Output

```
STRATEGIC ANALYSIS:
"Analyzing the trend data, I observe a clear inverse 
relationship between product categories. While traditional 
products are declining 8% YoY, emerging products show 
25% growth. This suggests a market evolution where 
consumers are shifting preferences..."

KEY INSIGHTS:
1. Market transition from Category A to Category B
2. Early adopters driving Category B adoption
3. Competitor activity in Category A unchanged
4. Supply chain optimization opportunity
5. Customer lifetime value shift evident

⚠️ WARNINGS & RISKS:
• Category A inventory overstocked by 40%
• Team expertise concentrated in declining category
• Competitor could launch Category B alternative within Q3

💡 OPPORTUNITIES:
• First-mover advantage in new segment
• Cross-category bundling could slow churn
• Talent redeployment to Category B could accelerate growth

STRATEGIC RECOMMENDATIONS:
→ Reduce Category A orders by 50% within 30 days
→ Invest in Category B production capacity expansion
→ Develop transition plan for affected teams
→ Monitor competitor moves in Category B closely
→ Launch customer education program for new products
```

## LLM Reasoning Examples

### Example 1: Trend Analysis

**Input Data:**
- Product A: Uptrend 15%, Forecast 120 units, 8 data points
- Product B: Downtrend -10%, Forecast 45 units, 3 data points
- Product C: Stable 0.5%, Forecast 80 units, 7 data points

**LLM Reasoning:**
```
"Looking at this portfolio, I see a three-segment strategy:

1. GROWTH ENGINE (Product A): Strong uptrend suggests 
   market pull. With 8 data points, forecast is reliable.
   Action: Invest and accelerate.

2. RISK EXPOSURE (Product B): Downtrend with limited data 
   (only 3 points) is concerning. Could be temporary 
   fluctuation or real decline - needs monitoring.
   Action: Stabilize and investigate.

3. CASH COW (Product C): Stable baseline. Critical for 
   portfolio stability. Protects against A's volatility.
   Action: Maintain and defend market share.

Strategic implication: Reallocate 20% from C to A, 
stabilize B while investigating root cause of decline."
```

### Example 2: Anomaly Detection

**Input Anomaly:**
- High forecast (200 units) but strong downtrend (-25%)

**LLM Reasoning:**
```
"This is a classic 'canary in the coal mine' signal. 
The high forecast appears to be lagging indicator - 
momentum is clearly reversing downward. This suggests:

1. Historical demand was strong (hence high forecast baseline)
2. Recent market events are causing rapid reversal
3. Forecast model hasn't caught up with reality yet

Risk: If we treat the forecast as reliable, we'll 
over-produce and create excess inventory.

Action: Investigate what changed. Interview sales team. 
Check competitor announcements. Reduce production forecast 
by 30% proactively."
```

### Example 3: Portfolio Optimization

**Input Data:**
- 12 products analyzed
- 3 uptrending (15%, 12%, 8% growth)
- 5 stable (±2% variance)
- 4 downtrending (-5%, -8%, -12%, -15%)

**LLM Reasoning:**
```
"This portfolio shows concerning concentration of decline.
The math is simple:

Growing volume: 3 × 12 = 36 units growth
Declining volume: 4 × 10 = -40 units decline
Net position: -4 units (NEGATIVE)

This is unsustainable. We need to either:
1. Accelerate growth products (expand to 7-8 products)
2. Stabilize declining products (reduce to 1-2)
3. Both

The stable products are valuable buffer - they reduce 
volatility. But we need them to grow, not just survive.

Strategic recommendation: Launch innovation pipeline 
to convert 2 stable products into growth products 
within 6 months. Divest from bottom 2 declining products."
```

## Fallback & Error Handling

### If LLM is Unavailable

```python
# LLM automatically disabled if:
- GROQ_API_KEY not set
- langchain_groq not installed
- API call fails

# System falls back to agent-only answer
# No degradation to user experience
# Fully functional data-backed analysis still provided
```

### If LLM Call Fails

```python
# Error handling:
1. LLM reasoning attempted
2. If timeout (>30 seconds): Use agent answer
3. If API error: Log and use agent answer
4. If response empty: Log and use agent answer
5. No errors propagated to user
```

## Performance Characteristics

| Component | Time |
|-----------|------|
| Agent Orchestration | 1-2 seconds |
| LLM Chain-of-Thought | 2-5 seconds |
| **Total with LLM** | **3-7 seconds** |
| Without LLM (fallback) | 1-2 seconds |

## Advanced Usage

### Custom System Prompts

```python
from rag_chatbot import llm_reasoner

# Customize the system prompt
llm_reasoner.SYSTEM_PROMPT_DEEP_REASONING = """
Your custom system prompt emphasizing specific expertise...
"""

# Or customize just the template
llm_reasoner.PROMPT_TEMPLATE_DEEP_ANALYSIS = """
Your custom analysis template...
"""
```

### Extraction of LLM Reasoning

```python
# Access reasoning in response
response = chatbot.chat_with_agents(message, thread_id)

# Check if LLM was used
if "reasoning_trail" in response:
    answer_agent_output = response["reasoning_trail"]["answer_agent"]
    if answer_agent_output["data"].get("llm_enhanced"):
        print("✓ LLM-enhanced answer generated")
```

### Separate LLM & Agent Analysis

```python
# Get agent-only answer
agent_answer = agents.AnswerAgent(analyzer)._compose_base_answer(...)

# Get LLM-enhanced answer
llm_answer = await llm_reasoner.apply_llm_reasoning(
    query=query,
    agent_outputs=all_outputs,
    retrieved_docs=docs,
    answer_agent_answer=agent_answer,
)

# Compare both
print("Agent-only:", agent_answer[:200])
print("\nLLM-enhanced:", llm_answer[:200])
```

## Monitoring & Debugging

### Check LLM Availability

```python
from rag_chatbot import llm_reasoner

reasoner = llm_reasoner.LLMReasoner()
if reasoner.is_available():
    print("✓ LLM reasoner is available")
else:
    print("✗ LLM reasoner is disabled")
```

### Enable Debug Logging

```python
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag.llm_reasoner")

# Now LLM reasoning will log all steps
```

### Inspect Full LLM Response

```python
response = chatbot.chat_with_agents(message, thread_id)

# The full LLM response is available
import json
reasoning_trail = response.get("reasoning_trail", {})
print(json.dumps(reasoning_trail, indent=2))
```

## Best Practices

### 1. Always Have Fallback
LLM reasoning is enhancement, not requirement:
```python
# System works with or without LLM
# Graceful degradation guaranteed
```

### 2. Monitor LLM Costs
```python
# Each call uses ~1000-2000 tokens
# Monitor usage to control costs
# Consider caching for common queries
```

### 3. Validate LLM Output
```python
# LLM insights are validated against data
# System ensures no hallucinations
# Data remains source of truth
```

### 4. Use for Strategic Decisions
```python
# LLM reasoning best for:
# ✓ Strategic planning
# ✓ Risk identification
# ✓ Pattern recognition
# ✓ Opportunity discovery

# Not ideal for:
# ✗ Real-time tactical decisions
# ✗ Exact inventory calculations
# ✗ Critical dependent calculations
```

## Troubleshooting

### "LLM reasoner not available"
**Solution:**
```bash
pip install langchain-groq
export GROQ_API_KEY=gsk_your_key
```

### "LLM call timed out"
**Solution:**
```python
# System automatically falls back to agent answer
# Increase timeout if needed:
# (Currently hardcoded to 30 seconds)
```

### "LLM reasoning seems generic"
**Solution:**
- Ensure data quality is high (DataAgent validation)
- Provide detailed query context
- Use specific product names in query
- Let agents populate analysis before LLM sees it

### "LLM output differs from data"
**Solution:**
- LLM is intentionally strategic, not literal
- Agent analysis is source of truth for metrics
- LLM adds interpretation layer
- Check reasoning trail for justification

## API Reference

### LLMReasoner Class

```python
class LLMReasoner:
    def __init__(self)
    def is_available() -> bool
    async def reason_about_data(query, agent_outputs, retrieved_docs) -> Dict
```

### Public Function

```python
async def apply_llm_reasoning(
    query: str,
    agent_outputs: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    answer_agent_answer: str,
) -> str
```

### Helper Functions

```python
def compose_llm_enhanced_answer(
    query: str,
    agent_outputs: Dict[str, Any],
    llm_reasoning: Dict[str, Any],
    answer_agent_answer: str,
) -> str
```

## Future Enhancements

1. **Multi-turn Reasoning**: Follow-up questions for deeper analysis
2. **Scenario Planning**: "What if" analysis with LLM
3. **Forecast Confidence**: LLM-assessed confidence intervals
4. **Competitive Intelligence**: LLM-powered competitor analysis
5. **Trend Prediction**: LLM forecasts based on patterns

## Summary

The LLM reasoning layer adds **strategic depth** to the multi-agent system:

✓ Reasons through data like a C-suite executive  
✓ Identifies patterns beyond statistical analysis  
✓ Discovers opportunities and risks proactively  
✓ Provides justification for all recommendations  
✓ Maintains data accuracy as source of truth  
✓ Gracefully degrades if LLM unavailable  
✓ Shows all reasoning for auditability  

**Result: Business-grade strategic analysis at scale!** 🚀
