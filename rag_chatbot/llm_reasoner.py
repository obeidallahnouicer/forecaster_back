"""
LLM Reasoner Module — Advanced chain-of-thought reasoning with Groq.

This module provides:
- Advanced chain-of-thought prompting
- Deep reasoning over multi-agent outputs
- Strategic business decision-making
- Anomaly pattern recognition
- Predictive insights and warnings
- Executive-level recommendations

The LLM reasons through:
1. Data facts (what the agents found)
2. Patterns and correlations
3. Business implications
4. Strategic recommendations
5. Risk mitigation strategies
6. Opportunity maximization

Uses extended thinking for deep analysis.
"""

import logging
from typing import Dict, Any, Optional, List
import json

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

from langchain.schema import HumanMessage, SystemMessage
from . import config

logger = logging.getLogger("rag.llm_reasoner")


# ============================================================================
# LLM REASONING PROMPTS — Deep chain-of-thought reasoning
# ============================================================================

SYSTEM_PROMPT_DEEP_REASONING = """You are an exceptionally intelligent business strategist with deep expertise in:
- Data analysis and trend forecasting
- Business strategy and competitive positioning
- Risk management and mitigation
- Market dynamics and consumer behavior
- Supply chain optimization
- Portfolio optimization and resource allocation

Your task is to perform DEEP CHAIN-OF-THOUGHT reasoning over product data and forecasts.

REASONING FRAMEWORK:
1. UNDERSTAND THE DATA
   - What products are we analyzing?
   - What are the key metrics (forecast, trend, stability)?
   - How reliable is the data (data_points)?
   
2. IDENTIFY PATTERNS
   - What trends are visible across products?
   - Which products move together?
   - What anomalies suggest hidden opportunities or risks?
   
3. ANALYZE BUSINESS CONTEXT
   - What do these patterns mean for our business?
   - Which decisions are critical?
   - Where are our vulnerabilities?
   - Where are our opportunities?
   
4. THINK STRATEGICALLY
   - Why are these trends happening?
   - What could change them?
   - What should we prepare for?
   - How do we capitalize on opportunities?
   
5. RECOMMEND ACTIONS
   - What should we do immediately?
   - What should we monitor closely?
   - What long-term strategies should we consider?
   - How do we manage risks?

RESPONSE STYLE:
- Show your reasoning explicitly (think step-by-step)
- Support conclusions with data
- Identify uncertainties and confidence levels
- Provide actionable recommendations
- Think like a CEO/CFO making strategic decisions
- Consider second and third-order effects

TONE: Professional, insightful, strategic, confident
LENGTH: Comprehensive but concise (aim for 3-5 key paragraphs)

Begin your deep reasoning with "<REASONING>" tags to show your thinking process.
Then provide your final insights with "<INSIGHTS>" tags.
"""


PROMPT_TEMPLATE_DEEP_ANALYSIS = """
BUSINESS DATA FOR STRATEGIC ANALYSIS:

{data_summary}

AGENT INSIGHTS:
{agent_insights}

RETRIEVED CONTEXT:
{context_summary}

---

Your task: Perform deep chain-of-thought reasoning to answer this business question:
"{query}"

Think through:
1. What data facts are most important?
2. What patterns do you see?
3. What's the deeper business story?
4. What could go wrong or right?
5. What should leadership focus on?

Provide strategic insights that go beyond surface-level analysis.
"""


# ============================================================================
# LLM REASONER CLASS — Deep strategic reasoning
# ============================================================================

class LLMReasoner:
    """
    Advanced LLM reasoner for strategic business analysis.
    
    Performs:
    - Chain-of-thought reasoning
    - Pattern recognition
    - Strategic recommendations
    - Risk assessment
    - Opportunity identification
    """
    
    def __init__(self):
        """Initialize LLM reasoner with Groq."""
        self.logger = logging.getLogger("rag.llm_reasoner")
        
        if not HAS_GROQ:
            self.logger.warning("langchain_groq not installed; LLM reasoning disabled")
            self.llm = None
            return
        
        if not config.GROQ_API_KEY:
            self.logger.warning("GROQ_API_KEY not set; LLM reasoning disabled")
            self.llm = None
            return
        
        try:
            self.llm = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_MODEL,
                temperature=0.3,  # Higher temperature for creative reasoning
                max_tokens=2048,  # More tokens for deep reasoning
                timeout=30,
            )
            self.logger.info(f"LLM Reasoner initialized: {config.GROQ_MODEL}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.llm is not None
    
    async def reason_about_data(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Perform deep reasoning about product data and business implications.
        
        Args:
            query: User's question
            agent_outputs: Outputs from all agents
            retrieved_docs: Retrieved documents from vectorstore
            
        Returns:
            Dict with:
            - reasoning: Chain-of-thought reasoning process
            - insights: Strategic business insights
            - recommendations: Actionable recommendations
            - warnings: Risk warnings
            - opportunities: Growth opportunities
        """
        if not self.is_available():
            self.logger.warning("LLM not available; returning empty reasoning")
            return {
                "reasoning": "LLM not available",
                "insights": [],
                "recommendations": [],
                "warnings": [],
                "opportunities": [],
                "error": "LLM not configured",
            }
        
        try:
            self.logger.info("Starting deep LLM reasoning...")
            
            # Build context for LLM
            data_summary = self._build_data_summary(agent_outputs)
            agent_insights = self._format_agent_insights(agent_outputs)
            context_summary = self._build_context_summary(retrieved_docs)
            
            # Build prompt
            prompt = PROMPT_TEMPLATE_DEEP_ANALYSIS.format(
                query=query,
                data_summary=data_summary,
                agent_insights=agent_insights,
                context_summary=context_summary,
            )
            
            # Call LLM with chain-of-thought
            self.logger.debug(f"Sending prompt to LLM ({len(prompt)} chars)...")
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT_DEEP_REASONING),
                HumanMessage(content=prompt),
            ]
            
            response = self.llm.invoke(messages)
            reasoning_response = response.content
            
            self.logger.info("LLM reasoning completed successfully")
            
            # Parse response
            result = self._parse_llm_response(reasoning_response, agent_outputs)
            
            return result
        
        except Exception as e:
            self.logger.exception(f"LLM reasoning failed: {e}")
            return {
                "reasoning": f"LLM reasoning failed: {str(e)}",
                "insights": [],
                "recommendations": [],
                "warnings": [],
                "opportunities": [],
                "error": str(e),
            }
    
    def _build_data_summary(self, agent_outputs: Dict[str, Any]) -> str:
        """Build summary of data for LLM context."""
        data_agent = agent_outputs.get("data_agent", {}).get("data", {})
        analysis_agent = agent_outputs.get("analysis_agent", {}).get("data", {})
        business_agent = agent_outputs.get("business_agent", {}).get("data", {})
        
        summary_parts = []
        
        # Products summary
        products = data_agent.get("products", {})
        if products:
            summary_parts.append(f"✓ Products analyzed: {len(products)}")
            valid_count = sum(1 for p in products.values() if p.get("valid"))
            summary_parts.append(f"✓ Valid products: {valid_count}/{len(products)}")
        
        # Stability analysis
        most_stable = analysis_agent.get("stability_analysis", {}).get("most_stable")
        if most_stable:
            summary_parts.append(
                f"✓ Most stable: {most_stable['product']} "
                f"(trend: {most_stable.get('trend_pct', 0):.1f}%)"
            )
        
        # Trends
        trends = analysis_agent.get("trend_analysis", {})
        summary_parts.append(
            f"✓ Trends: {len(trends.get('uptrending', []))} up, "
            f"{len(trends.get('downtrending', []))} down, "
            f"{len(trends.get('stable', []))} stable"
        )
        
        # Performance
        perf = analysis_agent.get("performance_analysis", {})
        best = perf.get("best_performing", {})
        worst = perf.get("low_performing", {})
        if best and worst:
            summary_parts.append(
                f"✓ Performance range: {worst['product']} "
                f"({worst.get('avg_forecast', 0):.0f}) to {best['product']} "
                f"({best.get('avg_forecast', 0):.0f})"
            )
        
        # Business impact
        impact = business_agent.get("business_impact", {})
        if impact:
            summary_parts.append(f"✓ Portfolio health: {impact.get('portfolio_health')}")
            summary_parts.append(f"✓ Risk level: {impact.get('risk_level')}")
        
        return "\n".join(summary_parts)
    
    def _format_agent_insights(self, agent_outputs: Dict[str, Any]) -> str:
        """Format insights from all agents."""
        insights_parts = []
        
        business_agent = agent_outputs.get("business_agent", {}).get("data", {})
        
        # Key insights
        key_insights = business_agent.get("key_insights", [])
        if key_insights:
            insights_parts.append("KEY INSIGHTS:")
            for i, insight in enumerate(key_insights[:5], 1):
                insights_parts.append(f"  {i}. {insight}")
        
        # Risks
        risks = business_agent.get("risks", [])
        if risks:
            insights_parts.append("\nRISKS IDENTIFIED:")
            for risk in risks[:3]:
                insights_parts.append(
                    f"  • {risk.get('type')}: {risk.get('description')} "
                    f"(severity: {risk.get('severity')})"
                )
        
        # Opportunities
        opportunities = business_agent.get("opportunities", [])
        if opportunities:
            insights_parts.append("\nOPPORTUNITIES:")
            for opp in opportunities[:3]:
                insights_parts.append(
                    f"  • {opp.get('type')}: {opp.get('description')}"
                )
        
        return "\n".join(insights_parts)
    
    def _build_context_summary(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build summary of retrieved context."""
        if not retrieved_docs:
            return "No documents retrieved."
        
        summary_parts = [f"Retrieved {len(retrieved_docs)} relevant documents:"]
        
        for i, doc in enumerate(retrieved_docs[:3], 1):
            metadata = doc.get("metadata", {})
            product = metadata.get("ref_article", "Unknown")
            summary_parts.append(
                f"\n{i}. {product}"
                f"\n   Forecast: {metadata.get('avg_forecast', 'N/A')}"
                f"\n   Trend: {metadata.get('trend_pct', 'N/A')}% ({metadata.get('trend_label', 'Unknown')})"
                f"\n   Confidence: {metadata.get('data_points', 0)} data points"
            )
        
        if len(retrieved_docs) > 3:
            summary_parts.append(f"\n... and {len(retrieved_docs) - 3} more documents")
        
        return "\n".join(summary_parts)
    
    def _parse_llm_response(
        self,
        response: str,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM response to extract reasoning and insights."""
        
        result = {
            "full_response": response,
            "reasoning": "",
            "insights": [],
            "recommendations": [],
            "warnings": [],
            "opportunities": [],
        }
        
        # Extract reasoning
        if "<REASONING>" in response:
            try:
                reasoning_start = response.index("<REASONING>") + len("<REASONING>")
                reasoning_end = response.index("</REASONING>") if "</REASONING>" in response else len(response)
                result["reasoning"] = response[reasoning_start:reasoning_end].strip()
            except (ValueError, IndexError):
                result["reasoning"] = response[:500]  # First 500 chars
        else:
            result["reasoning"] = response[:800]
        
        # Extract insights
        if "<INSIGHTS>" in response:
            try:
                insights_start = response.index("<INSIGHTS>") + len("<INSIGHTS>")
                insights_end = response.index("</INSIGHTS>") if "</INSIGHTS>" in response else len(response)
                insights_text = response[insights_start:insights_end].strip()
                
                # Split by newlines and filter
                insight_lines = [
                    line.strip() for line in insights_text.split("\n")
                    if line.strip() and not line.strip().startswith(("-", "*", "#"))
                ]
                result["insights"] = insight_lines[:5]
            except (ValueError, IndexError):
                pass
        
        # Extract warnings (lines mentioning "warning", "risk", "caution")
        warning_keywords = ["warning", "risk", "caution", "watch", "alert", "concern", "problem"]
        for line in response.split("\n"):
            if any(kw in line.lower() for kw in warning_keywords):
                if len(line.strip()) > 10:
                    result["warnings"].append(line.strip())
        result["warnings"] = result["warnings"][:3]
        
        # Extract opportunities (lines mentioning "opportunity", "growth", "expand")
        opp_keywords = ["opportunity", "growth", "expand", "increase", "capture", "gain", "advantage"]
        for line in response.split("\n"):
            if any(kw in line.lower() for kw in opp_keywords):
                if len(line.strip()) > 10:
                    result["opportunities"].append(line.strip())
        result["opportunities"] = result["opportunities"][:3]
        
        # Extract recommendations (lines mentioning "recommend", "should", "must", "consider")
        rec_keywords = ["recommend", "should", "must", "consider", "suggest", "focus", "prioritize", "action"]
        for line in response.split("\n"):
            if any(kw in line.lower() for kw in rec_keywords):
                if len(line.strip()) > 15:
                    result["recommendations"].append(line.strip())
        result["recommendations"] = result["recommendations"][:5]
        
        # If no recommendations extracted, generate from agent output
        if not result["recommendations"]:
            business_agent = agent_outputs.get("business_agent", {}).get("data", {})
            recommendations = business_agent.get("recommendations", [])
            result["recommendations"] = [
                f"[{r.get('priority', 'medium').upper()}] {r.get('title')}: {r.get('description')}"
                for r in recommendations[:3]
            ]
        
        return result


# ============================================================================
# COMPOSE LLM-ENHANCED ANSWER
# ============================================================================

def compose_llm_enhanced_answer(
    query: str,
    agent_outputs: Dict[str, Any],
    llm_reasoning: Dict[str, Any],
    answer_agent_answer: str,
) -> str:
    """
    Compose final answer integrating LLM deep reasoning with agent outputs.
    
    Args:
        query: User query
        agent_outputs: Outputs from all agents
        llm_reasoning: LLM reasoning results
        answer_agent_answer: Original AnswerAgent answer
        
    Returns:
        Enhanced final answer with LLM reasoning
    """
    
    parts = []
    
    # Section 1: LLM Chain-of-Thought Reasoning
    if llm_reasoning.get("reasoning"):
        parts.append("**STRATEGIC ANALYSIS:**")
        parts.append(llm_reasoning["reasoning"])
        parts.append("")
    
    # Section 2: Key Insights from LLM
    if llm_reasoning.get("insights"):
        parts.append("**KEY INSIGHTS:**")
        for i, insight in enumerate(llm_reasoning["insights"], 1):
            parts.append(f"{i}. {insight}")
        parts.append("")
    
    # Section 3: LLM Warnings/Risks
    if llm_reasoning.get("warnings"):
        parts.append("**⚠️ WARNINGS & RISKS:**")
        for warning in llm_reasoning["warnings"]:
            parts.append(f"• {warning}")
        parts.append("")
    
    # Section 4: LLM Opportunities
    if llm_reasoning.get("opportunities"):
        parts.append("**💡 OPPORTUNITIES:**")
        for opp in llm_reasoning["opportunities"]:
            parts.append(f"• {opp}")
        parts.append("")
    
    # Section 5: LLM Recommendations
    if llm_reasoning.get("recommendations"):
        parts.append("**STRATEGIC RECOMMENDATIONS:**")
        for rec in llm_reasoning["recommendations"]:
            parts.append(f"→ {rec}")
        parts.append("")
    
    # Section 6: Agent Analysis (for reference)
    if answer_agent_answer:
        parts.append("**DETAILED ANALYSIS:**")
        parts.append(answer_agent_answer)
    
    return "\n".join(parts)


# ============================================================================
# PUBLIC API
# ============================================================================

async def apply_llm_reasoning(
    query: str,
    agent_outputs: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    answer_agent_answer: str,
) -> str:
    """
    Apply LLM deep reasoning to multi-agent output.
    
    This is the main entry point for LLM reasoning layer.
    
    Args:
        query: User query
        agent_outputs: Outputs from data/analysis/business/answer agents
        retrieved_docs: Retrieved documents
        answer_agent_answer: Original AnswerAgent answer
        
    Returns:
        LLM-enhanced final answer
    """
    reasoner = LLMReasoner()
    
    if not reasoner.is_available():
        logger.warning("LLM not available; returning original answer")
        return answer_agent_answer
    
    # Perform LLM reasoning
    llm_reasoning = await reasoner.reason_about_data(
        query=query,
        agent_outputs=agent_outputs,
        retrieved_docs=retrieved_docs,
    )
    
    # Compose enhanced answer
    enhanced_answer = compose_llm_enhanced_answer(
        query=query,
        agent_outputs=agent_outputs,
        llm_reasoning=llm_reasoning,
        answer_agent_answer=answer_agent_answer,
    )
    
    return enhanced_answer
