"""
Reasoning Agent - Synthesizes analytical findings into human-interpretable insights.

This agent applies rule-based reasoning on AnalysisAgent outputs and generates
insights that bridge numerical results and business implications.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.context_manager import get_context_manager
try:
    from rag_chatbot.llm_reasoner import LLMReasoner
    HAS_LLM_REASONER = True
except Exception:
    LLMReasoner = None
    HAS_LLM_REASONER = False


class ReasoningAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ReasoningAgent", description="Synthesizes analytical findings into insights")

    def get_capabilities(self):
        return ["insight_synthesis", "pattern_recognition", "logical_reasoning"]

    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        session_id = agent_input.session_id
        reasoning_steps = []
        self.logger.info(f"ReasoningAgent.reason called for session={session_id}")

        try:
            # Consume prior AnalysisAgent output from context manager if available
            analysis_out = None
            try:
                if agent_input.context:
                    analysis_out = agent_input.context.get_agent_output("AnalysisAgent")
            except Exception:
                analysis_out = None
            # analysis_out may be an AgentOutput object or raw dict
            analysis_data = None
            if analysis_out:
                analysis_data = analysis_out["output"] if isinstance(analysis_out, dict) and "output" in analysis_out else analysis_out

            # Fallback to retrieved_documents analysis
            if not analysis_data:
                analysis_products = agent_input.retrieved_documents
            else:
                analysis_products = analysis_data.get("data", {}).get("products") if isinstance(analysis_data, dict) else None

            # If LLM reasoner is available, use it to interpret the analysis_data and produce prioritized recommendations
            insights = []
            llm_result = None
            if HAS_LLM_REASONER and analysis_data:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        # Provide agent outputs and retrieved docs for deep reasoning
                        agent_outputs = {"analysis_agent": analysis_data}
                        # The LLM should return prioritized interpretations and candidate actions with confidence/ROI/risk
                        llm_result = await llm.reason_about_data(agent_input.query, {"analysis_agent": analysis_data}, agent_input.retrieved_documents)
                        insights = llm_result.get("insights", []) or []
                        # Persist raw LLM reasoning for auditability
                        try:
                            cm = get_context_manager()
                            cm.update_context(session_id, 'ReasoningAgent', {'data': llm_result, 'confidence': 0.9})
                        except Exception:
                            pass
                        reasoning_steps.append("LLM reasoning applied to analysis results")
                    else:
                        reasoning_steps.append("LLM reasoner not available; falling back to rule-based insights")
                except Exception as e:
                    reasoning_steps.append(f"LLM reasoning failed: {e}; falling back to rule-based insights")

            # Fallback: rule-based synthesis if LLM not used or failed
            # Fallback: rule-based synthesis if LLM not used or failed
            if not insights and isinstance(analysis_products, dict):
                for product, pdat in analysis_products.items():
                    score = pdat.get("stability_score", 0)
                    disagreement = pdat.get("disagreement", 0)
                    trend = pdat.get("trend_label", "stable")

                    note = ""
                    if score >= 0.8 and trend == "growing":
                        note = "Consistent growth across models"
                    elif score >= 0.8 and trend == "stable":
                        note = "Stable and reliable forecasts"
                    elif disagreement > 0.2:
                        note = "Models disagree significantly — investigate"
                    elif score < 0.4:
                        note = "High volatility / low stability"

                    insights.append({
                        "product": product,
                        "trend": trend,
                        "stability_score": score,
                        "disagreement": disagreement,
                        "note": note,
                    })

            reasoning_steps.append(f"Generated {len(insights)} insights from analysis data")
            self.logger.info(f"Generated {len(insights)} insights")

            # Post-process: produce prioritized recommendations from insights (placeholder)
            # PLACEHOLDER: implement recommendation scoring (ROI estimate, risk assessment) here
            recommendations = []
            for ins in insights:
                # Simple heuristic fallback scoring
                prod = ins.get('product')
                score = ins.get('stability_score', 0)
                priority = 'medium'
                if score >= 0.8:
                    priority = 'high'
                elif score < 0.4:
                    priority = 'low'
                recommendations.append({
                    'product': prod,
                    'priority': priority,
                    'justification': ins.get('note') or ins.get('explanation', ''),
                    'confidence': ins.get('confidence', score)
                })

            data = {"insights": insights, "recommendations": recommendations}
            confidence = 0.9 if recommendations else 0.3

            # Persist ReasoningAgent output for downstream consumption and audit
            try:
                cm = get_context_manager()
                cm.update_context(session_id, 'ReasoningAgent', {'data': data, 'confidence': confidence})
                self.logger.info(f"ReasoningAgent output persisted to ContextManager for session={session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to persist ReasoningAgent output: {e}")

            return await self._create_output(success=True, data=data, reasoning_steps=reasoning_steps, confidence=confidence, execution_time_ms=0.0)

        except Exception as e:
            await self._log_reasoning_step(session_id, f"Reasoning failed: {e}", success=False, error=str(e))
            return await self._create_output(success=False, data={}, reasoning_steps=[f"Error: {e}"], confidence=0.0, execution_time_ms=0.0)
