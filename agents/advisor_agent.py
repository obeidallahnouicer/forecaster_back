"""
Advisor Agent - Translates insights into business/financial recommendations.

This agent maps analytical insights to prioritized business actions and includes
justifications linked to the underlying numerical evidence.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentInput, AgentOutput
import json
try:
    from rag_chatbot.llm_reasoner import LLMReasoner
    HAS_LLM = True
except Exception:
    LLMReasoner = None
    HAS_LLM = False
from core.context_manager import get_context_manager
import logging


class AdvisorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="AdvisorAgent", description="Generates business recommendations from insights")

    def get_capabilities(self):
        return ["business_recommendations", "risk_assessment", "strategy_planning"]

    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        session_id = agent_input.session_id
        reasoning_steps = []
        self.logger.info(f"AdvisorAgent.reason called for session={session_id}")

        try:

            reasoning_out = None
            try:
                if agent_input.context:
                    reasoning_out = agent_input.context.get_agent_output("ReasoningAgent")
            except Exception:
                reasoning_out = None
            reasoning_data = None
            if reasoning_out:
                reasoning_data = reasoning_out["output"] if isinstance(reasoning_out, dict) and "output" in reasoning_out else reasoning_out

            insights = reasoning_data.get("data", {}).get("insights") if isinstance(reasoning_data, dict) else None

            recommendations = []

            # Use LLM to craft prioritized recommendations when available
            if HAS_LLM and insights:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        # Build JSON payload of insights
                        payload = {"insights": insights}
                        human = (
                            "You are a senior business/financial advisor. Given the insights JSON below, "
                            "produce a JSON array named \"recommendations\" with entries: {product, action, priority (high|medium|low), expected_roi_estimate (optional numeric), risk_level (low|medium|high), justification}. \n\n"
                            + json.dumps(payload, default=str)
                        )
                        system = "Generate prioritized, auditable business recommendations with justification and risk/ROI estimates."
                        resp = llm.invoke_with_prompts(system, human)
                        # PLACEHOLDER: robust JSON parsing + schema validation here (pydantic)
                        try:
                            parsed = json.loads(resp)
                            if isinstance(parsed, dict) and 'recommendations' in parsed:
                                recommendations = parsed['recommendations']
                                data_audit = parsed
                            elif isinstance(parsed, list):
                                recommendations = parsed
                                data_audit = {'recommendations': parsed}
                            else:
                                data_audit = {'raw': resp}
                        except Exception:
                            # fallback to naive mapping
                            recommendations = []
                            data_audit = {'raw': resp}
                except Exception:
                    recommendations = []

            # Fallback rule-based if LLM not used or failed
            if not recommendations and isinstance(insights, list):
                for ins in insights:
                    prod = ins.get("product")
                    trend = ins.get("trend")
                    score = ins.get("stability_score", 0)
                    disagreement = ins.get("disagreement", 0)

                    rec = {"product": prod, "action": "Monitor", "priority": "medium", "justification": ""}

                    if score >= 0.75 and trend == "growing":
                        rec["action"] = "Invest / Expand Marketing"
                        rec["priority"] = "high"
                        rec["justification"] = f"Stability {score:.2f} and positive trend; consensus across models."
                    elif score >= 0.6 and trend == "stable":
                        rec["action"] = "Maintain Stock / Monitor"
                        rec["priority"] = "medium"
                        rec["justification"] = f"Stable forecasts (score={score:.2f}); keep current strategy."
                    elif disagreement > 0.2:
                        rec["action"] = "Model Review / Further Investigation"
                        rec["priority"] = "high"
                        rec["justification"] = "Significant disagreement between models; verify data and model assumptions."
                    elif score < 0.4 and trend == "declining":
                        rec["action"] = "Reduce Stock / Phase-Out"
                        rec["priority"] = "high"
                        rec["justification"] = "Low stability and declining forecasts across models."

                    recommendations.append(rec)

            reasoning_steps.append(f"Created {len(recommendations)} recommendations")

            # LLM: produce a reasoning chain / justification per recommendation
            llm_chain = None
            if HAS_LLM and recommendations:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        # Include prior agent outputs from ContextManager
                        cm = get_context_manager()
                        analysis_output = cm.get_context(agent_input.session_id).get_agent_output('AnalysisAgent') if cm.get_context(agent_input.session_id) else None
                        reasoning_output = cm.get_context(agent_input.session_id).get_agent_output('ReasoningAgent') if cm.get_context(agent_input.session_id) else None

                        payload = {"recommendations": recommendations, "analysis": analysis_output, "reasoning": reasoning_output}
                        human = (
                            "You are a CFO-level advisor. Given the analysis and preliminary reasoning, prioritize recommendations for CFO action.\n"
                            "Return JSON: {recommendations: [...], reasoning_chain: {...}, overall_priority_list: [...]}\n\n"
                            + json.dumps(payload, default=str)
                        )
                        system = "System: Provide prioritized recommendations, ROI and risk commentary, and a short chain-of-thought for each recommendation."
                        resp = llm.invoke_with_prompts(system, human)
                        try:
                            parsed = json.loads(resp)
                            if isinstance(parsed, dict):
                                # Use parsed recommendations if present
                                if 'recommendations' in parsed:
                                    recommendations = parsed.get('recommendations')
                                llm_chain = parsed.get('reasoning_chain', parsed)
                        except Exception:
                            llm_chain = {'raw': resp}
                except Exception:
                    llm_chain = None

            if llm_chain:
                result_data = {"recommendations": recommendations, "llm_chain": llm_chain}
            else:
                result_data = {"recommendations": recommendations}

            # Attach any LLM audit info
            if 'data_audit' in locals():
                result_data.setdefault('audit', {})['llm_advisor_raw'] = data_audit

            confidence = 0.9 if recommendations else 0.3

            # Persist advisor output into context (for Validator)
            try:
                cm = get_context_manager()
                cm.update_context(agent_input.session_id, 'AdvisorAgent', {'data': result_data, 'confidence': confidence})
                self.logger.info(f"AdvisorAgent output persisted to ContextManager for session={session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to persist AdvisorAgent output: {e}")

            return await self._create_output(success=True, data=result_data, reasoning_steps=reasoning_steps, confidence=confidence, execution_time_ms=0.0)

        except Exception as e:
            await self._log_reasoning_step(session_id, f"Advisor failed: {e}", success=False, error=str(e))
            return await self._create_output(success=False, data={}, reasoning_steps=[f"Error: {e}"], confidence=0.0, execution_time_ms=0.0)
