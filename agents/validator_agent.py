"""
Validator Agent - Cross-checks recommendations and ensures consistency with data.

This agent validates that insights and recommendations are grounded in the
retrieved data and flags uncertainty when necessary.
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


class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ValidatorAgent", description="Validates recommendations and checks for consistency")

    def get_capabilities(self):
        return ["data_validation", "consistency_checking", "confidence_assessment"]

    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        session_id = agent_input.session_id
        reasoning_steps = []
        self.logger.info(f"ValidatorAgent.reason called for session={session_id}")

        try:
            advisor_out = None
            analysis_out = None
            try:
                if agent_input.context:
                    advisor_out = agent_input.context.get_agent_output("AdvisorAgent")
                    analysis_out = agent_input.context.get_agent_output("AnalysisAgent")
            except Exception:
                advisor_out = None
                analysis_out = None

            advisor_data = advisor_out.get('output') if isinstance(advisor_out, dict) and 'output' in advisor_out else (advisor_out if isinstance(advisor_out, dict) else None)
            analysis_data = analysis_out.get('output') if isinstance(analysis_out, dict) and 'output' in analysis_out else (analysis_out if isinstance(analysis_out, dict) else None)

            recommendations = []
            if isinstance(advisor_data, dict):
                recommendations = advisor_data.get("data", {}).get("recommendations", [])

            issues = []
            validated = []

            # Basic cross-checks: ensure product exists in analysis and justification references values
            analysis_products = {}
            if isinstance(analysis_data, dict):
                analysis_products = analysis_data.get("data", {}).get("products", {})

            # LLM-assisted validation: send analysis + recommendations and ask for contradictions and confidence
            if HAS_LLM and recommendations and analysis_products:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        payload = {"analysis_products": analysis_products, "recommendations": recommendations}
                        human = (
                            "You are an expert validator. Given the analysis_products JSON and recommendations JSON below, "
                            "return a JSON object with keys: validated_recommendations (array of {product, valid:bool, notes}), issues (array), confidence (0-1).\n\n"
                            + json.dumps(payload, default=str)
                        )
                        system = "Validate logical consistency between analysis results and recommendations; flag contradictions and estimate overall confidence."
                        resp = llm.invoke_with_prompts(system, human)
                        # PLACEHOLDER: robust JSON parsing + schema validation
                        try:
                            parsed = json.loads(resp)
                            if isinstance(parsed, dict):
                                validated = parsed.get('validated_recommendations', [])
                                issues = parsed.get('issues', [])
                                confidence = parsed.get('confidence', 0.0)
                                reasoning_steps.append('LLM validation applied')
                                data = {"validated_recommendations": validated, "issues": issues, "confidence": confidence}
                                # Persist into context
                                try:
                                    cm = get_context_manager()
                                    cm.update_context(agent_input.session_id, 'ValidatorAgent', {'data': data, 'confidence': confidence})
                                except Exception:
                                    pass
                                self.logger.info(f"ValidatorAgent LLM validation completed for session={session_id}")
                                return await self._create_output(success=True, data=data, reasoning_steps=reasoning_steps, confidence=confidence, execution_time_ms=0.0)
                        except Exception:
                            # fallback to rule-based
                            data_audit = {'raw': resp}
                            self.logger.warning(f"ValidatorAgent LLM returned non-JSON response; saved raw audit")
                except Exception:
                    pass

            for rec in recommendations:
                prod = rec.get("product")
                justification = rec.get("justification", "")

                if prod not in analysis_products:
                    issues.append({"product": prod, "issue": "Product not found in analysis results"})
                    validated.append({"product": prod, "valid": False})
                    continue

                # Example numeric check: if recommendation says "high stability" ensure stability_score > 0.6
                stability = analysis_products.get(prod, {}).get("stability_score", None)
                if "Stability" in justification or "stability" in justification.lower():
                    if stability is None or stability < 0.5:
                        issues.append({"product": prod, "issue": "Justification claims stability but score is low"})
                        validated.append({"product": prod, "valid": False})
                        continue

                validated.append({"product": prod, "valid": True})

            confidence = 1.0 - (len(issues) / max(1, len(recommendations))) if recommendations else 0.0

            reasoning_steps.append(f"Validated {len(validated)} recommendations with {len(issues)} issues")

            data = {"validated_recommendations": validated, "issues": issues, "confidence": confidence}

            return await self._create_output(success=True, data=data, reasoning_steps=reasoning_steps, confidence=confidence, execution_time_ms=0.0)

        except Exception as e:
            await self._log_reasoning_step(session_id, f"Validation failed: {e}", success=False, error=str(e))
            return await self._create_output(success=False, data={}, reasoning_steps=[f"Error: {e}"], confidence=0.0, execution_time_ms=0.0)
