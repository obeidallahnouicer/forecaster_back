"""
Analysis Agent - Performs numerical and statistical analysis on retrieved data.

This agent computes per-product metrics across models, detects volatility,
measures model disagreement, ranks model reliability, and produces structured
analysis results that downstream agents can consume.
"""

import math
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentInput, AgentOutput
import json
try:
    from rag_chatbot.llm_reasoner import LLMReasoner
    HAS_LLM = True
except Exception:
    LLMReasoner = None
    HAS_LLM = False
from core.context_manager import get_context_manager
from rag_chatbot import prompt_templates


class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="AnalysisAgent", description="Performs numerical/statistical analysis on retrieved data")

    def get_capabilities(self):
        return ["statistical_analysis", "trend_detection", "volatility_detection", "model_comparison"]

    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        session_id = agent_input.session_id
        documents = agent_input.retrieved_documents or []
        reasoning_steps = []
        self.logger.info(f"AnalysisAgent.reason called for session={session_id} with {len(documents)} documents")

        try:
            await self._log_reasoning_step(session_id, "Starting analysis of retrieved documents")

            # Group forecasts by product and model (expect metadata to include model_type and forecast values)
            products: Dict[str, Dict[str, List[float]]] = {}

            for doc in documents:
                meta = doc.get("metadata", {})
                product = meta.get("ref_article") or doc.get("product_code") or meta.get("product_code")
                model = meta.get("model", meta.get("model_type", "unknown_model"))

                # Forecast values may be stored as a list under metadata['forecast_values'] or a single avg
                values = None
                if "forecast_values" in meta:
                    values = meta.get("forecast_values")
                elif "forecast_series" in meta:
                    values = meta.get("forecast_series")
                elif "avg_forecast" in meta:
                    values = [meta.get("avg_forecast")]

                if product is None:
                    continue

                products.setdefault(product, {})
                products[product].setdefault(model, [])

                if isinstance(values, list):
                    products[product][model].extend([float(v) for v in values if v is not None])
                elif values is not None:
                    try:
                        products[product][model].append(float(values))
                    except Exception:
                        pass

            reasoning_steps.append(f"Found {len(products)} products for analysis")
            self.logger.info(f"Found {len(products)} products for analysis")

            # Compute per-product statistics
            analysis_results = {}

            for product, model_dict in products.items():
                model_stats = {}
                all_model_means = []
                model_means = {}
                model_stds = {}

                for model_name, vals in model_dict.items():
                    if not vals:
                        continue
                    mean_v = statistics.mean(vals)
                    std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    model_stats[model_name] = {"mean": mean_v, "std": std_v, "count": len(vals)}
                    all_model_means.append(mean_v)
                    model_means[model_name] = mean_v
                    model_stds[model_name] = std_v

                # Overall metrics
                overall_mean = statistics.mean(all_model_means) if all_model_means else 0.0
                overall_std = statistics.pstdev(all_model_means) if len(all_model_means) > 1 else 0.0

                # Disagreement (variance across model means)
                disagreement = statistics.pvariance(all_model_means) if len(all_model_means) > 1 else 0.0

                # Stability score: 1 - (std / mean) (bounded)
                stability_score = None
                if overall_mean != 0:
                    stability_score = max(0.0, 1.0 - (overall_std / abs(overall_mean)))
                else:
                    stability_score = 0.0

                # Trend detection (simple average slope proxy using model means order)
                # When multiple model means exist, compute sign of overall_mean and relative trend
                trend_label = "stable"
                if overall_mean > 0 and overall_mean > abs(overall_std):
                    trend_label = "growing"
                elif overall_mean < 0 and abs(overall_mean) > abs(overall_std):
                    trend_label = "declining"

                # Model reliability ranking: based on lower std and higher count
                reliability_scores = {}
                for m, stats_v in model_stats.items():
                    count_factor = min(1.0, stats_v["count"] / 10)
                    std_factor = 1.0 - min(1.0, stats_v["std"] / (abs(stats_v["mean"]) + 1e-6))
                    reliability_scores[m] = max(0.0, min(1.0, 0.6 * count_factor + 0.4 * std_factor))

                sorted_models = sorted(reliability_scores.items(), key=lambda x: x[1], reverse=True)

                analysis_results[product] = {
                    "product": product,
                    "model_stats": model_stats,
                    "model_means": model_means,
                    "model_stds": model_stds,
                    "overall_mean": overall_mean,
                    "overall_std": overall_std,
                    "disagreement": disagreement,
                    "stability_score": stability_score,
                    "trend_label": trend_label,
                    "model_reliability_ranking": [ {"model": m, "score": s} for m, s in sorted_models ]
                }

            # Aggregate insights
            stable_products = [p for p, d in analysis_results.items() if d.get("stability_score", 0) >= 0.7]
            risky_products = [p for p, d in analysis_results.items() if d.get("stability_score", 0) < 0.4]
            disagreement_high = [p for p, d in analysis_results.items() if d.get("disagreement", 0) > 0.1]

            reasoning_steps.append(f"Analysis produced results for {len(analysis_results)} products")
            self.logger.info(f"Analysis produced results for {len(analysis_results)} products")

            data = {
                "products": analysis_results,
                "stable_products": stable_products,
                "risky_products": risky_products,
                "high_disagreement_products": disagreement_high,
                "summary": {
                    "total_products": len(analysis_results),
                    "stable_count": len(stable_products),
                    "risky_count": len(risky_products)
                }
            }

            # Aggregate views expected by other modules/tests
            try:
                # Stability analysis: identify most and least stable products
                most_stable = None
                least_stable = None
                if analysis_results:
                    sorted_by_stability = sorted(
                        ((p, d.get('stability_score', 0)) for p, d in analysis_results.items()),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    most_stable = {'product': sorted_by_stability[0][0], 'stability_score': sorted_by_stability[0][1]} if sorted_by_stability else None
                    least_stable = {'product': sorted_by_stability[-1][0], 'stability_score': sorted_by_stability[-1][1]} if sorted_by_stability else None

                # Trend analysis: bucket products by trend_label
                trend_analysis = {'uptrending': [], 'downtrending': [], 'stable': []}
                for p, d in analysis_results.items():
                    t = (d.get('trend_label') or 'stable').lower()
                    if 'up' in t:
                        trend_analysis['uptrending'].append(p)
                    elif 'down' in t:
                        trend_analysis['downtrending'].append(p)
                    else:
                        trend_analysis['stable'].append(p)

                # Performance analysis: find best and worst by overall_mean
                best_performing = None
                low_performing = None
                if analysis_results:
                    sorted_by_mean = sorted(
                        ((p, d.get('overall_mean', 0)) for p, d in analysis_results.items()),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    best_performing = {'product': sorted_by_mean[0][0], 'avg_forecast': sorted_by_mean[0][1]} if sorted_by_mean else None
                    low_performing = {'product': sorted_by_mean[-1][0], 'avg_forecast': sorted_by_mean[-1][1]} if sorted_by_mean else None

                # Reliability assessment: bucket by stability_score thresholds
                reliability_assessment = {'high_confidence': [], 'medium_confidence': [], 'low_confidence': []}
                for p, d in analysis_results.items():
                    s = d.get('stability_score', 0)
                    if s >= 0.75:
                        reliability_assessment['high_confidence'].append(p)
                    elif s >= 0.5:
                        reliability_assessment['medium_confidence'].append(p)
                    else:
                        reliability_assessment['low_confidence'].append(p)

                data['stability_analysis'] = {'most_stable': most_stable, 'least_stable': least_stable}
                data['trend_analysis'] = trend_analysis
                data['performance_analysis'] = {'best_performing': best_performing, 'low_performing': low_performing}
                data['reliability_assessment'] = reliability_assessment
            except Exception:
                # Non-critical; continue without these aggregates
                self.logger.debug("Failed to compute aggregated analysis views")

            # LLM interpretive augmentation: translate numeric metrics into natural-language insights
            llm_insights = None
            # LLM augmentation (optional). Provide a structured multi-step analysis prompt.
            if HAS_LLM and analysis_results:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        # Prepare concise summary for LLM
                        prod_summary = []
                        for p, info in analysis_results.items():
                            prod_summary.append({
                                'product': p,
                                'overall_mean': info.get('overall_mean'),
                                'overall_std': info.get('overall_std'),
                                'disagreement': info.get('disagreement'),
                                'stability_score': info.get('stability_score'),
                                'trend_label': info.get('trend_label')
                            })

                        human = (
                            "You are a senior business analyst. Given the following per-product numeric summary (JSON), "
                            "provide a JSON array named \"insights\" with entries: {product, insight, critical_metrics, suggested_secondary_analyses}.\n\n"
                            + json.dumps({'products': prod_summary}, default=str)
                        )
                        system = "Interpret numeric analysis results and translate them into concise business insights with suggested next analysis steps."
                        resp = llm.invoke_with_prompts(system, human)
                        # Attempt parse JSON from response
                        # PLACEHOLDER: insert robust JSON parsing and validation here (function-calling preferred)
                        try:
                            parsed = json.loads(resp)
                            # Expected shape: {"insights": [...]}
                            llm_insights = parsed.get('insights') if isinstance(parsed, dict) else None
                            # Store raw response for audit
                            data.setdefault('audit', {})['llm_raw_insights'] = resp
                        except Exception:
                            # Fallback: store raw text per product so nothing is lost
                            llm_insights = [{'product': p, 'raw': resp} for p in analysis_results.keys()]
                            data.setdefault('audit', {})['llm_raw_insights'] = resp
                except Exception:
                    llm_insights = None

            if llm_insights:
                data['llm_insights'] = llm_insights
                self.logger.info(f"LLM produced insights for {len(llm_insights)} products")

            # LLM chain-of-thought: request a reasoning chain for each product
            llm_chain = None
            if HAS_LLM and analysis_results:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        # Build context: include retrieved docs if present in agent_input
                        retrieved = agent_input.retrieved_documents or []
                        # Compose compact analysis summary
                        summary = {p: {
                            'overall_mean': v.get('overall_mean'),
                            'overall_std': v.get('overall_std'),
                            'disagreement': v.get('disagreement'),
                            'stability_score': v.get('stability_score'),
                            'trend_label': v.get('trend_label')
                        } for p, v in analysis_results.items()}

                        # Use centralized prompt template for AnalysisAgent
                        sys_prompt, human_prompt = prompt_templates.compose_agent_prompt(
                            prompt_templates.ANALYSIS_SYSTEM_PROMPT,
                            prompt_templates.ANALYSIS_HUMAN_TEMPLATE,
                            {'summary': summary, 'retrieved_sample': retrieved[:3]}
                        )

                        resp = llm.invoke_with_prompts(sys_prompt, human_prompt)
                        # Try parse JSON
                        # PLACEHOLDER: implement stricter JSON schema validation here (pydantic)
                        try:
                            parsed = json.loads(resp)
                            llm_chain = parsed.get('product_commentary') if isinstance(parsed, dict) else None
                            # capture summary_insights and overall_confidence too
                            if isinstance(parsed, dict):
                                data['summary_insights'] = parsed.get('summary_insights')
                                data['overall_confidence'] = parsed.get('overall_confidence')
                                data.setdefault('audit', {})['llm_full_parsed'] = parsed
                        except Exception:
                            # not JSON - store raw text under a generic key for auditing
                            llm_chain = {'raw': resp}
                            data.setdefault('audit', {})['llm_raw_chain'] = resp
                except Exception:
                    llm_chain = None

            if llm_chain:
                data['llm_chain'] = llm_chain
                self.logger.info("Persisting llm_chain into analysis data")

            # Compute confidence

            confidence = 0.9 if analysis_results else 0.0

            # Persist analysis output into shared context for downstream agents
            try:
                cm = get_context_manager()
                # Ensure we persist any LLM-chain and summary fields for auditability
                ctx_payload = {'data': data, 'confidence': confidence}
                # If llm_chain wasn't JSON-parsable and we captured raw text, keep it under audit
                if isinstance(data.get('llm_chain'), dict) and 'raw' in data.get('llm_chain'):
                    ctx_payload.setdefault('audit', {})
                    ctx_payload['audit']['llm_raw'] = data['llm_chain']['raw']
                cm.update_context(agent_input.session_id, 'AnalysisAgent', ctx_payload)
                self.logger.info(f"AnalysisAgent output persisted to ContextManager for session={session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to persist AnalysisAgent output to ContextManager: {e}")

            return await self._create_output(
                success=True,
                data=data,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                execution_time_ms=0.0
            )

        except Exception as e:
            await self._log_reasoning_step(session_id, f"Analysis failed: {e}", success=False, error=str(e))
            return await self._create_output(success=False, data={}, reasoning_steps=[f"Error: {e}"], confidence=0.0, execution_time_ms=0.0)
