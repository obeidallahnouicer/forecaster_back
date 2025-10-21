"""
Multi-agent reasoning layer for RAG-based chatbot.

This module implements a business-intelligent agent system for structured reasoning:

1. Data Agent: Extracts structured data from RAG results and validates against dataset
2. Analysis Agent: Performs trend analysis, identifies patterns, computes metrics
3. Business Agent: Interprets data in terms of business impact, risks, recommendations
4. Answer Agent: Composes final human-readable answer integrating all insights

Flow:
  Retrieve documents → Data Agent (parse) → Analysis Agent (compute) → 
  Business Agent (interpret) → Answer Agent (compose)

Each agent:
- Logs intermediate reasoning for audit/debug
- Validates data against the dataset
- Returns structured output (can run in parallel where applicable)
- Integrates with conversation memory for context

Key features:
- Async-ready design (can parallelize agents)
- Modular prompts for LLM pluggability
- Strict data validation to prevent hallucinations
- Business-focused reasoning and recommendations
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import re
from pathlib import Path

from . import dataset_analyzer, config

# Try to import LLM reasoner (optional)
try:
    from . import llm_reasoner
    HAS_LLM_REASONER = True
except ImportError:
    HAS_LLM_REASONER = False

logger = logging.getLogger("rag.agents")


# ============================================================================
# DATA STRUCTURES — Agent inputs/outputs
# ============================================================================

@dataclass
class AgentInput:
    """Input data for an agent."""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None  # From previous agents


@dataclass
class AgentOutput:
    """Output structure for all agents."""
    agent_name: str
    success: bool
    data: Dict[str, Any]  # Structured output from agent reasoning
    reasoning_steps: List[str]  # Audit trail of reasoning
    validation_passed: bool = True
    error: Optional[str] = None


@dataclass
class MultiAgentResult:
    """Final structured result from multi-agent orchestration."""
    answer: str
    source: str  # "multi_agent_rag", "fallback", "error"
    agents_used: List[str]
    validation: str  # "passed", "partial", "fallback"
    source_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    reasoning_trail: Dict[str, AgentOutput]  # All agent outputs for debugging


# ============================================================================
# BASE AGENT — Abstract class for all reasoning agents
# ============================================================================

class Agent(ABC):
    """Abstract base class for reasoning agents."""
    
    def __init__(self, name: str, analyzer: dataset_analyzer.DatasetAnalyzer):
        """
        Initialize agent.
        
        Args:
            name: Agent identifier
            analyzer: DatasetAnalyzer for data validation
        """
        self.name = name
        self.analyzer = analyzer
        self.logger = logging.getLogger(f"rag.agents.{name.lower()}")
    
    @abstractmethod
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """
        Perform reasoning and return structured output.
        
        Args:
            agent_input: Input data (query, docs, context)
            
        Returns:
            AgentOutput with results and reasoning trail
        """
        pass
    
    def _log_reasoning_step(self, step: str, details: Optional[str] = None) -> str:
        """Log a reasoning step for audit trail."""
        msg = step
        if details:
            msg = f"{step}: {details}"
        self.logger.debug(f"[{self.name}] {msg}")
        return msg


# ============================================================================
# DATA AGENT — Extracts and validates structured data from retrieval results
# ============================================================================

class DataAgent(Agent):
    """
    Extracts structured data from RAG results.
    
    Responsibilities:
    - Parse product codes, metrics, trends from retrieved documents
    - Extract key numbers (avg_forecast, trend_pct, data_points)
    - Validate all mentioned products exist in dataset
    - Organize data by product/category
    - Flag any data quality issues or missing values
    
    Output:
    {
        "products": {
            "PRODUCT_CODE": {
                "mentions": int,
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
            "metrics_found": List[str],
            "data_quality": str,  # "high", "medium", "low"
        },
        "raw_extracts": {
            "product_codes": List[str],
            "metrics": List[str],
            "trends": List[str],
        }
    }
    """
    
    def __init__(self, analyzer: dataset_analyzer.DatasetAnalyzer):
        super().__init__("DataAgent", analyzer)
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """Extract and validate data from retrieved documents."""
        reasoning_steps = []
        
        try:
            # Step 1: Extract product codes and metrics from documents
            reasoning_steps.append(
                self._log_reasoning_step("Extracting product codes and metrics from retrieved documents")
            )
            products_mentioned = {}
            metrics_found = set()
            
            for doc in agent_input.retrieved_docs:
                content = doc.get("page_content") or doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # Extract from metadata first (most reliable)
                if "ref_article" in metadata:
                    product = metadata["ref_article"]
                    if product not in products_mentioned:
                        products_mentioned[product] = {
                            "mentions": 0,
                            "sources": [],
                            "metadata": metadata,
                        }
                    products_mentioned[product]["mentions"] += 1
                    products_mentioned[product]["sources"].append(doc.get("id", "unknown"))
                
                # Extract key metrics from metadata
                metric_fields = ["avg_forecast", "trend_pct", "trend_label", "data_points"]
                for field in metric_fields:
                    if field in metadata:
                        metrics_found.add(field)
            
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Extracted products: {len(products_mentioned)} unique",
                    f"metrics: {sorted(metrics_found)}"
                )
            )
            
            # Step 2: Validate product existence and enrich with dataset info
            reasoning_steps.append(
                self._log_reasoning_step("Validating products against dataset")
            )
            valid_products = 0
            invalid_products = []
            
            for product_code in list(products_mentioned.keys()):
                # Verify product exists
                exists = self.analyzer.verify_product_exists(product_code)
                
                if exists:
                    # Enrich with full product info
                    info = self.analyzer.get_product_info(product_code)
                    if info:
                        products_mentioned[product_code].update({
                            "valid": True,
                            "avg_forecast": info.get("avg_forecast"),
                            "trend_pct": info.get("trend_pct"),
                            "trend_label": info.get("trend_label"),
                            "data_points": info.get("data_points"),
                        })
                        valid_products += 1
                else:
                    products_mentioned[product_code]["valid"] = False
                    invalid_products.append(product_code)
            
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Validation complete: {valid_products} valid, {len(invalid_products)} invalid"
                )
            )
            
            # Step 3: Extract product codes from query text (for additional context)
            reasoning_steps.append(
                self._log_reasoning_step("Extracting product codes from query text")
            )
            query_products = self._extract_product_codes_from_text(agent_input.query)
            reasoning_steps.append(
                self._log_reasoning_step(f"Found in query: {query_products}")
            )
            
            # Step 4: Assess data quality
            data_quality = self._assess_data_quality(
                products_mentioned, metrics_found, agent_input.retrieved_docs
            )
            reasoning_steps.append(
                self._log_reasoning_step(f"Data quality assessment: {data_quality}")
            )
            
            # Build output
            output_data = {
                "products": {
                    code: {
                        "mentions": info["mentions"],
                        "sources": info.get("sources", []),
                        "avg_forecast": info.get("avg_forecast"),
                        "trend_pct": info.get("trend_pct"),
                        "trend_label": info.get("trend_label"),
                        "data_points": info.get("data_points"),
                        "valid": info.get("valid", False),
                    }
                    for code, info in products_mentioned.items()
                },
                "key_metrics": {
                    "total_products_mentioned": len(products_mentioned),
                    "total_valid_products": valid_products,
                    "total_invalid_products": len(invalid_products),
                    "invalid_products": invalid_products,
                    "metrics_found": sorted(metrics_found),
                    "data_quality": data_quality,
                },
                "raw_extracts": {
                    "product_codes": list(products_mentioned.keys()),
                    "query_products": query_products,
                },
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=len(invalid_products) == 0,
            )
        
        except Exception as e:
            self.logger.exception(f"Data extraction failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
    
    def _extract_product_codes_from_text(self, text: str) -> List[str]:
        """Extract potential product codes from text."""
        patterns = [
            r'\b[A-Z]{2}[A-Z0-9_/]{0,}\b',  # Starts with 2+ letters
            r'\b\d{13,}\b',  # EAN codes (13+ digits)
        ]
        
        codes = []
        for pattern in patterns:
            codes.extend(re.findall(pattern, text))
        
        # Filter out common non-product words
        stopwords = {
            'AND', 'THE', 'WITH', 'FOR', 'THIS', 'THAT', 'WHICH', 'TREND',
            'ITS', 'HAS', 'MAY', 'CAN', 'ARE', 'WHICH', 'ALL', 'NOT'
        }
        
        return list(set([c for c in codes if c not in stopwords]))
    
    def _assess_data_quality(
        self,
        products: Dict[str, Any],
        metrics: set,
        docs: List[Dict[str, Any]]
    ) -> str:
        """Assess overall data quality from extraction."""
        # High quality: >80% of products valid, >3 key metrics found, >2 docs
        valid_count = sum(1 for p in products.values() if p.get("valid", False))
        valid_pct = valid_count / len(products) if products else 0
        metric_count = len(metrics)
        
        if valid_pct >= 0.8 and metric_count >= 3 and len(docs) >= 2:
            return "high"
        elif valid_pct >= 0.5 and metric_count >= 2 and len(docs) >= 1:
            return "medium"
        else:
            return "low"


# ============================================================================
# ANALYSIS AGENT — Performs business-focused trend and anomaly analysis
# ============================================================================

class AnalysisAgent(Agent):
    """
    Performs trend analysis and anomaly detection.
    
    Responsibilities:
    - Identify most stable products (closest to 0% trend)
    - Find uptrending and downtrending products
    - Detect low-performing outliers
    - Analyze data point reliability (more data = more reliable)
    - Compute ranking and comparisons
    - Identify anomalies (unexpected forecasts vs trends)
    
    Output:
    {
        "stability_analysis": {
            "most_stable": {"product": str, "trend_pct": float, ...},
            "stability_ranking": List[Dict],
        },
        "trend_analysis": {
            "uptrending": List[Dict],
            "downtrending": List[Dict],
            "stable": List[Dict],
        },
        "performance_analysis": {
            "best_performing": {"product": str, "avg_forecast": float, ...},
            "low_performing": {"product": str, "avg_forecast": float, ...},
        },
        "reliability_assessment": {
            "high_confidence": List[str],  # 5+ data points
            "medium_confidence": List[str],  # 3-4 data points
            "low_confidence": List[str],  # <3 data points
        },
        "anomalies": List[Dict],  # Unexpected patterns
    }
    """
    
    def __init__(self, analyzer: dataset_analyzer.DatasetAnalyzer):
        super().__init__("AnalysisAgent", analyzer)
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """Analyze retrieved products for trends, performance, and anomalies."""
        reasoning_steps = []
        
        try:
            # Get data from previous agent
            data_agent_output = agent_input.context.get("data_agent_output") if agent_input.context else None
            if not data_agent_output or not data_agent_output.get("products"):
                reasoning_steps.append(
                    self._log_reasoning_step("No valid product data from DataAgent, using dataset_analyzer")
                )
                return await self._fallback_analysis(reasoning_steps)
            
            products_data = data_agent_output["products"]
            reasoning_steps.append(
                self._log_reasoning_step(f"Analyzing {len(products_data)} products")
            )
            
            # Step 1: Stability Analysis
            reasoning_steps.append(self._log_reasoning_step("Computing stability analysis"))
            stability_analysis = self._compute_stability_analysis(products_data)
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Most stable product: {stability_analysis.get('most_stable', {}).get('product', 'N/A')}"
                )
            )
            
            # Step 2: Trend Analysis
            reasoning_steps.append(self._log_reasoning_step("Categorizing by trend"))
            trend_analysis = self._categorize_by_trend(products_data)
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Trends: {len(trend_analysis.get('uptrending', []))} up, "
                    f"{len(trend_analysis.get('downtrending', []))} down, "
                    f"{len(trend_analysis.get('stable', []))} stable"
                )
            )
            
            # Step 3: Performance Analysis
            reasoning_steps.append(self._log_reasoning_step("Analyzing performance metrics"))
            performance_analysis = self._compute_performance_analysis(products_data)
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Best: {performance_analysis.get('best_performing', {}).get('product', 'N/A')}, "
                    f"Worst: {performance_analysis.get('low_performing', {}).get('product', 'N/A')}"
                )
            )
            
            # Step 4: Reliability Assessment
            reasoning_steps.append(self._log_reasoning_step("Assessing data reliability by data_points"))
            reliability = self._assess_reliability(products_data)
            reasoning_steps.append(
                self._log_reasoning_step(
                    f"Confidence: {len(reliability.get('high_confidence', []))} high, "
                    f"{len(reliability.get('medium_confidence', []))} medium, "
                    f"{len(reliability.get('low_confidence', []))} low"
                )
            )
            
            # Step 5: Anomaly Detection
            reasoning_steps.append(self._log_reasoning_step("Detecting anomalies"))
            anomalies = self._detect_anomalies(products_data)
            reasoning_steps.append(
                self._log_reasoning_step(f"Found {len(anomalies)} potential anomalies")
            )
            
            # Build output
            output_data = {
                "stability_analysis": stability_analysis,
                "trend_analysis": trend_analysis,
                "performance_analysis": performance_analysis,
                "reliability_assessment": reliability,
                "anomalies": anomalies,
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=True,
            )
        
        except Exception as e:
            self.logger.exception(f"Analysis failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
    
    async def _fallback_analysis(self, reasoning_steps: List[str]) -> AgentOutput:
        """Fallback: use dataset_analyzer directly."""
        try:
            output_data = {
                "most_stable": self.analyzer.find_most_stable_product(),
                "low_performing": self.analyzer.find_low_performing_product(),
                "top_performers": self.analyzer.get_top_products_by_forecast(n=3),
                "uptrending": self.analyzer.get_products_by_uptrend(limit=3),
                "summary_stats": self.analyzer.get_summary_stats(),
            }
            
            reasoning_steps.append(
                self._log_reasoning_step("Using dataset_analyzer fallback")
            )
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=True,
            )
        except Exception as e:
            self.logger.exception(f"Fallback analysis failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
    
    def _compute_stability_analysis(self, products: Dict[str, Any]) -> Dict[str, Any]:
        """Compute stability metrics (trend_pct closest to 0)."""
        valid_products = [
            (code, data) for code, data in products.items()
            if data.get("valid") and data.get("trend_pct") is not None
        ]
        
        if not valid_products:
            return {"most_stable": None, "stability_ranking": []}
        
        # Sort by absolute trend_pct (closest to 0 = most stable)
        sorted_by_stability = sorted(
            valid_products,
            key=lambda x: abs(x[1]["trend_pct"])
        )
        
        most_stable = {
            "product": sorted_by_stability[0][0],
            **sorted_by_stability[0][1],
        }
        
        stability_ranking = [
            {"product": code, "trend_pct": data["trend_pct"], "abs_trend": abs(data["trend_pct"])}
            for code, data in sorted_by_stability[:5]
        ]
        
        return {
            "most_stable": most_stable,
            "stability_ranking": stability_ranking,
        }
    
    def _categorize_by_trend(self, products: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Categorize products by trend direction."""
        uptrending = []
        downtrending = []
        stable = []
        
        for code, data in products.items():
            if not data.get("valid"):
                continue
            
            trend_label = data.get("trend_label", "").lower()
            if "uptrend" in trend_label:
                uptrending.append({"product": code, **data})
            elif "downtrend" in trend_label:
                downtrending.append({"product": code, **data})
            else:
                stable.append({"product": code, **data})
        
        return {
            "uptrending": sorted(uptrending, key=lambda x: x.get("avg_forecast", 0), reverse=True)[:5],
            "downtrending": sorted(downtrending, key=lambda x: x.get("avg_forecast", 0))[:5],
            "stable": stable[:5],
        }
    
    def _compute_performance_analysis(self, products: Dict[str, Any]) -> Dict[str, Any]:
        """Identify best and worst performing products."""
        valid_products = [
            (code, data) for code, data in products.items()
            if data.get("valid") and data.get("avg_forecast") is not None
        ]
        
        if not valid_products:
            return {"best_performing": None, "low_performing": None}
        
        best = max(valid_products, key=lambda x: x[1]["avg_forecast"])
        worst = min(valid_products, key=lambda x: x[1]["avg_forecast"])
        
        return {
            "best_performing": {"product": best[0], **best[1]},
            "low_performing": {"product": worst[0], **worst[1]},
        }
    
    def _assess_reliability(self, products: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize by data reliability (based on data_points count)."""
        high_conf = []
        med_conf = []
        low_conf = []
        
        for code, data in products.items():
            if not data.get("valid"):
                continue
            
            data_pts = data.get("data_points", 0)
            if data_pts >= 5:
                high_conf.append(code)
            elif data_pts >= 3:
                med_conf.append(code)
            else:
                low_conf.append(code)
        
        return {
            "high_confidence": high_conf,
            "medium_confidence": med_conf,
            "low_confidence": low_conf,
        }
    
    def _detect_anomalies(self, products: Dict[str, Any]) -> List[Dict]:
        """Detect unexpected patterns (e.g., high forecast but downtrend)."""
        anomalies = []
        
        for code, data in products.items():
            if not data.get("valid"):
                continue
            
            forecast = data.get("avg_forecast", 0)
            trend = data.get("trend_pct", 0)
            label = data.get("trend_label", "").lower()
            
            # Anomaly: High forecast but negative trend
            if forecast > 0 and trend < -10 and "downtrend" in label:
                anomalies.append({
                    "type": "high_forecast_downtrend",
                    "product": code,
                    "forecast": forecast,
                    "trend": trend,
                    "description": f"High forecast ({forecast}) but strong downtrend ({trend}%)",
                })
            
            # Anomaly: Very few data points
            if data.get("data_points", 0) < 2:
                anomalies.append({
                    "type": "low_data_points",
                    "product": code,
                    "data_points": data.get("data_points"),
                    "description": "Very few data points for reliability assessment",
                })
        
        return anomalies


# ============================================================================
# BUSINESS AGENT — Interprets data in business context
# ============================================================================

class BusinessAgent(Agent):
    """
    Interprets analysis results in business context.
    
    Responsibilities:
    - Translate metrics into business impact (risk, opportunity)
    - Generate actionable recommendations
    - Identify key decision points
    - Flag products requiring attention
    - Assess portfolio implications
    
    Output:
    {
        "key_insights": List[str],  # High-level business findings
        "recommendations": List[Dict],  # Actionable items
        "risks": List[Dict],  # Potential issues
        "opportunities": List[Dict],  # Business opportunities
        "actions_required": List[Dict],  # Immediate actions
        "business_impact": Dict,  # Overall portfolio assessment
    }
    """
    
    def __init__(self, analyzer: dataset_analyzer.DatasetAnalyzer):
        super().__init__("BusinessAgent", analyzer)
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """Generate business insights and recommendations."""
        reasoning_steps = []
        
        try:
            # Get analysis from previous agents
            analysis_output = agent_input.context.get("analysis_agent_output") if agent_input.context else None

            # Normalize: allow either AgentOutput or plain dict
            if isinstance(analysis_output, AgentOutput):
                analysis_data = analysis_output.data or {}
            elif isinstance(analysis_output, dict):
                # If dict has nested 'data', prefer it; otherwise treat dict as data
                analysis_data = analysis_output.get("data", analysis_output)
            else:
                analysis_data = None

            if not analysis_output:
                reasoning_steps.append(
                    self._log_reasoning_step("No analysis data available, generating generic recommendations")
                )
                return await self._fallback_business_analysis(reasoning_steps)
            
            # analysis_data is now a dict with analysis content
            reasoning_steps.append(
                self._log_reasoning_step("Beginning business interpretation")
            )
            
            # Step 1: Generate key insights
            reasoning_steps.append(self._log_reasoning_step("Generating key business insights"))
            insights = self._generate_insights(analysis_data)
            reasoning_steps.append(
                self._log_reasoning_step(f"Generated {len(insights)} insights")
            )
            
            # Step 2: Identify recommendations
            reasoning_steps.append(self._log_reasoning_step("Formulating recommendations"))
            recommendations = self._formulate_recommendations(analysis_data)
            reasoning_steps.append(
                self._log_reasoning_step(f"Generated {len(recommendations)} recommendations")
            )
            
            # Step 3: Risk assessment
            reasoning_steps.append(self._log_reasoning_step("Assessing risks"))
            risks = self._assess_risks(analysis_data)
            reasoning_steps.append(
                self._log_reasoning_step(f"Identified {len(risks)} potential risks")
            )
            
            # Step 4: Opportunity analysis
            reasoning_steps.append(self._log_reasoning_step("Identifying opportunities"))
            opportunities = self._identify_opportunities(analysis_data)
            reasoning_steps.append(
                self._log_reasoning_step(f"Found {len(opportunities)} opportunities")
            )
            
            # Step 5: Action items
            reasoning_steps.append(self._log_reasoning_step("Defining action items"))
            actions = self._define_actions(analysis_data, insights, risks)
            reasoning_steps.append(
                self._log_reasoning_step(f"Defined {len(actions)} action items")
            )
            
            # Step 6: Overall impact assessment
            impact = self._assess_business_impact(analysis_data, insights, risks)
            reasoning_steps.append(
                self._log_reasoning_step(f"Overall portfolio health: {impact.get('portfolio_health')}")
            )
            
            output_data = {
                "key_insights": insights,
                "recommendations": recommendations,
                "risks": risks,
                "opportunities": opportunities,
                "actions_required": actions,
                "business_impact": impact,
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=True,
            )
        
        except Exception as e:
            self.logger.exception(f"Business analysis failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
    
    async def _fallback_business_analysis(self, reasoning_steps: List[str]) -> AgentOutput:
        """Generate generic business recommendations when analysis unavailable."""
        output_data = {
            "key_insights": [
                "Focus on data-driven decision making",
                "Prioritize products with high data reliability (5+ data points)",
                "Monitor uptrending products for growth opportunities",
            ],
            "recommendations": [
                {
                    "title": "Data Quality Review",
                    "priority": "high",
                    "description": "Ensure products have sufficient data points for reliable analysis",
                }
            ],
            "risks": [],
            "opportunities": [],
            "actions_required": [],
            "business_impact": {"portfolio_health": "unknown", "confidence": "low"},
        }
        
        reasoning_steps.append(
            self._log_reasoning_step("Using fallback business analysis")
        )
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data=output_data,
            reasoning_steps=reasoning_steps,
            validation_passed=True,
        )
    
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate key business insights from analysis."""
        insights = []
        
        # Insight 1: Stability insight
        most_stable = analysis.get("stability_analysis", {}).get("most_stable")
        if most_stable:
            insights.append(
                f"Product {most_stable['product']} offers the most stable forecast "
                f"(trend: {most_stable['trend_pct']:.1f}%), making it a reliable baseline for planning"
            )
        
        # Insight 2: Growth opportunity
        uptrending = analysis.get("trend_analysis", {}).get("uptrending", [])
        if uptrending:
            insights.append(
                f"{len(uptrending)} products are on uptrend, led by {uptrending[0]['product']}; "
                f"consider allocating resources to capitalize on growth"
            )
        
        # Insight 3: Risk exposure
        downtrending = analysis.get("trend_analysis", {}).get("downtrending", [])
        if downtrending:
            insights.append(
                f"{len(downtrending)} products show downtrend; monitor for further deterioration"
            )
        
        # Insight 4: Data quality
        reliability = analysis.get("reliability_assessment", {})
        low_conf = reliability.get("low_confidence", [])
        if low_conf and len(low_conf) > 2:
            insights.append(
                f"{len(low_conf)} products have limited data; "
                f"forecast confidence is lower for these items"
            )
        
        # Insight 5: Performance spread
        perf = analysis.get("performance_analysis", {})
        best = perf.get("best_performing", {})
        worst = perf.get("low_performing", {})
        if best and worst:
            forecast_best = best.get("avg_forecast", 0)
            forecast_worst = worst.get("avg_forecast", 0)
            if forecast_best > 0 and forecast_worst > 0:
                ratio = forecast_best / forecast_worst if forecast_worst != 0 else 0
                insights.append(
                    f"Significant performance variance: best-performing product "
                    f"({best['product']}) forecasts ~{ratio:.1f}x higher than lowest"
                )
        
        return insights
    
    def _formulate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Rec 1: Prioritize stable products
        recommendations.append({
            "title": "Baseline Planning",
            "priority": "high",
            "description": "Use most stable products as baseline for portfolio forecasting",
            "rationale": "Reduces forecast variance and improves planning accuracy",
        })
        
        # Rec 2: Growth focus
        uptrending = analysis.get("trend_analysis", {}).get("uptrending", [])
        if uptrending:
            recommendations.append({
                "title": "Growth Investment",
                "priority": "medium",
                "description": f"Increase investment in uptrending products, particularly {uptrending[0].get('product')}",
                "rationale": "Capitalize on positive market trends before competitors",
            })
        
        # Rec 3: Risk mitigation
        downtrending = analysis.get("trend_analysis", {}).get("downtrending", [])
        if downtrending:
            recommendations.append({
                "title": "Risk Mitigation",
                "priority": "medium",
                "description": f"Review and optimize downtrending product portfolio; consider consolidation",
                "rationale": "Reduce exposure to declining products",
            })
        
        # Rec 4: Data enhancement
        reliability = analysis.get("reliability_assessment", {})
        low_conf = reliability.get("low_confidence", [])
        if low_conf:
            recommendations.append({
                "title": "Data Collection",
                "priority": "low",
                "description": f"Enhance data collection for {len(low_conf)} low-confidence products",
                "rationale": "Improve forecast reliability over time",
            })
        
        return recommendations
    
    def _assess_risks(self, analysis: Dict[str, Any]) -> List[Dict]:
        """Identify business risks."""
        risks = []
        
        # Risk 1: Downtrend exposure
        downtrending = analysis.get("trend_analysis", {}).get("downtrending", [])
        if downtrending:
            risks.append({
                "type": "market_risk",
                "severity": "medium" if len(downtrending) < 3 else "high",
                "description": f"{len(downtrending)} products experiencing downtrend",
                "products": [p.get("product") for p in downtrending],
            })
        
        # Risk 2: Low data reliability
        reliability = analysis.get("reliability_assessment", {})
        low_conf = reliability.get("low_confidence", [])
        if low_conf and len(low_conf) > len(reliability.get("high_confidence", [])):
            risks.append({
                "type": "forecast_risk",
                "severity": "high",
                "description": f"Majority of products have insufficient data points for reliable forecasting",
                "products": low_conf,
            })
        
        # Risk 3: Anomalies
        anomalies = analysis.get("anomalies", [])
        if anomalies:
            risks.append({
                "type": "data_anomaly",
                "severity": "low",
                "description": f"{len(anomalies)} unusual patterns detected in data",
                "details": anomalies,
            })
        
        return risks
    
    def _identify_opportunities(self, analysis: Dict[str, Any]) -> List[Dict]:
        """Identify business opportunities."""
        opportunities = []
        
        # Opp 1: Growth products
        uptrending = analysis.get("trend_analysis", {}).get("uptrending", [])
        if uptrending:
            top_up = uptrending[0]
            opportunities.append({
                "type": "market_expansion",
                "strength": "high",
                "description": f"Product {top_up.get('product')} showing strong uptrend",
                "potential": f"Consider expanding market reach or increasing inventory",
            })
        
        # Opp 2: Performance optimization
        perf = analysis.get("performance_analysis", {})
        best = perf.get("best_performing", {})
        if best:
            opportunities.append({
                "type": "best_practice",
                "strength": "medium",
                "description": f"Product {best.get('product')} outperforming portfolio",
                "potential": "Replicate success factors to other products",
            })
        
        # Opp 3: Portfolio diversification
        reliability = analysis.get("reliability_assessment", {})
        high_conf = reliability.get("high_confidence", [])
        if high_conf and len(high_conf) >= 2:
            opportunities.append({
                "type": "portfolio_stability",
                "strength": "medium",
                "description": f"{len(high_conf)} high-confidence products provide stable base",
                "potential": "Build portfolio around stable performers while exploring growth",
            })
        
        return opportunities
    
    def _define_actions(
        self,
        analysis: Dict[str, Any],
        insights: List[str],
        risks: List[Dict]
    ) -> List[Dict]:
        """Define immediate action items."""
        actions = []
        
        # Action 1: Monitor downtrends
        downtrending = analysis.get("trend_analysis", {}).get("downtrending", [])
        if downtrending:
            actions.append({
                "action": "Weekly Review",
                "priority": "high",
                "owner": "Product Manager",
                "description": "Review performance of downtrending products",
                "target_date": "End of week",
            })
        
        # Action 2: Communicate forecast
        actions.append({
            "action": "Forecast Communication",
            "priority": "medium",
            "owner": "Analytics Team",
            "description": "Publish forecast summary with confidence intervals",
            "target_date": "Immediate",
        })
        
        # Action 3: Address anomalies
        anomalies = analysis.get("anomalies", [])
        if anomalies:
            actions.append({
                "action": "Investigate Anomalies",
                "priority": "low",
                "owner": "Data Team",
                "description": f"Investigate {len(anomalies)} data anomalies",
                "target_date": "This week",
            })
        
        return actions
    
    def _assess_business_impact(
        self,
        analysis: Dict[str, Any],
        insights: List[str],
        risks: List[Dict]
    ) -> Dict[str, Any]:
        """Assess overall business impact."""
        trend_dist = analysis.get("trend_analysis", {})
        uptrending_count = len(trend_dist.get("uptrending", []))
        downtrending_count = len(trend_dist.get("downtrending", []))
        
        # Calculate health score
        if uptrending_count > downtrending_count:
            health = "positive"
        elif downtrending_count > uptrending_count:
            health = "negative"
        else:
            health = "neutral"
        
        # Risk level
        high_risks = sum(1 for r in risks if r.get("severity") == "high")
        risk_level = "high" if high_risks >= 2 else "medium" if high_risks == 1 else "low"
        
        return {
            "portfolio_health": health,
            "risk_level": risk_level,
            "growth_momentum": "positive" if uptrending_count >= 3 else "neutral",
            "forecast_confidence": "high" if high_risks == 0 else "medium",
            "insights_count": len(insights),
        }


# ============================================================================
# ANSWER AGENT — Orchestrates multi-agent output into final answer
# ============================================================================

class AnswerAgent(Agent):
    """
    Composes final human-readable answer integrating all agent insights.
    
    With LLM enhancement (optional):
    - Uses deep chain-of-thought reasoning for strategic insights
    - Identifies patterns and business implications
    - Generates risk warnings and opportunity alerts
    - Falls back to data-only formatting if LLM unavailable
    
    Responsibilities:
    - Synthesize insights from all agents
    - Format answer for human consumption
    - Include relevant metrics and recommendations
    - Maintain clarity and conciseness
    - Preserve traceability to source data
    """
    
    def __init__(self, analyzer: dataset_analyzer.DatasetAnalyzer):
        super().__init__("AnswerAgent", analyzer)
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """Compose final answer from all agent outputs with optional LLM reasoning."""
        reasoning_steps = []
        
        try:
            reasoning_steps.append(
                self._log_reasoning_step("Synthesizing multi-agent insights into final answer")
            )
            
            # Extract all agent outputs
            context = agent_input.context or {}
            data_agent_output = context.get("data_agent_output", {})
            analysis_agent_output = context.get("analysis_agent_output", {})
            business_agent_output = context.get("business_agent_output", {})
            retrieved_docs = agent_input.retrieved_docs
            
            reasoning_steps.append(
                self._log_reasoning_step("Extracting insights from all agents")
            )
            
            # Build base answer from agents
            base_answer = self._compose_base_answer(
                agent_input.query,
                # Normalize: support AgentOutput objects
                data_agent_output.data if isinstance(data_agent_output, AgentOutput) else (data_agent_output if isinstance(data_agent_output, dict) else {}),
                analysis_agent_output.data if isinstance(analysis_agent_output, AgentOutput) else (analysis_agent_output if isinstance(analysis_agent_output, dict) else {}),
                business_agent_output.data if isinstance(business_agent_output, AgentOutput) else (business_agent_output if isinstance(business_agent_output, dict) else {})
            )
            reasoning_steps.append(
                self._log_reasoning_step("Composed base answer from agent insights")
            )
            
            # Try to enhance with LLM reasoning
            final_answer = base_answer
            llm_used = False
            
            if HAS_LLM_REASONER:
                reasoning_steps.append(
                    self._log_reasoning_step("Attempting LLM-enhanced reasoning...")
                )
                
                try:
                    # Prepare agent outputs dict for LLM
                    all_agent_outputs = {
                        "data_agent": {"data": data_agent_output},
                        "analysis_agent": {"data": analysis_agent_output},
                        "business_agent": {"data": business_agent_output},
                    }
                    
                    # Apply LLM reasoning
                    enhanced_answer = await llm_reasoner.apply_llm_reasoning(
                        query=agent_input.query,
                        agent_outputs=all_agent_outputs,
                        retrieved_docs=retrieved_docs,
                        answer_agent_answer=base_answer,
                    )
                    
                    if enhanced_answer and enhanced_answer != base_answer:
                        final_answer = enhanced_answer
                        llm_used = True
                        reasoning_steps.append(
                            self._log_reasoning_step("✓ LLM reasoning applied successfully")
                        )
                    else:
                        reasoning_steps.append(
                            self._log_reasoning_step("LLM reasoning returned empty; using base answer")
                        )
                
                except Exception as e:
                    self.logger.debug(f"LLM reasoning failed: {e}; using base answer")
                    reasoning_steps.append(
                        self._log_reasoning_step(f"LLM reasoning failed ({type(e).__name__}); using base answer")
                    )
            else:
                reasoning_steps.append(
                    self._log_reasoning_step("LLM reasoner not available; using base agent answer only")
                )
            
            reasoning_steps.append(
                self._log_reasoning_step(f"Final answer ready ({len(final_answer)} chars)")
            )
            
            output_data = {
                "answer": final_answer,
                "llm_enhanced": llm_used,
                "base_answer_included": True,
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=output_data,
                reasoning_steps=reasoning_steps,
                validation_passed=True,
            )
        
        except Exception as e:
            self.logger.exception(f"Answer composition failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                data={},
                reasoning_steps=reasoning_steps,
                validation_passed=False,
                error=str(e),
            )
    
    def _compose_base_answer(
        self,
        query: str,
        data_output: Dict,
        analysis_output: Dict,
        business_output: Dict
    ) -> str:
        """Compose base answer from agent outputs before LLM enhancement."""
        answer_sections = []
        
        # Section 1: Direct answer to query
        direct_answer = self._answer_query(
            query,
            data_output,
            analysis_output
        )
        if direct_answer:
            answer_sections.append(direct_answer)
        
        # Section 2: Key findings
        business_data = business_output.get("data", {})
        insights = business_data.get("key_insights", [])
        if insights:
            insights_text = self._format_insights(insights)
            answer_sections.append(insights_text)
        
        # Section 3: Recommendations
        recommendations = business_data.get("recommendations", [])
        if recommendations:
            rec_text = self._format_recommendations(recommendations)
            answer_sections.append(rec_text)
        
        # Section 4: Data quality/confidence note
        data_metrics = data_output.get("key_metrics", {})
        if data_metrics.get("data_quality"):
            quality_text = self._format_data_quality(data_metrics)
            answer_sections.append(quality_text)
        
        return "\n\n".join(answer_sections)
    
    def _answer_query(
        self,
        query: str,
        data_output: Dict,
        analysis_output: Dict
    ) -> Optional[str]:
        """Generate direct answer to user query."""
        query_lower = query.lower()
        
        # Pattern 1: Stability question
        if any(w in query_lower for w in ["stable", "stability", "volatile", "least volatile"]):
            most_stable = analysis_output.get("data", {}).get("stability_analysis", {}).get("most_stable")
            if most_stable:
                return (
                    f"**Most Stable Product:** {most_stable['product']}\n"
                    f"Trend: {most_stable.get('trend_pct', 0):.2f}% (closest to zero)\n"
                    f"Forecast: {most_stable.get('avg_forecast', 0):.2f}\n"
                    f"Confidence: {most_stable.get('data_points', 0)} data points"
                )
        
        # Pattern 2: Performance question
        if any(w in query_lower for w in ["best", "worst", "highest", "lowest", "performance", "top"]):
            perf = analysis_output.get("data", {}).get("performance_analysis", {})
            best = perf.get("best_performing")
            worst = perf.get("low_performing")
            
            if best and worst:
                return (
                    f"**Performance Overview:**\n"
                    f"Best: {best.get('product')} (forecast: {best.get('avg_forecast', 0):.2f})\n"
                    f"Lowest: {worst.get('product')} (forecast: {worst.get('avg_forecast', 0):.2f})\n"
                    f"Variance: {best.get('avg_forecast', 0) / (worst.get('avg_forecast', 1) or 1):.1f}x"
                )
        
        # Pattern 3: Trend question
        if any(w in query_lower for w in ["trend", "uptrend", "downtrend", "growth", "decline"]):
            trends = analysis_output.get("data", {}).get("trend_analysis", {})
            up = trends.get("uptrending", [])
            down = trends.get("downtrending", [])
            
            if up or down:
                answer = "**Trend Analysis:**\n"
                if up:
                    answer += f"Growing: {', '.join([p.get('product') for p in up[:3]])}\n"
                if down:
                    answer += f"Declining: {', '.join([p.get('product') for p in down[:3]])}"
                return answer
        
        # Default: Generic answer
        products_data = data_output.get("key_metrics", {})
        if products_data:
            return (
                f"**Data Analysis Summary:**\n"
                f"Products analyzed: {products_data.get('total_valid_products', 0)}\n"
                f"Data quality: {products_data.get('data_quality', 'unknown')}"
            )
        
        return None
    
    def _format_insights(self, insights: List[str]) -> str:
        """Format insights section."""
        if not insights:
            return ""
        
        lines = ["**Key Insights:**"]
        for i, insight in enumerate(insights[:5], 1):
            lines.append(f"{i}. {insight}")
        
        return "\n".join(lines)
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations section."""
        if not recommendations:
            return ""
        
        lines = ["**Recommendations:**"]
        for rec in recommendations[:3]:
            priority = rec.get("priority", "medium").upper()
            title = rec.get("title", "Action")
            desc = rec.get("description", "")
            lines.append(f"• [{priority}] {title}: {desc}")
        
        return "\n".join(lines)
    
    def _format_data_quality(self, metrics: Dict) -> str:
        """Format data quality assessment."""
        quality = metrics.get("data_quality", "unknown")
        valid = metrics.get("total_valid_products", 0)
        invalid = metrics.get("total_invalid_products", 0)
        
        return (
            f"**Data Quality:** {quality.upper()}\n"
            f"Valid products: {valid} | Data issues: {invalid}"
        )


# ============================================================================
# MULTI-AGENT ORCHESTRATOR — Coordinates all agents
# ============================================================================

class MultiAgentOrchestrator:
    """
    Coordinates data, analysis, business, and answer agents.
    
    Implements orchestration flow:
    1. DataAgent: Extract and validate product codes/metrics
    2. AnalysisAgent: Compute trends, stability, performance
    3. BusinessAgent: Interpret business impact and recommendations
    4. AnswerAgent: Compose final human-readable answer
    
    Supports both serial and parallel execution where dependencies permit.
    """
    
    def __init__(self, analyzer: dataset_analyzer.DatasetAnalyzer):
        """Initialize orchestrator with agents."""
        self.analyzer = analyzer
        self.logger = logging.getLogger("rag.agents.orchestrator")
        
        # Initialize agents
        self.data_agent = DataAgent(analyzer)
        self.analysis_agent = AnalysisAgent(analyzer)
        self.business_agent = BusinessAgent(analyzer)
        self.answer_agent = AnswerAgent(analyzer)
    
    async def orchestrate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        timeout_seconds: float = 30.0,
    ) -> MultiAgentResult:
        """
        Orchestrate all agents to produce final answer.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents from vectorstore
            timeout_seconds: Timeout for entire orchestration
            
        Returns:
            MultiAgentResult with final answer and reasoning trail
        """
        self.logger.info(f"Orchestrating agents for query: {query[:100]}...")
        
        reasoning_trail = {}
        agents_used = []
        
        try:
            # Step 1: Data Agent (must run first)
            self.logger.debug("Running DataAgent")
            start_time = __import__('time').time()
            
            data_input = AgentInput(
                query=query,
                retrieved_docs=retrieved_docs,
                context=None,
            )
            
            data_output = await asyncio.wait_for(
                self.data_agent.reason(data_input),
                timeout=timeout_seconds * 0.2
            )
            reasoning_trail["data_agent"] = data_output
            agents_used.append("DataAgent")
            
            self.logger.debug(f"DataAgent completed: {data_output.success}")
            
            # Step 2: Analysis Agent (depends on DataAgent)
            self.logger.debug("Running AnalysisAgent")
            
            analysis_input = AgentInput(
                query=query,
                retrieved_docs=retrieved_docs,
                context={"data_agent_output": data_output.data},
            )
            
            analysis_output = await asyncio.wait_for(
                self.analysis_agent.reason(analysis_input),
                timeout=timeout_seconds * 0.3
            )
            reasoning_trail["analysis_agent"] = analysis_output
            agents_used.append("AnalysisAgent")
            
            self.logger.debug(f"AnalysisAgent completed: {analysis_output.success}")
            
            # Step 3: Business Agent (depends on AnalysisAgent)
            self.logger.debug("Running BusinessAgent")
            
            business_input = AgentInput(
                query=query,
                retrieved_docs=retrieved_docs,
                context={
                    "data_agent_output": data_output.data,
                    "analysis_agent_output": analysis_output,
                },
            )
            
            business_output = await asyncio.wait_for(
                self.business_agent.reason(business_input),
                timeout=timeout_seconds * 0.3
            )
            reasoning_trail["business_agent"] = business_output
            agents_used.append("BusinessAgent")
            
            self.logger.debug(f"BusinessAgent completed: {business_output.success}")
            
            # Step 4: Answer Agent (depends on all previous)
            self.logger.debug("Running AnswerAgent")
            
            answer_input = AgentInput(
                query=query,
                retrieved_docs=retrieved_docs,
                context={
                    "data_agent_output": data_output.data,
                    "analysis_agent_output": analysis_output,
                    "business_agent_output": business_output,
                },
            )
            
            answer_output = await asyncio.wait_for(
                self.answer_agent.reason(answer_input),
                timeout=timeout_seconds * 0.2
            )
            reasoning_trail["answer_agent"] = answer_output
            agents_used.append("AnswerAgent")
            
            self.logger.debug(f"AnswerAgent completed: {answer_output.success}")
            
            # Determine validation status
            validation_status = "passed"
            if not data_output.validation_passed:
                validation_status = "partial"
            if not answer_output.success:
                validation_status = "fallback"
            
            # Build final result
            final_answer = answer_output.data.get("answer", "Unable to generate answer")
            
            result = MultiAgentResult(
                answer=final_answer,
                source="multi_agent_rag",
                agents_used=agents_used,
                validation=validation_status,
                source_documents=retrieved_docs,
                metadata={
                    "query_length": len(query),
                    "documents_retrieved": len(retrieved_docs),
                    "agents_executed": len(agents_used),
                    "execution_time_s": __import__('time').time() - start_time,
                },
                reasoning_trail=reasoning_trail,
            )
            
            self.logger.info(
                f"Orchestration complete: source={result.source}, "
                f"validation={result.validation}, agents={len(agents_used)}"
            )
            
            return result
        
        except asyncio.TimeoutError:
            self.logger.error(f"Orchestration timed out after {timeout_seconds}s")
            return MultiAgentResult(
                answer="Request processing timed out. Please try a simpler query.",
                source="error",
                agents_used=agents_used,
                validation="fallback",
                source_documents=retrieved_docs,
                metadata={"error": "timeout"},
                reasoning_trail=reasoning_trail,
            )
        
        except Exception as e:
            self.logger.exception(f"Orchestration failed: {e}")
            return MultiAgentResult(
                answer=f"Error during analysis: {str(e)}",
                source="error",
                agents_used=agents_used,
                validation="fallback",
                source_documents=retrieved_docs,
                metadata={"error": str(e)},
                reasoning_trail=reasoning_trail,
            )


# ============================================================================
# PUBLIC API — Convenience function for chatbot integration
# ============================================================================

async def orchestrate_multi_agent_reasoning(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    analyzer: Optional[dataset_analyzer.DatasetAnalyzer] = None,
) -> MultiAgentResult:
    """
    Convenience function to orchestrate all agents.
    
    This is the main entry point for chatbot integration.
    
    Args:
        query: User query
        retrieved_docs: Retrieved documents from RAG vectorstore
        analyzer: DatasetAnalyzer (uses global if not provided)
        
    Returns:
        MultiAgentResult with final answer and full reasoning trail
    """
    if analyzer is None:
        analyzer = dataset_analyzer.get_analyzer()
    
    orchestrator = MultiAgentOrchestrator(analyzer)
    return await orchestrator.orchestrate(query, retrieved_docs)
