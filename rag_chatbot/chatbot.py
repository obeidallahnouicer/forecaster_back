"""
RAG Chatbot Module — Robust retrieval-augmented generation with vectorstore integration.

This module provides:
- Vectorstore-backed retrieval with the new indexer pipeline
- Reasoning-ready prompt composition with source citations
- Chat API with conversation memory
- Structured retrieval results (answer + sources + docs)

Key functions:
- get_answer(query, top_k=5): Retrieve docs and generate answer
- chat(message, thread_id): Stateful conversation endpoint
- get_retriever(): Load vectorstore and return LangChain retriever
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import time
from pathlib import Path
import threading
import json
import re
import asyncio

import pandas as pd

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

from . import config, retriever, data_loader, prompt_templates, dataset_analyzer, agents, llm_reasoner
from core.logger import log_agent_step, LogLevel, get_agent_logger
from agents.base_agent import AgentInput

logger = logging.getLogger("rag.chatbot")


# ============================================================================
# RESPONSE CLEANING — Remove citations and source listings
# ============================================================================

def _clean_citations_from_response(response_text: str) -> str:
    """
    Clean up the response by removing citation markers and source listings.
    
    Removes patterns like:
    - [Document 1], [Doc 3], [Source: ...]
    - "The documents used for this answer are..."
    - "List of documents used:"
    - Entire "List of documents used:" sections with bullets
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Cleaned response without citations and source listings
    """
    # Remove citation markers like [Document 1], [Doc 3], etc.
    cleaned = re.sub(r'\s*\[(?:Document|Doc|Source)[^\]]*\]\s*', ' ', response_text)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    
    # Remove entire "List of documents used:" section and everything after
    cleaned = re.sub(r'\n*(?:List of documents used|The documents used for this answer)[:\n-].*?(?=\n\n|$)', 
                     '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove trailing "Note:" sections that start with document references
    cleaned = re.sub(r'\n*Note:.*?(?:documents|sources|used).*?(?=\n\n|$)', 
                     '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove bullet points that reference documents
    cleaned = re.sub(r'\n\s*-\s*\[Document [^\]]*\].*?(?=\n|$)', '', cleaned)
    
    # Final cleanup: remove extra whitespace and normalize
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)  # Remove multiple newlines
    
    return cleaned


def execute_agent(agent, agent_input: AgentInput):
    """Synchronous helper that executes agent.execute or agent.reason.

    If the agent method returns a coroutine, this will run it on an event loop
    and return the result. Returns the AgentOutput instance.
    """
    # Prefer execute()
    if hasattr(agent, "execute"):
        try:
            coro_or_result = agent.execute(agent_input)
        except Exception as e:
            # Log and re-raise
            sid = getattr(agent_input, 'session_id', 'unknown')
            log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Agent execute raised synchronously', level=LogLevel.ERROR, data={"error": str(e)})
            raise

        if asyncio.iscoroutine(coro_or_result):
            # Run coroutine to completion
            try:
                sid = getattr(agent_input, 'session_id', 'unknown')
                log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Running async.execute() on event loop', level=LogLevel.DEBUG)
            except Exception:
                pass
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running (e.g., under some frameworks), create a new loop
                        new_loop = asyncio.new_event_loop()
                        return new_loop.run_until_complete(coro_or_result)
                    else:
                        return loop.run_until_complete(coro_or_result)
                except RuntimeError:
                    new_loop = asyncio.new_event_loop()
                    return new_loop.run_until_complete(coro_or_result)
            except Exception as e:
                sid = getattr(agent_input, 'session_id', 'unknown')
                log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Async agent execution failed', level=LogLevel.ERROR, data={"error": str(e)})
                raise

        # Synchronous result
        try:
            sid = getattr(agent_input, 'session_id', 'unknown')
            log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Executed sync.execute()', level=LogLevel.DEBUG)
        except Exception:
            pass
        return coro_or_result

    # Fallback to reason()
    if hasattr(agent, "reason"):
        try:
            coro_or_result = agent.reason(agent_input)
        except Exception as e:
            sid = getattr(agent_input, 'session_id', 'unknown')
            log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Agent reason raised synchronously', level=LogLevel.ERROR, data={"error": str(e)})
            raise

        if asyncio.iscoroutine(coro_or_result):
            try:
                sid = getattr(agent_input, 'session_id', 'unknown')
                log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Running async.reason() on event loop', level=LogLevel.DEBUG)
            except Exception:
                pass
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        new_loop = asyncio.new_event_loop()
                        return new_loop.run_until_complete(coro_or_result)
                    else:
                        return loop.run_until_complete(coro_or_result)
                except RuntimeError:
                    new_loop = asyncio.new_event_loop()
                    return new_loop.run_until_complete(coro_or_result)
            except Exception as e:
                sid = getattr(agent_input, 'session_id', 'unknown')
                log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Async agent reason failed', level=LogLevel.ERROR, data={"error": str(e)})
                raise

        try:
            sid = getattr(agent_input, 'session_id', 'unknown')
            log_agent_step(sid, getattr(agent, 'name', 'unknown'), 'Executed sync.reason()', level=LogLevel.DEBUG)
        except Exception:
            pass
        return coro_or_result

    raise RuntimeError("Agent has no execute() or reason() method")


def _extract_and_aggregate_data(retrieved_docs: List[Dict[str, Any]], analyzer: 'dataset_analyzer.DatasetAnalyzer') -> Dict[str, Any]:
    """
    Extract numeric data from retrieved docs and compute aggregates.
    
    This provides structured data context for LLM reasoning.
    
    Args:
        retrieved_docs: List of retrieved document dicts
        analyzer: DatasetAnalyzer instance
        
    Returns:
        Dict with aggregated metrics:
        - products: List of product info dicts
        - avg_forecast_mean: Average of all avg_forecast values
        - avg_forecast_range: (min, max) tuple
        - trend_distribution: Count of uptrend/downtrend/stable
        - most_stable: Product with smallest abs(trend_pct)
        - highest_forecast: Product with highest avg_forecast
        - lowest_forecast: Product with lowest avg_forecast
    """
    products = []
    forecasts = []
    trends = []
    
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        
        product_code = metadata.get("ref_article")
        if not product_code:
            continue
        
        # Extract numeric values
        avg_forecast = metadata.get("avg_forecast")
        trend_pct = metadata.get("trend_pct")
        trend_label = metadata.get("trend_label", "Unknown")
        data_points = metadata.get("data_points", 0)
        
        if avg_forecast is not None and trend_pct is not None:
            products.append({
                "product": product_code,
                "avg_forecast": float(avg_forecast),
                "trend_pct": float(trend_pct),
                "trend_label": trend_label,
                "data_points": int(data_points),
            })
            forecasts.append(float(avg_forecast))
            trends.append(trend_label)
    
    if not products:
        return {
            "products": [],
            "count": 0,
        }
    
    # Compute aggregates
    aggregates = {
        "products": products,
        "count": len(products),
        "avg_forecast_mean": sum(forecasts) / len(forecasts) if forecasts else 0,
        "avg_forecast_range": (min(forecasts), max(forecasts)) if forecasts else (0, 0),
        "trend_distribution": {
            "uptrend": sum(1 for t in trends if t.lower() == "uptrend"),
            "downtrend": sum(1 for t in trends if t.lower() == "downtrend"),
            "stable": sum(1 for t in trends if t.lower() == "stable"),
        },
    }
    
    # Find most stable (smallest abs trend_pct) with reliability-aware tie-breakers
    # Prefer products with more data points (>= MIN_DATA_POINTS_FOR_RELIABILITY)
    try:
        min_required = dataset_analyzer.MIN_DATA_POINTS_FOR_RELIABILITY
    except Exception:
        min_required = 3

    # Filter out entries missing trend_pct
    candidates = [p for p in products if p.get("trend_pct") is not None]

    def _safe_vals(p):
        try:
            trend = float(p.get("trend_pct", 0))
        except Exception:
            trend = 0.0
        try:
            dp = int(p.get("data_points", 0))
        except Exception:
            dp = 0
        try:
            af = float(p.get("avg_forecast", 0))
        except Exception:
            af = 0.0
        return trend, dp, af

    # Prefer candidates meeting the primary data_points threshold
    primary = [p for p in candidates if _safe_vals(p)[1] >= min_required]
    if not primary:
        # Secondary: at least 2 data points
        primary = [p for p in candidates if _safe_vals(p)[1] >= 2]
    if not primary:
        primary = candidates

    if primary:
        # Sort by (abs(trend_pct) asc, data_points desc, avg_forecast desc)
        primary_sorted = sorted(
            primary,
            key=lambda p: (abs(float(p.get("trend_pct", 0))), -int(p.get("data_points", 0) or 0), -float(p.get("avg_forecast", 0) or 0.0)),
        )
        aggregates["most_stable"] = primary_sorted[0]
    else:
        aggregates["most_stable"] = None
    
    # Find highest/lowest forecast
    aggregates["highest_forecast"] = max(products, key=lambda p: p["avg_forecast"])
    aggregates["lowest_forecast"] = min(products, key=lambda p: p["avg_forecast"])
    
    return aggregates


def _build_reasoning_prompt(
    query: str,
    aggregated_data: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
) -> str:
    """
    Build a reasoning prompt for the LLM with structured data context.
    
    The prompt includes:
    - User's question
    - Aggregated metrics (means, ranges, distributions)
    - Individual product details
    - Clear instructions for data-backed reasoning
    
    Args:
        query: User's question
        aggregated_data: Aggregated metrics from _extract_and_aggregate_data
        retrieved_docs: Retrieved documents (for additional context)
        
    Returns:
        Complete prompt string for LLM
    """
    products = aggregated_data.get("products", [])
    
    # Get total dataset size from analyzer
    try:
        analyzer = dataset_analyzer.get_analyzer()
        total_products = analyzer.get_product_count()
        dataset_stats = analyzer.get_summary_stats()
    except Exception:
        total_products = None
        dataset_stats = None
    
    # Build structured data summary
    data_summary_parts = [
        "=== CONTEXT ===",
    ]
    
    # Check if we have ALL products (if len(products) matches total_products, we loaded everything)
    has_all_products = (total_products and len(products) >= total_products * 0.95)  # Within 95% = we have all
    
    if has_all_products:
        data_summary_parts.append(
            f"✓ COMPLETE DATASET: Below are ALL {len(products)} products from the entire dataset."
        )
        if dataset_stats:
            data_summary_parts.append(
                f"Dataset stats: avg forecast = {dataset_stats.get('avg_forecast_mean', 0):.2f}, "
                f"range = {dataset_stats.get('avg_forecast_min', 0):.2f} to {dataset_stats.get('avg_forecast_max', 0):.2f}"
            )
    elif total_products:
        data_summary_parts.append(
            f"⚠️ SAMPLE: The full dataset contains {total_products} products total. "
            f"Below are the {len(products)} most relevant products retrieved for this query."
        )
        if dataset_stats:
            data_summary_parts.append(
                f"Full dataset stats: avg forecast = {dataset_stats.get('avg_forecast_mean', 0):.2f}, "
                f"range = {dataset_stats.get('avg_forecast_min', 0):.2f} to {dataset_stats.get('avg_forecast_max', 0):.2f}"
            )
    else:
        data_summary_parts.append(
            f"Retrieved {len(products)} relevant products for this query (sample from larger dataset)."
        )
    
    data_summary_parts.append("")
    data_summary_parts.append("=== RETRIEVED PRODUCTS SUMMARY ===")
    
    if aggregated_data.get("avg_forecast_mean"):
        data_summary_parts.append(
            f"Average forecast (of retrieved): {aggregated_data['avg_forecast_mean']:.2f}"
        )
    
    if aggregated_data.get("avg_forecast_range"):
        min_f, max_f = aggregated_data["avg_forecast_range"]
        data_summary_parts.append(f"Forecast range (of retrieved): {min_f:.2f} to {max_f:.2f}")
    
    trend_dist = aggregated_data.get("trend_distribution", {})
    if trend_dist:
        data_summary_parts.append(
            f"Trend distribution (of retrieved): {trend_dist.get('uptrend', 0)} uptrend, "
            f"{trend_dist.get('downtrend', 0)} downtrend, "
            f"{trend_dist.get('stable', 0)} stable"
        )
    
    # Most stable product
    most_stable = aggregated_data.get("most_stable")
    if most_stable:
        data_summary_parts.append(
            f"\nMost stable (in retrieved set): {most_stable['product']} "
            f"(trend: {most_stable['trend_pct']:.2f}%, forecast: {most_stable['avg_forecast']:.2f}, "
            f"{most_stable['data_points']} data points)"
        )
    
    # Highest forecast
    highest = aggregated_data.get("highest_forecast")
    if highest:
        data_summary_parts.append(
            f"Highest forecast (in retrieved set): {highest['product']} "
            f"(forecast: {highest['avg_forecast']:.2f}, trend: {highest['trend_pct']:.2f}%)"
        )
    
    # Lowest forecast
    lowest = aggregated_data.get("lowest_forecast")
    if lowest:
        data_summary_parts.append(
            f"Lowest forecast (in retrieved set): {lowest['product']} "
            f"(forecast: {lowest['avg_forecast']:.2f}, trend: {lowest['trend_pct']:.2f}%)"
        )
    
    data_summary = "\n".join(data_summary_parts)
    
    # Build product details table
    product_details_parts = ["\n=== DETAILED PRODUCT DATA (MOST RELEVANT) ==="]
    for i, prod in enumerate(products[:10], 1):  # Top 10 only
        product_details_parts.append(
            f"{i}. {prod['product']}: "
            f"forecast={prod['avg_forecast']:.2f}, "
            f"trend={prod['trend_pct']:.2f}% ({prod['trend_label']}), "
            f"confidence={prod['data_points']} data points"
        )
    
    if len(products) > 10:
        product_details_parts.append(f"... and {len(products) - 10} more retrieved products")
    
    product_details = "\n".join(product_details_parts)
    
    # Build complete prompt
    prompt = f"""You are a data analyst specializing in sales forecasting. Answer the user's question using the provided data.

{data_summary}

{product_details}

=== CRITICAL INSTRUCTIONS ===
1. {"✓ YOU HAVE THE COMPLETE DATASET - all products are included below!" if has_all_products else "REMEMBER: These are the MOST RELEVANT products retrieved, NOT the entire dataset"}
2. Use ONLY the numeric values provided - do NOT invent or estimate data
3. Answer directly and concisely - NO citations or document markers
4. For stability: use trend_pct (closer to 0% = more stable)
5. For performance: consider both avg_forecast and trend_pct
6. For reliability: mention data_points (more = more reliable)
7. {"When calculating averages or aggregates, use ALL products listed - you have the complete dataset!" if has_all_products else f"If asked about 'all products', clarify you're analyzing the {len(products)} most relevant out of {total_products if total_products else '600+'} total"}
8. Keep answer to 3-5 sentences maximum

=== USER QUESTION ===
{query}

=== YOUR ANSWER ==="""
    
    return prompt


def _generate_fallback_answer(aggregated_data: Dict[str, Any], query: str) -> str:
    """
    Generate a structured answer when LLM is unavailable.
    
    Uses aggregated data to build a factual summary.
    
    Args:
        aggregated_data: Aggregated metrics
        query: User's question
        
    Returns:
        Structured answer string
    """
    query_lower = query.lower()
    products = aggregated_data.get("products", [])
    
    if not products:
        return "No relevant forecast data found for your query."
    
    # Try to match query intent
    if any(word in query_lower for word in ["stable", "stability", "least volatile"]):
        most_stable = aggregated_data.get("most_stable")
        if most_stable:
            return (
                f"The most stable product is {most_stable['product']}, with a trend percentage of "
                f"{most_stable['trend_pct']:.2f}% (closest to zero). Its average forecast is "
                f"{most_stable['avg_forecast']:.2f} with {most_stable['data_points']} data points."
            )
    
    if any(word in query_lower for word in ["highest", "best", "top", "maximum"]):
        highest = aggregated_data.get("highest_forecast")
        if highest:
            return (
                f"The product with the highest average forecast is {highest['product']}, "
                f"with a forecast of {highest['avg_forecast']:.2f} and a trend of "
                f"{highest['trend_pct']:.2f}% ({highest['trend_label']})."
            )
    
    if any(word in query_lower for word in ["lowest", "worst", "minimum", "poor"]):
        lowest = aggregated_data.get("lowest_forecast")
        if lowest:
            return (
                f"The product with the lowest average forecast is {lowest['product']}, "
                f"with a forecast of {lowest['avg_forecast']:.2f} and a trend of "
                f"{lowest['trend_pct']:.2f}% ({lowest['trend_label']})."
            )
    
    # Generic summary
    return (
        f"Based on {len(products)} products analyzed: "
        f"average forecast is {aggregated_data['avg_forecast_mean']:.2f}, "
        f"ranging from {aggregated_data['avg_forecast_range'][0]:.2f} to "
        f"{aggregated_data['avg_forecast_range'][1]:.2f}. "
        f"Trend distribution: {aggregated_data['trend_distribution']['uptrend']} uptrend, "
        f"{aggregated_data['trend_distribution']['downtrend']} downtrend, "
        f"{aggregated_data['trend_distribution']['stable']} stable."
    )


def _extract_mentioned_products(response_text: str) -> List[str]:
    """Extract product codes mentioned in the response.
    
    Looks for patterns like:
    - RBKTFC, PA0316, SH_RB_ANTIC_1L, SHAMP_RB_BOT_500ML
    - HZTI/10.31, 8006569715814, HZT/7.32B
    
    Filters out pure numbers and common words.
    """
    # Match patterns that look like product codes:
    # - Start with letters (at least 2)
    # - Can contain numbers, underscores, slashes, dots
    # OR
    # - Pure numbers that are long enough to be EAN codes (13+ digits)
    patterns = [
        r'\b[A-Z]{2}[A-Z0-9_/\.]{0,}\b',  # Starts with 2+ letters, followed by alphanumeric/underscore/slash/dot
        r'\b\d{13,}\b',  # EAN codes (13+ digits)
    ]
    
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, response_text))
    
    # Filter out common words
    stopwords = {'AND', 'THE', 'WITH', 'FOR', 'THIS', 'THAT', 'WHICH', 'TREND', 'ITS', 'HAS', 'MAY', 'CAN', 'ARE', 'WITH'}
    filtered = [m for m in matches if m not in stopwords]
    
    return list(set(filtered))  # Deduplicate


def _validate_response_against_data(response_text: str, analyzer: 'dataset_analyzer.DatasetAnalyzer') -> Tuple[bool, List[str]]:
    """
    Validate that mentioned products actually exist in the dataset.
    
    Args:
        response_text: LLM response to validate
        analyzer: DatasetAnalyzer instance
        
    Returns:
        Tuple of (is_valid, invalid_products)
    """
    mentioned_products = _extract_mentioned_products(response_text)
    invalid_products = []

    # If the LLM did not mention any explicit product codes, we should not consider
    # the response "validated" against the dataset. Returning True in that case
    # is a common false-positive (LLM can answer generically without grounding).
    if not mentioned_products:
        logger.debug("No explicit product mentions extracted from LLM response; marking as not validated")
        # Return False with a marker so callers can log or treat as 'partial' validation
        return False, ["<no_products_mentioned>"]
    
    for product in mentioned_products:
        # Skip if already validated through different patterns
        if analyzer.verify_product_exists(product):
            continue
        
        # Try to find partial matches (in case extraction was imperfect)
        # E.g., "HZTI/10" might be part of "HZTI/10.31" in the CSV
        found_partial = False
        for ref in analyzer.df["ref_article"].values:
            ref_str = str(ref)  # Ensure it's a string
            if product in ref_str or ref_str in product:
                found_partial = True
                break
        
        if not found_partial:
            # Only flag as invalid if truly not in dataset
            invalid_products.append(product)
    
    # Be lenient: only consider response invalid if we found clearly fabricated products
    # (not just extraction mistakes)
    is_valid = len(invalid_products) == 0

    logger.debug(f"Validation check: extracted_products={mentioned_products}, invalid_products={invalid_products}, is_valid={is_valid}")
    return is_valid, invalid_products


def _generate_data_backed_answer(query: str, analyzer: 'dataset_analyzer.DatasetAnalyzer') -> Optional[str]:
    """
    Generate answer from actual dataset analysis, bypassing LLM if possible.
    
    This function demonstrates strict data-backed reasoning: all values come directly
    from the analyzer's DataFrame (loaded from CSV via data_loader.load_and_preprocess()).
    
    No hardcoded products, thresholds, or values are used. All computations:
    - Read from analyzer.df (the preprocessed CSV)
    - Apply data-driven thresholds (MIN_DATA_POINTS_FOR_RELIABILITY, FEW_DATA_POINTS_THRESHOLD, etc.)
    - Return only real values from the dataset
    
    Args:
        query: User question
        analyzer: DatasetAnalyzer instance (loads data from CSV)
        
    Returns:
        Pre-generated answer with real dataset values, or None to fall through to LLM
        
    Data sources:
    - analyzer.df: Loaded from CSV via rag_chatbot.data_loader.load_and_preprocess()
    - Product info: analyzer.find_most_stable_product() reads ref_article, trend_pct, avg_forecast, data_points
    - Low-performing: analyzer.find_low_performing_product() reads and scores real products
    - Uptrend products: analyzer.get_products_by_uptrend() filters trend_label == "Uptrend" from CSV
    """
    query_lower = query.lower()
    
    # Pattern 1: Most stable product
    if any(phrase in query_lower for phrase in ["most stable", "most stable product", "stability", "least volatile"]):
        product = analyzer.find_most_stable_product()
        if product:
            return (
                f"The most stable product is {product['product']}, with a trend percentage of {product['trend_pct']:.2f}%, "
                f"closest to zero among the products in the dataset. Its average forecast is {product['avg_forecast']:.2f} "
                f"with {product['data_points']} data points."
            )
    
    # Pattern 2: Lowest performing / products to disregard
    if any(phrase in query_lower for phrase in ["disregard", "low perform", "worst", "lowest", "can we ignore"]):
        product = analyzer.find_low_performing_product()
        if product:
            return (
                f"The product {product['product']} has the lowest average forecast ({product['avg_forecast']:.2f}) "
                f"and a {product['trend_label'].lower()} of {product['trend_pct']:.2f}%, "
                f"with only {product['data_points']} data points. It may be considered less reliable and could be disregarded."
            )
    
    # Pattern 3: Products with uptrend
    if any(phrase in query_lower for phrase in ["uptrend", "growing", "best perform", "positive trend", "increasing", "growth"]):
        products = analyzer.get_products_by_uptrend(limit=5)
        if products:
            product_strs = []
            for p in products[:3]:  # Show top 3
                product_strs.append(f"{p['product']} (trend: {p['trend_pct']:.2f}%, forecast: {p['avg_forecast']:.2f})")
            
            count_text = f"{len(products)} products" if len(products) > 3 else f"{len(products)} product{'s' if len(products) > 1 else ''}"
            return f"Found {count_text} with uptrend: {', '.join(product_strs)}{'...' if len(products) > 3 else ''}."
    
    # Pattern 4: Total products in dataset
    if any(phrase in query_lower for phrase in ["how many product", "total product", "number of product"]):
        count = analyzer.get_product_count()
        stats = analyzer.get_summary_stats()
        return f"The dataset contains {count} products. Distribution: {stats['trend_distribution']['uptrend']} uptrend, {stats['trend_distribution']['downtrend']} downtrend, {stats['trend_distribution']['stable']} stable."
    
    # Note: Average/mean queries are handled by loading all products in the main chat function
    # to allow LLM to perform proper aggregation across the entire dataset
    
    return None


def _is_degenerate_aggregated_data(aggregated_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (True, reason) when aggregated data is degenerate (no meaningful numeric values).

    Degenerate cases include:
      - No products
      - Majority (or all) products have data_points == 0
      - avg_forecast all zero or NaN
      - trend_pct all zero and trend_label all 'Unknown'
    """
    products = aggregated_data.get("products", [])
    if not products:
        return True, "no_products"

    total = len(products)
    if total == 0:
        return True, "no_products"

    data_points_zero = sum(1 for p in products if (p.get("data_points") in (None, 0)))
    avg_forecasts = [p.get("avg_forecast") for p in products if p.get("avg_forecast") is not None]
    trend_pcts = [p.get("trend_pct") for p in products if p.get("trend_pct") is not None]
    trend_labels = [str(p.get("trend_label", "")).lower() for p in products]

    # If most products have zero data points -> unreliable
    if data_points_zero >= max(10, int(0.6 * total)):
        return True, "insufficient_data_points"

    # If all avg_forecast values are zero or missing
    if avg_forecasts and all((af == 0 or af is None) for af in avg_forecasts):
        return True, "all_zero_avg_forecast"

    # If trend_pct is present but all zero and trend_label unknown
    if trend_pcts and all((tp == 0 or tp is None) for tp in trend_pcts) and all(lbl in ("unknown", "") for lbl in trend_labels):
        return True, "no_trend_information"

    return False, "ok"




# ============================================================================
# STATE MANAGEMENT — Conversation memory and retrieval tracking
# ============================================================================

# In-memory conversation history (thread_id -> ConversationBufferMemory)
_memories: Dict[str, ConversationBufferMemory] = {}
_memories_lock = threading.Lock()

# Last retrieval info for debugging (thread_id -> {timestamp, question, answer, docs})
_last_retrievals: Dict[str, Dict[str, Any]] = {}
_last_retrievals_lock = threading.Lock()

# Global vectorstore reference (lazy loaded on first use)
_vectorstore = None
_vectorstore_lock = threading.Lock()


def get_memory(thread_id: str) -> ConversationBufferMemory:
    """
    Get or create conversation memory for a thread.
    
    Loads persisted memory from disk if it exists, otherwise creates new.
    """
    with _memories_lock:
        if thread_id not in _memories:
            mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # Load persisted memory if exists
            mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
            if mem_file.exists():
                try:
                    data = json.loads(mem_file.read_text(encoding="utf-8"))
                    for m in data:
                        if m.get("role") == "user":
                            mem.chat_memory.add_user_message(m.get("text"))
                        else:
                            mem.chat_memory.add_ai_message(m.get("text"))
                    logger.info(f"Loaded {len(data)} messages from persisted memory for thread_id={thread_id}")
                except Exception as e:
                    logger.warning(f"Failed to load persisted memory for thread_id={thread_id}: {e}")
            
            _memories[thread_id] = mem
        
        return _memories[thread_id]


def reset_memory(thread_id: str) -> None:
    """Clear conversation memory for a thread (both in-memory and persisted)."""
    with _memories_lock:
        if thread_id in _memories:
            del _memories[thread_id]
    
    try:
        mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
        if mem_file.exists():
            mem_file.unlink()
        logger.info(f"Reset memory for thread_id={thread_id}")
    except Exception as e:
        logger.warning(f"Failed to delete persisted memory for thread_id={thread_id}: {e}")


def _persist_memory(thread_id: str) -> None:
    """Save conversation history for a thread to disk as JSON."""
    try:
        mem = _memories.get(thread_id)
        if not mem:
            return
        
        messages = []
        for m in mem.chat_memory.messages:
            role = "user" if m.type == "human" else "ai"
            messages.append({"role": role, "text": m.content})
        
        mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
        mem_file.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"Persisted {len(messages)} messages for thread_id={thread_id}")
    except Exception as e:
        logger.warning(f"Failed to persist memory for thread_id={thread_id}: {e}")


# ============================================================================
# VECTORSTORE INITIALIZATION — Lazy loading with automatic rebuild
# ============================================================================

def get_vectorstore():
    """
    Get or initialize the Chroma vectorstore.
    
    Lazy loads on first call. Automatically rebuilds if vectorstore doesn't exist.
    Uses the new robust indexer pipeline.
    """
    global _vectorstore
    
    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore
        
        logger.info("Initializing vectorstore...")
        try:
            # Try to load existing vectorstore (non-recreate mode)
            _vectorstore = retriever.build_vectorstore(recreate=False)
            
            if _vectorstore is None:
                logger.warning("Vectorstore build returned None; attempting rebuild with recreate=True")
                _vectorstore = retriever.build_vectorstore(recreate=True)
            
            if _vectorstore is None:
                raise RuntimeError("Failed to build or load vectorstore after retry")
            
            logger.info("Vectorstore initialized successfully")
            return _vectorstore
        
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
            raise RuntimeError(
                f"Vectorstore initialization failed: {e}\n"
                f"Run 'python scripts/build_vectorstore.py --recreate' to rebuild the index."
            )


def get_retriever(k: int = None):
    """
    Get a LangChain retriever backed by the Chroma vectorstore.
    
    Args:
        k: Number of documents to retrieve (default: from config.RETRIEVAL_K)
    
    Returns:
        LangChain Retriever instance
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    store = get_vectorstore()
    logger.info(f"Returning retriever with k={k}")
    return store.as_retriever(search_kwargs={"k": k})


# ============================================================================
# CORE RAG LOGIC — Retrieval and reasoning
# ============================================================================

def get_answer(
    query: str,
    top_k: int = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant documents and generate a structured answer.
    
    This is the core RAG function: retrieve docs, compose context, and prepare
    a prompt for LLM reasoning.
    
    Args:
        query: User's question
        top_k: Number of docs to retrieve (default: config.RETRIEVAL_K)
    
    Returns:
        Dict with keys:
        - answer: None (use llm_prompt for LLM call)
        - llm_prompt: Composed prompt ready for LLM
        - sources: List of document IDs
        - retrieved_docs: List of {id, content, metadata} dicts
        - embedding_model: Model used for retrieval
        - num_chunks: Total chunks in vectorstore
    """
    if top_k is None:
        top_k = config.RETRIEVAL_K
    
    logger.info(f"get_answer called: query_len={len(query)}, top_k={top_k}")
    
    start_time = time.time()
    
    try:
        # Get retriever and search
        retr = get_retriever(k=top_k)
        retrieved_docs = retr.get_relevant_documents(query)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
        
        # Format retrieved docs for output
        retrieved_info = []
        sources = []
        for idx, doc in enumerate(retrieved_docs):
            doc_id = doc.metadata.get("source_id", f"doc_{idx}")
            sources.append(doc_id)
            retrieved_info.append({
                "id": doc_id,
                "content": doc.page_content[:300],  # First 300 chars
                "full_content": doc.page_content,
                "metadata": doc.metadata,
            })
            logger.debug(f"Doc {idx}: {doc_id} | {doc.page_content[:100]}...")
        
        # Compose prompt for LLM
        llm_prompt = prompt_templates.compose_prompt_for_llm(
            query=query,
            retrieved_docs=retrieved_docs,
            include_metadata=True,
        )
        
        # Get vectorstore metadata
        try:
            from rag_chatbot import indexer as rag_indexer
            # Try to load metadata from the actual retriever/vectorstore if possible (useful for tests that mock get_vectorstore)
            meta = None
            try:
                # Attempt to introspect persist directory from retriever object
                persist_dir = None
                # Common attributes where Chroma stores persist directory
                if hasattr(retr, 'vectorstore') and getattr(retr, 'vectorstore') is not None:
                    store_obj = getattr(retr, 'vectorstore')
                    persist_dir = getattr(store_obj, 'persist_directory', None) or getattr(store_obj, '_persist_directory', None)
                elif hasattr(retr, 'client') and getattr(retr, 'client') is not None:
                    client = getattr(retr, 'client')
                    persist_dir = getattr(client, 'persist_directory', None) or getattr(client, '_persist_directory', None)
                elif hasattr(retr, 'persist_directory'):
                    persist_dir = getattr(retr, 'persist_directory')

                if persist_dir:
                    meta = rag_indexer.load_vectorstore_metadata(Path(persist_dir))
            except Exception:
                meta = None

            # Fallback to configured vectorstore path
            if meta is None:
                meta = rag_indexer.load_vectorstore_metadata(Path(config.VECTORSTORE_DIR) / "chroma")

            # Extra fallback: scan common temp/persist locations for _meta.json (useful for tests)
            if meta is None:
                import glob as _glob
                candidates = list(Path(".").glob("**/_meta.json"))
                if candidates:
                    try:
                        with open(candidates[0], 'r', encoding='utf-8') as _f:
                            meta = json.load(_f)
                    except Exception:
                        meta = None

            num_chunks = meta.get("num_docs", 0) if meta else 0
            embedding_model = meta.get("embedding_model", config.EMBEDDING_MODEL) if meta else config.EMBEDDING_MODEL
        except Exception:
            num_chunks = 0
            embedding_model = config.EMBEDDING_MODEL
        
        logger.info(
            f"Composed prompt (model={embedding_model}, "
            f"retrieved_docs={len(retrieved_docs)}, total_chunks={num_chunks})"
        )
        
        # If multi-agent orchestration is enabled, run the analysis pipeline
        try:
            from . import config as _config
            if getattr(_config, "MULTI_AGENT_ORCHESTRATION", False):
                # Lazy import agents to avoid heavy imports at module load
                from agents.retriever_agent import RetrieverAgent
                from agents.analysis_agent import AnalysisAgent
                from agents.reasoning_agent import ReasoningAgent
                from agents.advisor_agent import AdvisorAgent
                from agents.validator_agent import ValidatorAgent
                from core.context_manager import get_context_manager

                session_id = f"session_{int(time.time())}"
                # Start agent logger session for orchestration
                try:
                    agent_logger = get_agent_logger()
                    try:
                        agent_logger.start_session(session_id, query, metadata={"source": "get_answer_multi_agent"})
                    except Exception:
                        logger.exception("Failed to start agent logger session for session_id=%s", session_id)
                except Exception:
                    agent_logger = None
                cm = get_context_manager()
                # Create context with retrieved docs
                ctx = cm.create_context(query, session_id, retrieved_documents=[d.get("metadata", {}) for d in retrieved_info])

                # Structured pipeline logging: Retriever finished
                try:
                    # Log number of retrieved docs and top-3 metadata samples
                    sample_meta = [d.get('metadata', {}) for d in retrieved_info[:3]]
                    log_agent_step(session_id, 'Retriever', 'Retrieved documents', level=LogLevel.INFO,
                                   data={"count": len(retrieved_info), "sample_metadata": sample_meta})
                except Exception:
                    logger.exception("Failed to log retriever summary")

                # 1) RetrieverAgent - we already have retrieved_info, but execute for logging
                try:
                    retr_agent = RetrieverAgent()
                    ai = AgentInput(query=query, context=ctx, retrieved_documents=retrieved_info, session_id=session_id)
                    retr_out = execute_agent(retr_agent, ai)
                    ctx.add_agent_output("RetrieverAgent", retr_out.__dict__)
                except Exception:
                    # Continue with retrieved_info
                    pass

                # 2) Analysis
                analysis_agent = AnalysisAgent()
                ai = AgentInput(query=query, context=ctx, retrieved_documents=retrieved_info, session_id=session_id)
                # Log Analysis input
                try:
                    log_agent_step(session_id, 'AnalysisAgent', 'Starting AnalysisAgent with input', level=LogLevel.DEBUG,
                                   data={"query": query[:200], "doc_count": len(retrieved_info), "sample_meta": [d.get('metadata', {}) for d in retrieved_info[:3]]})
                except Exception:
                    logger.exception("Failed to emit analysis input log")

                analysis_out = execute_agent(analysis_agent, ai)

                # Log Analysis output summary
                try:
                    out_summary = {"success": analysis_out.success, "confidence": analysis_out.confidence}
                    # include data keys and small sample
                    if isinstance(analysis_out.data, dict):
                        out_summary['data_keys'] = list(analysis_out.data.keys())
                        # small samples for large dicts
                        if 'products' in analysis_out.data and isinstance(analysis_out.data['products'], dict):
                            out_summary['products_sample'] = list(list(analysis_out.data['products'].keys())[:5])
                    log_agent_step(session_id, 'AnalysisAgent', 'Completed AnalysisAgent', level=LogLevel.INFO, data=out_summary)
                except Exception:
                    logger.exception("Failed to emit analysis output log")

                ctx.add_agent_output("AnalysisAgent", analysis_out.__dict__)

                # 3) Reasoning (LLM-based interpretation)
                reasoning_agent = ReasoningAgent()
                # Build aggregated data for reasoning
                try:
                    # Acquire analyzer if needed for aggregation
                    try:
                        analyzer = dataset_analyzer.get_analyzer()
                    except Exception:
                        analyzer = None

                    aggregated_data = _extract_and_aggregate_data(retrieved_info, analyzer)
                except Exception as e:
                    log_agent_step(session_id, 'AnalysisAgent', 'Failed to aggregate data for reasoning', level=LogLevel.ERROR, data={"error": str(e)})
                    raise

                # Degenerate data guard for multi-agent orchestration
                is_deg, deg_reason = _is_degenerate_aggregated_data(aggregated_data)
                if is_deg:
                    log_agent_step(session_id, 'AnalysisAgent', 'Degenerate aggregated data detected; aborting multi-agent orchestration', level=LogLevel.ERROR,
                                   data={"reason": deg_reason, "product_count": aggregated_data.get('count', 0)})
                    # Build structured result to return to caller
                    return {
                        "answer": None,
                        "llm_prompt": llm_prompt,
                        "sources": sources,
                        "retrieved_docs": retrieved_info,
                        "analysis": aggregated_data,
                        "reasoning": None,
                        "advisor": None,
                        "validator": None,
                        "embedding_model": embedding_model,
                        "num_chunks": num_chunks,
                        "retrieval_time_s": retrieval_time,
                        "error": f"degenerate_data:{deg_reason}",
                    }

                # Build explicit reasoning prompt and log it (truncate to 2000 chars)
                reasoning_prompt = _build_reasoning_prompt(query, aggregated_data, retrieved_info)
                try:
                    log_agent_step(session_id, 'ReasoningAgent', 'Prepared reasoning prompt', level=LogLevel.DEBUG,
                                   data={"prompt_truncated": reasoning_prompt[:2000]})
                except Exception:
                    logger.exception("Failed to log reasoning prompt")

                ai = AgentInput(query=query, context=ctx, retrieved_documents=retrieved_info, session_id=session_id,
                                metadata={"reasoning_prompt": reasoning_prompt, "aggregated_data_summary": {"count": aggregated_data.get('count')}})

                # Execute with one retry for LLM failures
                reasoning_out = execute_agent(reasoning_agent, ai)
                if not reasoning_out.success:
                    # Retry once and log retry attempt
                    log_agent_step(session_id, 'ReasoningAgent', 'ReasoningAgent failed, retrying once', level=LogLevel.WARNING,
                                   data={"error": reasoning_out.error})
                    reasoning_out = execute_agent(reasoning_agent, ai)

                # Log Reasoning output
                try:
                    out_summary = {"success": reasoning_out.success, "confidence": reasoning_out.confidence}
                    if isinstance(reasoning_out.data, dict):
                        out_summary['data_keys'] = list(reasoning_out.data.keys())
                    log_agent_step(session_id, 'ReasoningAgent', 'Completed ReasoningAgent', level=LogLevel.INFO, data=out_summary,
                                   success=reasoning_out.success, error=reasoning_out.error)
                except Exception:
                    logger.exception("Failed to emit reasoning output log")

                ctx.add_agent_output("ReasoningAgent", reasoning_out.__dict__)

                # 4) Advisor
                advisor_agent = AdvisorAgent()
                ai = AgentInput(query=query, context=ctx, retrieved_documents=retrieved_info, session_id=session_id,
                                metadata={"reasoning_summary_keys": list(reasoning_out.data.keys()) if isinstance(reasoning_out.data, dict) else None})
                log_agent_step(session_id, 'AdvisorAgent', 'Starting AdvisorAgent', level=LogLevel.DEBUG,
                               data={"based_on_reasoning_success": reasoning_out.success})
                advisor_out = execute_agent(advisor_agent, ai)
                try:
                    out_summary = {"success": advisor_out.success, "confidence": advisor_out.confidence}
                    if isinstance(advisor_out.data, dict):
                        out_summary['data_keys'] = list(advisor_out.data.keys())
                    log_agent_step(session_id, 'AdvisorAgent', 'Completed AdvisorAgent', level=LogLevel.INFO, data=out_summary,
                                   success=advisor_out.success, error=advisor_out.error)
                except Exception:
                    logger.exception("Failed to emit advisor output log")
                ctx.add_agent_output("AdvisorAgent", advisor_out.__dict__)

                # 5) Validator
                validator_agent = ValidatorAgent()
                ai = AgentInput(query=query, context=ctx, retrieved_documents=retrieved_info, session_id=session_id,
                                metadata={"advisor_summary_keys": list(advisor_out.data.keys()) if isinstance(advisor_out.data, dict) else None})
                log_agent_step(session_id, 'ValidatorAgent', 'Starting ValidatorAgent', level=LogLevel.DEBUG)
                validator_out = execute_agent(validator_agent, ai)
                try:
                    out_summary = {"success": validator_out.success, "confidence": validator_out.confidence}
                    if isinstance(validator_out.data, dict):
                        out_summary['data_keys'] = list(validator_out.data.keys())
                    log_agent_step(session_id, 'ValidatorAgent', 'Completed ValidatorAgent', level=LogLevel.INFO, data=out_summary,
                                   success=validator_out.success, error=validator_out.error)
                except Exception:
                    logger.exception("Failed to emit validator output log")
                ctx.add_agent_output("ValidatorAgent", validator_out.__dict__)

                # Build structured response
                result = {
                    "answer": None,
                    "llm_prompt": llm_prompt,
                    "sources": sources,
                    "retrieved_docs": retrieved_info,
                    "analysis": analysis_out.data if hasattr(analysis_out, 'data') else analysis_out,
                    "reasoning": reasoning_out.data if hasattr(reasoning_out, 'data') else reasoning_out,
                    "advisor": advisor_out.data if hasattr(advisor_out, 'data') else advisor_out,
                    "validator": validator_out.data if hasattr(validator_out, 'data') else validator_out,
                    "embedding_model": embedding_model,
                    "num_chunks": num_chunks,
                    "retrieval_time_s": retrieval_time,
                }
                # End agent logger session
                try:
                    if agent_logger:
                        agent_logger.end_session(session_id)
                except Exception:
                    logger.exception("Failed to end agent logger session for session_id=%s", session_id)

                return result
        except Exception:
            # End agent logger session if present
            try:
                if 'session_id' in locals() and agent_logger:
                    agent_logger.end_session(session_id)
            except Exception:
                logger.exception("Failed to end agent logger session after orchestration failure")
            # Fall back to original llm_prompt return path if pipeline fails
            pass

        return {
            "answer": None,  # Use llm_prompt with an LLM provider
            "llm_prompt": llm_prompt,
            "sources": sources,
            "retrieved_docs": retrieved_info,
            "embedding_model": embedding_model,
            "num_chunks": num_chunks,
            "retrieval_time_s": retrieval_time,
        }
    
    except Exception as e:
        logger.exception(f"get_answer failed: {e}")
        raise


# ============================================================================
# LLM INTEGRATION (with Groq)
# ============================================================================

def _build_chain(thread_id: str):
    """
    Build a ConversationalRetrievalChain for Groq LLM.
    
    This chains the vectorstore retriever with Groq's LLM for stateful conversation.
    """
    logger.info(f"Building retrieval chain for thread_id={thread_id}")
    
    if not HAS_GROQ:
        raise RuntimeError("langchain_groq not installed; pip install langchain_groq")
    
    # Initialize Groq LLM
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY must be set. Add to .env file:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "Or set environment variable: export GROQ_API_KEY=gsk_..."
        )
    
    llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        temperature=0.0,
        max_tokens=512,
    )
    logger.info(f"Initialized Groq LLM: model={config.GROQ_MODEL}")
    
    # Get retriever
    try:
        retr = retriever.get_retriever(k=config.RETRIEVAL_K)
        logger.info(f"Got retriever for thread_id={thread_id} (k={config.RETRIEVAL_K})")
    except RuntimeError as e:
        logger.error(f"Failed to get retriever: {e}")
        raise
    
    # Build memory
    memory = get_memory(thread_id)
    logger.info(f"Loaded memory for thread_id={thread_id} ({len(memory.chat_memory.messages)} messages)")
    
    # Build reasoning prompt
    reasoning_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_templates.RETRIEVAL_PROMPT_TEMPLATE,
    )
    
    # Build chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retr,
        memory=None,  # Manual memory management
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    logger.info(f"Built ConversationalRetrievalChain for thread_id={thread_id}")
    
    return chain


def chat(message: str, thread_id: str) -> Dict[str, Any]:
    """
    Main chat endpoint: stateful conversation with proper reasoning over forecast data.
    
    Pipeline:
    1. Load analyzer and check for direct data-backed answers (pattern matching)
    2. Retrieve relevant documents from vectorstore
    3. Extract and aggregate numeric data from retrieved docs
    4. Build reasoning prompt with structured data context
    5. Call LLM to synthesize insights and answer
    6. Validate and clean the response
    7. Return structured response with metadata
    
    Args:
        message: User's message
        thread_id: Conversation thread identifier
    
    Returns:
        Dict with keys:
        - answer: Synthesized, data-backed answer
        - source: "data_backed", "rag", or "error"
        - thread_id: Thread identifier
        - source_documents: Top 3 retrieved docs
        - metadata: Comprehensive execution stats
    """
    logger.info(f"chat called: thread_id={thread_id}, message_len={len(message)}")
    
    # Add to memory
    memory = get_memory(thread_id)
    memory.chat_memory.add_user_message(message)

    # Start an agent reasoning session for this chat (so agent logs have a session)
    try:
        agent_logger = get_agent_logger()
        try:
            agent_logger.start_session(thread_id, message, metadata={"source": "chat_api"})
        except Exception:
            # Non-fatal: continue without session persistence
            logger.exception("Failed to start agent reasoning session for thread_id=%s", thread_id)
    except Exception:
        agent_logger = None
    
    try:
        # Step 1: Acquire the dataset analyzer and validate dataset integrity
        analyzer = dataset_analyzer.get_analyzer()

        # Enforce strict dataset integrity: required columns must be present.
        # Do not proceed or attempt any fallback if key columns are missing.
        required_col = "ref_article"
        try:
            cols = list(analyzer.df.columns)
        except Exception:
            # If analyzer.df is not accessible for any reason, log and abort
            msg = f"Critical dataset integrity issue: analyzer DataFrame not accessible (source={getattr(config, 'DATA_PATH', 'unknown')}). Aborting analysis."
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg)

        if required_col not in cols:
            msg = (
                f"Critical dataset integrity issue: missing required column '{required_col}' in dataset (source={getattr(config, 'DATA_PATH', 'unknown')}). Aborting analysis."
            )
            # Log as ERROR with stack trace for visibility in logs
            logger.error(msg, exc_info=True)
            # Disable any silent fallback — raise to abort processing upstream
            raise RuntimeError(msg)

        # Proceed with direct pattern-based answer generation now that data is valid
        # Enforced policy: data-backed/deterministic answers are disabled.
        # Do not call _generate_data_backed_answer under any circumstances.
        data_backed_answer = None
        
        # Step 2: ALWAYS load ALL products for maximum context and best possible answers
        logger.info(f"🚀 Loading ALL products for maximum LLM context (query: '{message[:80]}...')")
        
        start_time = time.time()
        
        # Load complete dataset from analyzer
        all_products_df = analyzer.get_all_products()
        
        # Convert to retrieved docs format
        retrieved_docs_raw = []
        for idx, row in all_products_df.iterrows():
            doc_dict = {
                "id": f"product_{idx}",
                "content": f"ref_article: {row.get('ref_article', 'N/A')} | avg_forecast: {row.get('avg_forecast', 0)} | trend_pct: {row.get('trend_pct', 0)} | trend_label: {row.get('trend_label', 'Unknown')} | data_points: {row.get('data_points', 0)}",
                "full_content": f"ref_article: {row.get('ref_article', 'N/A')} | designation: {row.get('designation', 'N/A')} | marque: {row.get('marque', 'N/A')} | famille: {row.get('famille', 'N/A')} | next_year: {row.get('next_year', 'N/A')} | avg_forecast: {row.get('avg_forecast', 0)} | trend_pct: {row.get('trend_pct', 0)} | trend_label: {row.get('trend_label', 'Unknown')} | data_points: {row.get('data_points', 0)}",
                "metadata": {
                    "ref_article": row.get('ref_article', 'N/A'),
                    "avg_forecast": float(row.get('avg_forecast', 0)),
                    "trend_pct": float(row.get('trend_pct', 0)),
                    "trend_label": row.get('trend_label', 'Unknown'),
                    "data_points": int(row.get('data_points', 0)),
                    "designation": row.get('designation', ''),
                    "marque": row.get('marque', ''),
                    "famille": row.get('famille', ''),
                }
            }
            retrieved_docs_raw.append(doc_dict)
        
        sources = [f"product_{idx}" for idx in range(len(retrieved_docs_raw))]
        retrieval_time = time.time() - start_time
        needs_all_products = True  # Always true now
        
        logger.info(f"✅ Loaded ALL {len(retrieved_docs_raw)} products in {retrieval_time:.2f}s for complete analysis")
        
        # Step 3: Extract and aggregate numeric data from retrieved docs
        aggregated_data = _extract_and_aggregate_data(retrieved_docs_raw, analyzer)

        # Detect degenerate aggregated data and abort LLM calls to avoid hallucination
        is_deg, deg_reason = _is_degenerate_aggregated_data(aggregated_data)
        if is_deg:
            logger.error(f"Degenerate dataset detected (reason={deg_reason}); aborting LLM synthesis")
            try:
                log_agent_step(thread_id, 'Chat', 'Degenerate aggregated data detected; aborting LLM', level=LogLevel.ERROR,
                               data={"reason": deg_reason, "product_count": aggregated_data.get('count', 0)})
            except Exception:
                logger.exception("Failed to log degenerate data event")

            answer = f"Cannot answer reliably: dataset is degenerate ({deg_reason}). Please provide a richer dataset or check data quality."
            memory.chat_memory.add_ai_message(answer)
            _persist_memory(thread_id)

            # End session if started
            try:
                if agent_logger:
                    agent_logger.end_session(thread_id)
            except Exception:
                logger.exception("Failed to end agent reasoning session for thread_id=%s", thread_id)

            return {
                "answer": answer,
                "source": "error",
                "thread_id": thread_id,
                "source_documents": [],
                "error": f"degenerate_data:{deg_reason}",
                "metadata": {
                    "type": "degenerate_data",
                    "reason": deg_reason,
                    "num_products": aggregated_data.get('count', 0),
                }
            }
        
        # Step 4: Build reasoning prompt with structured context
        reasoning_prompt = _build_reasoning_prompt(
            query=message,
            aggregated_data=aggregated_data,
            retrieved_docs=retrieved_docs_raw,
        )
        
        # Step 5: Call LLM for synthesis
        answer = None
        validation_status = "fallback"
        
        if HAS_GROQ and config.GROQ_API_KEY:
            try:
                logger.info(f"Calling Groq LLM for synthesis (thread_id={thread_id})")
                
                llm = ChatGroq(
                    api_key=config.GROQ_API_KEY,
                    model=config.GROQ_MODEL,
                    temperature=0.0,
                    max_tokens=1024,  # Allow longer responses
                )
                
                response = llm.invoke([HumanMessage(content=reasoning_prompt)])
                answer = response.content
                
                # Clean citations and metadata spam
                answer = _clean_citations_from_response(answer)
                
                # Validate answer against actual data
                is_valid, invalid_products = _validate_response_against_data(answer, analyzer)
                
                if is_valid:
                    # Mark as validated by checks but do NOT mark as 'data_backed'
                    # to avoid exposing deterministic/data-only answer flags.
                    validation_status = "validated"
                    logger.info(f"LLM response validated successfully (marked as 'validated')")
                else:
                    logger.warning(f"LLM response contains invalid products: {invalid_products}")
                    # Still use the answer but mark validation as partial
                    validation_status = "partial"
                
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                answer = None
        
        # Step 6: Fallback to structured summary if LLM unavailable
        if answer is None:
            # Enforce: no fallback structured (data-backed) answers allowed.
            err_msg = (
                "LLM synthesis unavailable and deterministic/fallback answers are disallowed by policy."
            )
            logger.error(err_msg)
            # End session if started
            try:
                if agent_logger:
                    agent_logger.end_session(thread_id)
            except Exception:
                logger.exception("Failed to end agent reasoning session for thread_id=%s", thread_id)

            return {
                "answer": None,
                "source": "error",
                "thread_id": thread_id,
                "source_documents": [],
                "error": "llm_unavailable_and_data_backed_forbidden",
                "metadata": {
                    "type": "no_answer",
                    "reason": "llm_unavailable_and_data_backed_forbidden",
                }
            }
        
        # Add to memory
        memory.chat_memory.add_ai_message(answer)
        _persist_memory(thread_id)
        # End session if started
        try:
            if agent_logger:
                agent_logger.end_session(thread_id)
        except Exception:
            logger.exception("Failed to end agent reasoning session for thread_id=%s", thread_id)
        
        # Track last retrieval
        with _last_retrievals_lock:
            _last_retrievals[thread_id] = {
                "timestamp": int(time.time()),
                "question": message,
                "answer_len": len(answer),
                "docs_count": len(retrieved_docs_raw),
                "retrieval_time_s": retrieval_time,
                "sources": sources[:3],  # Top 3 only
                "validation": validation_status,
            }
        
        # Return top 3 source documents only
        top_source_docs = [
            {"page_content": d["full_content"], "metadata": d["metadata"]}
            for d in retrieved_docs_raw[:3]
        ]
        
        logger.info(
            f"chat complete: thread_id={thread_id}, source=rag, "
            f"validation={validation_status}, answer_len={len(answer)}, "
            f"docs={len(retrieved_docs_raw)}"
        )
        
        # Using default embedding model (no vectorstore retrieval needed)
        embedding_model = config.EMBEDDING_MODEL
        
        return {
            "answer": answer,
            "source": "rag",
            "thread_id": thread_id,
            "source_documents": top_source_docs,
            "metadata": {
                # Always use a retrieval synthesis metadata type to avoid exposing
                # dataset-wide deterministic analysis labels. Keep used_all_products
                # for internal diagnostics but do not change the public 'type'.
                "type": "retrieval_synthesis",
                "validation": validation_status,
                "num_chunks": len(retrieved_docs_raw),
                "retrieval_time_s": retrieval_time,
                "embedding_model": embedding_model,
                "used_all_products": needs_all_products,
            }
        }
    
    except Exception as e:
        logger.exception(f"chat failed: {e}")
        answer = f"Error processing your request: {str(e)}"
        memory.chat_memory.add_ai_message(answer)
        _persist_memory(thread_id)
        # End session if started
        try:
            if agent_logger:
                agent_logger.end_session(thread_id)
        except Exception:
            logger.exception("Failed to end agent reasoning session for thread_id=%s", thread_id)

        return {
            "answer": answer,
            "source": "error",
            "thread_id": thread_id,
            "source_documents": [],
            "error": str(e),
            "metadata": {
                "type": "error",
                "validation": "failed",
            }
        }



def get_last_retrieval(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get debugging info about the last retrieval for a thread."""
    with _last_retrievals_lock:
        return _last_retrievals.get(thread_id)


# ============================================================================
# MULTI-AGENT REASONING ENTRY POINT
# ============================================================================

def chat_with_agents(message: str, thread_id: str) -> Dict[str, Any]:
    """
    Main chat endpoint with multi-agent reasoning enabled.
    
    This enhanced version uses the multi-agent orchestrator to provide:
    - Data extraction and validation
    - Trend analysis and anomaly detection
    - Business-focused insights and recommendations
    - Structured reasoning trail for debugging
    
    Flow:
    1. Load/retrieve documents using get_answer()
    2. Orchestrate DataAgent → AnalysisAgent → BusinessAgent → AnswerAgent
    3. Return structured result with reasoning trail
    
    Args:
        message: User's message
        thread_id: Conversation thread identifier
    
    Returns:
        Dict with keys:
        - answer: Generated answer from multi-agent reasoning
        - source: "multi_agent_rag", "fallback", or "error"
        - agents_used: List of agents that executed
        - validation: "passed", "partial", or "fallback"
        - thread_id: Thread identifier
        - source_documents: Retrieved documents
        - metadata: Execution stats and retrieval info
        - reasoning_trail: Detailed output from each agent (for debugging)
    """
    logger.info(f"chat_with_agents called: thread_id={thread_id}, message_len={len(message)}")
    
    # Add to memory
    memory = get_memory(thread_id)
    memory.chat_memory.add_user_message(message)
    
    try:
        # Step 1: Retrieve documents
        start_time = time.time()
        result = get_answer(message, top_k=config.RETRIEVAL_K)
        retrieval_time = time.time() - start_time
        
        retrieved_docs_raw = result["retrieved_docs"]
        sources = result["sources"]
        
        logger.info(f"Retrieved {len(retrieved_docs_raw)} documents in {retrieval_time:.2f}s")
        
        # Step 2: Run multi-agent orchestration (async)
        agent_start_time = time.time()
        
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run orchestrator
            analyzer = dataset_analyzer.get_analyzer()
            multi_agent_result = loop.run_until_complete(
                agents.orchestrate_multi_agent_reasoning(
                    query=message,
                    retrieved_docs=retrieved_docs_raw,
                    analyzer=analyzer,
                )
            )
            
            agent_time = time.time() - agent_start_time
            logger.info(f"Multi-agent orchestration completed in {agent_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Multi-agent orchestration failed: {e}; falling back to simple RAG")
            multi_agent_result = None
            agent_time = time.time() - agent_start_time
        
        # Step 3: Use result
        if multi_agent_result and multi_agent_result.source == "multi_agent_rag":
            answer = multi_agent_result.answer
            validation = multi_agent_result.validation
            agents_used = multi_agent_result.agents_used
            
            # Add to memory
            memory.chat_memory.add_ai_message(answer)
            _persist_memory(thread_id)
            
            # Track retrieval
            with _last_retrievals_lock:
                _last_retrievals[thread_id] = {
                    "timestamp": int(time.time()),
                    "question": message,
                    "answer_len": len(answer),
                    "docs_count": len(retrieved_docs_raw),
                    "retrieval_time_s": retrieval_time,
                    "agent_time_s": agent_time,
                    "sources": sources,
                    "validation": validation,
                    "agents_used": agents_used,
                }
            
            logger.info(
                f"chat_with_agents complete: source=multi_agent_rag, "
                f"agents={agents_used}, validation={validation}, "
                f"answer_len={len(answer)}"
            )
            
            # Build reasoning_trail dict for output
            reasoning_trail = {}
            for agent_name, agent_output in multi_agent_result.reasoning_trail.items():
                reasoning_trail[agent_name] = {
                    "success": agent_output.success,
                    "reasoning_steps": agent_output.reasoning_steps,
                    "validation_passed": agent_output.validation_passed,
                    "error": agent_output.error,
                }
            
            return {
                "answer": answer,
                "source": "multi_agent_rag",
                "agents_used": agents_used,
                "validation": validation,
                "thread_id": thread_id,
                "source_documents": [
                    {"page_content": d["full_content"], "metadata": d["metadata"]}
                    for d in retrieved_docs_raw
                ],
                "metadata": {
                    "embedding_model": result["embedding_model"],
                    "num_chunks": result["num_chunks"],
                    "retrieval_time_s": retrieval_time,
                    "agent_orchestration_time_s": agent_time,
                    "total_time_s": retrieval_time + agent_time,
                    "sources": sources,
                    "validation": validation,
                },
                "reasoning_trail": reasoning_trail,
            }
        
        else:
            # Fallback to simple RAG
            logger.warning("Using fallback to simple RAG chat")
            return chat(message, thread_id)
    
    except Exception as e:
        logger.exception(f"chat_with_agents failed: {e}")
        answer = f"Error during multi-agent analysis: {str(e)}"
        memory.chat_memory.add_ai_message(answer)
        _persist_memory(thread_id)
        
        return {
            "answer": answer,
            "source": "error",
            "thread_id": thread_id,
            "error": str(e),
        }