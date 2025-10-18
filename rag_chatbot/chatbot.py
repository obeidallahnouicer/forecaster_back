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
    
    # Find most stable (smallest abs trend_pct)
    most_stable = min(products, key=lambda p: abs(p["trend_pct"]))
    aggregates["most_stable"] = most_stable
    
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
            meta = rag_indexer.load_vectorstore_metadata(Path(config.VECTORSTORE_DIR) / "chroma")
            num_chunks = meta.get("num_docs", 0) if meta else 0
            embedding_model = meta.get("embedding_model", config.EMBEDDING_MODEL) if meta else config.EMBEDDING_MODEL
        except Exception:
            num_chunks = 0
            embedding_model = config.EMBEDDING_MODEL
        
        logger.info(
            f"Composed prompt (model={embedding_model}, "
            f"retrieved_docs={len(retrieved_docs)}, total_chunks={num_chunks})"
        )
        
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
    
    try:
        # Step 1: Try direct pattern-based answer from dataset
        analyzer = dataset_analyzer.get_analyzer()
        data_backed_answer = _generate_data_backed_answer(message, analyzer)
        
        if data_backed_answer:
            logger.info(f"Generated data-backed answer for thread_id={thread_id}")
            memory.chat_memory.add_ai_message(data_backed_answer)
            _persist_memory(thread_id)
            
            return {
                "answer": data_backed_answer,
                "source": "data_backed",
                "thread_id": thread_id,
                "source_documents": [],
                "metadata": {
                    "type": "direct_analysis",
                    "validation": "data_backed",
                    "num_chunks": 0,
                    "retrieval_time_s": 0.0,
                }
            }
        
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
                    validation_status = "data_backed"
                    logger.info(f"LLM response validated successfully")
                else:
                    logger.warning(f"LLM response contains invalid products: {invalid_products}")
                    # Still use the answer but mark validation as partial
                    validation_status = "partial"
                
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                answer = None
        
        # Step 6: Fallback to structured summary if LLM unavailable
        if answer is None:
            answer = _generate_fallback_answer(aggregated_data, message)
            validation_status = "data_backed"
            logger.info("Using fallback structured summary")
        
        # Add to memory
        memory.chat_memory.add_ai_message(answer)
        _persist_memory(thread_id)
        
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
                "type": "dataset_wide_analysis" if needs_all_products else "retrieval_synthesis",
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