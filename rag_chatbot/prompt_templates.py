"""
Prompt templates for RAG-based reasoning and LLM synthesis.

This module provides:
- Retrieval prompt template (instructions for using retrieved context)
- Composed prompt helpers for downstream LLM consumption
- Citation formatting utilities
- Data-aware prompts to prevent hallucinations
"""

from typing import List, Dict, Any


RETRIEVAL_SYSTEM_PROMPT = """You are a business analyst. Answer questions using ONLY the provided documents and data.

CRITICAL RULES:
1. ONLY mention products, metrics, and values that appear in the provided documents
2. NEVER invent or estimate product codes, trend values, or metrics
3. If a product is not in the documents, do NOT mention it
4. Always cite specific numbers from the data (avg_forecast, trend_pct, data_points)
5. For stability: use the trend_pct value (closer to zero = more stable)
6. For low-performing: consider avg_forecast (lower), negative trend_pct, and fewer data_points

RESPONSE STYLE - BALANCED:
1. Answer the question clearly (1-2 sentences)
2. Include the specific metrics from the data that support your answer
3. NO citations or document markers - just state the facts
4. NO made-up products or invented data
5. If asked about a product not in the data: "No product in the current dataset matches the criteria."
6. Keep to 3-5 sentences max

Example correct response format:
"The most stable product is [PRODUCT_CODE], with a trend percentage of [TREND_PCT]%, closest to zero among the products in the dataset. Its average forecast is [FORECAST_VALUE] with [DATA_POINTS] data points."

Example incorrect (AVOID):
"The product [FAKE_PRODUCT] has the best trend" (if the product doesn't exist in data)
"""


RETRIEVAL_PROMPT_TEMPLATE = """Answer using only the context documents. Be helpful but balanced - show relevant details without being too wordy.

Context:
{context}

Question: {query}

Answer (include key details, keep to 3-5 sentences):"""


def compose_retrieved_context(docs: List[Dict[str, Any]], include_metadata: bool = True) -> str:
    """
    Compose retrieved documents into a formatted context string for LLM input.
    
    Includes comprehensive metadata for analysis: metrics, trends, data quality indicators.
    
    Args:
        docs: List of retrieved document dicts (or LangChain Document objects)
        include_metadata: Whether to include metadata fields in the output
        
    Returns:
        Formatted context string with all relevant fields
    """
    parts = []
    
    for idx, doc in enumerate(docs, 1):
        # Handle LangChain Document objects
        if hasattr(doc, "page_content"):
            content = doc.page_content
            metadata = getattr(doc, "metadata", {})
        else:
            # Handle dict-based docs
            content = doc.get("page_content") or doc.get("content", "")
            metadata = doc.get("metadata", {})
        
        # Format document header
        ref_article = metadata.get("ref_article", f"doc_{idx}")
        header = f"[Document {idx}] Product: {ref_article}"
        parts.append(header)
        parts.append(content)
        
        if include_metadata and metadata:
            # Add comprehensive metadata with business context
            meta_lines = []
            
            # Key business metrics
            key_metrics = [
                ("avg_forecast", "Average Forecast (next year sales volume)"),
                ("trend_pct", "Trend Percentage (% change)"),
                ("trend_label", "Trend Classification"),
                ("data_points", "Data Points (forecast reliability: more = better)"),
                ("next_year", "Forecast Year"),
            ]
            
            for field, description in key_metrics:
                if field in metadata:
                    value = metadata[field]
                    meta_lines.append(f"  {description}: {value}")
            
            # Source information
            if "source_row" in metadata:
                meta_lines.append(f"  Source Row: {metadata['source_row']}")
            
            if meta_lines:
                parts.append("\nMetadata:\n" + "\n".join(meta_lines))
        
        parts.append("")  # Blank line between docs
    
    return "\n".join(parts)


def compose_prompt_for_llm(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    system_prompt: str = RETRIEVAL_SYSTEM_PROMPT,
    include_metadata: bool = True,
) -> str:
    """
    Compose a complete prompt for LLM input, including system message, context, and query.
    
    This is a helper for plugging into OpenAI, Anthropic, or local LLM APIs.
    
    Args:
        query: The user's question
        retrieved_docs: Retrieved documents from vectorstore
        system_prompt: System message to prepend
        include_metadata: Whether to include document metadata
        
    Returns:
        Complete prompt string ready for LLM
    """
    context = compose_retrieved_context(retrieved_docs, include_metadata)
    
    prompt = (
        f"System:\n{system_prompt}\n\n"
        f"Context Documents:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )
    
    return prompt


# ---------------------------------------------------------------------------
# Agent-specific high-quality prompt templates (Beast Mode)
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM_PROMPT = """You are a senior business analyst and model auditor. Work in a structured, auditable way and prefer data-backed statements.

Work steps (strict):
1) Observe: List which numeric fields you used (avg_forecast, trend_pct, data_points, trend_label) and any assumptions.
2) Analyze: For each product, compute stability (absolute trend_pct), reliability (data_points), and performance (avg_forecast).
3) Compare: Highlight conflicts or disagreements across models or metrics and quantify them (e.g., delta between model means).
4) Deduce: Produce concise business conclusions and rank them by confidence (0.0-1.0).
5) Recommend: Provide prioritized next steps and expected impact (qualitative or estimate) for each recommendation.

Output rules:
- Prefer returning a strict JSON object matching the Analysis schema (see the human template for exact keys).
- If you cannot return valid JSON, return a minimal JSON: {"raw": "<full human-readable analysis text here>"}
- Always include a short 'chain_of_thought' array of 1-3 short steps explaining the highest-impact deduction (kept separate from the numeric output to preserve auditable traces).
"""

ANALYSIS_HUMAN_TEMPLATE = """
You are given a structured JSON input (per-product numeric summaries and optional example documents).

Task: produce a validated, auditable JSON Analysis report that follows the schema below. Be conservative: use only numbers present in the input. If data is insufficient to compute a value, set it to null and explain in the 'notes' field.

Required Analysis JSON schema (example structure):

{
    "products": [
        {
            "product": "<product_code>",
            "critical_metrics": {"avg_forecast": 12.3, "trend_pct": 0.5, "data_points": 10, "trend_label": "Stable"},
            "alternatives": [
                {"explanation": "...", "likelihood": 0.7, "impact": 0.4, "chain_of_thought": ["step1","step2"], "suggested_next_steps": ["reforecast with X"]}
            ],
            "recommendations": [{"action": "...", "priority": "high|medium|low", "expected_roi": "estimate or null", "confidence": 0.0}]
        }
    ],
    "summary_insights": ["short bullet 1", "short bullet 2"],
    "overall_confidence": 0.0,
    "chain_of_thought": ["one-line reasoning step", "another step"]
}

Instructions:
- Use ONLY values from the provided input JSON. Do not invent product codes or numeric values.
- For each product, provide at least two alternative interpretations where feasible (if insufficient data, explain in 'notes').
- Provide short, auditable 'chain_of_thought' entries (1-3 short sentences) separate from the JSON numeric fields.
- Output must be valid JSON. If you cannot produce valid JSON for any reason, return exactly: {"raw": "<analysis text>"} where <analysis text> is your full human-readable analysis.

Input:
{input_json}

Respond ONLY with the JSON object described above.
"""


ADVISOR_SYSTEM_PROMPT = """You are a CFO-level strategic advisor focused on concise, prioritized, and cost-aware recommendations suitable for executive decision-makers.

Behavioral rules:
- Base every recommendation on Analysis outputs; list which metrics and products you used.
- For each recommendation, provide: action (one-line), priority (high|medium|low), expected_roi_estimate (numeric or categorical), risk_level (low|medium|high), estimated_confidence (0.0-1.0), and a one-sentence justification that cites numeric evidence.
- Keep executive summary to 2-3 sentences and the JSON recommendations machine-parseable.

Output rules:
- Return a JSON object with 'executive_summary' (string) and 'recommendations' (array) per the schema below. Also include a short 'audit' section listing the Analysis keys used.
- If you cannot produce valid JSON, return {"raw": "<executive text>"} and include an 'audit' key in that raw text describing missing data.
"""


VALIDATOR_SYSTEM_PROMPT = """You are an expert validator and critic. Your job is to verify that each recommendation and numeric claim is grounded in the provided Analysis outputs and underlying data.

Validation steps:
1) Check that every numeric claim references a field present in the Analysis input (avg_forecast, trend_pct, data_points, etc.).
2) If a claim references a product not present in the Analysis input, flag it as fabricated.
3) For each recommendation, assess whether expected ROI and confidence are consistent with the supporting metrics; if not, suggest a priority adjustment and explain why.
4) Produce a machine-readable JSON critique and a short human summary.

Output rules:
- Return JSON with keys: validated_recommendations (array), issues (array), suggested_followups (array), overall_confidence (0.0-1.0), and an 'audit' field listing which inputs were checked.
- If you cannot return valid JSON, return {"raw": "<critique text>"}.
"""

VALIDATOR_HUMAN_TEMPLATE = """
You will receive a chained context object containing retrieved_docs, analysis, reasoning, and advisor outputs.

Task: run the Validation steps (see system prompt) and produce the following JSON schema:

{
    "validated_recommendations": [{"product": "X", "valid": true, "notes": "...", "suggested_priority_adjustment": "none|up|down"}],
    "issues": ["short issue 1", "short issue 2"],
    "suggested_followups": [{"product": "X", "action": "reforecast with Y", "reason": "..."}],
    "overall_confidence": 0.0,
    "audit": {"analysis_keys_checked": ["avg_forecast","trend_pct","data_points"]}
}

Input:
{chained_context}

Respond ONLY with valid JSON following the schema above. If unable, return exactly {"raw": "<critique text>"}.
"""


def compose_agent_prompt(system_prompt: str, human_template: str, context: Dict[str, Any]) -> tuple:
        """Helper to render an agent system + human prompt pair from a template and context dict.

        Returns (system_prompt, human_prompt) ready for LLM consumption.
        """
        import json as _json
        sys = system_prompt
        human = human_template.format(input_json=_json.dumps(context, default=str), chained_context=_json.dumps(context, default=str))
        return sys, human


def extract_citations(response_text: str) -> List[str]:
    """
    Extract citation references from LLM response.
    
    Looks for patterns like [Doc 1], [Document 5], [Source: filename].
    
    Args:
        response_text: Response text from LLM
        
    Returns:
        List of citation strings found
    """
    import re
    
    patterns = [
        r"\[Doc\s+(\d+)\]",
        r"\[Document\s+(\d+)\]",
        r"\[Source:\s*([^\]]+)\]",
        r"\[([^\]]*?(?:row|source)[^\]]*?)\]",
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        citations.extend(matches)
    
    return list(set(citations))  # Deduplicate


# Example usage and documentation

EXAMPLE_USAGE = """
# Example: Building a complete LLM call with retrieved context

from rag_chatbot.prompt_templates import (
    compose_prompt_for_llm,
    extract_citations,
)

# After retrieving docs from Chroma vectorstore
retrieved_docs = vectorstore.similarity_search("What products had uptrend?", k=3)

# Compose full prompt
prompt = compose_prompt_for_llm(
    query="What products had uptrend?",
    retrieved_docs=retrieved_docs,
    include_metadata=True,
)

# Call your LLM of choice (example with OpenAI)
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

answer = response["choices"][0]["message"]["content"]
print(answer)

# Extract citations from response
citations = extract_citations(answer)
print(f"Document references: {citations}")
"""


if __name__ == "__main__":
    # Print example
    print(EXAMPLE_USAGE)
