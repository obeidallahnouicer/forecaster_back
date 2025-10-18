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
