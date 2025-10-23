"""
InsightAgent: data-driven analysis using LangChain chains.

Computes business KPIs, detects patterns/trends, and generates actionable
recommendations using deterministic rules + LangChain LLM analysis.
"""
import logging
import statistics
import json
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from core.config import GROQ_API_KEY, GROQ_MODEL
from prompts.insight_generation import INSIGHT_GENERATION_CHAT_PROMPT, generate_file_schema

logger = logging.getLogger("agents.insight_agent")


def _compute_stats(rows: List[Dict], numeric_cols: List[str]) -> Dict[str, Any]:
    """Compute statistics for numeric columns."""
    stats = {}
    
    for col in numeric_cols:
        try:
            values = [float(row.get(col, 0)) for row in rows if row.get(col) is not None]
            if values:
                stats[col] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'total': sum(values),
                    'count': len(values)
                }
                
                if len(values) > 1:
                    stats[col]['std'] = statistics.stdev(values)
        except Exception as e:
            logger.debug(f"Failed to compute stats for {col}: {e}")
    
    return stats


def _detect_numeric_columns(rows: List[Dict]) -> List[str]:
    """Detect which columns contain numeric data."""
    if not rows:
        return []
    
    numeric_cols = []
    sample = rows[0]
    
    for col, val in sample.items():
        if isinstance(val, (int, float)):
            numeric_cols.append(col)
        elif isinstance(val, str):
            try:
                float(val)
                numeric_cols.append(col)
            except:
                pass
    
    return numeric_cols


def _generate_insights(rows: List[Dict], stats: Dict, question: str) -> List[str]:
    """Generate deterministic insights based on statistics."""
    insights = []
    
    # Basic insights
    insights.append(f"Found {len(rows)} records")
    
    # Statistical insights
    for col, col_stats in stats.items():
        mean = col_stats.get('mean', 0)
        total = col_stats.get('total', 0)
        
        if total > 0:
            insights.append(
                f"{col}: total={total:,.2f}, avg={mean:,.2f}, "
                f"range=[{col_stats.get('min', 0):,.2f} - {col_stats.get('max', 0):,.2f}]"
            )
        
        # Detect outliers
        if 'std' in col_stats and col_stats['std'] > 0:
            cv = col_stats['std'] / abs(mean) if mean != 0 else 0
            if cv > 0.5:
                insights.append(f"⚠️ High variability detected in {col} (CV={cv:.2f})")
    
    return insights


def _generate_recommendations(rows: List[Dict], stats: Dict, question: str) -> List[str]:
    """Generate actionable recommendations based on data patterns."""
    recommendations = []
    
    if not rows:
        recommendations.append("No data found. Check filters or date ranges.")
        return recommendations
    
    # Check for common business patterns
    for col, col_stats in stats.items():
        mean = col_stats.get('mean', 0)
        
        # Low values warnings
        if 'stock' in col.lower() or 'inventory' in col.lower():
            if mean < 100:
                recommendations.append(f"⚠️ Low {col} detected (avg: {mean:.0f}). Consider restocking.")
        
        # High variability warnings
        if 'std' in col_stats:
            cv = col_stats['std'] / abs(mean) if mean != 0 else 0
            if cv > 0.7:
                recommendations.append(
                    f"📊 {col} shows high variability. Review pricing or demand patterns."
                )
    
    # Growth opportunities
    if len(rows) > 10:
        recommendations.append(f"✅ Sufficient data ({len(rows)} records) for trend analysis.")
    elif len(rows) < 3:
        recommendations.append("ℹ️ Limited data. Consider expanding time range or filters.")
    
    return recommendations


def summarize_results(
    rows: List[Dict], 
    columns: List[str], 
    question: str,
    sql: str = ""
) -> Dict[str, Any]:
    """
    Analyze SQL query results using LangChain and generate business insights.
    
    Args:
        rows: Query result rows
        columns: Column names
        question: Original user question
        sql: The executed SQL query
    
    Returns:
        {
            'summary': str,
            'insights': List[str],
            'recommendations': List[str],
            'kpis': Dict,
            'stats': Dict,
            'confidence': float
        }
    """
    result = {
        'row_count': len(rows),
        'columns': columns,
        'insights': [],
        'recommendations': [],
        'kpis': {},
        'stats': {},
        'summary': '',
        'confidence': 0.0
    }
    
    if not rows:
        result['summary'] = "No data found for your query."
        result['recommendations'].append("Try adjusting filters or date ranges.")
        return result
    
    # Compute statistics
    numeric_cols = _detect_numeric_columns(rows)
    stats = _compute_stats(rows, numeric_cols)
    result['stats'] = stats
    
    # KPIs
    result['kpis']['total_rows'] = len(rows)
    result['kpis']['total_columns'] = len(columns)
    
    for col, col_stats in stats.items():
        result['kpis'][f'{col}_total'] = col_stats.get('total', 0)
        result['kpis'][f'{col}_avg'] = col_stats.get('mean', 0)
    
    # Generate deterministic insights
    insights = _generate_insights(rows, stats, question)
    result['insights'] = insights
    
    # Generate recommendations
    recommendations = _generate_recommendations(rows, stats, question)
    result['recommendations'] = recommendations
    
    # Build context for LLM summary using LangChain
    sample_rows_str = json.dumps(rows[:3], default=str, indent=2) if rows else "No rows"
    
    key_metrics_str = "\n".join([
        f"- {col}: total={s.get('total', 0):,.2f}, avg={s.get('mean', 0):,.2f}"
        for col, s in list(stats.items())[:5]
    ])
    
    try:
        if not GROQ_API_KEY:
            result['summary'] = ". ".join(insights[:3]) if insights else "Analysis complete."
            result['confidence'] = 0.3
            return result
        
        # Create ChatGroq LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=200
        )
        
        # Create output parser
        parser = StrOutputParser()
        
        # Build the chain
        chain = INSIGHT_GENERATION_CHAT_PROMPT | llm | parser
        
        # Invoke the chain
        # generate file/column schema to give the model context about available files
        file_schema = generate_file_schema()

        summary_text = chain.invoke({
            "question": question,
            "sql": sql[:200],
            "row_count": len(rows),
            "columns": ", ".join(columns[:5]),
            "stats": json.dumps({k: v for k, v in list(stats.items())[:3]}, default=str),
            "key_metrics": key_metrics_str,
            "sample_rows": sample_rows_str,
            "deterministic_insights": "; ".join(insights[:3]),
            "file_schema": file_schema
        })
        
        result['summary'] = summary_text.strip() if summary_text else ". ".join(insights[:3])
        result['confidence'] = 0.85
        
    except Exception as e:
        logger.warning(f"LangChain insight generation failed: {e}", exc_info=True)
        result['summary'] = ". ".join(insights[:3]) if insights else "Analysis complete."
        result['confidence'] = 0.5
    
    return result


class InsightAgent:
    """Thin compatibility wrapper expected by the API.

    The rest of the module implements the analysis logic as pure functions
    (summarize_results). The router expects an object `InsightAgent` with
    an `analyze(...)` method; previously that class was removed which caused
    an ImportError and made the API skip insight generation (hence empty
    recommendations). This minimal wrapper delegates to summarize_results
    and preserves the historical API shape.
    """

    def __init__(self):
        # no state for now; kept for future extensibility
        pass

    def analyze(self, *, question, sql_query, results, columns):
        """Analyze query results and return the same structure as summarize_results.

        Args:
            question: original user question
            sql_query: executed SQL
            results: list of row dicts returned by executor
            columns: list of column names

        Returns:
            dict matching the expected insight result shape
        """
        # Delegate to the pure function
        return summarize_results(results, columns, question, sql=sql_query)

