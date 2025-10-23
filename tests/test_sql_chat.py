"""
Integration test for Text-to-SQL chatbot.

Tests the complete pipeline:
1. Natural language -> SQL generation
2. SQL validation
3. Query execution
4. Insight generation
"""
import pytest
from agents.query_agent import generate_sql
from agents.sql_validator import validate_sql
from agents.executor_agent import execute_query
from agents.insight_agent import summarize_results


def test_sql_generation():
    """Test that QueryAgent generates valid SQL."""
    result = generate_sql("How many products are in the stock table?")
    
    assert 'sql' in result
    assert 'params' in result
    assert 'explanation' in result
    assert 'SELECT' in result['sql'].upper()


def test_sql_validation():
    """Test SQL validator accepts SELECT and rejects dangerous queries."""
    # Valid SELECT
    valid, reason, adjusted = validate_sql("SELECT * FROM t_stock")
    assert valid is True
    assert 'LIMIT' in adjusted.upper()  # Should add LIMIT
    
    # Invalid: semicolon
    valid, reason, adjusted = validate_sql("SELECT * FROM t_stock; DROP TABLE t_stock")
    assert valid is False
    assert 'semicolon' in reason.lower()
    
    # Invalid: not SELECT
    valid, reason, adjusted = validate_sql("DELETE FROM t_stock")
    assert valid is False


def test_query_execution():
    """Test ExecutorAgent executes queries safely."""
    sql = "SELECT COUNT(*) as count FROM t_base LIMIT 1"
    result = execute_query(sql, {})
    
    assert 'columns' in result
    assert 'rows' in result
    assert 'rowcount' in result
    assert result['rowcount'] >= 0


def test_insight_generation():
    """Test InsightAgent generates meaningful insights."""
    rows = [
        {'product': 'A', 'revenue': 1000},
        {'product': 'B', 'revenue': 2000},
        {'product': 'C', 'revenue': 1500}
    ]
    columns = ['product', 'revenue']
    
    insights = summarize_results(rows, columns, "What are the top products?")
    
    assert 'summary' in insights
    assert 'insights' in insights
    assert 'recommendations' in insights
    assert 'kpis' in insights
    assert insights['row_count'] == 3


def test_full_pipeline():
    """Test complete Text-to-SQL pipeline end-to-end."""
    question = "Show me 5 products from the base table"
    
    # Step 1: Generate SQL
    generated = generate_sql(question)
    sql = generated['sql']
    params = generated.get('params', {})
    
    # Step 2: Validate
    valid, reason, adjusted = validate_sql(sql)
    assert valid, f"SQL validation failed: {reason}"
    
    # Step 3: Execute
    result = execute_query(adjusted, params, max_rows=5)
    assert result['rowcount'] >= 0
    
    # Step 4: Analyze
    analysis = summarize_results(
        result['rows'], 
        result['columns'], 
        question,
        adjusted
    )
    
    assert analysis['summary']
    assert isinstance(analysis['insights'], list)
    assert isinstance(analysis['recommendations'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
