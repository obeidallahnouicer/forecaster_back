"""
api/routers/sql_chat.py

SQL Chat router implementing the strict validation pipeline:
- Input PII validation
- LLM SQL generation (LangChain/Groq, lazy)
- SQL output validation
- Execution (only validated SQL)
- Insights generation (best-effort)

Endpoints:
- POST /api/sql-chat
- GET  /api/sql-chat/health
- GET  /api/sql-chat/schema
- GET  /api/sql-chat/validation-info
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import logging
import time

from agents.query_agent import QueryAgent
from agents.executor_agent import ExecutorAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sql-chat", tags=["sql-chat"])


class SQLChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: Optional[str] = None


class SQLChatResponse(BaseModel):
    success: bool
    question: str
    answer: Optional[str] = None
    sql: Optional[str] = None
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    rows_preview: List[Dict] = Field(default_factory=list)
    rowcount: int = 0
    execution_time_ms: float = 0.0
    validation_status: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


@router.post("", response_model=SQLChatResponse)
async def sql_chat(request: SQLChatRequest):
    """
    Process natural language question with strict validation pipeline.
    
    PIPELINE STEPS:
    1. INPUT VALIDATION: Check for PII → REJECT if unsafe
    2. LLM GENERATION: Generate SQL → REJECT if LLM fails
    3. OUTPUT VALIDATION: Validate SQL → REJECT if invalid/unsafe
    4. EXECUTION: Execute validated SQL → REJECT if execution fails
    5. INSIGHTS: Analyze results → Return insights
    
    Each step can fail - workflow stops immediately on failure.
    
    Args:
        request: SQLChatRequest with question
        
    Returns:
        SQLChatResponse with results or error details
    """
    start = time.time()
    validation_status = {
        "input_validation": "not_started",
        "llm_generation": "not_started",
        "output_validation": "not_started",
        "execution": "not_started",
        "insights": "not_started",
    }

    try:
        logger.info(f"SQL chat request: {request.question}")

        # =====================================================
        # STAGE 1-3: INPUT VALIDATION + LLM + OUTPUT VALIDATION
        # =====================================================
        query_agent = QueryAgent()
        sql_result = query_agent.generate_sql(request.question)

        # Update validation status based on results
        stage = sql_result.get("validation_stage")
        
        if stage == "input_validation":
            validation_status["input_validation"] = "failed"
            elapsed = (time.time() - start) * 1000
            logger.error(f"INPUT VALIDATION FAILED: {sql_result.get('error')}")
            return SQLChatResponse(
                success=False,
                question=request.question,
                error=sql_result.get("error", "Input validation failed"),
                execution_time_ms=elapsed,
                validation_status=validation_status
            )

        validation_status["input_validation"] = "passed"

        if stage == "llm_generation":
            validation_status["llm_generation"] = "failed"
            elapsed = (time.time() - start) * 1000
            logger.error(f"LLM GENERATION FAILED: {sql_result.get('error')}")
            return SQLChatResponse(
                success=False,
                question=request.question,
                error=sql_result.get("error", "SQL generation failed"),
                execution_time_ms=elapsed,
                validation_status=validation_status
            )

        validation_status["llm_generation"] = "passed"

        if stage == "output_validation":
            validation_status["output_validation"] = "failed"
            elapsed = (time.time() - start) * 1000
            logger.error(f"OUTPUT VALIDATION FAILED: {sql_result.get('error')}")
            return SQLChatResponse(
                success=False,
                question=request.question,
                error=sql_result.get("error", "SQL validation failed"),
                sql=sql_result.get("generated_sql"),
                execution_time_ms=elapsed,
                validation_status=validation_status
            )

        validation_status["output_validation"] = "passed"

        # All validations PASSED - extract SQL
        if not sql_result.get("success"):
            raise HTTPException(status_code=500, detail="Unexpected validation failure")

        validated_sql = sql_result["sql"]
        logger.info(f"✓✓✓ ALL VALIDATIONS PASSED - SQL: {validated_sql}")

        # =====================================================
        # STAGE 4: EXECUTION
        # =====================================================
        executor = ExecutorAgent()
        # Pass named parameters produced by the QueryAgent (if any)
        params = sql_result.get('params') or {}
        try:
            exec_result = executor.execute(validated_sql, validation_passed=True, params=params)
        except Exception as e:
            # Catch unexpected execution-time exceptions and return a structured
            # response rather than letting the server return HTTP 500.
            validation_status["execution"] = "failed"
            elapsed = (time.time() - start) * 1000
            logger.exception(f"Unexpected execution exception: {e}")
            return SQLChatResponse(
                success=False,
                question=request.question,
                error=f"Execution exception: {str(e)}",
                sql=validated_sql,
                execution_time_ms=elapsed,
                validation_status=validation_status
            )

        if not exec_result.get("success"):
            validation_status["execution"] = "failed"
            elapsed = (time.time() - start) * 1000
            logger.error(f"EXECUTION FAILED: {exec_result.get('error')}")
            return SQLChatResponse(
                success=False,
                question=request.question,
                error=exec_result.get("error", "Query execution failed"),
                sql=validated_sql,
                execution_time_ms=elapsed,
                validation_status=validation_status
            )

        validation_status["execution"] = "passed"

        results = exec_result.get("data", [])
        rowcount = exec_result.get("row_count", 0)
        columns = exec_result.get("columns", [])

        logger.info(f"✓ EXECUTION COMPLETED: {rowcount} rows returned")

        # =====================================================
        # STAGE 5: INSIGHTS GENERATION
        # =====================================================
        # Insights generation is best-effort - we return results even if it fails
        validation_status["insights"] = "skipped"
        answer = f"Query returned {rowcount} rows."
        insights = []
        recommendations = []

        # Try to generate insights if InsightAgent is available
        try:
            from agents.insight_agent import InsightAgent
            insight_agent = InsightAgent()
            insights_result = insight_agent.analyze(
                question=request.question,
                sql_query=validated_sql,
                results=results,
                columns=columns
            )
            
            validation_status["insights"] = "passed"
            answer = insights_result.get("summary", answer)
            insights = insights_result.get("insights", [])
            recommendations = insights_result.get("recommendations", [])
            logger.info(f"✓ INSIGHTS GENERATED")
            
        except ImportError:
            logger.info("InsightAgent not available - skipping insights")
        except Exception as e:
            logger.warning(f"Insights generation failed: {e}")
            validation_status["insights"] = "failed"

        # =====================================================
        # SUCCESS - RETURN COMPLETE RESPONSE
        # =====================================================
        elapsed = (time.time() - start) * 1000
        
        logger.info("=" * 80)
        logger.info(f"✓✓✓ SQL CHAT COMPLETED SUCCESSFULLY ({elapsed:.1f}ms)")
        logger.info("=" * 80)

        return SQLChatResponse(
            success=True,
            question=request.question,
            answer=answer,
            sql=validated_sql,
            insights=insights,
            recommendations=recommendations,
            rows_preview=results[:10],
            rowcount=rowcount,
            execution_time_ms=elapsed,
            validation_status=validation_status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"SQL chat failed with unexpected error: {e}")
        elapsed = (time.time() - start) * 1000
        
        return SQLChatResponse(
            success=False,
            question=request.question,
            error=f"Unexpected error: {str(e)}",
            execution_time_ms=elapsed,
            validation_status=validation_status
        )


@router.get("/health")
async def health_check():
    """
    Health check for SQL chat service.
    
    Returns:
        Service status and configuration
    """
    try:
        from core.schema_loader import get_schema_snapshot
        from core import config
        
        snapshot = get_schema_snapshot(limit_sample=0)
        
        return {
            "status": "healthy",
            "service": "sql-chat",
            "pipeline": "LangChain + Guardrails AI",
            "llm": config.GROQ_MODEL,
            "validation": {
                "input_validator": "PIIInputValidator",
                "output_validator": "SQLOutputValidator"
            },
            "tables_available": len(snapshot.get('tables', {}))
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@router.get("/schema")
async def get_schema():
    """
    Get available database schema.
    
    Returns:
        Database schema information
    """
    try:
        from core.schema_loader import get_schema_snapshot
        snapshot = get_schema_snapshot(limit_sample=1)
        
        # Format for readability
        tables_info = {}
        for table_name, info in snapshot.get('tables', {}).items():
            tables_info[table_name] = {
                'columns': info.get('columns', []),
                'sample_count': len(info.get('samples', []))
            }
        
        return {
            "tables": tables_info,
            "total_tables": len(tables_info),
            "note": "This is the schema used for SQL generation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation-info")
async def get_validation_info():
    """
    Get information about the validation pipeline.
    
    Returns:
        Details about each validation stage
    """
    return {
        "pipeline": {
            "1_input_validation": {
                "purpose": "Check for PII/sensitive data in user input",
                "action_on_fail": "REJECT - Do not proceed to LLM",
                "validator": "PIIInputValidator",
                "checks": ["email", "phone", "ssn", "credit_card", "sensitive_keywords"]
            },
            "2_llm_generation": {
                "purpose": "Generate SQL from natural language",
                "action_on_fail": "REJECT - Return error to user",
                "llm": "Groq Llama 3.3 70B",
                "framework": "LangChain"
            },
            "3_output_validation": {
                "purpose": "Validate generated SQL for safety and correctness",
                "action_on_fail": "REJECT - Do not execute SQL",
                "validator": "SQLOutputValidator",
                "checks": ["syntax", "sql_injection", "forbidden_operations", "structure"]
            },
            "4_execution": {
                "purpose": "Execute validated SQL query",
                "action_on_fail": "REJECT - Return execution error",
                "mode": "read-only",
                "safety": "Only executes validated queries"
            },
            "5_insights": {
                "purpose": "Generate insights and recommendations",
                "action_on_fail": "Return results without insights",
                "llm": "Groq Llama 3.3 70B",
                "framework": "LangChain"
            }
        },
        "safety_guarantees": [
            "No PII/sensitive data sent to LLM",
            "No invalid SQL executed",
            "No SQL injection possible",
            "Only SELECT queries allowed",
            "All validations must pass before execution"
        ]
    }
