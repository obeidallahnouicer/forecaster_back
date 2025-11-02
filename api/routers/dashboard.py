"""
Dashboard API for serving forecast data with filters, sorting, pagination, and aggregated metrics.

Endpoints:
  - GET /documents: Filtered, sorted, paginated forecast rows
  - GET /metrics: Aggregated metrics over filtered rows
  - GET /status: Forecast cache metadata

Note: This reads from the session's forecast cache (via REGISTRY), not hardcoded files.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query, HTTPException

from core.registry import REGISTRY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

# Global cache directory
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
GLOBAL_SUMMARY = CACHE_DIR / "summary" / "summary.parquet"


def load_forecast_summary(session_id: Optional[str] = None, frequency: str = "yearly") -> pd.DataFrame:
    """
    Load forecast summary from session or global cache.
    
    Priority:
    1. If session_id provided: Load from that session's forecaster
    2. Otherwise: Load from global summary cache (cache/summary/summary.parquet)
    
    Args:
        session_id: Optional session ID to load specific session data
        frequency: Frequency filter (yearly/monthly) - only used for global cache
    
    Returns:
        DataFrame with forecast summary
    """
    try:
        # Option 1: Load from specific session
        if session_id:
            info = REGISTRY.get(session_id)
            if not info:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
            
            forecaster = info.forecaster
            if not forecaster:
                raise HTTPException(status_code=500, detail=f"Forecaster not initialized for session '{session_id}'")
            
            # Try session-specific cache first
            summary_path = forecaster._summary_cache_path()
            if summary_path.exists():
                df = pd.read_parquet(summary_path)
                logger.info(f"Loaded summary from session cache: {len(df)} rows (session: {session_id})")
                return df.copy()
            
            # Fallback to generating summary
            df = forecaster.generate_summary(force_recompute=False)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"No forecast data available for session '{session_id}'")
            
            logger.info(f"Generated summary for session {session_id}: {len(df)} rows")
            return df.copy()
        
        # Option 2: Load from global cache
        if GLOBAL_SUMMARY.exists():
            df = pd.read_parquet(GLOBAL_SUMMARY)
            
            # Apply frequency filter if needed
            if frequency and 'frequency' in df.columns:
                df = df[df['frequency'] == frequency]
            
            logger.info(f"Loaded global summary cache: {len(df)} rows (frequency: {frequency})")
            return df.copy()
        
        # No data available
        raise HTTPException(
            status_code=404,
            detail="No forecast summary available. Please run forecast generation first via /api/forecasts/forecast/all/{session_id} endpoint."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to load forecast summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load forecast summary: {str(e)}")


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters."""
    if isinstance(text, str) and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def apply_filters(df: pd.DataFrame, 
                  marque: Optional[str] = None,
                  famille: Optional[str] = None,
                  next_period: Optional[str] = None,
                  frequency: Optional[str] = None) -> pd.DataFrame:
    """Apply exact-match filters to DataFrame.
    
    Updated to work with new dual forecasting structure (sales + quantities).
    """
    filtered_df = df.copy()
    
    filters_applied = {}
    
    if marque:
        if 'marque' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["marque"] == marque]
            filters_applied["marque"] = marque
    
    if famille:
        if 'famille' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["famille"] == famille]
            filters_applied["famille"] = famille
    
    if next_period:
        if 'next_period' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["next_period"].astype(str) == str(next_period)]
            filters_applied["next_period"] = next_period
    
    if frequency:
        if 'frequency' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["frequency"] == frequency]
            filters_applied["frequency"] = frequency
    
    if filters_applied:
        logger.info(f"Applied filters: {filters_applied}, result: {len(filtered_df)} rows")
    
    return filtered_df


def apply_sorting(df: pd.DataFrame,
                  sort_by: Optional[str] = None,
                  sort_order: str = "asc") -> pd.DataFrame:
    """Sort DataFrame by a numeric column."""
    if sort_by:
        if sort_by not in df.columns:
            logger.warning(f"Sort column '{sort_by}' not found in DataFrame")
            return df
        
        ascending = sort_order.lower() in ["asc", "ascending"]
        sorted_df = df.sort_values(by=sort_by, ascending=ascending)
        logger.info(f"Sorted by {sort_by} ({sort_order})")
        return sorted_df
    
    return df


def apply_pagination(df: pd.DataFrame, limit: int = 20, offset: int = 0) -> tuple:
    """Paginate DataFrame and return (sliced_df, total_count)."""
    total = len(df)
    
    # Enforce safe max limit
    max_limit = 100
    if limit > max_limit:
        logger.warning(f"Limit {limit} exceeds max {max_limit}, capping")
        limit = max_limit
    
    if limit < 1:
        limit = 20
    
    if offset < 0:
        offset = 0
    
    sliced_df = df.iloc[offset : offset + limit]
    
    logger.info(f"Paginated: offset={offset}, limit={limit}, total={total}, returned={len(sliced_df)}")
    
    return sliced_df, total


@router.get("/status")
def get_status(
    session_id: Optional[str] = Query(None, description="Session ID to load specific forecast data"),
    frequency: str = Query("yearly", description="Forecast frequency (yearly/monthly)")
):
    """
    Get forecast summary metadata: existence, row count, and column names.
    
    Query Parameters:
      - session_id: Optional session ID to load specific forecast
      - frequency: Fallback frequency if no session_id provided
    
    Returns:
        {
            "summary_exists": bool,
            "total_rows": int,
            "columns": List[str],
            "frequency": str,
            "session_id": str (if provided)
        }
    """
    try:
        df = load_forecast_summary(session_id=session_id, frequency=frequency)
        response = {
            "summary_exists": True,
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "frequency": frequency,
            "data_source": "session_cache" if session_id else "global_cache"
        }
        if session_id:
            response["session_id"] = session_id
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@router.get("/documents")
def get_documents(
    session_id: Optional[str] = Query(None, description="Session ID to load specific forecast data"),
    marque: Optional[str] = Query(None, description="Filter by marque (exact match)"),
    famille: Optional[str] = Query(None, description="Filter by famille (exact match)"),
    next_period: Optional[str] = Query(None, description="Filter by next_period"),
    frequency: str = Query("yearly", description="Forecast frequency (yearly/monthly)"),
    sort_by: Optional[str] = Query(None, description="Sort by column (e.g., sales_avg_forecast, qty_avg_forecast)"),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    limit: int = Query(20, description="Number of rows to return (max 100)"),
    offset: int = Query(0, description="Number of rows to skip"),
) -> Dict[str, Any]:
    """
    Get filtered, sorted, and paginated forecast rows with truncated text.
    
    Updated to work with dual forecasting (sales AND quantities).
    
    Query Parameters:
      - session_id: Optional session ID to load specific forecast
      - marque: Filter by marque (exact match)
      - famille: Filter by famille (exact match)
      - next_period: Filter by next forecast period
      - frequency: Forecast frequency (yearly/monthly) - DEFAULT: yearly
      - sort_by: Sort by column (e.g., sales_avg_forecast, qty_avg_forecast)
      - sort_order: asc or desc
      - limit: Rows per page (max 100)
      - offset: Pagination offset
    
    Returns:
        {
            "data": List[Dict],
            "total": int,
            "limit": int,
            "offset": int,
            "data_source": str
        }
    """
    try:
        # Load from session or global cache
        df = load_forecast_summary(session_id=session_id, frequency=frequency)
        total_docs = len(df)
        
        logger.info(f"GET /documents: marque={marque}, famille={famille}, next_period={next_period}, "
                   f"frequency={frequency}, sort_by={sort_by}, sort_order={sort_order}, "
                   f"limit={limit}, offset={offset}")
        
        # Apply filters (frequency filter not needed since we already loaded that frequency)
        filtered_df = apply_filters(df, marque=marque, famille=famille, 
                                   next_period=next_period, frequency=None)
        filtered_docs = len(filtered_df)
        
        # Apply sorting
        sorted_df = apply_sorting(filtered_df, sort_by=sort_by, sort_order=sort_order)
        
        # Apply pagination
        paginated_df, _ = apply_pagination(sorted_df, limit=limit, offset=offset)
        
        # Truncate text columns and convert to dict
        documents = []
        for _, row in paginated_df.iterrows():
            documents.append({
                "ref_article": str(row.get("ref_article", "")),
                "designation": truncate_text(str(row.get("designation", "")), max_length=500),
                "marque": str(row.get("marque", "")),
                "famille": str(row.get("famille", "")),
                "frequency": str(row.get("frequency", "")),
                "next_period": str(row.get("next_period", "")),
                "sales_avg_forecast": float(row.get("sales_avg_forecast", 0)),
                "qty_avg_forecast": float(row.get("qty_avg_forecast", 0)),
                "sales_trend_pct": float(row.get("sales_trend_pct", 0)),
                "qty_trend_pct": float(row.get("qty_trend_pct", 0)),
                "data_points": int(row.get("data_points", 0))
            })
        
        return {
            "data": documents,
            "total": total_docs,
            "filtered": filtered_docs,
            "limit": limit,
            "offset": offset,
            "data_source": "forecaster_response"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@router.get("/metrics")
def get_metrics(
    session_id: Optional[str] = Query(None, description="Session ID to load specific forecast data"),
    marque: Optional[str] = Query(None, description="Filter by marque (exact match)"),
    famille: Optional[str] = Query(None, description="Filter by famille (exact match)"),
    next_period: Optional[str] = Query(None, description="Filter by next_period"),
    frequency: str = Query("yearly", description="Forecast frequency (yearly/monthly)"),
) -> Dict[str, Any]:
    """
    Get rich aggregated metrics over filtered rows.
    
    Updated to work with dual forecasting (sales AND quantities).
    
    Includes:
      - Total rows and mean forecasts (sales & quantities)
      - Sales and quantity trend statistics
      - Top articles by sales and quantity forecasts
      - Aggregations by famille and marque
      - Products grouped by famille and marque (for dashboard dropdowns)
      - Average data_points
    
    Query Parameters:
      - marque: Filter by marque
      - famille: Filter by famille
      - next_period: Filter by next_period
      - frequency: Forecast frequency (yearly/monthly) - DEFAULT: yearly
    
    Returns:
        {
            "total_rows": int,
            "sales_avg_forecast": float,        # Mean of sales_avg_forecast
            "qty_avg_forecast": float,          # Mean of qty_avg_forecast
            "avg_data_points": float,
            "sales_trend_avg": float,           # Average sales trend %
            "qty_trend_avg": float,             # Average quantity trend %
            "top_articles_by_sales": [
                {"ref_article": str, "designation": str, "sales_avg_forecast": float},
                ...
            ],
            "top_articles_by_qty": [
                {"ref_article": str, "designation": str, "qty_avg_forecast": float},
                ...
            ],
            "top_marques_by_sales": [
                {"marque": str, "mean_sales_forecast": float, "count": int},
                ...
            ],
            "top_marques_by_qty": [
                {"marque": str, "mean_qty_forecast": float, "count": int},
                ...
            ],
            "top_familles_by_sales": [
                {"famille": str, "mean_sales_forecast": float, "count": int},
                ...
            ],
            "top_familles_by_qty": [
                {"famille": str, "mean_qty_forecast": float, "count": int},
                ...
            ],
            "products_by_famille": {
                "LISSAGE": ["designation1", "designation2", ...],
                ...
            },
            "products_by_marque": {
                "CADIVEU": ["designation1", "designation2", ...],
                ...
            }
        }
    """
    try:
        logger.info(f"GET /metrics: session_id={session_id}, marque={marque}, famille={famille}, "
                   f"next_period={next_period}, frequency={frequency}")
        
        # Load from session or global cache
        df = load_forecast_summary(session_id=session_id, frequency=frequency)
        
        # Apply filters (frequency filter not needed since we already loaded that frequency)
        filtered_df = apply_filters(df, marque=marque, famille=famille, 
                                   next_period=next_period, frequency=None)
        
        total_rows = len(filtered_df)
        
        if total_rows == 0:
            logger.warning("No rows match the given filters")
            return {
                "total_rows": 0,
                "sales_avg_forecast": 0,
                "qty_avg_forecast": 0,
                "avg_data_points": 0,
                "sales_trend_avg": 0,
                "qty_trend_avg": 0,
                "top_articles_by_sales": [],
                "top_articles_by_qty": [],
                "top_marques_by_sales": [],
                "top_marques_by_qty": [],
                "top_familles_by_sales": [],
                "top_familles_by_qty": [],
                "products_by_famille": {},
                "products_by_marque": {}
            }
        
        # Basic aggregations
        sales_avg_forecast = float(filtered_df["sales_avg_forecast"].mean()) if "sales_avg_forecast" in filtered_df.columns else 0
        qty_avg_forecast = float(filtered_df["qty_avg_forecast"].mean()) if "qty_avg_forecast" in filtered_df.columns else 0
        avg_data_points = float(filtered_df["data_points"].mean()) if "data_points" in filtered_df.columns else 0
        sales_trend_avg = float(filtered_df["sales_trend_pct"].mean()) if "sales_trend_pct" in filtered_df.columns else 0
        qty_trend_avg = float(filtered_df["qty_trend_pct"].mean()) if "qty_trend_pct" in filtered_df.columns else 0
        
        # Top articles by sales forecast (top 10)
        top_articles_by_sales = []
        if "sales_avg_forecast" in filtered_df.columns:
            top_sales_df = filtered_df.nlargest(10, "sales_avg_forecast")[
                ["ref_article", "designation", "sales_avg_forecast"]
            ]
            top_articles_by_sales = [
                {
                    "ref_article": str(row["ref_article"]),
                    "designation": truncate_text(str(row["designation"]), max_length=200),
                    "sales_avg_forecast": float(row["sales_avg_forecast"])
                }
                for _, row in top_sales_df.iterrows()
            ]
        
        # Top articles by quantity forecast (top 10)
        top_articles_by_qty = []
        if "qty_avg_forecast" in filtered_df.columns:
            top_qty_df = filtered_df.nlargest(10, "qty_avg_forecast")[
                ["ref_article", "designation", "qty_avg_forecast"]
            ]
            top_articles_by_qty = [
                {
                    "ref_article": str(row["ref_article"]),
                    "designation": truncate_text(str(row["designation"]), max_length=200),
                    "qty_avg_forecast": float(row["qty_avg_forecast"])
                }
                for _, row in top_qty_df.iterrows()
            ]
        
        # Top marques by mean sales forecast
        top_marques_by_sales = []
        if "marque" in filtered_df.columns and "sales_avg_forecast" in filtered_df.columns:
            top_marques_sales_df = filtered_df.groupby("marque").agg({
                "sales_avg_forecast": ["mean", "count"]
            }).reset_index()
            top_marques_sales_df.columns = ["marque", "mean_sales_forecast", "count"]
            top_marques_sales_df = top_marques_sales_df.nlargest(10, "mean_sales_forecast")
            top_marques_by_sales = [
                {
                    "marque": str(row["marque"]),
                    "mean_sales_forecast": float(row["mean_sales_forecast"]),
                    "count": int(row["count"])
                }
                for _, row in top_marques_sales_df.iterrows()
            ]
        
        # Top marques by mean quantity forecast
        top_marques_by_qty = []
        if "marque" in filtered_df.columns and "qty_avg_forecast" in filtered_df.columns:
            top_marques_qty_df = filtered_df.groupby("marque").agg({
                "qty_avg_forecast": ["mean", "count"]
            }).reset_index()
            top_marques_qty_df.columns = ["marque", "mean_qty_forecast", "count"]
            top_marques_qty_df = top_marques_qty_df.nlargest(10, "mean_qty_forecast")
            top_marques_by_qty = [
                {
                    "marque": str(row["marque"]),
                    "mean_qty_forecast": float(row["mean_qty_forecast"]),
                    "count": int(row["count"])
                }
                for _, row in top_marques_qty_df.iterrows()
            ]
        
        # Top familles by mean sales forecast
        top_familles_by_sales = []
        if "famille" in filtered_df.columns and "sales_avg_forecast" in filtered_df.columns:
            top_familles_sales_df = filtered_df.groupby("famille").agg({
                "sales_avg_forecast": ["mean", "count"]
            }).reset_index()
            top_familles_sales_df.columns = ["famille", "mean_sales_forecast", "count"]
            top_familles_sales_df = top_familles_sales_df.nlargest(10, "mean_sales_forecast")
            top_familles_by_sales = [
                {
                    "famille": str(row["famille"]),
                    "mean_sales_forecast": float(row["mean_sales_forecast"]),
                    "count": int(row["count"])
                }
                for _, row in top_familles_sales_df.iterrows()
            ]
        
        # Top familles by mean quantity forecast
        top_familles_by_qty = []
        if "famille" in filtered_df.columns and "qty_avg_forecast" in filtered_df.columns:
            top_familles_qty_df = filtered_df.groupby("famille").agg({
                "qty_avg_forecast": ["mean", "count"]
            }).reset_index()
            top_familles_qty_df.columns = ["famille", "mean_qty_forecast", "count"]
            top_familles_qty_df = top_familles_qty_df.nlargest(10, "mean_qty_forecast")
            top_familles_by_qty = [
                {
                    "famille": str(row["famille"]),
                    "mean_qty_forecast": float(row["mean_qty_forecast"]),
                    "count": int(row["count"])
                }
                for _, row in top_familles_qty_df.iterrows()
            ]
        
        # Products by famille (for dropdown)
        products_by_famille_dict = {}
        if "famille" in filtered_df.columns and "designation" in filtered_df.columns:
            for famille in filtered_df["famille"].dropna().unique():
                products = filtered_df[filtered_df["famille"] == famille]["designation"].dropna().unique().tolist()
                products_by_famille_dict[str(famille)] = [str(p) for p in products]
        
        # Products by marque (for dropdown)
        products_by_marque_dict = {}
        if "marque" in filtered_df.columns and "designation" in filtered_df.columns:
            for marque in filtered_df["marque"].dropna().unique():
                products = filtered_df[filtered_df["marque"] == marque]["designation"].dropna().unique().tolist()
                products_by_marque_dict[str(marque)] = [str(p) for p in products]
        
        return {
            "total_rows": total_rows,
            "sales_avg_forecast": sales_avg_forecast,
            "qty_avg_forecast": qty_avg_forecast,
            "avg_data_points": avg_data_points,
            "sales_trend_avg": sales_trend_avg,
            "qty_trend_avg": qty_trend_avg,
            "top_articles_by_sales": top_articles_by_sales,
            "top_articles_by_qty": top_articles_by_qty,
            "top_marques_by_sales": top_marques_by_sales,
            "top_marques_by_qty": top_marques_by_qty,
            "top_familles_by_sales": top_familles_by_sales,
            "top_familles_by_qty": top_familles_by_qty,
            "products_by_famille": products_by_famille_dict,
            "products_by_marque": products_by_marque_dict,
            "data_source": "forecaster_response"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")
