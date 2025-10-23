"""
Dashboard API for serving forecast data with filters, sorting, pagination, and aggregated metrics.

Endpoints:
  - GET /documents: Filtered, sorted, paginated forecast rows
  - GET /metrics: Aggregated metrics over filtered rows
  - GET /status: CSV metadata
"""

import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

# Path to the forecast CSV
CSV_PATH = Path(__file__).resolve().parents[2] / "forecast-summary.csv"

# Cache for the dataframe (reload on demand or if file changes)
_df_cache = None
_csv_mtime_cache = None


def load_csv():
    """Load CSV into a pandas DataFrame with basic validation."""
    global _df_cache, _csv_mtime_cache
    
    if not CSV_PATH.exists():
        logger.error(f"CSV not found at {CSV_PATH}")
        raise HTTPException(status_code=404, detail=f"CSV file not found: {CSV_PATH}")
    
    try:
        current_mtime = os.path.getmtime(CSV_PATH)
        # Reload if file changed or cache is empty
        if _df_cache is None or _csv_mtime_cache != current_mtime:
            logger.info(f"Loading CSV from {CSV_PATH}")
            df = pd.read_csv(CSV_PATH)
            _df_cache = df
            _csv_mtime_cache = current_mtime
            logger.info(f"Loaded {len(df)} rows from CSV")
        return _df_cache.copy()
    except Exception as e:
        logger.exception(f"Failed to load CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load CSV: {str(e)}")


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters."""
    if isinstance(text, str) and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def apply_filters(df: pd.DataFrame, 
                  marque: Optional[str] = None,
                  famille: Optional[str] = None,
                  trend_label: Optional[str] = None,
                  next_year: Optional[int] = None) -> pd.DataFrame:
    """Apply exact-match filters to DataFrame."""
    filtered_df = df.copy()
    
    filters_applied = {}
    
    if marque:
        filtered_df = filtered_df[filtered_df["marque"] == marque]
        filters_applied["marque"] = marque
    
    if famille:
        filtered_df = filtered_df[filtered_df["famille"] == famille]
        filters_applied["famille"] = famille
    
    if trend_label:
        filtered_df = filtered_df[filtered_df["trend_label"] == trend_label]
        filters_applied["trend_label"] = trend_label
    
    if next_year is not None:
        filtered_df = filtered_df[filtered_df["next_year"] == next_year]
        filters_applied["next_year"] = next_year
    
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
def get_status():
    """
    Get CSV metadata: existence, row count, and column names.
    
    Returns:
        {
            "csv_exists": bool,
            "csv_path": str,
            "total_rows": int,
            "columns": List[str]
        }
    """
    try:
        df = load_csv()
        return {
            "csv_exists": True,
            "csv_path": str(CSV_PATH),
            "total_rows": len(df),
            "columns": df.columns.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@router.get("/documents")
def get_documents(
    marque: Optional[str] = Query(None, description="Filter by marque (exact match)"),
    famille: Optional[str] = Query(None, description="Filter by famille (exact match)"),
    trend_label: Optional[str] = Query(None, description="Filter by trend_label: Uptrend/Downtrend/Stable"),
    next_year: Optional[int] = Query(None, description="Filter by next_year"),
    sort_by: Optional[str] = Query(None, description="Sort by column (e.g., avg_forecast, trend_pct)"),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    limit: int = Query(20, description="Number of rows to return (max 100)"),
    offset: int = Query(0, description="Number of rows to skip"),
) -> Dict[str, Any]:
    """
    Get filtered, sorted, and paginated forecast rows with truncated text.
    
    Query Parameters:
      - marque: Filter by marque (exact match)
      - famille: Filter by famille (exact match)
      - trend_label: Filter by trend_label (Uptrend, Downtrend, Stable)
      - next_year: Filter by next_year
      - sort_by: Sort by numeric column (avg_forecast, trend_pct, data_points, etc.)
      - sort_order: asc or desc
      - limit: Rows per page (default 20, max 100)
      - offset: Skip N rows (default 0)
    
    Returns:
        {
            "total_docs": int,           # Total rows in CSV
            "filtered_docs": int,        # Rows after filtering
            "limit": int,
            "offset": int,
            "documents": [               # Paginated rows
                {
                    "ref_article": str,
                    "designation": str,  # Truncated to 500 chars
                    "marque": str,
                    "famille": str,
                    "next_year": int,
                    "avg_forecast": float,
                    "trend_pct": float,
                    "trend_label": str,
                    "data_points": int
                }
            ]
        }
    """
    try:
        logger.info(f"GET /documents: marque={marque}, famille={famille}, trend_label={trend_label}, "
                   f"next_year={next_year}, sort_by={sort_by}, sort_order={sort_order}, "
                   f"limit={limit}, offset={offset}")
        
        df = load_csv()
        total_docs = len(df)
        
        # Apply filters
        filtered_df = apply_filters(df, marque=marque, famille=famille, 
                                   trend_label=trend_label, next_year=next_year)
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
                "next_year": int(row.get("next_year", 0)),
                "avg_forecast": float(row.get("avg_forecast", 0)),
                "trend_pct": float(row.get("trend_pct", 0)),
                "trend_label": str(row.get("trend_label", "")),
                "data_points": int(row.get("data_points", 0))
            })
        
        return {
            "total_docs": total_docs,
            "filtered_docs": filtered_docs,
            "limit": limit,
            "offset": offset,
            "documents": documents
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@router.get("/metrics")
def get_metrics(
    marque: Optional[str] = Query(None, description="Filter by marque (exact match)"),
    famille: Optional[str] = Query(None, description="Filter by famille (exact match)"),
    trend_label: Optional[str] = Query(None, description="Filter by trend_label: Uptrend/Downtrend/Stable"),
    next_year: Optional[int] = Query(None, description="Filter by next_year"),
) -> Dict[str, Any]:
    """
    Get rich aggregated metrics over filtered rows.
    
    Includes:
      - Total rows and mean avg_forecast (per-row average)
      - Overall trend label (based on majority of trend_labels)
      - Trend label counts
      - Top articles by avg_forecast and trend_pct
      - Aggregations by famille and marque (mean-based)
      - Products grouped by famille and marque (for dashboard dropdowns)
      - Average data_points
    
    Query Parameters:
      - marque: Filter by marque
      - famille: Filter by famille
      - trend_label: Filter by trend_label
      - next_year: Filter by next_year
    
    Returns:
        {
            "total_rows": int,
            "total_avg_forecast": float,        # Mean of avg_forecast (per row)
            "overall_trend": str,               # Majority trend: "Uptrend", "Downtrend", "Stable", or "Unknown"
            "avg_data_points": float,
            "trend_counts": {"Uptrend": int, "Downtrend": int, "Stable": int},
            "top_articles": [
                {"ref_article": str, "designation": str, "avg_forecast": float},
                ...
            ],
            "top_marques": [
                {"marque": str, "mean_avg_forecast": float, "count": int},
                ...
            ],
            "top_familles": [
                {"famille": str, "mean_avg_forecast": float, "count": int},
                ...
            ],
            "products_by_famille": {
                "LISSAGE": ["designation1", "designation2", ...],
                ...
            },
            "products_by_marque": {
                "CADIVEU": ["designation1", "designation2", ...],
                ...
            },
            "avg_forecast_by_famille": {
                "LISSAGE": float,
                ...
            },
            "avg_forecast_by_marque": {
                "CADIVEU": float,
                ...
            },
            "top_trend_pct_articles": [
                {"ref_article": str, "designation": str, "trend_pct": float},
                ...
            ]
        }
    """
    try:
        logger.info(f"GET /metrics: marque={marque}, famille={famille}, trend_label={trend_label}, "
                   f"next_year={next_year}")
        
        df = load_csv()
        
        # Apply filters
        filtered_df = apply_filters(df, marque=marque, famille=famille, 
                                   trend_label=trend_label, next_year=next_year)
        
        total_rows = len(filtered_df)
        
        if total_rows == 0:
            logger.warning("No rows match the given filters")
            return {
                "total_rows": 0,
                "total_avg_forecast": 0,
                "overall_trend": "Unknown",
                "avg_data_points": 0,
                "trend_counts": {},
                "top_articles": [],
                "top_marques": [],
                "top_familles": [],
                "products_by_famille": {},
                "products_by_marque": {},
                "avg_forecast_by_famille": {},
                "avg_forecast_by_marque": {},
                "top_trend_pct_articles": []
            }
        
        # Basic aggregations
        total_avg_forecast = float(filtered_df["avg_forecast"].mean())
        avg_data_points = float(filtered_df["data_points"].mean())
        
        # Trend counts and overall trend label
        trend_counts = filtered_df["trend_label"].value_counts().to_dict()
        trend_counts = {k: int(v) for k, v in trend_counts.items()}
        
        # Compute overall_trend based on majority
        if trend_counts:
            overall_trend = max(trend_counts, key=trend_counts.get)
        else:
            overall_trend = "Unknown"
        
        # Top articles by avg_forecast (top 10)
        top_articles_df = filtered_df.nlargest(10, "avg_forecast")[
            ["ref_article", "designation", "avg_forecast"]
        ]
        top_articles = [
            {
                "ref_article": str(row["ref_article"]),
                "designation": truncate_text(str(row["designation"]), max_length=200),
                "avg_forecast": float(row["avg_forecast"])
            }
            for _, row in top_articles_df.iterrows()
        ]
        
        # Top articles by trend_pct (top 10)
        top_trend_pct_df = filtered_df.nlargest(10, "trend_pct")[
            ["ref_article", "designation", "trend_pct"]
        ]
        top_trend_pct_articles = [
            {
                "ref_article": str(row["ref_article"]),
                "designation": truncate_text(str(row["designation"]), max_length=200),
                "trend_pct": float(row["trend_pct"])
            }
            for _, row in top_trend_pct_df.iterrows()
        ]
        
        # Top marques by mean avg_forecast
        top_marques_df = filtered_df.groupby("marque").agg({
            "avg_forecast": ["mean", "count"]
        }).reset_index()
        top_marques_df.columns = ["marque", "mean_avg_forecast", "count"]
        top_marques_df = top_marques_df.nlargest(10, "mean_avg_forecast")
        top_marques = [
            {
                "marque": str(row["marque"]),
                "mean_avg_forecast": float(row["mean_avg_forecast"]),
                "count": int(row["count"])
            }
            for _, row in top_marques_df.iterrows()
        ]
        logger.info(f"Computed top_marques (by mean avg_forecast): {len(top_marques)} marques")
        
        # Top familles by mean avg_forecast
        top_familles_df = filtered_df.groupby("famille").agg({
            "avg_forecast": ["mean", "count"]
        }).reset_index()
        top_familles_df.columns = ["famille", "mean_avg_forecast", "count"]
        top_familles_df = top_familles_df.nlargest(10, "mean_avg_forecast")
        top_familles = [
            {
                "famille": str(row["famille"]),
                "mean_avg_forecast": float(row["mean_avg_forecast"]),
                "count": int(row["count"])
            }
            for _, row in top_familles_df.iterrows()
        ]
        logger.info(f"Computed top_familles (by mean avg_forecast): {len(top_familles)} familles")
        
        # Products by famille (for dropdown)
        products_by_famille_dict = {}
        for famille in filtered_df["famille"].unique():
            products = filtered_df[filtered_df["famille"] == famille]["designation"].unique().tolist()
            products_by_famille_dict[str(famille)] = [str(p) for p in products]
        
        # Products by marque (for dropdown)
        products_by_marque_dict = {}
        for marque in filtered_df["marque"].unique():
            products = filtered_df[filtered_df["marque"] == marque]["designation"].unique().tolist()
            products_by_marque_dict[str(marque)] = [str(p) for p in products]
        
        # Average avg_forecast by famille
        avg_forecast_by_famille = {}
        for famille in filtered_df["famille"].unique():
            avg_val = float(filtered_df[filtered_df["famille"] == famille]["avg_forecast"].mean())
            avg_forecast_by_famille[str(famille)] = avg_val
        
        # Average avg_forecast by marque
        avg_forecast_by_marque = {}
        for marque in filtered_df["marque"].unique():
            avg_val = float(filtered_df[filtered_df["marque"] == marque]["avg_forecast"].mean())
            avg_forecast_by_marque[str(marque)] = avg_val
        
        return {
            "total_rows": total_rows,
            "total_avg_forecast": total_avg_forecast,
            "overall_trend": overall_trend,
            "avg_data_points": avg_data_points,
            "trend_counts": trend_counts,
            "top_articles": top_articles,
            "top_marques": top_marques,
            "top_familles": top_familles,
            "products_by_famille": products_by_famille_dict,
            "products_by_marque": products_by_marque_dict,
            "avg_forecast_by_famille": avg_forecast_by_famille,
            "avg_forecast_by_marque": avg_forecast_by_marque,
            "top_trend_pct_articles": top_trend_pct_articles
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")
