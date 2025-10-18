from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.core.registry import REGISTRY
from app.schemas import UploadResponse, ArticleListResponse, ForecastArticleRequest, ForecastArticleResponse, SummaryResponse
import pandas as pd
from rag_chatbot.data_loader import load_and_preprocess
from typing import Optional, List
from pathlib import Path
import logging
import math
import numpy as np
import datetime
import hashlib

# Global cache directory (project-root / "cache")
# file is at app/api/routers/forecasts.py -> parents[3] == project root
GLOBAL_CACHE = Path(__file__).resolve().parents[3] / "cache"
GLOBAL_FORECASTS = GLOBAL_CACHE / "forecasts"
GLOBAL_SUMMARY = GLOBAL_CACHE / "summary" / "summary.parquet"

router = APIRouter()
logger = logging.getLogger("app.forecasts")


def _sanitize_value(v):
    # numpy / pandas scalar types
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        try:
            if isinstance(v, np.floating):
                val = float(v)
                return val if math.isfinite(val) else None
            if isinstance(v, np.integer):
                return int(v)
            return bool(v)
        except Exception:
            return None

    # native float
    if isinstance(v, float):
        return v if math.isfinite(v) else None

    # pandas Period objects (for monthly aggregation)
    if isinstance(v, pd.Period):
        try:
            return str(v)
        except Exception:
            return None

    # datetimes: pandas Timestamp, numpy.datetime64, builtin datetime
    if isinstance(v, (pd.Timestamp, np.datetime64, datetime.datetime)):
        try:
            ts = pd.Timestamp(v)
            if pd.isna(ts):
                return None
            return ts.isoformat()
        except Exception:
            try:
                return str(v)
            except Exception:
                return None

    # numpy arrays, pandas Series or Index -> list
    if isinstance(v, (np.ndarray, pd.Series, pd.Index)):
        try:
            return [_sanitize_value(x) for x in list(v)]
        except Exception:
            return None

    # python lists/tuples/sets -> list
    if isinstance(v, (list, tuple, set)):
        try:
            return [_sanitize_value(x) for x in v]
        except Exception:
            return None

    # bytes -> decode
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode('utf-8', errors='replace')
        except Exception:
            return None

    # dicts handled by sanitize(), leave other types as-is
    return v


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return _sanitize_value(obj)


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), frequency: str = Form("yearly")):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel")

    # Validate frequency
    if frequency.lower() not in ["yearly", "monthly"]:
        raise HTTPException(status_code=400, detail="Frequency must be 'yearly' or 'monthly'")

    # save temp
    tmp = Path("./tmp_uploads")
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / file.filename
    with out.open("wb") as f:
        f.write(await file.read())

    try:
        info = REGISTRY.create_session_from_file(str(out), frequency=frequency.lower())
        logger.info(f"Created session {info.session_id} from upload {file.filename} (frequency={frequency})")
        # use the loaded dataframe to report rows to avoid re-reading with wrong encoding
        rows = getattr(info.forecaster, "df_raw", None).shape[0] if getattr(info.forecaster, "df_raw", None) is not None else 0
        return {"session_id": info.session_id, "rows": int(rows)}
    except RuntimeError as e:
        # expected when reading file fails due to encoding/parsing
        logger.exception(f"Failed to create session from upload (read error): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create session from upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/articles/{session_id}", response_model=ArticleListResponse)
async def list_articles(session_id: str):
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    if f.grouped_data is None:
        f.prepare_data()
    articles = sorted(f.grouped_data[f.ref_col].unique().tolist())
    logger.info(f"List articles for session {session_id}: {len(articles)} articles")
    return {"count": len(articles), "articles": articles}


@router.get("/sessions")
async def list_sessions():
    keys = REGISTRY.list_sessions()
    return {"count": len(keys), "sessions": keys}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = REGISTRY.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info(f"Deleted session {session_id}")
    return {"detail": "deleted"}


@router.get("/forecast/article/{session_id}")
async def forecast_article(session_id: str, ref: str, period: int = 3, alpha: float = 0.3,
                           force_recompute: Optional[bool] = False, fast_mode: Optional[bool] = True,
                           include_methods: Optional[str] = None):
    """
    Forecast a single article with full parameter control.
    
    Query Parameters:
      - ref: Article reference (required)
      - period: SMA period (default 3)
      - alpha: Exponential smoothing alpha (default 0.3)
      - force_recompute: Ignore cache and recompute (default False)
      - fast_mode: Use simplified methods for short series (default True)
      - include_methods: CSV of methods to include, e.g., "SMA,ExpSmoothing,ARIMA"
    
    Returns:
        Full forecast result with all methods, metrics, and historical data
    """
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    
    # Parse include_methods (comma-separated) into list expected by SalesForecaster
    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]
        # Validate method names
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = [m for m in methods_list if m in valid_methods]
        if not methods_list:
            raise HTTPException(status_code=400, detail=f"No valid methods specified. Valid: {','.join(valid_methods)}")

    # ✅ FIX: Respect force_recompute parameter (was hardcoded to True)
    res = f.forecast_article(ref, period=period, alpha=alpha, force_recompute=force_recompute, 
                             fast_mode=fast_mode, include_methods=methods_list, return_metrics=True)
    if res is None:
        raise HTTPException(status_code=404, detail="Article not found or no historical data")
    logger.info(f"Forecast article {ref} in session {session_id}: avg={res.get('avg_forecast')}, trend={res.get('trend_label')}, "
               f"force_recompute={force_recompute}, methods={methods_list}")
    return JSONResponse(content=sanitize(res))


@router.get("/data/{session_id}")
async def preview_data(session_id: str):
    """Return first 10 rows of the uploaded raw dataset for the session."""
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    df_raw = getattr(f, "df_raw", None)
    if df_raw is None or df_raw.empty:
        return JSONResponse(content={"count": 0, "rows": []})
    rows = df_raw.head(10).to_dict(orient="records")
    rows = sanitize(rows)
    logger.info(f"Preview data requested for session {session_id}: {len(rows)} rows")
    return JSONResponse(content={"count": int(len(rows)), "rows": rows})


@router.post("/forecast/all/{session_id}")
async def forecast_all(session_id: str, period: int = Form(3), alpha: float = Form(0.3),
                       force_recompute: Optional[bool] = Form(False), fast_mode: Optional[bool] = Form(True),
                       include_methods: Optional[str] = Form(None)):
    """
    Forecast all articles with full parameter control.
    
    Form Parameters:
      - period: SMA period (default 3)
      - alpha: Exponential smoothing alpha (default 0.3)
      - force_recompute: Ignore cache and recompute all (default False)
      - fast_mode: Use simplified methods for short series (default True)
      - include_methods: CSV of methods, e.g., "SMA,ExpSmoothing,ARIMA"
    
    Returns:
        Summary with count, preview (first 10), and top products by forecast
    """
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster

    def progress(i, tot):
        pass

    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]
        # Validate method names
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = [m for m in methods_list if m in valid_methods]
        if not methods_list:
            raise HTTPException(status_code=400, detail=f"No valid methods specified. Valid: {','.join(valid_methods)}")

    # ✅ FIX: Respect force_recompute parameter (was hardcoded to True)
    try:
        df = f.forecast_all_articles(period=period, alpha=alpha, force_recompute=force_recompute,
                                     fast_mode=fast_mode, include_methods=methods_list, progress_callback=progress)
        cache_mode = "bypassed" if force_recompute else "used"
        logger.info(f"Forecasted all articles for session {session_id}: {len(df)} results (cache {cache_mode}, "
                   f"period={period}, alpha={alpha}, fast_mode={fast_mode}, methods={methods_list})")
    except Exception as e:
        logger.exception(f"Failed to compute forecasts for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Persist session-specific summary cache and a project-global summary for RAG ingestion
    try:
        # session-specific
        summary_path = f._summary_cache_path()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(summary_path, index=False)
        logger.info(f"Wrote session summary cache to {summary_path}")
    except Exception:
        logger.exception("Failed to write session summary cache")

    try:
        # project-global summary (overwrite)
        GLOBAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(GLOBAL_SUMMARY, index=False)
        logger.info(f"Wrote global summary cache to {GLOBAL_SUMMARY}")
    except Exception:
        logger.exception("Failed to write global summary cache")

    # NOTE: We intentionally DO NOT trigger RAG reindexing here to avoid
    # indexing during compute-heavy forecast operations. Reindexing the
    # vectorstore is now an explicit action and will only be performed when
    # the user downloads the summary (see download_summary) or calls the
    # dedicated /reindex endpoint.

    preview = df.head(10).to_dict(orient='records') if not df.empty else []
    preview = sanitize(preview)

    # Compute top products by ensemble average forecast so callers can immediately answer questions
    top_products = []
    try:
        if not df.empty and 'avg_forecast' in df.columns:
            top_df = df.sort_values('avg_forecast', ascending=False).head(5)
            top_products = top_df[['ref_article', 'designation', 'avg_forecast']].to_dict(orient='records')
            top_products = sanitize(top_products)
    except Exception:
        logger.exception("Failed to compute top_products from summary dataframe")

    return JSONResponse(content={
        "count": int(len(df)),
        "preview": preview,
        "top_products": top_products,
        "note": "RAG reindexing disabled during forecast. To index this summary for RAG, download the summary or call the /reindex endpoint."
    })


@router.get("/summary/{session_id}")
async def get_summary(session_id: str, force_recompute: Optional[bool] = False):
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    # Prefer reading on-disk summary cache when available unless force_recompute
    summary_path = f._summary_cache_path()
    df = None
    # 1) try project-global summary cache
    if not force_recompute:
        try:
            if GLOBAL_SUMMARY.exists():
                df = pd.read_parquet(GLOBAL_SUMMARY)
                logger.info(f"Loaded global summary cache: {len(df)} rows")
        except Exception:
            df = None

    # 2) fallback to session-specific summary cache
    if df is None and (not force_recompute) and summary_path.exists():
        try:
            df = pd.read_parquet(summary_path)
            logger.info(f"Loaded summary cache for session {session_id}: {len(df)} rows")
        except Exception:
            df = None

    if df is None:
        df = f.generate_summary(force_recompute=force_recompute)

    if df is None or df.empty:
        return JSONResponse(content={"count": 0, "rows": []})

    logger.info(f"Generated summary for session {session_id}: {len(df)} rows")
    rows = df.to_dict(orient='records')
    rows = sanitize(rows)
    return JSONResponse(content={"count": int(len(df)), "rows": rows})


@router.get("/download/summary/{session_id}")
async def download_summary(session_id: str):
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    df = f.generate_summary(force_recompute=False)
    out = Path("./server_data") / session_id / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote summary CSV for session {session_id} to {out}")

    # Build/update the RAG vectorstore to index exactly this downloadable summary.
    # This ensures the vectorstore contains only the summary (not other datasets)
    # and avoids indexing during forecasting operations.
    try:
        from rag_chatbot import retriever as _retriever
        _store = _retriever.build_vectorstore(recreate=True, df=df)
        logger.info(f"Reindexed vectorstore from summary for session {session_id}: ok={_store is not None}")
    except Exception:
        logger.exception("Failed to build vectorstore from summary for session %s", session_id)

    return FileResponse(str(out), filename="summary.csv", media_type="text/csv")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/chatbotdf")
async def get_chatbotdf(limit: Optional[int] = 1000, preview: Optional[bool] = False, columns: Optional[str] = None,
                        page: Optional[int] = 1, page_size: Optional[int] = 1000):
    """Return rows from project-level `chatbotdf.csv` (raw sales data for dashboard).

    Query params:
    - limit: max rows to return (default 1000, set to 0 for all)
    - preview: if true, returns only the head(limit) rows
    - columns: comma-separated list of columns to include (optional)
    - page: page number for pagination (default 1, starts from 1)
    - page_size: rows per page (default 1000, overrides limit if set)
    """
    csv_path = Path(__file__).resolve().parents[3] / "chatbotdf.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="chatbotdf.csv not found")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        logger.exception("Failed to load chatbotdf.csv: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    if df is None or df.empty:
        return JSONResponse(content={"count": 0, "rows": []})

    total = len(df)

    # filter columns if requested
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        # keep only existing columns
        cols = [c for c in cols if c in df.columns]
        if cols:
            df = df[cols]

    # Determine effective limit
    effective_limit = limit
    if page_size and page_size > 0:
        effective_limit = page_size
    elif limit == 0:
        effective_limit = total  # all rows

    # Apply preview or limit
    if preview or (effective_limit > 0 and effective_limit < total):
        df_out = df.head(effective_limit)
    else:
        df_out = df

    # Apply pagination if page > 1
    if page and page > 1 and page_size and page_size > 0:
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df_out = df.iloc[start_idx:end_idx]

    rows = df_out.to_dict(orient="records")
    rows = sanitize(rows)

    return JSONResponse(content={
        "count": int(total),
        "returned": int(len(rows)),
        "page": page or 1,
        "page_size": page_size or effective_limit,
        "rows": rows
    })


@router.get("/download/chatbotdf")
async def download_chatbotdf(columns: Optional[str] = None):
    """Download chatbotdf.csv as CSV file.

    Query params:
    - columns: comma-separated list of columns to include (optional, all if not specified)
    """
    try:
        df = load_and_preprocess()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="chatbotdf.csv not found")
    except Exception as e:
        logger.exception("Failed to load chatbotdf: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    # filter columns if requested
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df = df[cols]

    # Write to temp CSV
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    logger.info(f"Prepared chatbotdf CSV download: {len(df)} rows")
    return FileResponse(temp_path, filename="chatbotdf.csv", media_type="text/csv")


# ==================== MONTHLY SUPPORT ENDPOINTS ====================

@router.get("/monthly/summary/{session_id}")
async def get_monthly_summary(session_id: str):
    """
    Get monthly aggregated summary for a session.
    
    Returns:
    - Total articles
    - Total monthly periods
    - Aggregate sales by month
    - Trend analysis by month
    """
    try:
        info = REGISTRY.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        f = info.forecaster
        if f.frequency != "monthly":
            raise HTTPException(status_code=400, detail="Session is not monthly frequency")
        
        # Get the grouped data (already aggregated by period)
        if f.grouped_data is None:
            f.prepare_data()
        
        df = f.grouped_data.copy()
        
        # Group by period and sum sales
        monthly_summary = df.groupby('period').agg({
            f.sales_col: 'sum',
            f.ref_col: 'nunique'
        }).reset_index()
        
        monthly_summary.columns = ['period', 'total_sales', 'article_count']
        monthly_summary = monthly_summary.sort_values('period')
        
        # Calculate trend (compare last 3 months to previous 3 months)
        if len(monthly_summary) >= 6:
            recent_avg = monthly_summary['total_sales'].tail(3).mean()
            previous_avg = monthly_summary['total_sales'].iloc[-6:-3].mean()
            trend_pct = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        else:
            trend_pct = 0.0
        
        result = {
            "session_id": session_id,
            "frequency": "monthly",
            "total_articles": int(df[f.ref_col].nunique()),
            "total_periods": len(monthly_summary),
            "total_sales": float(monthly_summary['total_sales'].sum()),
            "avg_monthly_sales": float(monthly_summary['total_sales'].mean()),
            "trend_pct": trend_pct,
            "date_range_start": str(monthly_summary.iloc[0]['period']),
            "date_range_end": str(monthly_summary.iloc[-1]['period']),
            "monthly_breakdown": sanitize(monthly_summary.to_dict(orient='records'))
        }
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting monthly summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monthly/articles/{session_id}/{ref}")
async def get_monthly_article(session_id: str, ref: str):
    """
    Get monthly data and forecast for a single article.
    
    Parameters:
    - session_id: Session identifier
    - ref: Article reference code
    
    Query params:
    - period: SMA window (default 3)
    - alpha: Exponential smoothing parameter (default 0.3)
    - include_methods: CSV of methods to include (default all)
    """
    try:
        info = REGISTRY.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        f = info.forecaster
        if f.frequency != "monthly":
            raise HTTPException(status_code=400, detail="Session is not monthly frequency")
        
        # Get monthly data for article from grouped_data
        if f.grouped_data is None:
            f.prepare_data()
        
        df = f.grouped_data[f.grouped_data[f.ref_col] == str(ref)].copy()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Article {ref} not found")
        
        df = df.sort_values('period')
        
        # Get article metadata
        metadata = df.iloc[0]
        
        # Calculate monthly trend
        if len(df) >= 2:
            trend_pct = ((df[f.sales_col].iloc[-1] - df[f.sales_col].iloc[-2]) / 
                        df[f.sales_col].iloc[-2] * 100) if df[f.sales_col].iloc[-2] > 0 else 0
        else:
            trend_pct = 0.0
        
        # Build response
        monthly_data = df[['period', f.sales_col]].copy()
        monthly_data.columns = ['period', 'sales']
        
        result = {
            "ref_article": ref,
            "designation": metadata.get('Designation') if 'Designation' in df.columns else None,
            "marque": metadata.get('Marque') if 'Marque' in df.columns else None,
            "famille": metadata.get('Famille') if 'Famille' in df.columns else None,
            "frequency": "monthly",
            "total_sales": float(monthly_data['sales'].sum()),
            "avg_monthly_sales": float(monthly_data['sales'].mean()),
            "max_monthly_sales": float(monthly_data['sales'].max()),
            "min_monthly_sales": float(monthly_data['sales'].min()),
            "data_points": len(monthly_data),
            "trend_pct": trend_pct,
            "trend_label": "uptrend" if trend_pct > 5 else "downtrend" if trend_pct < -5 else "stable",
            "date_range_start": str(monthly_data.iloc[0]['period']),
            "date_range_end": str(monthly_data.iloc[-1]['period']),
            "monthly_history": sanitize(monthly_data.to_dict(orient='records'))
        }
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting monthly article data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monthly/periods/{session_id}")
async def get_monthly_periods(session_id: str):
    """
    Get list of all monthly periods available in the session.
    
    Returns list of period strings (e.g., ['2024-01', '2024-02', ...])
    """
    try:
        info = REGISTRY.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        f = info.forecaster
        if f.frequency != "monthly":
            raise HTTPException(status_code=400, detail="Session is not monthly frequency")
        
        # Get unique periods from grouped_data
        if f.grouped_data is None:
            f.prepare_data()
        
        periods = sorted(f.grouped_data['period'].unique())
        periods_str = [str(p) for p in periods]
        
        return JSONResponse(content={
            "session_id": session_id,
            "frequency": "monthly",
            "total_periods": len(periods_str),
            "periods": periods_str
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting monthly periods: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monthly/forecast/article/{session_id}/{ref}")
async def forecast_monthly_article(
    session_id: str,
    ref: str,
    period: int = Form(3),
    alpha: float = Form(0.3),
    force_recompute: bool = Form(False),
    fast_mode: bool = Form(True),
    include_methods: Optional[str] = Form(None)
):
    """
    Forecast next month for a single article with full parameter control.
    
    Parameters (via form):
    - period: SMA window size (default 3)
    - alpha: Exponential smoothing parameter (default 0.3)
    - force_recompute: Bypass cache and recompute (default False)
    - fast_mode: Skip expensive models for short series <6 points (default True)
    - include_methods: CSV of methods to include (default all)
      Valid: 'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'
    
    Returns:
    - next_month forecast
    - Per-method forecasts
    - Ensemble (average) forecast
    - Historical monthly data
    - Trend analysis
    """
    try:
        info = REGISTRY.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        f = info.forecaster
        if f.frequency != "monthly":
            raise HTTPException(status_code=400, detail="Session is not monthly frequency")
        
        # Validate parameters
        if not (1 <= period <= 12):
            raise HTTPException(status_code=400, detail="period must be between 1 and 12")
        if not (0.1 <= alpha <= 0.9):
            raise HTTPException(status_code=400, detail="alpha must be between 0.1 and 0.9")
        
        # Parse methods
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = None
        if include_methods:
            methods_list = [m.strip() for m in include_methods.split(',')]
            invalid = set(methods_list) - valid_methods
            if invalid:
                raise HTTPException(status_code=400, detail=f"Invalid methods: {invalid}")
        
        # Forecast the article
        logger.info(f"[MONTHLY FORECAST] ref={ref}, session={session_id}, "
                   f"period={period}, alpha={alpha}, "
                   f"force_recompute={force_recompute}, fast_mode={fast_mode}, "
                   f"methods={methods_list or 'all'}")
        
        res = f.forecast_article(
            ref, 
            period=period, 
            alpha=alpha, 
            force_recompute=force_recompute,
            fast_mode=fast_mode,
            include_methods=methods_list,
            return_metrics=True
        )
        
        if res is None:
            raise HTTPException(status_code=404, detail=f"Article {ref} not found")
        
        result = sanitize(res)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error forecasting monthly article: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monthly/forecast/all/{session_id}")
async def forecast_all_monthly(
    session_id: str,
    period: Optional[str] = Form(None),
    alpha: Optional[str] = Form(None),
    force_recompute: Optional[str] = Form(None),
    fast_mode: Optional[str] = Form(None),
    include_methods: Optional[str] = Form(None)
):
    """
    Forecast all articles for next month with full parameter control.
    
    Parameters (via form):
    - period: SMA window size (default 3)
    - alpha: Exponential smoothing parameter (default 0.3)
    - force_recompute: Bypass cache and recompute all (default False)
    - fast_mode: Skip expensive models for short series <6 points (default True)
    - include_methods: CSV of methods to include (default all)
      Valid: 'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'
    
    Returns:
    - count: total forecasts generated
    - preview: first 10 forecasts
    - top_products: top 5 by ensemble average forecast
    """
    try:
        info = REGISTRY.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        f = info.forecaster
        if f.frequency != "monthly":
            raise HTTPException(status_code=400, detail="Session is not monthly frequency")
        
        # Coerce form string values to proper types with defaults
        try:
            period = int(period) if period else 3
        except (ValueError, TypeError):
            period = 3
        
        try:
            alpha = float(alpha) if alpha else 0.3
        except (ValueError, TypeError):
            alpha = 0.3
        
        force_recompute = force_recompute and force_recompute.lower() in ('true', '1', 'yes') if force_recompute else False
        fast_mode = fast_mode and fast_mode.lower() in ('true', '1', 'yes') if fast_mode else True
        
        # Validate parameters
        if not (1 <= period <= 12):
            raise HTTPException(status_code=400, detail="period must be between 1 and 12")
        if not (0.1 <= alpha <= 0.9):
            raise HTTPException(status_code=400, detail="alpha must be between 0.1 and 0.9")
        
        # Parse methods
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = None
        if include_methods:
            methods_list = [m.strip() for m in include_methods.split(',')]
            invalid = set(methods_list) - valid_methods
            if invalid:
                raise HTTPException(status_code=400, detail=f"Invalid methods: {invalid}")
        
        # Progress callback stub
        def progress(i, tot):
            pass
        
        # Forecast all articles for monthly
        logger.info(f"[MONTHLY FORECAST ALL] session={session_id}, "
                   f"period={period}, alpha={alpha}, "
                   f"force_recompute={force_recompute}, fast_mode={fast_mode}, "
                   f"methods={methods_list or 'all'}")
        
        df = f.forecast_all_articles(
            period=period,
            alpha=alpha,
            force_recompute=force_recompute,
            fast_mode=fast_mode,
            include_methods=methods_list,
            progress_callback=progress
        )
        
        logger.info(f"Forecasted all articles (monthly) for session {session_id}: {len(df)} results")
        
        # Prepare response
        preview = df.head(10).to_dict(orient='records') if not df.empty else []
        preview = sanitize(preview)
        
        # Compute top products
        top_products = []
        try:
            if not df.empty and 'avg_forecast' in df.columns:
                top_df = df.sort_values('avg_forecast', ascending=False).head(5)
                top_products = top_df[['ref_article', 'designation', 'avg_forecast']].to_dict(orient='records')
                top_products = sanitize(top_products)
        except Exception:
            logger.exception("Failed to compute top_products from monthly summary")
        
        return JSONResponse(content={
            "count": int(len(df)),
            "preview": preview,
            "top_products": top_products,
            "frequency": "monthly"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error forecasting all monthly articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))
