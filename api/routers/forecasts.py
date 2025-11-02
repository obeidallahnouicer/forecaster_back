from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from core.registry import REGISTRY
from api.schemas import UploadResponse, ArticleListResponse
import pandas as pd
from typing import Optional, List
from pathlib import Path
import logging
import math
import numpy as np
import datetime
import time
import hashlib
from pydantic import BaseModel

# Import new cache managers
from cache import UploadCacheManager, ForecastCacheManager

# Global cache directory (project-root / "cache")
# file is at api/routers/forecasts.py -> parents[2] == project root
GLOBAL_CACHE = Path(__file__).resolve().parents[2] / "cache"
GLOBAL_FORECASTS = GLOBAL_CACHE / "forecasts"
GLOBAL_SUMMARY = GLOBAL_CACHE / "summary" / "summary.parquet"

router = APIRouter()
logger = logging.getLogger("app.forecasts")

# Initialize cache managers
upload_cache = UploadCacheManager(
    cache_dir=GLOBAL_CACHE,
    max_retries=3,
    retry_delay=60,
    stale_threshold=3600,  # 1 hour
    ttl=604800  # 7 days
)

forecast_cache = ForecastCacheManager(
    cache_dir=GLOBAL_CACHE,
    default_ttl=86400,  # 24 hours
    model_version="v1"
)


def _parse_metrics_field(val):
    """Parse a metrics field which may be None, a JSON string, or a dict.

    Returns a dict or None.
    """
    if val is None:
        return None
    try:
        # If it's already a dict-like, return as-is
        if isinstance(val, dict):
            return val
        # If it's a string, try to parse as JSON-like
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return None
            # Some stored metric strings use single quotes; normalize
            try:
                import json
                return json.loads(s)
            except Exception:
                # Fallback: attempt to eval safely for simple dicts
                try:
                    return eval(s, {"__builtins__": None}, {})
                except Exception:
                    return None
    except Exception:
        return None


def _parse_json_list_field(val):
    """Parse a JSON string field (e.g., historical_periods, historical_sales).
    
    Returns a list or None.
    """
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return None
            import json
            return json.loads(s)
    except Exception:
        return None


def _ensure_metric_keys(d):
    """Ensure the metric dict contains MAE, MSE, RMSE, MAPE, R2 keys.

    Convert numeric-like values to float where possible; missing keys set to None.
    """
    if d is None:
        return {"MAE": None, "MSE": None, "RMSE": None, "MAPE": None, "R2": None}
    out = {}
    for k in ("MAE", "MSE", "RMSE", "MAPE", "R2"):
        v = None
        # try case-insensitive matches and common lowercase keys
        for key_variant in (k, k.lower(), k.upper()):
            if key_variant in d:
                v = d.get(key_variant)
                break
        # try some common alternative names
        if v is None:
            for alt in ("mae", "mse", "rmse", "mape", "r2"):
                if alt in d:
                    v = d.get(alt)
                    break
        # coerce numeric-like values
        try:
            if v is not None:
                if isinstance(v, (int, float, np.floating, np.integer)):
                    out[k] = float(v)
                else:
                    out[k] = float(str(v))
            else:
                out[k] = None
        except Exception:
            out[k] = None
    return out


def normalize_metrics_in_df(df):
    """Normalize metrics columns in the summary dataframe in-place.

    For each metrics column (sales_sma_metrics, sales_es_metrics, etc., and qty_sma_metrics, qty_es_metrics, etc.),
    parse strings/dicts and expand to JSON/dict with guaranteed keys: MAE,MSE,RMSE,MAPE,R2.
    Also parse JSON list fields for historical data.
    """
    if df is None or df.empty:
        return df

    # Metrics columns for both sales and quantities
    metrics_cols = [
        'sales_sma_metrics', 'sales_es_metrics', 'sales_lr_metrics', 
        'sales_arima_metrics', 'sales_prophet_metrics', 'sales_xgb_metrics',
        'qty_sma_metrics', 'qty_es_metrics', 'qty_lr_metrics',
        'qty_arima_metrics', 'qty_prophet_metrics', 'qty_xgb_metrics'
    ]

    for col in metrics_cols:
        if col not in df.columns:
            continue

        def _norm_cell(v):
            parsed = _parse_metrics_field(v)
            return _ensure_metric_keys(parsed)

        try:
            df[col] = df[col].apply(_norm_cell)
        except Exception:
            # As a last resort, set all rows to None-keys dict
            df[col] = [{"MAE": None, "MSE": None, "RMSE": None, "MAPE": None, "R2": None} for _ in range(len(df))]

    # Parse JSON list fields
    json_list_cols = ['historical_periods', 'historical_sales', 'historical_quantities']
    for col in json_list_cols:
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].apply(_parse_json_list_field)
        except Exception:
            df[col] = [None for _ in range(len(df))]

    return df


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


def normalize_metrics_in_result(res):
    """Normalize per-article forecast result dict in-place.

    Ensures each method metrics field is a dict with keys MAE,MSE,RMSE,MAPE,R2.
    Handles both sales and quantity metrics.
    """
    if not isinstance(res, dict):
        return res
    
    # Metrics columns for both sales and quantities
    metrics_cols = [
        'sales_sma_metrics', 'sales_es_metrics', 'sales_lr_metrics',
        'sales_arima_metrics', 'sales_prophet_metrics', 'sales_xgb_metrics',
        'qty_sma_metrics', 'qty_es_metrics', 'qty_lr_metrics',
        'qty_arima_metrics', 'qty_prophet_metrics', 'qty_xgb_metrics'
    ]
    
    for col in metrics_cols:
        if col in res:
            try:
                parsed = _parse_metrics_field(res.get(col))
                res[col] = _ensure_metric_keys(parsed)
            except Exception:
                res[col] = {"MAE": None, "MSE": None, "RMSE": None, "MAPE": None, "R2": None}
    
    # Parse JSON list fields if present
    json_list_cols = ['historical_periods', 'historical_sales', 'historical_quantities']
    for col in json_list_cols:
        if col in res:
            try:
                res[col] = _parse_json_list_field(res.get(col))
            except Exception:
                res[col] = None
    
    return res


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return _sanitize_value(obj)


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), frequency: str = Form("yearly")):
    """
    Upload dataset with robust cache tracking.
    
    Features:
    - Upload state tracking (pending -> processing -> completed/failed)
    - Automatic retry for failed uploads
    - Persistent upload history
    """
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel")

    # Validate frequency
    if frequency.lower() not in ["yearly", "monthly"]:
        raise HTTPException(status_code=400, detail="Frequency must be 'yearly' or 'monthly'")

    # Generate upload ID
    upload_id = f"upload_{int(time.time())}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
    
    # Save file - use absolute path from project root
    project_root = Path(__file__).resolve().parents[2]
    tmp = project_root / "tmp_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / file.filename
    
    file_content = await file.read()
    file_size = len(file_content)
    
    with out.open("wb") as f:
        f.write(file_content)
    
    # Create upload cache entry (PENDING state)
    upload_cache.create_upload(
        upload_id=upload_id,
        file_path=str(out),
        file_name=file.filename,
        file_size=file_size,
        frequency=frequency.lower(),
        metadata={"original_filename": file.filename}
    )
    logger.info(f"Upload cached: {upload_id} ({file.filename}, {file_size} bytes)")
    
    # Mark as processing
    upload_cache.mark_processing(upload_id)
    
    try:
        # Create session
        info = REGISTRY.create_session_from_file(str(out), frequency=frequency.lower())
        rows = getattr(info.forecaster, "df_raw", None).shape[0] if getattr(info.forecaster, "df_raw", None) is not None else 0
        
        # Mark upload as completed
        upload_cache.mark_completed(upload_id, info.session_id)
        
        logger.info(f"Upload completed: {upload_id} -> session {info.session_id} ({rows} rows)")
        
        return {"session_id": info.session_id, "rows": int(rows)}
        
    except RuntimeError as e:
        # Expected when reading file fails due to encoding/parsing
        error_msg = f"Data read error: {str(e)}"
        upload_cache.mark_failed(upload_id, error_msg)
        logger.exception(f"Failed to create session from upload (read error): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected error - mark for retry
        error_msg = f"Processing error: {str(e)}"
        upload_cache.mark_failed(upload_id, error_msg)
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


class SessionCreateRequest(BaseModel):
    # Server-side path to an uploaded CSV/XLSX file that already exists on the server.
    # If omitted, the endpoint will attempt to use the repository-level 'forecast-summary.csv'.
    file_path: Optional[str] = None
    frequency: str = "yearly"


@router.post("/sessions")
async def create_session(req: SessionCreateRequest):
    """Create a new forecasting session from an existing server-side file.

    Request body:
      - file_path (optional): absolute or repo-relative path to a CSV/XLSX file already present on the server.
      - frequency: 'yearly' or 'monthly' (default 'yearly')

    If file_path is omitted, the endpoint will try to use the repository root file 'forecast-summary.csv'.
    """
    # Resolve file path
    try:
        if req.file_path:
            fp = Path(req.file_path)
            # Allow repo-relative paths
            if not fp.exists():
                repo_root = Path(__file__).resolve().parents[2]
                fp = (repo_root / req.file_path).resolve()

            if not fp.exists():
                raise HTTPException(status_code=400, detail=f"file_path does not exist on server: {req.file_path}")
        else:
            repo_root = Path(__file__).resolve().parents[2]
            fp = repo_root / "forecast-summary.csv"
            if not fp.exists():
                raise HTTPException(status_code=400, detail=(
                    "No file_path provided and repository-level 'forecast-summary.csv' not found. "
                    "Upload via /api/upload or provide an existing server-side file_path."))

        info = REGISTRY.create_session_from_file(str(fp), frequency=req.frequency)
        logger.info(f"Created session {info.session_id} from server file {fp} (frequency={req.frequency})")
        rows = getattr(info.forecaster, "df_raw", None).shape[0] if getattr(info.forecaster, "df_raw", None) is not None else 0
        return {"session_id": info.session_id, "rows": int(rows)}

    except RuntimeError as e:
        logger.exception(f"Failed to create session from file (read error): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create session from file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    """List all active forecast sessions."""
    sessions = REGISTRY.list_sessions()
    
    # Get details for each session
    session_details = []
    for session_id in sessions:
        info = REGISTRY.get(session_id)
        if info:
            session_details.append({
                "session_id": info.session_id,
                "frequency": info.frequency,
                "created_at": info.created_at,
                "rows": info.rows,
                "status": info.status,
                "file_path": Path(info.file_path).name if info.file_path else None
            })
    
    logger.info(f"Listed {len(session_details)} sessions")
    return {
        "count": len(session_details),
        "sessions": session_details
    }


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
    Forecast a single article with production-grade caching.
    
    Query Parameters:
      - ref: Article reference (required)
      - period: SMA period (default 3)
      - alpha: Exponential smoothing alpha (default 0.3)
      - force_recompute: Ignore cache and recompute (default False)
      - fast_mode: Use simplified methods for short series (default True)
      - include_methods: CSV of methods to include, e.g., "SMA,ExpSmoothing,ARIMA"
    
    Returns:
        Full forecast result with all methods, metrics, and historical data
        
    Cache behavior:
      - Cache key includes: ref_article, data_fingerprint, model_version, parameters
      - Automatic invalidation on data/model/parameter changes
      - 24-hour TTL by default
    """
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    
    # Parse include_methods
    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = [m for m in methods_list if m in valid_methods]
        if not methods_list:
            raise HTTPException(status_code=400, detail=f"No valid methods specified. Valid: {','.join(valid_methods)}")
    
    # Build cache parameters (single-user optimized - no session dependency)
    params = {
        "period": period,
        "alpha": alpha,
        "fast_mode": fast_mode,
        "methods": methods_list or "all",
        "frequency": f.frequency
    }
    
    # Compute data fingerprint for cache validation
    data_fingerprint = forecast_cache._compute_fingerprint(f.df_raw)
    
    # Try cache first (unless force_recompute)
    if not force_recompute:
        cached_result = forecast_cache.get_article_forecast(
            ref_article=ref,
            params=params,
            data_fingerprint=data_fingerprint
        )
        if cached_result:
            logger.info(f"[CACHE HIT] Article {ref} forecast")
            try:
                cached_result = normalize_metrics_in_result(cached_result)
            except Exception:
                logger.exception("Failed to normalize metrics for cached article result")
            return JSONResponse(content=sanitize(cached_result))
    
    # Cache miss or force recompute - compute forecast
    start_time = time.time()
    res = f.forecast_article(ref, period=period, alpha=alpha, force_recompute=True,  # Always force in SalesForecaster to bypass its internal cache
                             fast_mode=fast_mode, include_methods=methods_list, return_metrics=True)
    computation_time_ms = (time.time() - start_time) * 1000
    
    if res is None:
        raise HTTPException(status_code=404, detail="Article not found or no historical data")
    
    # Cache the result (single-user: fingerprint stored in metadata, not in key)
    try:
        res = normalize_metrics_in_result(res)
    except Exception:
        logger.exception("Failed to normalize metrics for computed article result")

    forecast_cache.set_article_forecast(
        ref_article=ref,
        forecast_data=res,
        data_fingerprint=data_fingerprint,
        frequency=f.frequency,
        params=params,
        computation_time_ms=computation_time_ms
    )
    
    logger.info(f"[CACHE MISS] Forecast article {ref}: sales_avg={res.get('sales_avg_forecast')}, "
               f"qty_avg={res.get('qty_avg_forecast')}, computed in {computation_time_ms:.0f}ms")
    
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
    Forecast all articles with production-grade caching.
    
    Form Parameters:
      - period: SMA period (default 3)
      - alpha: Exponential smoothing alpha (default 0.3)
      - force_recompute: Ignore cache and recompute all (default False)
      - fast_mode: Use simplified methods for short series (default True)
      - include_methods: CSV of methods, e.g., "SMA,ExpSmoothing,ARIMA"
    
    Returns:
        Summary with count, preview (first 10), and top products by forecast
        
    Cache behavior:
      - Summary cache key includes: data_fingerprint, model_version, parameters
      - Automatic invalidation on data/model/parameter changes
      - 24-hour TTL by default
    """
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster

    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]
        valid_methods = {'SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'}
        methods_list = [m for m in methods_list if m in valid_methods]
        if not methods_list:
            raise HTTPException(status_code=400, detail=f"No valid methods specified. Valid: {','.join(valid_methods)}")
    
    # Build cache parameters (single-user optimized)
    params = {
        "period": period,
        "alpha": alpha,
        "fast_mode": fast_mode,
        "methods": methods_list or "all",
        "frequency": f.frequency
    }
    
    # Compute data fingerprint for validation
    data_fingerprint = forecast_cache._compute_fingerprint(f.df_raw)
    
    # Try cache first (unless force_recompute)
    if not force_recompute:
        cached_summary = forecast_cache.get_summary_forecast(
            params=params,
            data_fingerprint=data_fingerprint
        )
        if cached_summary is not None:
            df = cached_summary
            logger.info(f"[CACHE HIT] Summary forecast ({len(df)} articles)")
            # Normalize metrics in cached dataframe to ensure fields exist
            try:
                df = normalize_metrics_in_df(df)
            except Exception:
                logger.exception("Failed to normalize metrics for cached summary")

            # Build response from cached data
            preview = df.head(10).to_dict(orient='records') if not df.empty else []
            preview = sanitize(preview)
            
            top_products = []
            try:
                if not df.empty and 'sales_avg_forecast' in df.columns:
                    top_df = df.sort_values('sales_avg_forecast', ascending=False).head(5)
                    top_products = top_df[['ref_article', 'designation', 'sales_avg_forecast']].to_dict(orient='records')
                    top_products = sanitize(top_products)
            except Exception:
                logger.exception("Failed to compute top_products from cached summary")
            
            return JSONResponse(content={
                "count": int(len(df)),
                "preview": preview,
                "top_products": top_products,
                "cached": True
            })
    
    # Cache miss or force recompute - compute forecasts
    def progress(i, tot):
        pass

    try:
        start_time = time.time()
        df = f.forecast_all_articles(period=period, alpha=alpha, force_recompute=True,  # Always force in SalesForecaster
                                     fast_mode=fast_mode, include_methods=methods_list, progress_callback=progress)
        computation_time_ms = (time.time() - start_time) * 1000

        # Normalize metrics in the computed dataframe before caching/saving
        try:
            df = normalize_metrics_in_df(df)
        except Exception:
            logger.exception("Failed to normalize metrics for computed summary")

        logger.info(f"[CACHE MISS] Forecasted all articles: {len(df)} results, "
                   f"computed in {computation_time_ms:.0f}ms (period={period}, alpha={alpha}, fast_mode={fast_mode}, methods={methods_list})")

        # Cache the summary (single-user: fingerprint in metadata, not key)
        forecast_cache.set_summary_forecast(
            summary_df=df,
            data_fingerprint=data_fingerprint,
            frequency=f.frequency,
            params=params,
            computation_time_ms=computation_time_ms
        )

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
        if not df.empty and 'sales_avg_forecast' in df.columns:
            top_df = df.sort_values('sales_avg_forecast', ascending=False).head(5)
            top_products = top_df[['ref_article', 'designation', 'sales_avg_forecast']].to_dict(orient='records')
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


# ==================== CACHE MANAGEMENT ENDPOINTS ====================

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics for uploads and forecasts"""
    try:
        upload_stats = upload_cache.get_stats()
        forecast_stats = forecast_cache.get_stats()
        
        return {
            "uploads": upload_stats,
            "forecasts": forecast_stats
        }
    except Exception as e:
        logger.exception(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/uploads")
async def list_cached_uploads(status: Optional[str] = None, frequency: Optional[str] = None):
    """
    List cached uploads with optional filters.
    
    Query params:
    - status: Filter by status (pending, processing, completed, failed)
    - frequency: Filter by frequency (yearly, monthly)
    """
    try:
        uploads = upload_cache.list_uploads(status=status, frequency=frequency)
        
        return {
            "count": len(uploads),
            "uploads": [u.to_dict() for u in uploads]
        }
    except Exception as e:
        logger.exception(f"Failed to list uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/uploads/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get status of specific upload"""
    try:
        entry = upload_cache.get_upload(upload_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return entry.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/uploads/{upload_id}/retry")
async def retry_failed_upload(upload_id: str):
    """Retry a failed upload"""
    try:
        entry = upload_cache.get_upload(upload_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        if not upload_cache.can_retry(upload_id):
            return {
                "success": False,
                "message": f"Upload cannot be retried (status={entry.status}, retries={entry.retry_count})"
            }
        
        # Attempt to reprocess
        upload_cache.mark_processing(upload_id)
        
        try:
            info = REGISTRY.create_session_from_file(entry.file_path, frequency=entry.frequency)
            rows = getattr(info.forecaster, "df_raw", None).shape[0] if getattr(info.forecaster, "df_raw", None) is not None else 0
            
            upload_cache.mark_completed(upload_id, info.session_id)
            
            logger.info(f"Upload retry successful: {upload_id} -> session {info.session_id}")
            
            return {
                "success": True,
                "session_id": info.session_id,
                "rows": int(rows)
            }
        except Exception as e:
            error_msg = f"Retry failed: {str(e)}"
            upload_cache.mark_failed(upload_id, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to retry upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/uploads/{upload_id}")
async def delete_cached_upload(upload_id: str):
    """Delete cached upload entry"""
    try:
        deleted = upload_cache.delete_upload(upload_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/uploads/cleanup")
async def cleanup_stale_uploads():
    """Cleanup stale uploads that are stuck in processing"""
    try:
        count = upload_cache.cleanup_stale()
        return {
            "cleaned_up": count,
            "message": f"Cleaned up {count} stale uploads"
        }
    except Exception as e:
        logger.exception(f"Failed to cleanup uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/forecasts/invalidate")
async def invalidate_forecast_cache(
    ref_article: Optional[str] = None,
    frequency: Optional[str] = None
):
    """
    Invalidate forecast cache (single-user optimized).
    
    Query params:
    - ref_article: Invalidate for specific article
    - frequency: Invalidate for specific frequency (yearly/monthly)
    - If none specified, invalidates all
    
    Note: Since this is a single-user system, cache keys are not session-based.
    """
    try:
        count = 0
        
        if ref_article:
            # Invalidate specific article (all parameter combinations)
            count = forecast_cache.invalidate_article(ref_article)
        elif frequency:
            # Invalidate by frequency
            count = forecast_cache.invalidate_all(frequency=frequency)
        else:
            # Invalidate all forecasts
            count = forecast_cache.invalidate_all()
        
        return {
            "invalidated": count,
            "message": f"Invalidated {count} forecast cache entries"
        }
    except Exception as e:
        logger.exception(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_all_cache():
    """Clear all cache (uploads and forecasts) - use with caution!"""
    try:
        upload_count = upload_cache.cache.clear()
        forecast_count = forecast_cache.cache.clear()
        
        return {
            "cleared": {
                "uploads": upload_count,
                "forecasts": forecast_count
            },
            "message": f"Cleared {upload_count} uploads and {forecast_count} forecasts"
        }
    except Exception as e:
        logger.exception(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    csv_path = Path(__file__).resolve().parents[2] / "chatbotdf.csv"
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
        # Normalize metrics to ensure consumers always receive standard metric keys
        try:
            df = normalize_metrics_in_df(df)
        except Exception:
            logger.exception("Failed to normalize metrics for monthly computed summary")

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
