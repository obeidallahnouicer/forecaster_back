from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.core.registry import REGISTRY
from app.schemas import UploadResponse, ArticleListResponse, ForecastArticleRequest, ForecastArticleResponse, SummaryResponse
import pandas as pd
from typing import Optional, List
from pathlib import Path
import logging
import math
import numpy as np
import pandas as pd
import datetime

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
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel")

    # save temp
    tmp = Path("./tmp_uploads")
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / file.filename
    with out.open("wb") as f:
        f.write(await file.read())

    try:
        info = REGISTRY.create_session_from_file(str(out))
        logger.info(f"Created session {info.session_id} from upload {file.filename}")
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
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    # parse include_methods (comma-separated) into list expected by SalesForecaster
    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]
    # Prefer on-disk per-article cache when available (and not forcing recompute)
    res = None
    try:
        if not force_recompute:
            cached_df = f._load_forecast_cache(ref, use_cache=True)
            if cached_df is not None and not cached_df.empty:
                # single-row cache
                res = cached_df.iloc[0].to_dict()
    except Exception:
        # fall back to computing if cache read fails
        res = None

    if res is None:
        res = f.forecast_article(ref, period=period, alpha=alpha, force_recompute=force_recompute, fast_mode=fast_mode,
                                 include_methods=methods_list)
    if res is None:
        raise HTTPException(status_code=404, detail="Article not found or no historical data")
    logger.info(f"Forecast article {ref} in session {session_id}: avg={res.get('avg_forecast')}")
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
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster

    def progress(i, tot):
        pass

    methods_list = None
    if include_methods:
        methods_list = [m.strip() for m in include_methods.split(",") if m.strip()]

    # If a summary cache exists and recompute not requested, return it immediately
    try:
        summary_path = f._summary_cache_path()
        if (not force_recompute) and summary_path.exists():
            df = pd.read_parquet(summary_path)
            logger.info(f"Loaded summary cache for session {session_id}: {len(df)} rows")
        else:
            df = f.forecast_all_articles(period=period, alpha=alpha, force_recompute=force_recompute,
                                         fast_mode=fast_mode, include_methods=methods_list, progress_callback=progress)
            logger.info(f"Forecasted all articles for session {session_id}: {len(df)} results")
    except Exception as e:
        logger.exception(f"Failed to compute or read summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    preview = df.head(10).to_dict(orient='records') if not df.empty else []
    preview = sanitize(preview)
    return JSONResponse(content={"count": int(len(df)), "preview": preview})


@router.get("/summary/{session_id}")
async def get_summary(session_id: str, force_recompute: Optional[bool] = False):
    info = REGISTRY.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    f = info.forecaster
    # Prefer reading on-disk summary cache when available unless force_recompute
    summary_path = f._summary_cache_path()
    df = None
    if (not force_recompute) and summary_path.exists():
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
    df.to_csv(out, index=False)
    logger.info(f"Wrote summary CSV for session {session_id} to {out}")
    return FileResponse(str(out), filename="summary.csv", media_type="text/csv")


@router.get("/health")
async def health():
    return {"status": "ok"}
