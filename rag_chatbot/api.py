from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from . import chatbot, config
from .config import API_HOST, API_PORT
from . import retriever
import traceback
from pathlib import Path
from app.api.routers.forecasts import sanitize

# Import sales_forecaster lazily inside the endpoint to avoid import-time failures


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ResetRequest(BaseModel):
    thread_id: str


app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required")
    # Fail fast with a friendly message if the LLM API key is not configured
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail=(
            "GROQ_API_KEY is not set. Add your Groq API key to a `.env` file at the project root or set the "
            "environment variable GROQ_API_KEY. Example in PowerShell:\n"
            "  $env:GROQ_API_KEY = 'your_key_here'\n"
            "Or create a `.env` containing: GROQ_API_KEY=your_key_here"
        ))

    logger = logging.getLogger("rag.api")
    logger.info("/chat called thread_id=%s message=%s", req.thread_id, req.message[:80])
    try:
        res = chatbot.chat(req.message, req.thread_id)
        logger.info("/chat success thread_id=%s", req.thread_id)
        return sanitize(res)
    except RuntimeError as e:
        logger.error("/chat runtime error for thread_id=%s: %s", req.thread_id, e)
        # likely missing API key or retriever/index issue
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("/chat failed for thread_id=%s: %s", req.thread_id, e)
        # return a friendly 500 with the error message
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    chatbot.reset_memory(req.thread_id)
    return {"status": "reset", "thread_id": req.thread_id}


@app.post("/reindex")
def reindex_endpoint():
    """Trigger reindexing / embedding of the CSV into the vectorstore.

    This will build the Chroma index and cache it under cache/vectorstore/chroma.
    """
    try:
        # Prefer indexing a project-level CSV named 'forecast-summary.csv' if
        # present at the repository root (this is the single file the user
        # requested to be RAG-indexed). If it's missing, fall back to the
        # cached summary parquet file.
        project_root = Path(__file__).resolve().parents[1]
        forecast_csv = project_root / "forecast-summary.csv"
        summary_path = project_root / "cache" / "summary" / "summary.parquet"
        df = None

        # Try repository-level CSV first
        if forecast_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(forecast_csv, low_memory=False)
            except Exception:
                df = None

        # Fallback to cached parquet summary
        if df is None and summary_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(summary_path)
            except Exception:
                df = None

        if df is None:
            # Fail fast: do not index other raw datasets by default.
            raise RuntimeError("No forecast summary found to index. Place 'forecast-summary.csv' at the repo root or generate the summary cache first.")

        # Build the vectorstore from the chosen dataframe only (Chroma)
        store = retriever.build_vectorstore(recreate=True, df=df)

        # Attempt to report success: Chroma stores don't expose a uniform docstore
        ok = store is not None
        return {"status": "ok" if ok else "failed", "docs_indexed": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/last_retrieval")
def debug_last_retrieval(thread_id: str = "default"):
    """Return the last retrieval details for a thread (for debugging).

    Useful to inspect which documents were retrieved and the augmented question.
    """
    try:
        try:
            from . import chatbot as _chatbot
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to import chatbot module: {e}")

        info = _chatbot.get_last_retrieval(thread_id)
        if info is None:
            return {"ok": True, "found": False, "thread_id": thread_id}
        return {"ok": True, "found": True, "thread_id": thread_id, "retrieval": sanitize(info)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/build_chain")
def debug_build_chain(thread_id: str = "debug"):
    """Debug endpoint: attempt to build the ConversationalRetrievalChain and return success or error.

    This does not run the chain, only builds it and reports any construction-time errors
    (useful to surface validation errors from langchain/Pydantic).
    """
    try:
        # Attempt to build the chain; capture any runtime exceptions and return them to caller
        chain = None
        try:
            chain = __import__("rag_chatbot.chatbot", fromlist=["_build_chain"])._build_chain(thread_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build chain: {e}")
        # Basic sanity: ensure chain has a retriever attribute or callable
        has_retriever = hasattr(chain, "retriever") or getattr(chain, "_callable", None) is not None
        return {"ok": True, "has_chain": True, "has_retriever": bool(has_retriever)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/run_chain")
def debug_run_chain(thread_id: str = "debug", question: str = "what product got best prediction for next year ?"):
    """Build the chain and execute it once with a provided question. Returns the chain output or full traceback on error.

    This endpoint is temporary for debugging and should be removed after the issue is diagnosed.
    """
    try:
        try:
            chain = __import__("rag_chatbot.chatbot", fromlist=["_build_chain"])._build_chain(thread_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build chain: {e}")

        try:
            # Some chain implementations expect a `chat_history` key; provide
            # an empty history during debug runs to avoid missing-key errors.
            inputs = {"question": question, "chat_history": []}
            try:
                res = chain(inputs)
                return {"ok": True, "result": sanitize(res)}
            except ValueError as ve:
                # LangChain may raise a ValueError about missing input keys; try
                # alternate call styles to be robust for debug runs.
                err_text = str(ve)
                # Try chain.run(question) if available
                if hasattr(chain, "run"):
                    try:
                        run_res = chain.run(question)
                        return {"ok": True, "result": sanitize(run_res)}
                    except Exception:
                        pass

                # Try other common single-input keys
                for alt_key in ("input", "query", "question"):
                    try:
                        alt_res = chain({alt_key: question})
                        return {"ok": True, "result": sanitize(alt_res)}
                    except Exception:
                        continue

                tb = traceback.format_exc()
                return {"ok": False, "error": err_text, "traceback": tb}
        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/compute_and_reindex")
def debug_compute_and_reindex(force_recompute: bool = True):
    """Debug helper: compute forecasts for all articles (writes per-product CSVs)
    then trigger retriever.build_vectorstore(recreate=True) so the RAG index ingests
    the freshly computed per-product forecasts. This is a convenience for testing
    and should be removed or protected in production.
    """
    try:
        # Load data using the same loader the rest of the app uses
        from rag_chatbot import data_loader
        df = data_loader.load_and_preprocess()
        # Lazy import SalesForecaster to avoid optional deps failing at server start
        try:
            from sales_forecaster import SalesForecaster
        except Exception as e:
            return {"ok": False, "error": "Failed to import SalesForecaster", "detail": str(e)}

        sf = SalesForecaster(df, cache_dir=str(Path(__file__).resolve().parent.parent / "cache"))
        # Force recompute all forecasts and write per-product CSVs
        df_all = sf.forecast_all_articles(force_recompute=force_recompute)
        num_forecasts = 0 if df_all is None else len(df_all)

        # Rebuild the vectorstore from the freshly computed summary dataframe
        try:
            store = retriever.build_vectorstore(recreate=True, df=df_all)
            docs_indexed = getattr(store, 'docstore', None) is not None
        except Exception as e:
            return {"ok": False, "error": "Forecasts computed but reindex failed", "forecast_count": num_forecasts, "reindex_error": str(e)}

        return {"ok": True, "forecast_count": num_forecasts, "docs_indexed": bool(docs_indexed)}
    except Exception as e:
        tb = traceback.format_exc()
        return {"ok": False, "error": str(e), "traceback": tb}


# Expose app for uvicorn: rag_chatbot.api:app
