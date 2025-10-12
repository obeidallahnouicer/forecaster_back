from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from . import chatbot
from .config import API_HOST, API_PORT
from . import retriever


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
    res = chatbot.chat(req.message, req.thread_id)
    return res


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    chatbot.reset_memory(req.thread_id)
    return {"status": "reset", "thread_id": req.thread_id}


@app.post("/reindex")
def reindex_endpoint():
    """Trigger reindexing / embedding of the CSV into the vectorstore.

    This will build the FAISS index and cache it under cache/vectorstore.
    """
    try:
        store = retriever.build_vectorstore(recreate=True)
        return {"status": "ok", "docs_indexed": getattr(store, 'docstore', None) is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Expose app for uvicorn: rag_chatbot.api:app
