from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agent import Chat2DBQueryAgent
from .logger import logger

app = FastAPI(title="Text2SQL Service")


class QueryIn(BaseModel):
    question: str
    db_url: str | None = None


@app.post("/query")
async def query_endpoint(payload: QueryIn):
    try:
        agent = Chat2DBQueryAgent(db_url=payload.db_url)
        result = agent.generate_and_run(payload.question)
        return result
    except Exception as exc:
        logger.exception("API error during query")
        raise HTTPException(status_code=500, detail=str(exc))
