from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import forecasts, dashboard
from rag_chatbot import api
import logging


def create_app() -> FastAPI:
    # configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    app = FastAPI(title="Sales Forecaster API (modular)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # simple request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = logging.getLogger("app.requests")
        client = request.client.host if request.client else "unknown"
        logger.info(f"Incoming request: {request.method} {request.url.path} from {client}")
        try:
            response = await call_next(request)
            logger.info(f"Completed {request.method} {request.url.path} -> {response.status_code}")
            return response
        except Exception as exc:
            logger.exception(f"Error handling request {request.method} {request.url.path}: {exc}")
            raise

    app.include_router(forecasts.router, prefix="/api")
    app.include_router(dashboard.router, prefix="/api")
    app.include_router(api.app.router, prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
