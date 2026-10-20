from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router as api_router
from app.database.db import db_manager
from app.cache.embeddings import EmbeddingFactory

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smartroute")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info(f"Starting {settings.APP_NAME} in '{settings.APP_ENV}' mode...")
    # Initialize DB schema
    _ = db_manager.get_connection()
    # Warm up embedding model
    _ = EmbeddingFactory.get_embedding_model()
    logger.info("System warmup complete. Gateway ready for traffic.")
    yield
    logger.info("Shutting down SmartRoute Gateway...")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "SmartRoute: Semantic Caching and Cost-Aware LLM Routing AI Gateway. "
        "Reduces LLM API costs and latency while preserving response quality."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards and client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api/v1 and root
app.include_router(api_router, prefix="/api/v1", tags=["Gateway"])
app.include_router(api_router, tags=["Gateway"])  # Also expose directly at root


@app.get("/", summary="Gateway Root Information")
def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "online",
        "provider_mode": settings.LLM_PROVIDER,
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred in SmartRoute Gateway."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
