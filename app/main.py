"""Punto de entrada de la API (FastAPI)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "API iniciada",
        extra={
            "version": __version__,
            "modelo_llm": settings.gemini_llm_model,
            "modelo_embeddings": settings.gemini_embedding_model,
        },
    )
    yield
    logger.info("API detenida")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="API RAG para consultar las normas de trabajos de grado.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _error_no_controlado(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado", extra={"ruta": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})

    app.include_router(health.router)
    return app


app = create_app()
