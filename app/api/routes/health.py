"""Ruta de salud: indica que la API está arriba y con qué configuración corre."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.core.config import Settings, get_settings

router = APIRouter(tags=["salud"])


class HealthResponse(BaseModel):
    estado: str
    version: str
    modelo_llm: str
    modelo_embeddings: str
    clave_gemini_configurada: bool
    indice_existe: bool


@router.get("/health", response_model=HealthResponse, summary="Estado del servicio")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """No llama a Gemini: responde rápido y sin consumir cuota."""
    indice = settings.chroma_dir
    return HealthResponse(
        estado="ok",
        version=__version__,
        modelo_llm=settings.gemini_llm_model,
        modelo_embeddings=settings.gemini_embedding_model,
        clave_gemini_configurada=settings.google_api_key is not None,
        indice_existe=indice.is_dir() and any(p.name != ".gitkeep" for p in indice.iterdir()),
    )
