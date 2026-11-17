"""Rutas administrativas: reindexar el corpus sin reiniciar el servidor (HU04)."""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps import SettingsDep, verificar_admin
from app.rag.index_store import ModeloEmbeddingsDistintoError
from app.services.ingestion import ingestar

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["administración"])

# Evita dos ingestas simultáneas sobre el mismo índice.
_candado = threading.Lock()


class ArchivoIngesta(BaseModel):
    archivo: str
    estado: str
    paginas: int
    fragmentos: int
    vectores: int


class IngestaResponse(BaseModel):
    documentos: int
    paginas: int
    fragmentos: int
    vectores: int
    dimension: int
    modelo_embeddings: str
    verificacion_ok: bool
    segundos: float
    archivos: list[ArchivoIngesta]


@router.post(
    "/ingest",
    response_model=IngestaResponse,
    summary="Actualiza el índice con los PDF de data/corpus",
    dependencies=[Depends(verificar_admin)],
    responses={401: {"description": "Clave inválida"}, 409: {"description": "Ingesta en curso"}},
)
def ingest(settings: SettingsDep, reset: bool = False) -> IngestaResponse:
    """Ingesta incremental: solo procesa PDF nuevos o modificados (``reset=true`` reconstruye)."""
    if not _candado.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "Ya hay una ingesta en curso.")
    try:
        resultado = ingestar(settings, reset=reset)
    except ModeloEmbeddingsDistintoError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    finally:
        _candado.release()
    logger.info("Ingesta solicitada por API", extra={"vectores": resultado.vectores})
    return IngestaResponse(
        **{k: v for k, v in resultado.como_dict().items() if k in IngestaResponse.model_fields}
    )
