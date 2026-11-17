"""POST /api/chat (HU06, HU07)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import obtener_servicio_chat
from app.schemas.chat import ChatRequest, ChatResponse, Fuente
from app.services.chat import ChatService, ServicioNoDisponibleError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Responde una pregunta con base en los documentos oficiales",
    responses={
        422: {"description": "Pregunta vacía o demasiado larga"},
        503: {"description": "Gemini o el índice no están disponibles (reintentar)"},
    },
)
def chat(
    solicitud: ChatRequest,
    servicio: Annotated[ChatService, Depends(obtener_servicio_chat)],
) -> ChatResponse:
    """Recupera los fragmentos más similares, genera la respuesta con Gemini y devuelve las
    fuentes (documento y página) de los fragmentos usados.

    El campo ``tipo`` indica si la pregunta fue respondida, si la información no está en los
    documentos (HU08) o si está fuera del dominio de los trabajos de grado (HU09).
    """
    try:
        resultado = servicio.responder(solicitud.pregunta)
    except ServicioNoDisponibleError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return ChatResponse(
        respuesta=resultado.respuesta,
        tipo=resultado.tipo,
        fuentes=[Fuente(**vars(f)) for f in resultado.fuentes],
    )
