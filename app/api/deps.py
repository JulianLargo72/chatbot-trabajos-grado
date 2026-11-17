"""Dependencias compartidas por las rutas."""

import secrets
import threading
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


def verificar_admin(
    settings: SettingsDep,
    x_admin_key: Annotated[str | None, Header(description="Clave de administrador")] = None,
) -> None:
    """Exige la cabecera ``X-Admin-Key`` con el valor de ``ADMIN_API_KEY``."""
    if settings.admin_api_key is None or not settings.admin_api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Endpoint administrativo deshabilitado: configure ADMIN_API_KEY.",
        )
    esperado = settings.admin_api_key.get_secret_value().encode()
    if x_admin_key is None or not secrets.compare_digest(x_admin_key.encode(), esperado):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Clave de administrador inválida."
        )


# ------------------------------------------------------------------ Servicio de chat
# Se construye una sola vez por proceso (abre el índice y prepara los modelos) y se reutiliza
# en todas las consultas. Una reindexación administrativa lo invalida para recargarlo.
_servicio_chat = None
_candado_chat = threading.Lock()


def obtener_servicio_chat(settings: SettingsDep):
    """Devuelve el ``ChatService`` compartido o responde 503 si no hay índice."""
    global _servicio_chat
    if _servicio_chat is None:
        with _candado_chat:
            if _servicio_chat is None:
                from app.rag.models import build_llm
                from app.rag.retriever import crear_recuperador
                from app.services.chat import ChatService

                try:
                    _servicio_chat = ChatService(crear_recuperador(settings), build_llm(settings))
                except RuntimeError as exc:  # índice vacío, falta la clave o modelo distinto
                    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return _servicio_chat


def invalidar_servicio_chat() -> None:
    global _servicio_chat
    with _candado_chat:
        _servicio_chat = None
