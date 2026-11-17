"""Dependencias compartidas por las rutas."""

import secrets
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
