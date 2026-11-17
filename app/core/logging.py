"""Logging estructurado.

En modo JSON cada línea es un objeto con fecha, nivel, módulo, mensaje y los campos extra
pasados con ``logger.info("...", extra={"campo": valor})``. Sirve como evidencia verificable
(logs) para la definición de terminado.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Atributos estándar de LogRecord que no se repiten como campos extra.
_ATRIBUTOS_ESTANDAR = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _campos_extra(record: logging.LogRecord) -> dict[str, object]:
    """Campos pasados con ``extra={...}`` (todo lo que no es estándar de LogRecord)."""
    return {
        clave: valor
        for clave, valor in vars(record).items()
        if clave not in _ATRIBUTOS_ESTANDAR and not clave.startswith("_")
    }


class TextFormatter(logging.Formatter):
    """Formato legible para la terminal que también muestra los campos extra."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        linea = super().format(record)
        extra = _campos_extra(record)
        if extra:
            linea += " | " + " ".join(f"{clave}={valor}" for clave, valor in extra.items())
        return linea


class JsonFormatter(logging.Formatter):
    """Convierte cada registro en una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        datos: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "nivel": record.levelname,
            "logger": record.name,
            "mensaje": record.getMessage(),
        }
        datos.update(_campos_extra(record))
        if record.exc_info:
            datos["excepcion"] = self.formatException(record.exc_info)
        return json.dumps(datos, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configura el logger raíz. Es idempotente: se puede llamar varias veces."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_format else TextFormatter())

    raiz = logging.getLogger()
    raiz.handlers.clear()
    raiz.addHandler(handler)
    raiz.setLevel(level)

    # Librerías muy verbosas: solo advertencias.
    for ruidoso in ("httpx", "httpcore", "chromadb", "google_genai", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
