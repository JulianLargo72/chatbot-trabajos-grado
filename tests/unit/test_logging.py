"""Pruebas del logging estructurado (F1)."""

import json
import logging

from app.core.logging import JsonFormatter, TextFormatter, configure_logging


def test_formato_json_incluye_campos_extra() -> None:
    registro = logging.makeLogRecord(
        {"name": "prueba", "levelname": "INFO", "msg": "hola %s", "args": ("mundo",)}
    )
    registro.fragmentos = 42
    datos = json.loads(JsonFormatter().format(registro))
    assert datos["mensaje"] == "hola mundo"
    assert datos["nivel"] == "INFO"
    assert datos["fragmentos"] == 42
    assert "ts" in datos


def test_configure_logging_es_idempotente() -> None:
    configure_logging("DEBUG")
    configure_logging("WARNING")
    raiz = logging.getLogger()
    assert len(raiz.handlers) == 1
    assert raiz.level == logging.WARNING


def test_formato_texto_muestra_campos_extra() -> None:
    registro = logging.makeLogRecord(
        {"name": "app.cli", "levelname": "ERROR", "msg": "Falló el modelo"}
    )
    registro.error = "404 NOT_FOUND"
    linea = TextFormatter().format(registro)
    assert "Falló el modelo" in linea
    assert "error=404 NOT_FOUND" in linea
