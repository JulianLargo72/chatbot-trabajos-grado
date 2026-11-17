"""Prueba de humo de la fase F0: el paquete y sus capas se importan correctamente."""

import importlib

import pytest

import app

CAPAS = [
    "app.core",
    "app.api",
    "app.api.routes",
    "app.schemas",
    "app.services",
    "app.rag",
]


def test_version_definida() -> None:
    assert app.__version__


@pytest.mark.parametrize("modulo", CAPAS)
def test_capas_importables(modulo: str) -> None:
    assert importlib.import_module(modulo) is not None
