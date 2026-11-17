"""Pruebas de POST /api/admin/ingest (HU04)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import admin
from app.core.config import Settings, get_settings
from app.main import app
from tests.fakes import EmbeddingFalso
from tests.pdf_utils import crear_pdf


@pytest.fixture
def cliente_admin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(corpus / "formato.pdf", [["El resumen tiene maximo 300 palabras."]])
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        admin_api_key="clave-admin",
        google_api_key="x",
    )
    monkeypatch.setattr("app.rag.models.build_embed_model", lambda _s: EmbeddingFalso())
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as cliente:
        yield cliente, settings
    app.dependency_overrides.clear()


def test_sin_clave_responde_401(cliente_admin) -> None:
    cliente, _ = cliente_admin
    assert cliente.post("/api/admin/ingest").status_code == 401
    assert cliente.post("/api/admin/ingest", headers={"X-Admin-Key": "mala"}).status_code == 401


def test_con_clave_indexa(cliente_admin) -> None:
    cliente, _ = cliente_admin
    r = cliente.post("/api/admin/ingest", headers={"X-Admin-Key": "clave-admin"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["vectores"] == datos["fragmentos"] == 1
    assert datos["verificacion_ok"] is True
    assert datos["archivos"][0]["estado"] == "nuevo"


def test_sin_admin_key_configurada_responde_503(cliente_admin) -> None:
    cliente, settings = cliente_admin
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"admin_api_key": None}
    )
    r = cliente.post("/api/admin/ingest", headers={"X-Admin-Key": "clave-admin"})
    assert r.status_code == 503


def test_ingesta_en_curso_responde_409(cliente_admin) -> None:
    cliente, _ = cliente_admin
    admin._candado.acquire()
    try:
        r = cliente.post("/api/admin/ingest", headers={"X-Admin-Key": "clave-admin"})
        assert r.status_code == 409
    finally:
        admin._candado.release()
