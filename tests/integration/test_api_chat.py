"""Pruebas de integración de POST /api/chat con LLM y embeddings simulados (HU06, HU07)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.core.config import Settings, get_settings
from app.main import app
from app.rag.retriever import crear_recuperador
from app.services.chat import ChatService
from app.services.ingestion import ingestar
from tests.fakes import EmbeddingFalso, LLMFalso
from tests.pdf_utils import crear_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def entorno(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(
        corpus / "Formato_Proyecto_de_Grado.pdf",
        [["El resumen tiene una extension maxima de 300 palabras"]],
    )
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        gemini_embedding_model="falso",
        similarity_threshold=0.2,
        admin_api_key="clave",
        google_api_key="x",
    )
    modelo = EmbeddingFalso()
    ingestar(settings, lambda _s: modelo)
    llm = LLMFalso(respuesta="Máximo 300 palabras (Formato de proyecto de grado, p. 1).")
    servicio = ChatService(crear_recuperador(settings, modelo), llm)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[deps.obtener_servicio_chat] = lambda: servicio
    with TestClient(app) as cliente:
        yield cliente, llm, settings
    app.dependency_overrides.clear()


def test_chat_responde_200_con_respuesta_fuentes_y_tipo(entorno) -> None:
    cliente, _, _ = entorno
    r = cliente.post("/api/chat", json={"pregunta": "¿Cuál es la extensión máxima del resumen?"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["tipo"] == "respondida"
    assert "300 palabras" in datos["respuesta"]
    assert datos["fuentes"][0] == {
        "documento": "Formato_Proyecto_de_Grado.pdf",
        "pagina": "1",
        "puntaje": datos["fuentes"][0]["puntaje"],
    }
    assert 0 < datos["fuentes"][0]["puntaje"] <= 1


@pytest.mark.parametrize(
    "cuerpo", [{}, {"pregunta": ""}, {"pregunta": "  a "}, {"pregunta": "x" * 1001}]
)
def test_pregunta_invalida_responde_422(entorno, cuerpo) -> None:
    cliente, llm, _ = entorno
    assert cliente.post("/api/chat", json=cuerpo).status_code == 422
    assert llm.llamadas == 0  # no se gasta cuota con solicitudes inválidas


def test_fallo_de_gemini_responde_503(entorno) -> None:
    cliente, llm, _ = entorno
    llm.error = "Deadline exceeded"
    r = cliente.post("/api/chat", json={"pregunta": "extension del resumen"})
    assert r.status_code == 503
    assert "no está disponible" in r.json()["detail"]


def test_endpoint_documentado_en_swagger(entorno) -> None:
    cliente, _, _ = entorno
    esquema = cliente.get("/openapi.json").json()
    operacion = esquema["paths"]["/api/chat"]["post"]
    assert "503" in operacion["responses"]
    assert "ChatResponse" in str(operacion["responses"]["200"])
    assert cliente.get("/docs").status_code == 200


def test_cors_permite_el_frontend(entorno) -> None:
    cliente, _, _ = entorno
    r = cliente.options(
        "/api/chat",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_sin_indice_responde_503(tmp_path: Path) -> None:
    vacio = Settings(_env_file=None, chroma_dir=tmp_path / "vacio", google_api_key="x")
    app.dependency_overrides[get_settings] = lambda: vacio
    deps.invalidar_servicio_chat()
    try:
        with TestClient(app) as cliente:
            r = cliente.post("/api/chat", json={"pregunta": "¿Qué es el resumen?"})
        assert r.status_code == 503
        assert "ingest" in r.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        deps.invalidar_servicio_chat()
