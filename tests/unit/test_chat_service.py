"""Pruebas del servicio de chat con LLM y embeddings simulados (HU06-HU09)."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag import prompts
from app.rag.retriever import FragmentoRecuperado, crear_recuperador
from app.services.chat import (
    ChatService,
    ServicioNoDisponibleError,
    clasificar,
    fuentes_unicas,
)
from app.services.ingestion import ingestar
from tests.fakes import EmbeddingFalso, LLMFalso
from tests.pdf_utils import crear_pdf


@pytest.fixture
def recuperador(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(
        corpus / "Formato_Proyecto_de_Grado.pdf",
        [["El resumen tiene una extension maxima de 300 palabras"], ["Objetivos del proyecto"]],
    )
    crear_pdf(
        corpus / "Manual_IEEE.pdf",
        [["El abstract tiene maximo 150 palabras"], ["Las referencias van entre corchetes"]],
        primera_etiqueta=11,
    )
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        gemini_embedding_model="falso",
        similarity_threshold=0.3,
        top_k=3,
    )
    modelo = EmbeddingFalso()
    ingestar(settings, lambda _s: modelo)
    return crear_recuperador(settings, modelo)


def test_respuesta_con_fuentes_de_los_fragmentos_usados(recuperador) -> None:
    llm = LLMFalso(respuesta="Máximo 300 palabras (Formato de proyecto de grado, p. 1).")
    r = ChatService(recuperador, llm).responder("extension maxima del resumen en palabras")
    assert r.tipo == "respondida"
    assert r.respuesta.startswith("Máximo 300 palabras")
    assert r.fuentes[0].documento == "Formato_Proyecto_de_Grado.pdf"
    assert r.fuentes[0].pagina == "1"
    # HU07: las fuentes son exactamente los fragmentos enviados como contexto.
    contexto = llm.mensajes[0][1].content
    for f in r.fuentes:
        assert f"Página: {f.pagina}" in contexto


def test_el_prompt_incluye_reglas_y_contexto(recuperador) -> None:
    llm = LLMFalso()
    ChatService(recuperador, llm).responder("extension maxima del resumen")
    sistema, usuario = llm.mensajes[0]
    assert sistema.content == prompts.SYSTEM_PROMPT
    assert "prevalece el formato institucional" in sistema.content
    assert "CONTEXTO:" in usuario.content and "extension maxima del resumen" in usuario.content


def test_etiqueta_sin_informacion_da_mensaje_estandar(recuperador) -> None:
    llm = LLMFalso(respuesta=prompts.ETIQUETA_SIN_INFORMACION)
    r = ChatService(recuperador, llm).responder("extension del resumen y fecha de entrega")
    assert r.tipo == "sin_informacion"
    assert r.respuesta == prompts.MENSAJE_SIN_INFORMACION
    assert r.fuentes == []


def test_etiqueta_fuera_de_dominio_da_mensaje_estandar(recuperador) -> None:
    llm = LLMFalso(respuesta="[FUERA_DE_DOMINIO]")
    r = ChatService(recuperador, llm).responder("cual es la capital de Australia")
    assert r.tipo == "fuera_de_dominio"
    assert r.respuesta == prompts.MENSAJE_FUERA_DE_DOMINIO
    assert r.fuentes == []


def test_sin_fragmentos_relevantes_nunca_responde_inventando(recuperador) -> None:
    """Si nada supera el umbral y el modelo responde igual, se fuerza 'sin información'."""
    recuperador.umbral = 0.99
    llm = LLMFalso(respuesta="La fecha de entrega es el 15 de diciembre.")
    r = ChatService(recuperador, llm).responder("cual es la fecha de entrega")
    assert r.tipo == "sin_informacion"
    assert r.fuentes == []
    assert "No se recuperaron fragmentos" in llm.mensajes[0][1].content


def test_sin_fragmentos_puede_detectar_fuera_de_dominio(recuperador) -> None:
    recuperador.umbral = 0.99
    r = ChatService(recuperador, LLMFalso(respuesta="[FUERA_DE_DOMINIO]")).responder("receta")
    assert r.tipo == "fuera_de_dominio"


def test_fallo_del_llm_lanza_servicio_no_disponible(recuperador) -> None:
    with pytest.raises(ServicioNoDisponibleError):
        ChatService(recuperador, LLMFalso(error="timeout")).responder("resumen")


def test_fuentes_unicas_por_documento_y_pagina() -> None:
    frags = [
        FragmentoRecuperado("1", "a.pdf", "1", 0.70, "x"),
        FragmentoRecuperado("2", "a.pdf", "1", 0.80, "y"),
        FragmentoRecuperado("3", "b.pdf", "4", 0.75, "z"),
    ]
    fuentes = fuentes_unicas(frags)
    assert [(f.documento, f.pagina, f.puntaje) for f in fuentes] == [
        ("a.pdf", "1", 0.80),
        ("b.pdf", "4", 0.75),
    ]


@pytest.mark.parametrize(
    ("texto", "tipo"),
    [
        ("[SIN_INFORMACION]", "sin_informacion"),
        ("Lo siento. [fuera_de_dominio]", "fuera_de_dominio"),
        ("El resumen tiene 300 palabras.", "respondida"),
    ],
)
def test_clasificar(texto: str, tipo: str) -> None:
    assert clasificar(texto) == tipo


def test_nombres_legibles() -> None:
    assert prompts.nombre_legible("Formato_Proyecto_de_Grado_ETIC_2026_limpio.pdf") == (
        "Formato de proyecto de grado"
    )
    assert prompts.nombre_legible("Manual_IEEE_Javeriana_limpio.pdf") == "Manual de norma IEEE"
    assert prompts.nombre_legible("Otro_documento.pdf") == "Otro documento"
