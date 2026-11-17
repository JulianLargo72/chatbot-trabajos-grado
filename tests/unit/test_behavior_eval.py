"""Pruebas de la evaluación de comportamiento (HU08, HU09) con LLM simulado."""

from pathlib import Path

import pytest

from app import cli
from app.core.config import Settings
from app.rag import prompts
from app.rag.retriever import crear_recuperador
from app.services.behavior_eval import evaluar_comportamiento, guardar_reporte
from app.services.chat import ChatService
from app.services.evaluation import Banco
from app.services.ingestion import ingestar
from tests.fakes import EmbeddingFalso, LLMFalso
from tests.pdf_utils import crear_pdf

BANCO = Banco.model_validate(
    {
        "version": "t",
        "documentos": {"proyecto": "formato.pdf"},
        "preguntas": [
            {
                "id": "D02",
                "categoria": "A",
                "pregunta": "extension maxima del resumen",
                "respuesta_esperada": "300",
                "documentos": ["proyecto"],
            },
            {
                "id": "S01",
                "categoria": "B",
                "pregunta": "fecha de entrega",
                "respuesta_esperada": "-",
            },
            {
                "id": "F01",
                "categoria": "C",
                "pregunta": "capital de Australia",
                "respuesta_esperada": "-",
            },
        ],
    }
)


class LLMPorPregunta(LLMFalso):
    """Responde según la pregunta, como lo haría un modelo que sigue el System Prompt."""

    def chat(self, messages, **kwargs):
        usuario = messages[-1].content
        if "Australia" in usuario:
            self.respuesta = prompts.ETIQUETA_FUERA_DE_DOMINIO
        elif "fecha" in usuario:
            self.respuesta = prompts.ETIQUETA_SIN_INFORMACION
        else:
            self.respuesta = "Máximo 300 palabras (Formato de proyecto de grado, p. 1)."
        return super().chat(messages, **kwargs)


@pytest.fixture
def servicio(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(corpus / "formato.pdf", [["El resumen tiene una extension maxima de 300 palabras"]])
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        gemini_embedding_model="falso",
        similarity_threshold=0.2,
        google_api_key="x",
    )
    modelo = EmbeddingFalso()
    ingestar(settings, lambda _s: modelo)
    return ChatService(crear_recuperador(settings, modelo), LLMPorPregunta()), settings, tmp_path


def test_por_defecto_evalua_b_y_c(servicio) -> None:
    chat, settings, _ = servicio
    rep = evaluar_comportamiento(BANCO, chat, settings)
    assert [r.id for r in rep.resultados] == ["S01", "F01"]
    assert rep.aciertos("B") == (1, 1)
    assert rep.aciertos("C") == (1, 1)


def test_ids_adicionales_y_reporte(servicio) -> None:
    chat, settings, tmp = servicio
    rep = evaluar_comportamiento(BANCO, chat, settings, ids=("D02",))
    d02 = next(r for r in rep.resultados if r.id == "D02")
    assert d02.correcto and d02.fuentes
    _, md = guardar_reporte(rep, tmp / "r")
    texto = md.read_text(encoding="utf-8")
    assert "**1 de 1**" in texto and "### F01" in texto and "System Prompt:** v" in texto


def test_error_del_llm_cuenta_como_incorrecto(servicio) -> None:
    chat, settings, _ = servicio
    chat._llm.error = "timeout"
    rep = evaluar_comportamiento(BANCO, chat, settings, categorias=("B",))
    assert rep.resultados[0].tipo_obtenido == "error"
    assert not rep.resultados[0].correcto


def test_cli_prompt(capsys) -> None:
    assert cli.main(["prompt"]) == 0
    salida = capsys.readouterr().out
    assert f"System Prompt v{prompts.VERSION_PROMPT}" in salida
    assert prompts.MENSAJE_FUERA_DE_DOMINIO in salida
