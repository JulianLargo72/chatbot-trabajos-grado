"""Pruebas del índice persistente e ingesta incremental (HU04)."""

import math
from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag import index_store
from app.services.ingestion import ELIMINADO, MODIFICADO, NUEVO, SIN_CAMBIOS, ingestar
from tests.fakes import EmbeddingFalso
from tests.pdf_utils import crear_pdf


@pytest.fixture
def entorno(tmp_path: Path) -> tuple[Settings, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(
        corpus / "formato.pdf",
        [["El resumen tiene maximo 300 palabras."], ["Los objetivos inician con verbo."]],
    )
    crear_pdf(
        corpus / "ieee.pdf",
        [["Las referencias se numeran entre corchetes."], ["Las figuras llevan titulo."]],
        primera_etiqueta=4,
    )
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        gemini_embedding_model="modelo-falso",
    )
    return settings, corpus


def _fabrica(modelo: EmbeddingFalso):
    return lambda _settings: modelo


def _no_llamar(_settings):
    raise AssertionError("No debía crear el modelo de embeddings (no hay cambios)")


def _estados(resultado) -> dict[str, str]:
    return {a.archivo: a.estado for a in resultado.archivos}


def test_primera_ingesta_indexa_todo_y_verifica(entorno) -> None:
    settings, _ = entorno
    r = ingestar(settings, _fabrica(EmbeddingFalso()))
    assert _estados(r) == {"formato.pdf": NUEVO, "ieee.pdf": NUEVO}
    assert r.fragmentos == r.vectores == 4
    assert r.verificacion_ok is True
    assert r.dimension == 64
    assert r.tamano_indice_mb > 0


def test_segunda_ingesta_no_duplica_ni_llama_a_la_api(entorno) -> None:
    settings, _ = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    r = ingestar(settings, _no_llamar)
    assert set(_estados(r).values()) == {SIN_CAMBIOS}
    assert r.vectores == 4
    assert r.paginas == 4
    assert r.verificacion_ok is True


def test_agregar_un_pdf_actualiza_solo_ese(entorno) -> None:
    """Criterio de HU04: al añadir un PDF se actualiza el índice."""
    settings, corpus = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    crear_pdf(corpus / "anteproyecto.pdf", [["El cronograma usa diagrama de Gantt."]])
    modelo = EmbeddingFalso()
    r = ingestar(settings, _fabrica(modelo))
    assert _estados(r)["anteproyecto.pdf"] == NUEVO
    assert _estados(r)["formato.pdf"] == SIN_CAMBIOS
    assert modelo.llamadas == 1  # solo se embebió el fragmento nuevo
    assert r.vectores == 5 and r.verificacion_ok


def test_pdf_modificado_reemplaza_sus_vectores(entorno) -> None:
    settings, corpus = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    crear_pdf(corpus / "formato.pdf", [["Version nueva con una sola pagina."]])
    r = ingestar(settings, _fabrica(EmbeddingFalso()))
    assert _estados(r)["formato.pdf"] == MODIFICADO
    assert r.vectores == 3 and r.verificacion_ok


def test_pdf_eliminado_sale_del_indice(entorno) -> None:
    settings, corpus = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    (corpus / "ieee.pdf").unlink()
    r = ingestar(settings, _no_llamar)
    assert _estados(r)["ieee.pdf"] == ELIMINADO
    assert r.documentos == 1
    assert r.vectores == 2 and r.verificacion_ok


def test_reset_reconstruye(entorno) -> None:
    settings, _ = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    r = ingestar(settings, _fabrica(EmbeddingFalso()), reset=True)
    assert set(_estados(r).values()) == {NUEVO}
    assert r.vectores == 4


def test_ids_deterministas(entorno) -> None:
    settings, _ = entorno
    ids_1 = {n.node_id for n in ingestar(settings, _fabrica(EmbeddingFalso())).nodos}
    ids_2 = {n.node_id for n in ingestar(settings, _fabrica(EmbeddingFalso()), reset=True).nodos}
    assert ids_1 == ids_2


def test_modelo_distinto_se_rechaza(entorno) -> None:
    settings, _ = entorno
    ingestar(settings, _fabrica(EmbeddingFalso()))
    otro = settings.model_copy(update={"gemini_embedding_model": "otro-modelo"})
    with pytest.raises(index_store.ModeloEmbeddingsDistintoError, match="--reset"):
        index_store.obtener_coleccion(otro)


def test_recarga_desde_disco_y_recupera_con_metadatos(entorno) -> None:
    settings, _ = entorno
    modelo = EmbeddingFalso()
    ingestar(settings, _fabrica(modelo))
    indice = index_store.cargar_indice(settings, modelo)
    resultados = indice.as_retriever(similarity_top_k=1).retrieve(
        "como se numeran las referencias entre corchetes"
    )
    nodo = resultados[0].node
    assert nodo.metadata["file_name"] == "ieee.pdf"
    assert nodo.metadata["page_label"] == "4"
    assert 0.0 < index_store.similitud_coseno(resultados[0].score) <= 1.0


def test_indice_vacio_da_error_claro(entorno) -> None:
    settings, _ = entorno
    with pytest.raises(RuntimeError, match="ingest"):
        index_store.cargar_indice(settings, EmbeddingFalso())


@pytest.mark.parametrize("similitud", [1.0, 0.8, 0.5, 0.0])
def test_conversion_a_similitud_coseno(similitud: float) -> None:
    puntaje_chroma = math.exp(-(1 - similitud))  # lo que devuelve ChromaVectorStore
    assert index_store.similitud_coseno(puntaje_chroma) == pytest.approx(similitud)
