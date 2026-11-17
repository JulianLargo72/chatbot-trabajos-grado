"""Pruebas de fragmentación (HU03)."""

from llama_index.core import Document

from app.core.config import Settings
from app.rag.chunker import fragmentar


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _doc(texto: str, pagina: str = "7") -> Document:
    return Document(text=texto, metadata={"file_name": "manual.pdf", "page_label": pagina})


def test_texto_corto_produce_un_fragmento() -> None:
    nodos = fragmentar([_doc("Las referencias se numeran en orden de aparición.")], _settings())
    assert len(nodos) == 1


def test_texto_largo_se_divide_y_hereda_metadatos() -> None:
    texto = " ".join(f"Esta es la oración número {i} del documento." for i in range(300))
    nodos = fragmentar([_doc(texto)], _settings(chunk_size=128, chunk_overlap=16))
    assert len(nodos) > 1
    for nodo in nodos:
        assert nodo.metadata["file_name"] == "manual.pdf"
        assert nodo.metadata["page_label"] == "7"


def test_un_fragmento_nunca_mezcla_paginas() -> None:
    docs = [_doc("Contenido de la página cuatro.", "4"), _doc("Contenido de la cinco.", "5")]
    nodos = fragmentar(docs, _settings())
    assert [n.metadata["page_label"] for n in nodos] == ["4", "5"]


def test_chunk_size_menor_genera_mas_fragmentos() -> None:
    texto = " ".join(f"Regla {i}: el título va centrado y en mayúscula." for i in range(200))
    grandes = fragmentar([_doc(texto)], _settings(chunk_size=512, chunk_overlap=32))
    pequenos = fragmentar([_doc(texto)], _settings(chunk_size=128, chunk_overlap=16))
    assert len(pequenos) > len(grandes)
