"""Pruebas de lectura y limpieza de PDF (HU03)."""

from pathlib import Path

import pytest

from app.rag.loader import (
    PaginaPdf,
    cargar_corpus,
    detectar_lineas_repetidas,
    leer_pdf,
    limpiar_paginas,
    listar_pdfs,
    paginas_a_documentos,
)
from tests.pdf_utils import crear_pdf


def test_metadatos_por_pagina_con_etiquetas(tmp_path: Path) -> None:
    ruta = crear_pdf(
        tmp_path / "manual.pdf",
        [["Introduccion al manual"], ["Formato general"], ["Referencias"]],
        primera_etiqueta=4,
    )
    paginas = leer_pdf(ruta)
    assert [p.page_label for p in paginas] == ["4", "5", "6"]
    assert [p.page_index for p in paginas] == [1, 2, 3]
    assert all(p.file_name == "manual.pdf" for p in paginas)
    assert "Formato general" in paginas[1].texto


def test_sin_etiquetas_usa_numero_de_pagina(tmp_path: Path) -> None:
    ruta = crear_pdf(tmp_path / "formato.pdf", [["Resumen"], ["Objetivos"]])
    assert [p.page_label for p in leer_pdf(ruta)] == ["1", "2"]


def test_documentos_conservan_metadatos_y_ocultan_tecnicos(tmp_path: Path) -> None:
    ruta = crear_pdf(tmp_path / "f.pdf", [["Texto uno"], ["Texto dos"]], primera_etiqueta=10)
    docs = paginas_a_documentos(leer_pdf(ruta))
    assert docs[1].metadata["file_name"] == "f.pdf"
    assert docs[1].metadata["page_label"] == "11"
    assert "page_index" in docs[1].excluded_embed_metadata_keys
    assert "page_label" not in docs[1].excluded_embed_metadata_keys


def test_quita_numeros_de_pagina_y_encabezados_repetidos() -> None:
    textos = [
        "p. 4\nUniversidad Ejemplo - Manual\nContenido A",
        "p. 5\nUniversidad Ejemplo - Manual\nContenido B",
        "p. 6\nUniversidad Ejemplo - Manual\nContenido C\n12",
    ]
    limpias = limpiar_paginas(textos)
    assert limpias == ["Contenido A", "Contenido B", "Contenido C"]


def test_no_quita_lineas_que_no_se_repiten() -> None:
    textos = ["1. INTRODUCCION\nTexto", "2. OBJETIVOS\nTexto distinto"]
    assert limpiar_paginas(textos) == textos


def test_un_solo_documento_de_una_pagina_no_detecta_repetidas() -> None:
    assert detectar_lineas_repetidas([["Titulo", "Texto"]]) == set()


def test_normaliza_vinetas_de_word_y_espacios() -> None:
    texto = "  El problema   que se va a resolver"
    assert limpiar_paginas([texto]) == ["- El problema que se va a resolver"]


def test_omite_paginas_vacias() -> None:
    paginas = [
        PaginaPdf("a.pdf", "1", 1, "Texto"),
        PaginaPdf("a.pdf", "2", 2, "   "),
    ]
    docs = paginas_a_documentos(paginas)
    assert len(docs) == 1
    assert docs[0].metadata["page_label"] == "1"


def test_listar_pdfs_ordenado_e_ignora_otros(tmp_path: Path) -> None:
    crear_pdf(tmp_path / "b.pdf", [["B"]])
    crear_pdf(tmp_path / "a.PDF", [["A"]])
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in listar_pdfs(tmp_path)] == ["a.PDF", "b.pdf"]


def test_corpus_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(cargar_corpus(tmp_path / "no-existe"))
