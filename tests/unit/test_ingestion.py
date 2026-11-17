"""Pruebas del servicio de ingesta y del comando ingest (HU03)."""

import json
from pathlib import Path

from app import cli
from app.core.config import Settings
from app.services.ingestion import exportar_fragmentos, procesar_corpus
from tests.fakes import EmbeddingFalso
from tests.pdf_utils import crear_pdf


def _corpus(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(corpus / "formato.pdf", [["Resumen: maximo 300 palabras."], ["Objetivos."]])
    crear_pdf(corpus / "ieee.pdf", [["Citas numeradas."]], primera_etiqueta=4)
    return Settings(_env_file=None, corpus_dir=corpus)


def test_procesar_corpus_cuenta_documentos_paginas_y_fragmentos(tmp_path: Path) -> None:
    resultado = procesar_corpus(_corpus(tmp_path))
    assert resultado.documentos == 2
    assert resultado.paginas == 3
    assert resultado.fragmentos == len(resultado.nodos) == 3
    assert [a.archivo for a in resultado.archivos] == ["formato.pdf", "ieee.pdf"]
    assert resultado.nodos[-1].metadata["page_label"] == "4"


def test_exportar_fragmentos_jsonl(tmp_path: Path) -> None:
    resultado = procesar_corpus(_corpus(tmp_path))
    destino = tmp_path / "reportes" / "fragmentos.jsonl"
    exportar_fragmentos(resultado.nodos, destino)
    registros = [json.loads(linea) for linea in destino.read_text(encoding="utf-8").splitlines()]
    assert len(registros) == 3
    assert registros[0]["documento"] == "formato.pdf"
    assert registros[0]["pagina"] == "1"
    assert "300 palabras" in registros[0]["texto"]


def test_cli_ingest_solo_fragmentar(tmp_path: Path, monkeypatch, capsys) -> None:
    settings = _corpus(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["ingest", "--solo-fragmentar"]) == 0
    salida = capsys.readouterr().out
    assert "Documentos: 2 | Páginas: 3 | Fragmentos: 3" in salida
    assert "Vectores" not in salida


def test_cli_ingest_indice_y_buscar(tmp_path: Path, monkeypatch, capsys) -> None:
    settings = _corpus(tmp_path).model_copy(
        update={"chroma_dir": tmp_path / "chroma", "google_api_key": None}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("app.rag.models.build_embed_model", lambda _s: EmbeddingFalso())

    assert cli.main(["ingest"]) == 0
    salida = capsys.readouterr().out
    assert "Vectores en el índice: 3" in salida
    assert "Verificación fragmentos = vectores: OK" in salida

    assert cli.main(["indice"]) == 0
    assert "Vectores: 3" in capsys.readouterr().out

    assert cli.main(["buscar", "citas numeradas", "--top-k", "1"]) == 0
    assert "ieee.pdf | pág. 4" in capsys.readouterr().out


def test_cli_ingest_corpus_vacio_devuelve_1(tmp_path: Path, monkeypatch) -> None:
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, corpus_dir=vacio))
    assert cli.main(["ingest", "--solo-fragmentar"]) == 1


def test_cli_ingest_corpus_inexistente_devuelve_1(tmp_path: Path, monkeypatch) -> None:
    falta = Settings(_env_file=None, corpus_dir=tmp_path / "no")
    monkeypatch.setattr(cli, "get_settings", lambda: falta)
    assert cli.main(["ingest"]) == 1
