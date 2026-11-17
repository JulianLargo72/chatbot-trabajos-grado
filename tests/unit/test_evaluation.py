"""Pruebas de la evaluación de recuperación (HU05)."""

import json
from pathlib import Path

import pytest

from app import cli
from app.core.config import Settings
from app.rag.retriever import FragmentoRecuperado, crear_recuperador
from app.services.evaluation import (
    Banco,
    barrer_umbrales,
    cargar_banco,
    evaluar,
    guardar_reporte,
    hit_rate,
    mrr,
    posicion_documento,
    posicion_pagina,
    umbral_sugerido,
)
from app.services.ingestion import ingestar
from tests.fakes import EmbeddingFalso
from tests.pdf_utils import crear_pdf

BANCO_REAL = Path(__file__).resolve().parents[2] / "eval" / "banco_preguntas.json"


def _frag(doc: str, pag: str, sim: float = 0.8) -> FragmentoRecuperado:
    return FragmentoRecuperado("id", doc, pag, sim, "texto")


# ----------------------------------------------------------------- banco real
def test_banco_real_coincide_con_anexo_a() -> None:
    banco = cargar_banco(BANCO_REAL)
    categorias = [p.categoria for p in banco.preguntas]
    assert len(banco.preguntas) == 37
    assert (categorias.count("A"), categorias.count("B"), categorias.count("C")) == (27, 5, 5)
    principal = [p.documentos[0] for p in banco.preguntas if p.categoria == "A"]
    assert principal.count("proyecto") == 14
    assert principal.count("anteproyecto") == 5
    assert principal.count("ieee") == 8


def test_banco_real_categoria_a_tiene_paginas() -> None:
    banco = cargar_banco(BANCO_REAL)
    for p in banco.preguntas:
        if p.categoria == "A":
            assert p.paginas, p.id
            assert set(p.paginas) <= set(p.documentos), p.id


def test_banco_rechaza_ids_repetidos_y_documentos_desconocidos() -> None:
    base = {"version": "1", "documentos": {"x": "x.pdf"}}
    p = {"id": "D01", "categoria": "A", "pregunta": "?", "respuesta_esperada": "r"}
    with pytest.raises(ValueError, match="repetidos"):
        Banco.model_validate({**base, "preguntas": [{**p, "documentos": ["x"]}] * 2})
    with pytest.raises(ValueError, match="desconocidos"):
        Banco.model_validate({**base, "preguntas": [{**p, "documentos": ["y"]}]})
    with pytest.raises(ValueError, match="categoría A"):
        Banco.model_validate({**base, "preguntas": [p]})


# ----------------------------------------------------------------- métricas
def test_posiciones() -> None:
    frags = [_frag("a.pdf", "1"), _frag("b.pdf", "4"), _frag("b.pdf", "5")]
    assert posicion_documento(frags, {"b.pdf"}) == 2
    assert posicion_documento(frags, {"c.pdf"}) is None
    assert posicion_pagina(frags, {("b.pdf", "5")}) == 3


def test_hit_rate_y_mrr() -> None:
    posiciones = [1, 2, None, 4]
    assert hit_rate(posiciones, 1) == 0.25
    assert hit_rate(posiciones, 3) == 0.5
    assert hit_rate(posiciones, 5) == 0.75
    assert mrr(posiciones) == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert hit_rate([], 3) == 0.0


def test_barrido_y_umbral_sugerido() -> None:
    sim_a = [0.80, 0.75, 0.72]
    sim_bc = [0.55, 0.60]
    puntos = barrer_umbrales(sim_a, sim_bc, 0.50, 0.85)
    por_umbral = {p.umbral: p for p in puntos}
    assert por_umbral[0.65].exactitud == 1.0
    assert por_umbral[0.50].bc_rechazadas == 0
    sugerido = umbral_sugerido(puntos)
    assert 0.60 < sugerido <= 0.72


def test_umbral_sugerido_usa_el_tramo_continuo_mas_largo() -> None:
    """Caso real: 0.65-0.69 empatan en exactitud y 0.71 también, pero aislado."""
    from app.services.evaluation import PuntoUmbral

    exactitudes = {0.64: 0.9, 0.65: 1, 0.66: 1, 0.67: 1, 0.68: 1, 0.69: 1, 0.70: 0.9, 0.71: 1}
    puntos = [PuntoUmbral(u, 0, 0, e) for u, e in exactitudes.items()]
    assert umbral_sugerido(puntos) == 0.67


# ----------------------------------------------------------------- extremo a extremo
@pytest.fixture
def mini(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    crear_pdf(corpus / "formato.pdf", [["El resumen tiene maximo 300 palabras"], ["Objetivos"]])
    crear_pdf(corpus / "ieee.pdf", [["Las referencias van numeradas entre corchetes"]], 19)
    settings = Settings(
        _env_file=None,
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        gemini_embedding_model="falso",
        google_api_key="x",
    )
    modelo = EmbeddingFalso()
    ingestar(settings, lambda _s: modelo)
    banco = {
        "version": "t",
        "documentos": {"proyecto": "formato.pdf", "ieee": "ieee.pdf"},
        "preguntas": [
            {
                "id": "D01",
                "categoria": "A",
                "pregunta": "maximo de palabras del resumen",
                "respuesta_esperada": "300",
                "documentos": ["proyecto"],
                "paginas": {"proyecto": ["1"]},
            },
            {
                "id": "D02",
                "categoria": "A",
                "pregunta": "referencias numeradas entre corchetes",
                "respuesta_esperada": "[n]",
                "documentos": ["ieee"],
                "paginas": {"ieee": ["19"]},
            },
            {
                "id": "F01",
                "categoria": "C",
                "pregunta": "capital de Australia",
                "respuesta_esperada": "rechazar",
            },
        ],
    }
    ruta = tmp_path / "banco.json"
    ruta.write_text(json.dumps(banco), encoding="utf-8")
    return settings, modelo, ruta, tmp_path


def test_evaluar_calcula_hit_rate_y_guarda_reportes(mini) -> None:
    settings, modelo, ruta, tmp = mini
    recuperador = crear_recuperador(settings, modelo)
    reporte = evaluar(cargar_banco(ruta), recuperador, settings, max_k=3)
    assert reporte.metricas["hit_rate_documento"]["1"] == 1.0
    assert reporte.metricas["hit_rate_pagina"]["3"] == 1.0
    assert reporte.cumple_meta
    assert reporte.por_documento["proyecto"]["preguntas"] == 1
    assert reporte.por_documento["ieee"]["hit_rate_top3"] == 1.0
    assert reporte.umbral["similitud_max_por_categoria"]["C"]["max"] < 0.5
    ruta_json, ruta_md = guardar_reporte(reporte, tmp / "reports")
    assert json.loads(ruta_json.read_text(encoding="utf-8"))["metricas"]["mrr_documento"] == 1.0
    texto = ruta_md.read_text(encoding="utf-8")
    assert "CUMPLE" in texto and "| D02 |" in texto


def test_recuperador_filtra_por_umbral(mini) -> None:
    settings, modelo, _, _ = mini
    exigente = settings.model_copy(update={"similarity_threshold": 0.5})
    recuperador = crear_recuperador(exigente, modelo)
    fragmentos = recuperador.recuperar("referencias numeradas entre corchetes", top_k=3)
    assert fragmentos == sorted(fragmentos, key=lambda f: f.similitud, reverse=True)
    filtrados = recuperador.filtrar(fragmentos)
    assert filtrados and all(f.similitud >= 0.5 for f in filtrados)
    assert len(filtrados) < len(fragmentos)


def test_cli_eval(mini, monkeypatch, capsys) -> None:
    settings, modelo, ruta, tmp = mini
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("app.rag.models.build_embed_model", lambda _s: modelo)
    codigo = cli.main(["eval", "--banco", str(ruta), "--salida", str(tmp / "r"), "--max-k", "3"])
    salida = capsys.readouterr().out
    assert codigo == 0
    assert "Meta HU05 (hit rate top-3 >= 80 %): CUMPLE (100.0 %)" in salida
    assert list((tmp / "r").glob("hit_rate_*.md"))
