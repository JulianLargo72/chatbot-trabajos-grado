"""Comandos de línea: ``python -m app.cli <comando>``.

- ``check``  verifica la clave de Gemini y los dos modelos configurados (usa la API real).
- ``ingest`` lectura, limpieza y fragmentación del corpus (F2); la indexación llega en F3.
- ``eval``   hit rate top-k sobre el banco de preguntas (fase F4).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger("app.cli")


def cmd_check() -> int:
    """Hace una llamada mínima a cada modelo para confirmar que la clave y los nombres sirven."""
    from app.rag.models import build_embed_model, build_llm

    settings = get_settings()
    ok = True

    try:
        inicio = time.perf_counter()
        vector = build_embed_model(settings).get_text_embedding("prueba de conexión")
        logger.info(
            "Embeddings OK",
            extra={
                "modelo": settings.gemini_embedding_model,
                "dimension": len(vector),
                "ms": round((time.perf_counter() - inicio) * 1000),
            },
        )
    except Exception as exc:
        ok = False
        logger.error(
            "Falló el modelo de embeddings",
            extra={"modelo": settings.gemini_embedding_model, "error": str(exc)},
        )

    try:
        inicio = time.perf_counter()
        respuesta = build_llm(settings).complete("Responde solo con la palabra: listo")
        logger.info(
            "LLM OK",
            extra={
                "modelo": settings.gemini_llm_model,
                "respuesta": respuesta.text.strip()[:50],
                "ms": round((time.perf_counter() - inicio) * 1000),
            },
        )
    except Exception as exc:
        ok = False
        logger.error(
            "Falló el modelo de lenguaje",
            extra={"modelo": settings.gemini_llm_model, "error": str(exc)},
        )

    return 0 if ok else 1


def _imprimir_resumen(resultado) -> None:
    """Tabla legible para la terminal (evidencia de HU03, Figura 2)."""
    print()
    print(f"{'Archivo':<50} {'Págs':>5} {'Frag':>5} {'Tok prom':>9} {'Tok min':>8} {'Tok max':>8}")
    print("-" * 90)
    for a in resultado.archivos:
        print(
            f"{a.archivo:<50} {a.paginas:>5} {a.fragmentos:>5} "
            f"{a.tokens_promedio:>9} {a.tokens_min:>8} {a.tokens_max:>8}"
        )
    print("-" * 90)
    print(
        f"Documentos: {resultado.documentos} | Páginas: {resultado.paginas} | "
        f"Fragmentos: {resultado.fragmentos} | chunk_size={resultado.chunk_size} "
        f"chunk_overlap={resultado.chunk_overlap} | {resultado.segundos} s"
    )
    print()


def cmd_ingest(exportar: Path | None) -> int:
    from app.services.ingestion import exportar_fragmentos, procesar_corpus

    resultado = procesar_corpus(get_settings())
    _imprimir_resumen(resultado)
    if exportar:
        exportar_fragmentos(resultado.nodos, exportar)
        logger.info("Fragmentos exportados", extra={"archivo": str(exportar)})
    return 0 if resultado.fragmentos > 0 else 1


def _pendiente(fase: str) -> int:
    logger.warning("Comando aún no implementado", extra={"fase": fase})
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("check", help="Verifica la clave y los modelos de Gemini")
    p_ingest = sub.add_parser("ingest", help="Lee y fragmenta los PDF de data/corpus")
    p_ingest.add_argument(
        "--exportar",
        type=Path,
        metavar="RUTA",
        help="Guarda los fragmentos en JSONL (p. ej. eval/reports/fragmentos.jsonl)",
    )
    sub.add_parser("eval", help="Evalúa la recuperación con el banco de preguntas (F4)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, json_format=False)

    try:
        if args.comando == "check":
            return cmd_check()
        if args.comando == "ingest":
            return cmd_ingest(args.exportar)
        return _pendiente("F4")
    except (RuntimeError, FileNotFoundError) as exc:  # p. ej., falta la clave o el corpus
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
