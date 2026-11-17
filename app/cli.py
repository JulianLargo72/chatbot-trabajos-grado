"""Comandos de línea: ``python -m app.cli <comando>``.

- ``check``  verifica la clave de Gemini y los dos modelos configurados (usa la API real).
- ``ingest`` lectura, fragmentación e indexación del corpus (fase F2/F3).
- ``eval``   hit rate top-k sobre el banco de preguntas (fase F4).
"""

import argparse
import logging
import sys
import time

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


def _pendiente(fase: str) -> int:
    logger.warning("Comando aún no implementado", extra={"fase": fase})
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("check", help="Verifica la clave y los modelos de Gemini")
    sub.add_parser("ingest", help="Indexa los PDF de data/corpus (F2/F3)")
    sub.add_parser("eval", help="Evalúa la recuperación con el banco de preguntas (F4)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, json_format=False)

    try:
        if args.comando == "check":
            return cmd_check()
        if args.comando == "ingest":
            return _pendiente("F2")
        return _pendiente("F4")
    except RuntimeError as exc:  # p. ej., falta la clave
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
