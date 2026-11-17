"""Comandos de línea: ``python -m app.cli <comando>``.

- ``check``   verifica la clave de Gemini y los dos modelos configurados (usa la API real).
- ``ingest``  lectura, fragmentación, embeddings e indexación incremental en Chroma (HU03, HU04).
- ``indice``  estadísticas del índice persistente, sin reprocesar los PDF (HU04).
- ``buscar``  recupera los fragmentos más similares a una pregunta (prueba manual del índice).
- ``eval``    hit rate top-k sobre el banco de preguntas (fase F4).
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
    """Tabla legible para la terminal (evidencia de HU03 y HU04)."""
    print()
    encabezado = f"{'Archivo':<48} {'Estado':<12} {'Págs':>5} {'Frag':>5}"
    if resultado.indexado:
        encabezado += f" {'Vect':>5}"
    encabezado += f" {'Tok prom':>9} {'Tok min':>8} {'Tok max':>8}"
    print(encabezado)
    print("-" * len(encabezado))
    for a in resultado.archivos:
        fila = f"{a.archivo:<48} {a.estado:<12} {a.paginas:>5} {a.fragmentos:>5}"
        if resultado.indexado:
            fila += f" {a.vectores:>5}"
        tokens = (
            f" {a.tokens_promedio:>9} {a.tokens_min:>8} {a.tokens_max:>8}"
            if a.tokens_max
            else f" {'-':>9} {'-':>8} {'-':>8}"
        )
        print(fila + tokens)
    print("-" * len(encabezado))
    print(
        f"Documentos: {resultado.documentos} | Páginas: {resultado.paginas} | "
        f"Fragmentos: {resultado.fragmentos} | chunk_size={resultado.chunk_size} "
        f"chunk_overlap={resultado.chunk_overlap} | {resultado.segundos} s"
    )
    if resultado.indexado:
        verificacion = "OK" if resultado.verificacion_ok else "ERROR"
        print(
            f"Vectores en el índice: {resultado.vectores} | Dimensión: {resultado.dimension} | "
            f"Modelo: {resultado.modelo_embeddings} | Índice: {resultado.tamano_indice_mb} MB"
        )
        print(
            f"Nuevos: {resultado.contar('nuevo')} | Modificados: {resultado.contar('modificado')}"
            f" | Sin cambios: {resultado.contar('sin cambios')}"
            f" | Eliminados: {resultado.contar('eliminado')}"
            f" | Verificación fragmentos = vectores: {verificacion}"
        )
    print()


def cmd_ingest(exportar: Path | None, reset: bool, solo_fragmentar: bool) -> int:
    from app.services.ingestion import exportar_fragmentos, ingestar, procesar_corpus

    settings = get_settings()
    resultado = procesar_corpus(settings) if solo_fragmentar else ingestar(settings, reset=reset)
    _imprimir_resumen(resultado)
    if exportar:
        exportar_fragmentos(resultado.nodos, exportar)
        logger.info(
            "Fragmentos exportados",
            extra={"archivo": str(exportar), "fragmentos": len(resultado.nodos)},
        )
    if resultado.documentos == 0:
        return 1
    return 0 if resultado.verificacion_ok in (None, True) else 1


def cmd_indice() -> int:
    """Abre el índice existente y muestra sus datos, sin leer los PDF ni llamar a Gemini."""
    from app.rag import index_store

    settings = get_settings()
    inicio = time.perf_counter()
    coleccion = index_store.obtener_coleccion(settings)
    archivos = index_store.archivos_indexados(coleccion)
    ms = round((time.perf_counter() - inicio) * 1000)
    print()
    print(f"{'Archivo':<48} {'Vectores':>8}  Hash (SHA-256)")
    print("-" * 80)
    for archivo in sorted(archivos.values(), key=lambda a: a.file_name):
        print(f"{archivo.file_name:<48} {archivo.vectores:>8}  {archivo.file_hash[:16]}…")
    print("-" * 80)
    print(
        f"Colección: {settings.chroma_collection} | Vectores: {coleccion.count()} | "
        f"Dimensión: {index_store.dimension(coleccion)} | "
        f"Modelo: {(coleccion.metadata or {}).get(index_store.CLAVE_MODELO)} | "
        f"Disco: {round(index_store.tamano_en_disco(settings.chroma_dir) / 2**20, 2)} MB | "
        f"Carga: {ms} ms"
    )
    print()
    return 0 if coleccion.count() > 0 else 1


def cmd_buscar(pregunta: str, top_k: int | None) -> int:
    """Recupera los fragmentos más parecidos a la pregunta (no usa el LLM)."""
    from app.rag import index_store
    from app.rag.models import build_embed_model

    settings = get_settings()
    k = top_k or settings.top_k
    indice = index_store.cargar_indice(settings, build_embed_model(settings))
    resultados = indice.as_retriever(similarity_top_k=k).retrieve(pregunta)
    print(f"\nPregunta: {pregunta}\n")
    for posicion, r in enumerate(resultados, start=1):
        similitud = index_store.similitud_coseno(r.score or 0.0)
        marca = "" if similitud >= settings.similarity_threshold else "  (bajo el umbral)"
        print(
            f"{posicion}. {r.node.metadata.get('file_name')} | pág. "
            f"{r.node.metadata.get('page_label')} | similitud {similitud:.3f}{marca}"
        )
        texto = r.node.get_content().replace("\n", " ")
        print(f"   {texto[:220]}…\n")
    return 0 if resultados else 1


def _pendiente(fase: str) -> int:
    logger.warning("Comando aún no implementado", extra={"fase": fase})
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("check", help="Verifica la clave y los modelos de Gemini")

    p_ingest = sub.add_parser("ingest", help="Indexa los PDF de data/corpus en Chroma")
    p_ingest.add_argument(
        "--exportar",
        type=Path,
        metavar="RUTA",
        help="Guarda los fragmentos procesados en JSONL (p. ej. eval/reports/fragmentos.jsonl)",
    )
    p_ingest.add_argument(
        "--reset", action="store_true", help="Borra el índice y lo reconstruye desde cero"
    )
    p_ingest.add_argument(
        "--solo-fragmentar",
        action="store_true",
        help="Solo lee y fragmenta (sin embeddings ni índice; no consume cuota)",
    )

    sub.add_parser("indice", help="Muestra el estado del índice sin reprocesar")
    p_buscar = sub.add_parser("buscar", help="Recupera los fragmentos más similares")
    p_buscar.add_argument("pregunta", help="Pregunta en lenguaje natural (entre comillas)")
    p_buscar.add_argument("--top-k", type=int, default=None, help="Cantidad de fragmentos")

    sub.add_parser("eval", help="Evalúa la recuperación con el banco de preguntas (F4)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, json_format=False)

    try:
        if args.comando == "check":
            return cmd_check()
        if args.comando == "ingest":
            return cmd_ingest(args.exportar, args.reset, args.solo_fragmentar)
        if args.comando == "indice":
            return cmd_indice()
        if args.comando == "buscar":
            return cmd_buscar(args.pregunta, args.top_k)
        return _pendiente("F4")
    except (RuntimeError, FileNotFoundError) as exc:  # falta la clave, el corpus o el índice
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
