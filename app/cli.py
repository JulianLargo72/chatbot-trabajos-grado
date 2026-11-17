"""Comandos de línea: ``python -m app.cli <comando>``.

- ``check``   verifica la clave de Gemini y los dos modelos configurados (usa la API real).
- ``ingest``  lectura, fragmentación, embeddings e indexación incremental en Chroma (HU03, HU04).
- ``indice``  estadísticas del índice persistente, sin reprocesar los PDF (HU04).
- ``buscar``  recupera los fragmentos más similares a una pregunta (prueba manual del índice).
- ``preguntar`` hace una consulta completa (recuperación + Gemini), igual que POST /api/chat.
- ``eval``    hit rate top-k y análisis del umbral con el banco de preguntas (HU05).
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
    from app.rag.retriever import crear_recuperador

    settings = get_settings()
    recuperador = crear_recuperador(settings)
    fragmentos = recuperador.recuperar(pregunta, top_k=top_k)
    print(f"\nPregunta: {pregunta}\n")
    for posicion, f in enumerate(fragmentos, start=1):
        marca = "" if f.similitud >= settings.similarity_threshold else "  (bajo el umbral)"
        print(f"{posicion}. {f.documento} | pág. {f.pagina} | similitud {f.similitud:.3f}{marca}")
        print(f"   {f.texto.replace(chr(10), ' ')[:220]}…\n")
    return 0 if fragmentos else 1


def cmd_preguntar(pregunta: str) -> int:
    """Consulta completa desde la terminal: usa 1 embedding y 1 llamada al LLM."""
    from app.rag.models import build_llm
    from app.rag.retriever import crear_recuperador
    from app.services.chat import ChatService, ServicioNoDisponibleError

    settings = get_settings()
    servicio = ChatService(crear_recuperador(settings), build_llm(settings))
    try:
        r = servicio.responder(pregunta)
    except ServicioNoDisponibleError as exc:
        logger.error(str(exc))
        return 1
    print(f"\nPregunta: {pregunta}\nTipo: {r.tipo} | {r.ms} ms\n\n{r.respuesta}\n")
    for f in r.fuentes:
        print(f"  - {f.documento} | pág. {f.pagina} | puntaje {f.puntaje:.3f}")
    print()
    return 0


def cmd_eval(banco: Path, salida: Path, max_k: int, solo_a: bool, pausa: float) -> int:
    """Calcula el hit rate (HU05), guarda el reporte y sugiere el umbral de similitud."""
    from app.rag.retriever import crear_recuperador
    from app.services.evaluation import cargar_banco, evaluar, guardar_reporte

    settings = get_settings()
    datos = cargar_banco(banco)
    recuperador = crear_recuperador(settings)

    def progreso(n: int, total: int, pid: str) -> None:
        print(f"\r  Evaluando {n}/{total} ({pid})   ", end="", flush=True)

    reporte = evaluar(
        datos,
        recuperador,
        settings,
        max_k=max_k,
        categorias=("A",) if solo_a else ("A", "B", "C"),
        pausa=pausa,
        progreso=progreso,
    )
    print("\r" + " " * 40 + "\r", end="")
    ruta_json, ruta_md = guardar_reporte(reporte, salida)

    m, u = reporte.metricas, reporte.umbral
    print()
    print(f"Preguntas categoría A: {m['preguntas_categoria_a']} | chunk_size={settings.chunk_size}")
    for k, v in m["hit_rate_documento"].items():
        print(
            f"  Hit rate documento @{k}: {v * 100:5.1f} %   página @{k}: "
            f"{m['hit_rate_pagina'][k] * 100:5.1f} %"
        )
    print(f"  MRR documento: {m['mrr_documento']:.3f}")
    for d in reporte.por_documento.values():
        print(f"  {d['archivo']:<48} top-3: {d['hit_rate_top3'] * 100:5.1f} %")
    fallos = [
        r
        for r in reporte.preguntas
        if r.categoria == "A" and (r.pos_documento is None or r.pos_documento > 3)
    ]
    if fallos:
        print("  Fallos en top-3: " + ", ".join(r.id for r in fallos))
    estado = "CUMPLE" if reporte.cumple_meta else "NO CUMPLE"
    print(f"\nMeta HU05 (hit rate top-3 >= 80 %): {estado} ({reporte.hit_rate_top3 * 100:.1f} %)")
    for cat, est in u["similitud_max_por_categoria"].items():
        if est:
            print(
                f"  Similitud máx. cat. {cat}: min {est['min']:.3f} | "
                f"prom {est['promedio']:.3f} | max {est['max']:.3f}"
            )
    if u["sugerido"] is not None:
        print(f"  Umbral actual: {u['actual']} | sugerido: {u['sugerido']}")
    print(f"\nReporte: {ruta_md}\n         {ruta_json}\n")
    return 0 if reporte.cumple_meta else 1


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

    p_preg = sub.add_parser("preguntar", help="Consulta completa con Gemini (como /api/chat)")
    p_preg.add_argument("pregunta", help="Pregunta en lenguaje natural (entre comillas)")

    p_eval = sub.add_parser("eval", help="Hit rate top-k con el banco de preguntas (HU05)")
    p_eval.add_argument(
        "--banco", type=Path, default=Path("eval/banco_preguntas.json"), help="Banco JSON"
    )
    p_eval.add_argument(
        "--salida", type=Path, default=Path("eval/reports"), help="Carpeta de reportes"
    )
    p_eval.add_argument("--max-k", type=int, default=5, help="Fragmentos por pregunta")
    p_eval.add_argument(
        "--solo-a", action="store_true", help="Solo categoría A (sin análisis de umbral)"
    )
    p_eval.add_argument(
        "--pausa", type=float, default=0.0, help="Segundos entre preguntas (cuota gratuita)"
    )
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
        if args.comando == "preguntar":
            return cmd_preguntar(args.pregunta)
        return cmd_eval(args.banco, args.salida, args.max_k, args.solo_a, args.pausa)
    except (RuntimeError, FileNotFoundError) as exc:  # falta la clave, el corpus o el índice
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
