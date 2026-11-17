"""Servicio de ingesta: lectura → limpieza → fragmentación → embeddings → Chroma.

Lo usan la CLI (``python -m app.cli ingest``) y el endpoint ``POST /api/admin/ingest``,
de modo que ambos producen exactamente el mismo resultado.

La ingesta es incremental e idempotente: se compara el SHA-256 de cada PDF con el guardado
en el índice; solo se procesan los archivos nuevos o modificados y se eliminan del índice
los que ya no están en el corpus. Ejecutarla dos veces seguidas no duplica vectores.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tokenizer

from app.core.config import Settings
from app.rag import index_store
from app.rag.chunker import fragmentar
from app.rag.loader import cargar_pdf, hash_archivo, listar_pdfs

logger = logging.getLogger(__name__)

NUEVO, MODIFICADO, SIN_CAMBIOS, ELIMINADO = "nuevo", "modificado", "sin cambios", "eliminado"


@dataclass
class ResumenArchivo:
    archivo: str
    estado: str = NUEVO
    paginas: int = 0
    fragmentos: int = 0
    vectores: int = 0
    tokens_promedio: int = 0
    tokens_min: int = 0
    tokens_max: int = 0


@dataclass
class ResultadoIngesta:
    documentos: int = 0
    paginas: int = 0
    fragmentos: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    segundos: float = 0.0
    archivos: list[ResumenArchivo] = field(default_factory=list)
    # Solo cuando se indexa (no en --solo-fragmentar):
    indexado: bool = False
    modelo_embeddings: str = ""
    vectores: int = 0
    dimension: int = 0
    verificacion_ok: bool | None = None
    tamano_indice_mb: float = 0.0
    nodos: list[BaseNode] = field(default_factory=list, repr=False)

    def como_dict(self) -> dict:
        datos = asdict(self)
        datos.pop("nodos")
        return datos

    def contar(self, estado: str) -> int:
        return sum(1 for a in self.archivos if a.estado == estado)


def _contar_tokens(nodos: list[BaseNode]) -> list[int]:
    tokenizar = get_tokenizer()
    return [len(tokenizar(n.get_content(metadata_mode=MetadataMode.NONE))) for n in nodos]


def _procesar_archivo(
    ruta: Path, settings: Settings, file_hash: str
) -> tuple[ResumenArchivo, list[BaseNode]]:
    documentos = cargar_pdf(ruta, file_hash)
    nodos = fragmentar(documentos, settings)
    tokens = _contar_tokens(nodos) or [0]
    resumen = ResumenArchivo(
        archivo=ruta.name,
        paginas=len(documentos),
        fragmentos=len(nodos),
        tokens_promedio=round(sum(tokens) / len(tokens)),
        tokens_min=min(tokens),
        tokens_max=max(tokens),
    )
    return resumen, nodos


def _totalizar(resultado: ResultadoIngesta, inicio: float) -> None:
    presentes = [a for a in resultado.archivos if a.estado != ELIMINADO]
    resultado.documentos = len(presentes)
    resultado.paginas = sum(a.paginas for a in presentes)
    resultado.fragmentos = sum(a.fragmentos for a in presentes)
    resultado.segundos = round(time.perf_counter() - inicio, 2)


def procesar_corpus(settings: Settings) -> ResultadoIngesta:
    """Solo lee y fragmenta (sin embeddings ni índice). Útil para ajustar ``chunk_size``."""
    inicio = time.perf_counter()
    resultado = ResultadoIngesta(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    for ruta in listar_pdfs(settings.corpus_dir):
        resumen, nodos = _procesar_archivo(ruta, settings, hash_archivo(ruta))
        logger.info("Archivo fragmentado", extra=asdict(resumen))
        resultado.archivos.append(resumen)
        resultado.nodos.extend(nodos)
    _totalizar(resultado, inicio)
    if resultado.documentos == 0:
        logger.warning("No hay PDF en el corpus", extra={"carpeta": str(settings.corpus_dir)})
    return resultado


def ingestar(
    settings: Settings,
    embed_model_factory: Callable[[Settings], BaseEmbedding] | None = None,
    reset: bool = False,
) -> ResultadoIngesta:
    """Ingesta completa e incremental hacia el índice persistente de Chroma.

    ``embed_model_factory`` solo se invoca si hay archivos que indexar; así, una ejecución
    sin cambios no llama a la API de Gemini.
    """
    if embed_model_factory is None:
        from app.rag.models import build_embed_model

        embed_model_factory = build_embed_model

    inicio = time.perf_counter()
    resultado = ResultadoIngesta(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        indexado=True,
        modelo_embeddings=settings.gemini_embedding_model,
    )

    if reset:
        index_store.eliminar_coleccion(settings)
    coleccion = index_store.obtener_coleccion(settings)
    indexados = index_store.archivos_indexados(coleccion)
    embed_model: BaseEmbedding | None = None
    rutas = listar_pdfs(settings.corpus_dir)

    for ruta in rutas:
        file_hash = hash_archivo(ruta)
        previo = indexados.get(ruta.name)
        if previo and previo.file_hash == file_hash:
            resumen = ResumenArchivo(
                archivo=ruta.name,
                estado=SIN_CAMBIOS,
                fragmentos=previo.vectores,
                vectores=previo.vectores,
            )
            # Las páginas se leen del PDF (rápido, sin API) para que el reporte sea completo.
            resumen.paginas = len(cargar_pdf(ruta, file_hash))
            resultado.archivos.append(resumen)
            logger.info("Archivo sin cambios, se omite", extra={"archivo": ruta.name})
            continue

        resumen, nodos = _procesar_archivo(ruta, settings, file_hash)
        resumen.estado = MODIFICADO if previo else NUEVO
        if previo:
            index_store.eliminar_archivo(coleccion, ruta.name)
        if embed_model is None:
            embed_model = embed_model_factory(settings)
        index_store.agregar_nodos(coleccion, nodos, embed_model)
        resumen.vectores = len(coleccion.get(where={"file_name": ruta.name})["ids"])
        resultado.archivos.append(resumen)
        resultado.nodos.extend(nodos)
        logger.info("Archivo indexado", extra=asdict(resumen))

    en_disco = {r.name for r in rutas}
    for nombre, previo in indexados.items():
        if nombre not in en_disco:
            index_store.eliminar_archivo(coleccion, nombre)
            resultado.archivos.append(
                ResumenArchivo(archivo=nombre, estado=ELIMINADO, vectores=previo.vectores)
            )
            logger.info("Archivo retirado del índice", extra={"archivo": nombre})

    _totalizar(resultado, inicio)
    resultado.vectores = coleccion.count()
    resultado.dimension = index_store.dimension(coleccion)
    resultado.verificacion_ok = (
        all(a.vectores == a.fragmentos for a in resultado.archivos if a.estado != ELIMINADO)
        and resultado.vectores == resultado.fragmentos
    )
    resultado.tamano_indice_mb = round(index_store.tamano_en_disco(settings.chroma_dir) / 2**20, 2)

    nivel = logging.INFO if resultado.verificacion_ok else logging.ERROR
    logger.log(
        nivel,
        "Ingesta completada",
        extra={k: v for k, v in resultado.como_dict().items() if k != "archivos"},
    )
    return resultado


def exportar_fragmentos(nodos: list[BaseNode], destino: Path) -> None:
    """Guarda los fragmentos en JSONL para revisarlos a mano (evidencia de HU03)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as archivo:
        for nodo in nodos:
            registro = {
                "id": nodo.node_id,
                "documento": nodo.metadata.get("file_name"),
                "pagina": nodo.metadata.get("page_label"),
                "texto": nodo.get_content(metadata_mode=MetadataMode.NONE),
            }
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
