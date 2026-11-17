"""Índice vectorial persistente en ChromaDB (HU04).

- La colección usa distancia coseno y guarda en sus metadatos el modelo de embeddings con
  el que se construyó: si se intenta consultar o actualizar con otro modelo, se detiene con
  un error claro (si los modelos difieren, la búsqueda falla en silencio).
- Cada vector lleva ``file_name``, ``page_label`` y ``file_hash``. Con el hash se decide qué
  archivos son nuevos, cuáles cambiaron y cuáles ya están indexados (ingesta incremental).
- El índice se recarga desde disco sin reprocesar los PDF.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import Settings

logger = logging.getLogger(__name__)

CLAVE_MODELO = "embedding_model"


class ModeloEmbeddingsDistintoError(RuntimeError):
    """El índice se construyó con un modelo de embeddings diferente al configurado."""


@dataclass(frozen=True)
class ArchivoIndexado:
    file_name: str
    file_hash: str
    vectores: int


def _cliente(settings: Settings) -> chromadb.ClientAPI:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def obtener_coleccion(settings: Settings) -> Collection:
    """Abre (o crea) la colección y verifica que el modelo de embeddings coincida."""
    coleccion = _cliente(settings).get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine", CLAVE_MODELO: settings.gemini_embedding_model},
    )
    modelo_indice = (coleccion.metadata or {}).get(CLAVE_MODELO)
    if modelo_indice and modelo_indice != settings.gemini_embedding_model:
        raise ModeloEmbeddingsDistintoError(
            f"El índice se construyó con '{modelo_indice}' y la configuración usa "
            f"'{settings.gemini_embedding_model}'. Ejecuta: python -m app.cli ingest --reset"
        )
    return coleccion


def eliminar_coleccion(settings: Settings) -> None:
    cliente = _cliente(settings)
    if settings.chroma_collection in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(settings.chroma_collection)
        logger.info("Colección eliminada", extra={"coleccion": settings.chroma_collection})


def archivos_indexados(coleccion: Collection) -> dict[str, ArchivoIndexado]:
    """Archivos presentes en el índice con su hash y número de vectores."""
    metadatos = coleccion.get(include=["metadatas"])["metadatas"] or []
    conteo: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for meta in metadatos:
        nombre = str(meta.get("file_name", ""))
        conteo[nombre] = conteo.get(nombre, 0) + 1
        hashes[nombre] = str(meta.get("file_hash", ""))
    return {n: ArchivoIndexado(n, hashes[n], conteo[n]) for n in conteo}


def eliminar_archivo(coleccion: Collection, file_name: str) -> None:
    coleccion.delete(where={"file_name": file_name})


def agregar_nodos(coleccion: Collection, nodos: list[BaseNode], embed_model: BaseEmbedding) -> int:
    """Calcula los embeddings de los nodos y los guarda en Chroma. Devuelve la dimensión."""
    if not nodos:
        return 0
    textos = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodos]
    vectores = embed_model.get_text_embedding_batch(textos)
    for nodo, vector in zip(nodos, vectores, strict=True):
        nodo.embedding = vector
    ChromaVectorStore(chroma_collection=coleccion).add(nodos)
    return len(vectores[0])


def dimension(coleccion: Collection) -> int:
    muestra = coleccion.get(limit=1, include=["embeddings"])["embeddings"]
    return len(muestra[0]) if muestra is not None and len(muestra) else 0


def tamano_en_disco(directorio: Path) -> int:
    """Bytes que ocupa el índice persistente."""
    if not directorio.exists():
        return 0
    return sum(p.stat().st_size for p in directorio.rglob("*") if p.is_file())


def cargar_indice(settings: Settings, embed_model: BaseEmbedding) -> VectorStoreIndex:
    """Recarga el índice desde disco (no relee ni re-embebe los PDF)."""
    coleccion = obtener_coleccion(settings)
    if coleccion.count() == 0:
        raise RuntimeError("El índice está vacío. Ejecuta primero: python -m app.cli ingest")
    return VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=coleccion), embed_model=embed_model
    )


def similitud_coseno(puntaje_llamaindex: float) -> float:
    """Convierte el puntaje del adaptador de Chroma en similitud coseno real.

    ``ChromaVectorStore`` devuelve ``exp(-distancia)``; con espacio coseno,
    ``distancia = 1 - similitud``, así que ``similitud = 1 + ln(puntaje)``.
    """
    if puntaje_llamaindex <= 0:
        return -1.0
    return max(-1.0, min(1.0, 1.0 + math.log(puntaje_llamaindex)))
