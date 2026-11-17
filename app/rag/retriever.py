"""Recuperación semántica sobre el índice (HU05; la usan la evaluación y el chat).

Devuelve los ``top_k`` fragmentos más similares con su documento, página y similitud
coseno real (ver ``index_store.similitud_coseno``). El filtro por umbral se aplica aparte,
para que la evaluación pueda analizar todos los puntajes y el chat descarte los débiles.
"""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import MetadataMode

from app.core.config import Settings
from app.rag import index_store


@dataclass(frozen=True)
class FragmentoRecuperado:
    node_id: str
    documento: str
    pagina: str
    similitud: float
    texto: str


class Recuperador:
    def __init__(self, indice: VectorStoreIndex, top_k: int, umbral: float) -> None:
        self._indice = indice
        self.top_k = top_k
        self.umbral = umbral

    def recuperar(self, pregunta: str, top_k: int | None = None) -> list[FragmentoRecuperado]:
        """Fragmentos ordenados de mayor a menor similitud (sin filtrar por umbral)."""
        k = top_k or self.top_k
        resultados = self._indice.as_retriever(similarity_top_k=k).retrieve(pregunta)
        fragmentos = [
            FragmentoRecuperado(
                node_id=r.node.node_id,
                documento=str(r.node.metadata.get("file_name", "")),
                pagina=str(r.node.metadata.get("page_label", "")),
                similitud=round(index_store.similitud_coseno(r.score or 0.0), 4),
                texto=r.node.get_content(metadata_mode=MetadataMode.NONE),
            )
            for r in resultados
        ]
        return sorted(fragmentos, key=lambda f: f.similitud, reverse=True)

    def filtrar(self, fragmentos: list[FragmentoRecuperado]) -> list[FragmentoRecuperado]:
        """Conserva solo los fragmentos con similitud >= umbral (base de HU08)."""
        return [f for f in fragmentos if f.similitud >= self.umbral]


def crear_recuperador(settings: Settings, embed_model: BaseEmbedding | None = None) -> Recuperador:
    if embed_model is None:
        from app.rag.models import build_embed_model

        embed_model = build_embed_model(settings)
    indice = index_store.cargar_indice(settings, embed_model)
    return Recuperador(indice, settings.top_k, settings.similarity_threshold)
