"""Fragmentación de documentos (HU03).

Se usa ``SentenceSplitter`` de LlamaIndex: corta respetando oraciones y párrafos, con
``chunk_size`` y ``chunk_overlap`` medidos en tokens. Como cada ``Document`` es una página,
ningún fragmento mezcla dos páginas y la cita (documento + página) siempre es exacta.
"""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from app.core.config import Settings


def build_splitter(settings: Settings) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        paragraph_separator="\n\n",
        include_prev_next_rel=False,  # los IDs se reasignan; no se usan relaciones prev/next
    )


def fragmentar(documentos: list[Document], settings: Settings) -> list[BaseNode]:
    """Divide los documentos en fragmentos que heredan los metadatos de su página.

    El identificador de cada fragmento es determinista (``<id de página>-c<n>``): procesar
    el mismo PDF dos veces produce los mismos IDs, lo que hace la ingesta idempotente.
    """
    nodos = build_splitter(settings).get_nodes_from_documents(documentos)
    contador: dict[str, int] = {}
    for nodo in nodos:
        pagina = nodo.ref_doc_id or "doc"
        contador[pagina] = contador.get(pagina, 0) + 1
        nodo.id_ = f"{pagina}-c{contador[pagina]}"
    return nodos
