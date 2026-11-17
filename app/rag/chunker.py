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
    )


def fragmentar(documentos: list[Document], settings: Settings) -> list[BaseNode]:
    """Divide los documentos en fragmentos que heredan los metadatos de su página."""
    return build_splitter(settings).get_nodes_from_documents(documentos)
