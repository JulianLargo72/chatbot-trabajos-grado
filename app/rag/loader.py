"""Lectura de los PDF oficiales (HU03).

Cada página se convierte en un ``Document`` de LlamaIndex con los metadatos que luego
permiten citar la fuente (HU07):

- ``file_name``: nombre del archivo PDF.
- ``page_label``: número de página impreso (etiqueta del PDF si existe; si no, 1, 2, 3…).
- ``page_index``: posición física de la página dentro del archivo (base 1).

Antes de fragmentar se eliminan encabezados y pies de página repetidos y los números de
página sueltos, para que no contaminen los fragmentos ni los embeddings.
"""

import hashlib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Metadatos que no aportan significado: no se incluyen en el texto que se convierte en
# embedding ni en el que ve el LLM (se conservan en el nodo para trazabilidad).
METADATOS_TECNICOS = ["page_index", "total_pages", "file_hash"]

_NUMERO_PAGINA = re.compile(r"^(p(á|a)g(ina)?\.?|p\.)?\s*\d{1,4}(\s*(de|/)\s*\d{1,4})?$", re.I)
_DIGITOS = re.compile(r"\d+")
_ESPACIOS = re.compile(r"[ \t\u00a0]+")
# Viñetas de Word/Symbol que pypdf extrae como caracteres de uso privado (p. ej. U+F0B7).
_VINETA_PRIVADA = re.compile(r"^[\uf000-\uf8ff\u2022\u25aa\u25e6\u25cf\u25a0]\s*")
_USO_PRIVADO = re.compile(r"[\uf000-\uf8ff]")
# Cuántas líneas del inicio y del final de cada página se revisan como posible encabezado/pie.
_LINEAS_BORDE = 2


@dataclass(frozen=True)
class PaginaPdf:
    """Texto crudo de una página con su ubicación."""

    file_name: str
    page_label: str
    page_index: int
    texto: str


def _normalizar(linea: str) -> str:
    """Clave de comparación: ignora mayúsculas y números (p. ej. «p. 4» ≈ «p. 5»)."""
    return _DIGITOS.sub("#", linea.strip().lower())


def _limpiar_linea(linea: str) -> str:
    linea = _ESPACIOS.sub(" ", linea).strip()
    linea = _VINETA_PRIVADA.sub("- ", linea)
    return _USO_PRIVADO.sub("", linea).strip()


def _lineas(texto: str) -> list[str]:
    return [_limpiar_linea(linea) for linea in texto.splitlines()]


def detectar_lineas_repetidas(paginas: list[list[str]], min_frecuencia: float = 0.5) -> set[str]:
    """Devuelve las líneas (normalizadas) que se repiten en el borde de muchas páginas.

    Una línea se considera encabezado o pie si aparece entre las primeras o últimas
    ``_LINEAS_BORDE`` líneas de al menos ``min_frecuencia`` de las páginas (y en 2 o más).
    """
    if len(paginas) < 2:
        return set()
    conteo: Counter[str] = Counter()
    for lineas in paginas:
        no_vacias = [linea for linea in lineas if linea]
        borde = no_vacias[:_LINEAS_BORDE] + no_vacias[-_LINEAS_BORDE:]
        conteo.update({_normalizar(linea) for linea in borde})
    minimo = max(2, round(len(paginas) * min_frecuencia))
    return {clave for clave, veces in conteo.items() if veces >= minimo}


def limpiar_paginas(textos: list[str]) -> list[str]:
    """Quita encabezados/pies repetidos, números de página sueltos y espacios sobrantes."""
    paginas = [_lineas(texto) for texto in textos]
    repetidas = detectar_lineas_repetidas(paginas)
    limpias: list[str] = []
    for lineas in paginas:
        conservadas = [
            linea
            for linea in lineas
            if linea and _normalizar(linea) not in repetidas and not _NUMERO_PAGINA.match(linea)
        ]
        limpias.append("\n".join(conservadas))
    return limpias


def leer_pdf(ruta: Path) -> list[PaginaPdf]:
    """Extrae el texto de cada página conservando la etiqueta de página del PDF."""
    lector = PdfReader(ruta)
    try:
        etiquetas = list(lector.page_labels)
    except Exception:  # PDF con etiquetas mal formadas: se usa la posición
        etiquetas = []
    paginas: list[PaginaPdf] = []
    for indice, pagina in enumerate(lector.pages):
        etiqueta = etiquetas[indice] if indice < len(etiquetas) else str(indice + 1)
        paginas.append(
            PaginaPdf(
                file_name=ruta.name,
                page_label=etiqueta,
                page_index=indice + 1,
                texto=pagina.extract_text() or "",
            )
        )
    return paginas


def paginas_a_documentos(paginas: list[PaginaPdf], extra: dict | None = None) -> list[Document]:
    """Limpia las páginas de un mismo PDF y las convierte en ``Document`` (omite vacías)."""
    textos = limpiar_paginas([p.texto for p in paginas])
    documentos: list[Document] = []
    for pagina, texto in zip(paginas, textos, strict=True):
        if not texto.strip():
            logger.warning(
                "Página sin texto extraíble (¿imagen escaneada?)",
                extra={"archivo": pagina.file_name, "pagina": pagina.page_label},
            )
            continue
        file_hash = (extra or {}).get("file_hash", "")
        documentos.append(
            Document(
                id_=f"{file_hash[:16] or pagina.file_name}-p{pagina.page_index}",
                text=texto,
                metadata={
                    "file_name": pagina.file_name,
                    "page_label": pagina.page_label,
                    "page_index": pagina.page_index,
                    "total_pages": len(paginas),
                    **(extra or {}),
                },
                excluded_embed_metadata_keys=METADATOS_TECNICOS,
                excluded_llm_metadata_keys=METADATOS_TECNICOS,
            )
        )
    return documentos


def listar_pdfs(directorio: Path) -> list[Path]:
    """PDF del corpus en orden alfabético (la ingesta es determinista)."""
    if not directorio.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del corpus: {directorio}")
    return sorted(p for p in directorio.iterdir() if p.suffix.lower() == ".pdf")


def hash_archivo(ruta: Path) -> str:
    """SHA-256 del contenido: cambia si el PDF cambia, aunque conserve el nombre."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 16), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_pdf(ruta: Path, file_hash: str | None = None) -> list[Document]:
    """Lee un PDF y devuelve sus páginas limpias como documentos con ``file_hash``."""
    return paginas_a_documentos(
        leer_pdf(ruta), extra={"file_hash": file_hash or hash_archivo(ruta)}
    )


def cargar_corpus(directorio: Path) -> Iterable[tuple[Path, list[Document]]]:
    """Lee cada PDF del corpus y entrega sus documentos (una página = un documento)."""
    for ruta in listar_pdfs(directorio):
        yield ruta, cargar_pdf(ruta)
