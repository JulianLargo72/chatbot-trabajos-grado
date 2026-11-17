"""Genera PDF pequeños con texto para las pruebas (sin depender del corpus real)."""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)


def _escapar(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def crear_pdf(ruta: Path, paginas: list[list[str]], primera_etiqueta: int | None = None) -> Path:
    """Crea un PDF con una línea de texto por elemento de cada lista.

    Si se da ``primera_etiqueta``, las páginas se etiquetan desde ese número (como el
    manual IEEE, que empieza en la página 4).
    """
    escritor = PdfWriter()
    fuente = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    for lineas in paginas:
        pagina = escritor.add_blank_page(width=612, height=792)
        flujo = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for linea in lineas:
            flujo.append(f"({_escapar(linea)}) Tj T*")
        flujo.append("ET")
        contenido = DecodedStreamObject()
        contenido.set_data("\n".join(flujo).encode("cp1252"))
        pagina[NameObject("/Contents")] = escritor._add_object(contenido)
        pagina[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): fuente})}
        )
    if primera_etiqueta is not None:
        escritor.set_page_label(0, len(paginas) - 1, style="/D", start=primera_etiqueta)
    with ruta.open("wb") as archivo:
        escritor.write(archivo)
    return ruta


__all__ = ["crear_pdf"]
