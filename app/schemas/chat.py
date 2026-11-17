"""Contrato de la API de chat (HU06, HU07)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TipoRespuesta = Literal["respondida", "sin_informacion", "fuera_de_dominio"]

EJEMPLO_PREGUNTA = "¿Cuál es la extensión máxima del resumen?"


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"pregunta": EJEMPLO_PREGUNTA}]})

    pregunta: str = Field(
        min_length=3,
        max_length=1000,
        description="Pregunta en lenguaje natural sobre la elaboración del trabajo de grado.",
    )

    @field_validator("pregunta")
    @classmethod
    def _sin_espacios_sobrantes(cls, valor: str) -> str:
        valor = " ".join(valor.split())
        if len(valor) < 3:
            raise ValueError("La pregunta debe tener al menos 3 caracteres.")
        return valor


class Fuente(BaseModel):
    documento: str = Field(description="Nombre del archivo PDF del corpus.")
    pagina: str = Field(description="Página impresa del documento donde está el fragmento.")
    puntaje: float = Field(description="Similitud coseno entre la pregunta y el fragmento (0-1).")


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "respuesta": "Según el formato institucional, el resumen tiene una extensión "
                    "máxima de 300 palabras (Formato de proyecto de grado, p. 1).",
                    "fuentes": [
                        {
                            "documento": "Formato_Proyecto_de_Grado_ETIC_2026_limpio.pdf",
                            "pagina": "1",
                            "puntaje": 0.701,
                        }
                    ],
                    "tipo": "respondida",
                }
            ]
        }
    )

    respuesta: str
    fuentes: list[Fuente] = Field(
        default_factory=list,
        description="Fragmentos efectivamente usados como contexto (vacío si no hubo respuesta).",
    )
    tipo: TipoRespuesta = Field(
        description="respondida | sin_informacion (HU08) | fuera_de_dominio (HU09)"
    )
