"""Servicio de chat: recuperación → umbral → generación con Gemini → fuentes (HU06-HU09).

Flujo:
1. Se recuperan los ``top_k`` fragmentos y se descartan los que no superan el umbral.
2. Con fragmentos relevantes, Gemini responde usando solo ese contexto y citando la fuente.
3. Sin fragmentos relevantes, se consulta a Gemini sin contexto solo para distinguir entre
   «del dominio sin información» (HU08) y «fuera de dominio» (HU09). Si el modelo intentara
   responder de todas formas, se fuerza el mensaje de «sin información»: sin contexto no
   puede haber una respuesta sustentada.
4. Las etiquetas del modelo se reemplazan por mensajes estándar y, en esos casos, la lista de
   fuentes queda vacía. Las fuentes son exactamente los fragmentos enviados como contexto.
"""

import logging
import time
from dataclasses import dataclass, field

from llama_index.core.llms import LLM, ChatMessage, MessageRole

from app.rag import prompts
from app.rag.retriever import FragmentoRecuperado, Recuperador
from app.schemas.chat import TipoRespuesta

logger = logging.getLogger(__name__)


class ServicioNoDisponibleError(RuntimeError):
    """El modelo de lenguaje o el de embeddings no respondió (timeout, cuota, red)."""


@dataclass
class FuenteUsada:
    documento: str
    pagina: str
    puntaje: float


@dataclass
class RespuestaChat:
    respuesta: str
    tipo: TipoRespuesta
    fuentes: list[FuenteUsada] = field(default_factory=list)
    similitud_max: float = 0.0
    ms: int = 0


def fuentes_unicas(fragmentos: list[FragmentoRecuperado]) -> list[FuenteUsada]:
    """Una fuente por (documento, página), con el mayor puntaje, en orden de relevancia."""
    vistas: dict[tuple[str, str], FuenteUsada] = {}
    for f in fragmentos:
        clave = (f.documento, f.pagina)
        if clave not in vistas or f.similitud > vistas[clave].puntaje:
            vistas[clave] = FuenteUsada(f.documento, f.pagina, round(f.similitud, 3))
    return sorted(vistas.values(), key=lambda x: x.puntaje, reverse=True)


def clasificar(texto: str) -> TipoRespuesta:
    marcado = texto.upper()
    if prompts.ETIQUETA_FUERA_DE_DOMINIO in marcado:
        return "fuera_de_dominio"
    if prompts.ETIQUETA_SIN_INFORMACION in marcado:
        return "sin_informacion"
    return "respondida"


class ChatService:
    def __init__(self, recuperador: Recuperador, llm: LLM) -> None:
        self._recuperador = recuperador
        self._llm = llm

    def _generar(self, pregunta: str, fragmentos: list[FragmentoRecuperado]) -> str:
        mensajes = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompts.SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=prompts.construir_mensaje_usuario(pregunta, fragmentos),
            ),
        ]
        try:
            return (self._llm.chat(mensajes).message.content or "").strip()
        except Exception as exc:
            logger.error(
                "Fallo del modelo de lenguaje",
                extra={"error_tipo": type(exc).__name__, "error": str(exc)[:300]},
            )
            raise ServicioNoDisponibleError(
                "El modelo de lenguaje no está disponible en este momento. Intenta de nuevo."
            ) from exc

    def responder(self, pregunta: str) -> RespuestaChat:
        inicio = time.perf_counter()
        try:
            recuperados = self._recuperador.recuperar(pregunta)
        except Exception as exc:
            logger.error(
                "Fallo en la recuperación",
                extra={"error_tipo": type(exc).__name__, "error": str(exc)[:300]},
            )
            raise ServicioNoDisponibleError(
                "No fue posible consultar los documentos en este momento. Intenta de nuevo."
            ) from exc

        relevantes = self._recuperador.filtrar(recuperados)
        texto = self._generar(pregunta, relevantes)
        tipo = clasificar(texto)

        if not relevantes and tipo == "respondida":
            # Sin contexto no hay respuesta sustentada posible (protección anti-alucinación).
            tipo = "sin_informacion"

        if tipo == "sin_informacion":
            texto, fuentes = prompts.MENSAJE_SIN_INFORMACION, []
        elif tipo == "fuera_de_dominio":
            texto, fuentes = prompts.MENSAJE_FUERA_DE_DOMINIO, []
        else:
            fuentes = fuentes_unicas(relevantes)

        resultado = RespuestaChat(
            respuesta=texto,
            tipo=tipo,
            fuentes=fuentes,
            similitud_max=recuperados[0].similitud if recuperados else 0.0,
            ms=round((time.perf_counter() - inicio) * 1000),
        )
        logger.info(
            "Consulta respondida",
            extra={
                "tipo": resultado.tipo,
                "fuentes": len(resultado.fuentes),
                "relevantes": len(relevantes),
                "similitud_max": resultado.similitud_max,
                "ms": resultado.ms,
                "prompt": prompts.VERSION_PROMPT,
            },
        )
        return resultado
