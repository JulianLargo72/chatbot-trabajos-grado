"""Dobles de prueba sin red: modelo de embeddings determinista y LLM simulado."""

import hashlib
import math
import re

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM
from pydantic import PrivateAttr

_PALABRA = re.compile(r"\w+", re.UNICODE)


class EmbeddingFalso(BaseEmbedding):
    """Bolsa de palabras con hashing: textos con palabras en común quedan cerca.

    Cuenta las llamadas para verificar que una ingesta sin cambios no usa la API.
    """

    dim: int = 64
    _llamadas: int = PrivateAttr(default=0)

    @property
    def llamadas(self) -> int:
        return self._llamadas

    def _vector(self, texto: str) -> list[float]:
        self._llamadas += 1
        v = [0.0] * self.dim
        for palabra in _PALABRA.findall(texto.lower()):
            indice = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % self.dim
            v[indice] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


class LLMFalso(CustomLLM):
    """LLM simulado: devuelve una respuesta fija (o lanza un error) y guarda los mensajes."""

    respuesta: str = "Respuesta simulada (Formato de proyecto de grado, p. 1)."
    error: str | None = None
    _mensajes: list = PrivateAttr(default_factory=list)

    @property
    def mensajes(self) -> list:
        return self._mensajes

    @property
    def llamadas(self) -> int:
        return len(self._mensajes)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="llm-falso", is_chat_model=True)

    def chat(self, messages, **kwargs) -> ChatResponse:
        self._mensajes.append(list(messages))
        if self.error:
            raise TimeoutError(self.error)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.respuesta))

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(text=self.respuesta)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
        yield CompletionResponse(text=self.respuesta, delta=self.respuesta)
