"""Fábricas del LLM y del modelo de embeddings de Gemini.

Ingesta y consulta usan estas mismas funciones, así se garantiza que el modelo de
embeddings sea idéntico en ambos lados (si difiere, la búsqueda falla en silencio).

Notas para la familia Gemini 3.x:
- La temperatura solo se envía si está configurada (Google recomienda dejar la de por defecto).
- El nivel de razonamiento (``thinking_level``) se fija explícitamente para controlar latencia.
- Se declaran ``max_tokens`` y ``context_window`` para que crear el objeto no haga
  una llamada de red (LlamaIndex consulta los metadatos del modelo si faltan).
- Los embeddings usan automáticamente ``RETRIEVAL_DOCUMENT`` al indexar y
  ``RETRIEVAL_QUERY`` al consultar (lo hace ``GoogleGenAIEmbedding``).
"""

from google.genai import types
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from app.core.config import Settings


def build_generation_config(settings: Settings) -> types.GenerateContentConfig:
    """Parámetros de generación que se envían a Gemini en cada llamada."""
    thinking = (
        types.ThinkingConfig(thinking_level=settings.llm_thinking_level.upper())
        if settings.llm_thinking_level
        else None
    )
    return types.GenerateContentConfig(
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        thinking_config=thinking,
    )


def build_llm(settings: Settings) -> GoogleGenAI:
    return GoogleGenAI(
        model=settings.gemini_llm_model,
        api_key=settings.require_google_api_key(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
        context_window=settings.llm_context_window,
        max_retries=settings.llm_max_retries,
        generation_config=build_generation_config(settings),
        http_options=types.HttpOptions(timeout=settings.llm_timeout_seconds * 1000),
    )


def build_embed_model(settings: Settings) -> GoogleGenAIEmbedding:
    return GoogleGenAIEmbedding(
        model_name=settings.gemini_embedding_model,
        api_key=settings.require_google_api_key(),
    )
