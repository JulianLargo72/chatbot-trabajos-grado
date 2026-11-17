"""Configuración centralizada de la aplicación.

Todos los parámetros ajustables se leen de variables de entorno o del archivo ``.env``.
Ningún valor sensible ni experimental queda fijo en el código.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    """Parámetros de la aplicación (ver ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Aplicación ---
    app_name: str = "Chatbot de trabajos de grado"
    log_level: LogLevel = "INFO"
    log_json: bool = True

    # --- Gemini ---
    google_api_key: SecretStr | None = None
    gemini_llm_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    # Gemini 3.x está optimizado para su temperatura por defecto (1.0); Google advierte que
    # bajarla puede causar bucles o peor desempeño. Vacío = valor por defecto del modelo.
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Profundidad de razonamiento interno de Gemini 3.x. "low" reduce latencia y consumo de
    # tokens; suficiente para responder a partir de fragmentos recuperados. Vacío = por defecto.
    llm_thinking_level: ThinkingLevel | None = "low"
    llm_max_output_tokens: int = Field(default=1024, gt=0)
    # Se declara para que LlamaIndex no consulte los metadatos del modelo por red al crearlo.
    llm_context_window: int = Field(default=131072, gt=0)
    llm_timeout_seconds: int = Field(default=30, gt=0)

    # --- Rutas ---
    corpus_dir: Path = Path("data/corpus")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "reglamentos"

    # --- Fragmentación y recuperación ---
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    similarity_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # --- API ---
    admin_api_key: SecretStr | None = None
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _separar_origenes(cls, valor: object) -> object:
        """Permite escribir los orígenes separados por comas en el ``.env``."""
        if isinstance(valor, str):
            return [origen.strip() for origen in valor.split(",") if origen.strip()]
        return valor

    @field_validator("llm_temperature", "llm_thinking_level", mode="before")
    @classmethod
    def _vacio_es_none(cls, valor: object) -> object:
        """``LLM_TEMPERATURE=`` (vacío) en el ``.env`` significa «usar el valor del modelo»."""
        if isinstance(valor, str):
            valor = valor.strip().lower()
            return None if valor in ("", "none", "null") else valor
        return valor

    @field_validator("log_level", mode="before")
    @classmethod
    def _nivel_en_mayusculas(cls, valor: object) -> object:
        return valor.upper() if isinstance(valor, str) else valor

    @model_validator(mode="after")
    def _validar_fragmentacion(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        return self

    def require_google_api_key(self) -> str:
        """Devuelve la clave de Gemini o explica cómo configurarla."""
        if self.google_api_key is None or not self.google_api_key.get_secret_value().strip():
            raise RuntimeError(
                "Falta GOOGLE_API_KEY. Cópiala en el archivo .env (ver .env.example)."
            )
        return self.google_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Devuelve una única instancia de ``Settings`` (se lee el ``.env`` una sola vez)."""
    return Settings()
