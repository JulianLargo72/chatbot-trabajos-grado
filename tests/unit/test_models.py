"""Pruebas de las fábricas de modelos (sin red)."""

import pytest

from app.core.config import Settings
from app.rag.models import build_embed_model, build_generation_config, build_llm


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, google_api_key="clave-de-prueba", **kwargs)


def test_por_defecto_no_envia_temperatura_y_usa_thinking_low() -> None:
    config = build_generation_config(_settings())
    assert config.temperature is None
    assert config.thinking_config.thinking_level == "LOW"
    assert config.max_output_tokens == 1024


def test_temperatura_y_thinking_configurables() -> None:
    config = build_generation_config(_settings(llm_temperature=0.3, llm_thinking_level="minimal"))
    assert config.temperature == 0.3
    assert config.thinking_config.thinking_level == "MINIMAL"


def test_thinking_vacio_usa_valor_del_modelo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_THINKING_LEVEL", "")
    monkeypatch.setenv("LLM_TEMPERATURE", "")
    s = Settings(_env_file=None, google_api_key="x")
    assert s.llm_thinking_level is None
    assert s.llm_temperature is None
    assert build_generation_config(s).thinking_config is None


def test_crear_llm_no_hace_llamadas_de_red() -> None:
    """Si LlamaIndex consultara los metadatos del modelo, esta prueba fallaría sin red."""
    llm = build_llm(_settings())
    assert llm.model == "gemini-3.5-flash-lite"
    assert llm.metadata.context_window == 131072
    assert llm.metadata.num_output == 1024


def test_crear_embeddings_usa_modelo_configurado() -> None:
    assert build_embed_model(_settings()).model_name == "gemini-embedding-001"
