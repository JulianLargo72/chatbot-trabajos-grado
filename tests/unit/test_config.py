"""Pruebas de la configuración (F1)."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_valores_por_defecto() -> None:
    s = Settings(_env_file=None)
    assert s.gemini_llm_model == "gemini-3.5-flash-lite"
    assert s.gemini_embedding_model == "gemini-embedding-001"
    assert s.top_k == 3
    assert 0 <= s.similarity_threshold <= 1


def test_lee_variables_de_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://midominio.edu.co")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    s = Settings(_env_file=None)
    assert s.top_k == 5
    assert s.chunk_size == 1024
    assert s.cors_origins == ["http://localhost:3000", "https://midominio.edu.co"]
    assert s.log_level == "DEBUG"


def test_lee_archivo_env(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_LLM_MODEL=otro-modelo\nSIMILARITY_THRESHOLD=0.7\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.gemini_llm_model == "otro-modelo"
    assert s.similarity_threshold == 0.7


def test_overlap_mayor_que_chunk_es_invalido() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("similarity_threshold", 1.5), ("top_k", 0), ("llm_temperature", -1)],
)
def test_rangos_invalidos(campo: str, valor: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{campo: valor})


def test_la_clave_no_se_imprime() -> None:
    s = Settings(_env_file=None, google_api_key="secreto-123")
    assert "secreto-123" not in repr(s)
    assert s.require_google_api_key() == "secreto-123"


def test_clave_faltante_da_mensaje_claro() -> None:
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        Settings(_env_file=None).require_google_api_key()
