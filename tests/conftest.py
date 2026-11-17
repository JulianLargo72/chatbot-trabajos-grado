"""Fixtures compartidas. Las pruebas nunca leen el .env real ni usan la red."""

import pytest

from app.core.config import Settings, get_settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="clave-de-prueba",
        chroma_dir=tmp_path / "chroma",
        corpus_dir=tmp_path / "corpus",
    )


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import app

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as cliente:
        yield cliente
    app.dependency_overrides.clear()
