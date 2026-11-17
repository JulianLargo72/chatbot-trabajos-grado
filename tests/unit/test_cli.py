"""Pruebas de la CLI sin red."""

from app import cli
from app.core.config import Settings


def test_check_sin_clave_falla_con_codigo_1(monkeypatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.main(["check"]) == 1


def test_eval_sin_indice_falla_con_mensaje(monkeypatch, tmp_path) -> None:
    settings = Settings(_env_file=None, chroma_dir=tmp_path / "chroma", google_api_key="x")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["eval"]) == 1
