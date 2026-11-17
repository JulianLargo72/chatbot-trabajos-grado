"""Pruebas de la CLI sin red (F1)."""

from app import cli
from app.core.config import Settings


def test_check_sin_clave_falla_con_codigo_1(monkeypatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.main(["check"]) == 1


def test_comando_eval_pendiente(monkeypatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.main(["eval"]) == 2
