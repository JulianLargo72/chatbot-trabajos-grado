"""El System Prompt documentado debe coincidir con el del código (criterio 3 de HU09)."""

from pathlib import Path

from app.rag import prompts

DOC = Path(__file__).resolve().parents[2] / "docs" / "system_prompt.md"


def test_documento_coincide_con_el_codigo() -> None:
    texto = DOC.read_text(encoding="utf-8")
    assert f"v{prompts.VERSION_PROMPT}" in texto.splitlines()[0]
    assert prompts.SYSTEM_PROMPT in texto
    assert prompts.MENSAJE_SIN_INFORMACION in texto
    assert prompts.MENSAJE_FUERA_DE_DOMINIO in texto
