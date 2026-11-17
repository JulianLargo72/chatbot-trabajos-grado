"""Pruebas de GET /health (F1)."""


def test_health_responde_ok(client) -> None:
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["estado"] == "ok"
    assert datos["modelo_llm"] == "gemini-3.5-flash-lite"
    assert datos["clave_gemini_configurada"] is True
    assert datos["indice_existe"] is False


def test_health_no_expone_la_clave(client) -> None:
    assert "clave-de-prueba" not in client.get("/health").text


def test_documentacion_swagger_disponible(client) -> None:
    assert client.get("/docs").status_code == 200
    assert "/health" in client.get("/openapi.json").json()["paths"]
