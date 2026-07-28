"""Endpoint contract tests: schemas, status codes, and error paths."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.serve.conftest import MAX_LEN


def test_health_is_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["uptime_s"] >= 0


def test_health_is_ok_even_without_a_model(unloaded_client: TestClient) -> None:
    # Liveness must not depend on the model, or a slow-loading pod gets killed
    # and restarted forever without ever finishing.
    assert unloaded_client.get("/health").status_code == 200


def test_ready_reflects_model_state(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    assert r.json()["model_version"]


def test_ready_is_503_without_a_model(unloaded_client: TestClient) -> None:
    r = unloaded_client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_generate_returns_requested_samples(client: TestClient) -> None:
    r = client.post("/generate", json={"n_samples": 3, "length": 16, "steps": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["texts"]) == 3
    assert body["nfe"] > 0
    assert body["latency_ms"] > 0
    assert body["model_version"]
    assert body["tokens"] is None  # not requested


def test_generate_can_return_tokens(client: TestClient) -> None:
    r = client.post(
        "/generate", json={"n_samples": 2, "length": 8, "steps": 3, "return_tokens": True}
    )
    tokens = r.json()["tokens"]
    assert len(tokens) == 2
    assert all(len(row) == 8 for row in tokens)


def test_generate_rejects_out_of_range_parameters(client: TestClient) -> None:
    assert client.post("/generate", json={"steps": 0}).status_code == 422
    assert client.post("/generate", json={"n_samples": 999}).status_code == 422
    assert client.post("/generate", json={"temperature": -1}).status_code == 422
    assert client.post("/generate", json={"sampler": "beam"}).status_code == 422


def test_generate_rejects_length_beyond_the_model(client: TestClient) -> None:
    # Within the schema's bound but longer than this checkpoint supports.
    r = client.post("/generate", json={"length": MAX_LEN + 1, "steps": 2})
    assert r.status_code == 422
    assert "maximum sequence length" in r.json()["detail"]


def test_generate_is_503_without_a_model(unloaded_client: TestClient) -> None:
    r = unloaded_client.post("/generate", json={"length": 8, "steps": 2})
    assert r.status_code == 503


def test_model_info_reports_geometry(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["n_heads"] == 4  # recorded in the checkpoint, not inferred
    assert body["max_seq_len"] == MAX_LEN
    assert body["step"] == 4242
    assert body["params"] > 0


def test_metrics_exposes_prometheus_series(client: TestClient) -> None:
    client.post("/generate", json={"n_samples": 1, "length": 8, "steps": 2})
    text = client.get("/metrics").text
    for series in (
        "aether_requests_total",
        "aether_request_latency_seconds",
        "aether_queue_depth",
        "aether_model_loaded",
        "aether_nfe_total",
        "aether_tokens_generated_total",
    ):
        assert series in text


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/generate" in schema["paths"]
    assert "/generate/stream" in schema["paths"]
    assert "/health" in schema["paths"]
