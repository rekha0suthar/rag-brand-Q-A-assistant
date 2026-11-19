"""API contract tests with a stubbed chain -- no models, index or keys."""

from fastapi.testclient import TestClient

import chain
from api.main import app
from utils.errors import QUOTA_MESSAGE, QuotaExceededError

ANSWER = {"answer": "15% off storewide.", "sources": ["campaign_summer_2025.md"], "latency": 0.01}


def client_with(monkeypatch, fake_load_chain):
    """TestClient whose lifespan initializes the app with a stubbed load_chain."""
    monkeypatch.setattr(chain, "load_chain", fake_load_chain)
    return TestClient(app)


def test_ask_returns_chain_output(monkeypatch):
    with client_with(monkeypatch, lambda: lambda q: ANSWER) as client:
        r = client.post("/ask", json={"question": "What discount ran?"})
    assert r.status_code == 200
    assert r.json() == ANSWER


def test_healthz_ok(monkeypatch):
    with client_with(monkeypatch, lambda: lambda q: ANSWER) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_quota_error_maps_to_429_with_friendly_message(monkeypatch):
    def raising_ask(q):
        raise QuotaExceededError()

    with client_with(monkeypatch, lambda: raising_ask) as client:
        r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert r.json()["detail"] == QUOTA_MESSAGE


def test_generic_error_does_not_leak_details(monkeypatch):
    def raising_ask(q):
        raise ValueError("secret internal state: gsk_notarealkey")

    with client_with(monkeypatch, lambda: raising_ask) as client:
        r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert "gsk_" not in r.text


def test_init_failure_degrades_instead_of_crashing(monkeypatch):
    def broken_load_chain():
        raise RuntimeError("index missing")

    with client_with(monkeypatch, broken_load_chain) as client:
        health = client.get("/healthz")
        ask = client.post("/ask", json={"question": "q"})
    assert health.json()["status"] == "degraded"
    assert ask.status_code == 503


def test_get_ask_hints_at_post(monkeypatch):
    with client_with(monkeypatch, lambda: lambda q: ANSWER) as client:
        r = client.get("/ask")
    assert r.status_code == 405
    assert "POST /ask" in r.json()["detail"]
