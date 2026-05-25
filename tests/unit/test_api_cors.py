r"""CORS configuration tests for beverage_ai.api.main.

Verifies the three modes:
  - default (no env)        → localhost-only dev fallback
  - explicit origin list    → BEVERAGE_AI_CORS_ORIGINS=a,b,c
  - origin regex            → BEVERAGE_AI_CORS_ORIGIN_REGEX=^https://.*\.example\.com$
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _reload_app(monkeypatch, **env):
    for k in (
        "BEVERAGE_AI_CORS_ORIGINS",
        "BEVERAGE_AI_CORS_ORIGIN_REGEX",
        "BEVERAGE_AI_CORS_ALLOW_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from beverage_ai.api import main as api_main

    importlib.reload(api_main)
    return api_main.app


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v2/pipeline",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_cors_default_dev_localhost(monkeypatch):
    app = _reload_app(monkeypatch)
    client = TestClient(app)

    r = _preflight(client, "http://localhost:3000")
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    r2 = _preflight(client, "https://app.example.com")
    # Not in allow list → no allow-origin header (browser will block)
    assert "access-control-allow-origin" not in {k.lower() for k in r2.headers.keys()}


def test_cors_explicit_origin_list(monkeypatch):
    app = _reload_app(
        monkeypatch,
        BEVERAGE_AI_CORS_ORIGINS="https://app.example.com,https://admin.example.com",
    )
    client = TestClient(app)

    r = _preflight(client, "https://app.example.com")
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    r2 = _preflight(client, "https://admin.example.com")
    assert r2.headers.get("access-control-allow-origin") == "https://admin.example.com"

    # localhost no longer allowed once you set the env explicitly
    r3 = _preflight(client, "http://localhost:3000")
    assert "access-control-allow-origin" not in {k.lower() for k in r3.headers.keys()}


def test_cors_origin_regex(monkeypatch):
    app = _reload_app(
        monkeypatch,
        BEVERAGE_AI_CORS_ORIGIN_REGEX=r"^https://([a-z0-9-]+\.)*example\.com$",
    )
    client = TestClient(app)

    for ok in (
        "https://example.com",
        "https://app.example.com",
        "https://staging.app.example.com",
    ):
        r = _preflight(client, ok)
        assert r.headers.get("access-control-allow-origin") == ok, ok

    r = _preflight(client, "https://evil.example.org")
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers.keys()}


def test_cors_wildcard_forces_credentials_off(monkeypatch):
    app = _reload_app(
        monkeypatch,
        BEVERAGE_AI_CORS_ORIGINS="*",
        BEVERAGE_AI_CORS_ALLOW_CREDENTIALS="true",
    )
    client = TestClient(app)
    r = _preflight(client, "https://anywhere.example.org")
    # `*` is sent verbatim (no per-origin echo)
    assert r.headers.get("access-control-allow-origin") == "*"
    # credentials must be off when wildcard is used
    assert r.headers.get("access-control-allow-credentials") != "true"


def test_cors_allow_credentials_when_explicit(monkeypatch):
    app = _reload_app(
        monkeypatch,
        BEVERAGE_AI_CORS_ORIGINS="https://app.example.com",
        BEVERAGE_AI_CORS_ALLOW_CREDENTIALS="true",
    )
    client = TestClient(app)
    r = _preflight(client, "https://app.example.com")
    assert r.headers.get("access-control-allow-credentials") == "true"
