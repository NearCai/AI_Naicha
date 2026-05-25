"""Smoke tests for `beverage_ai.api.main` (FastAPI v2 endpoints).

Skipped automatically when fastapi is not installed; run with
    pip install -e .[api]
to enable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from beverage_ai.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/api/v2/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] > 50


def test_pipeline_minimal(client: TestClient):
    r = client.post(
        "/api/v2/pipeline",
        json={
            "prompt": "夏季年轻女性低糖, 定价 18-22 元",
            "n_candidates": 5,
            "pool_size": 30,
            "seed": 42,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    assert "spec" in body
    assert body["stats"]["n_generated"] > 0
    assert 1 <= len(body["candidates"]) <= 5
    first = body["candidates"][0]
    for key in (
        "candidate_id",
        "ingredients",
        "ingredients_display",
        "predicted_preference",
        "preference_sigma",
        "cost_cny",
        "nutrition",
        "feasibility",
    ):
        assert key in first
    # ingredients_display must use Chinese names from vocab
    assert all("name" in d and "amount" in d for d in first["ingredients_display"])


def test_pipeline_with_constraints(client: TestClient):
    r = client.post(
        "/api/v2/pipeline",
        json={
            "prompt": "夏季奶茶",
            "constraints": {
                "season": "夏季",
                "targetAudience": "健康轻负担",
                "priceBand": "18-22元",
                "maxIngredientCost": "8元",
                "maxMakeTime": "60秒",
                "sweetness": "低糖",
                "temperature": "冰",
            },
            "n_candidates": 5,
            "pool_size": 30,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # frontend constraints should be merged into the spec
    assert body["spec"]["health"]["sugar_limit_g"] <= 15.0
    assert "price_range_cny" in body["spec"]


def test_pipeline_validates_bounds(client: TestClient):
    r = client.post(
        "/api/v2/pipeline",
        json={"prompt": "x", "n_candidates": 999},
    )
    assert r.status_code == 422


def test_pipeline_rejects_empty_prompt(client: TestClient):
    r = client.post("/api/v2/pipeline", json={"prompt": ""})
    assert r.status_code == 422
