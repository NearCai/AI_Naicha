"""Smoke tests for the P1 endpoints: /api/v2/feedback and /api/v2/update.

`feedback` writes a recipe + N panel rows into a temp DuckDB file.
`update` shells out to scripts/update_from_feedback.py — we exercise the
job lifecycle with a fast `--skip-stage2` invocation against a small
synthetic session.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="function")
def isolated_app(tmp_path, monkeypatch):
    """Re-import the api module with env vars pointing at a tmpdir DB
    so concurrent tests don't share state."""
    monkeypatch.setenv("BEVERAGE_AI_FEEDBACK_DB", str(tmp_path / "feedback.duckdb"))
    monkeypatch.setenv("BEVERAGE_AI_AUDIT_DIR", str(tmp_path / "audit"))

    # Force reload so the module-level _FEEDBACK_DB picks up the new env.
    import importlib

    from beverage_ai.api import main as api_main

    importlib.reload(api_main)
    return api_main, tmp_path


@pytest.fixture
def client(isolated_app):
    api_main, _ = isolated_app
    return TestClient(api_main.app)


@pytest.fixture
def feedback_db_path(isolated_app):
    _, tmp_path = isolated_app
    return tmp_path / "feedback.duckdb"


def _post_feedback(client: TestClient, *, recipe_id: str, session_id: str, score: float):
    return client.post(
        "/api/v2/feedback",
        json={
            "session_id": session_id,
            "recipe_id": recipe_id,
            "recipe": {
                "name": "测试金萱厚乳茶",
                "description": "测试用",
                "style": "奶茶",
                "cup_volume_ml": 500,
                "sugar_level": "五分",
                "ingredients": [
                    {"name": "金萱乌龙", "amount": "250g"},
                    {"name": "厚乳", "amount": "110g"},
                    {"name": "白砂糖", "amount": "13g"},
                    {"name": "黑糖珍珠", "amount": "35g"},
                    {"name": "冰块", "amount": "92g"},
                ],
                "steps": ["试1", "试2", "试3", "试4"],
            },
            "feedbacks": [
                {"score": score, "panelist_id": "rater_a"},
                {"score": max(0.0, min(10.0, score - 1)), "panelist_id": "rater_b"},
            ],
            "constraints": {"sweetness": "五分", "season": "夏季"},
            "library": "recipe_skill_library",
        },
    )


# -----------------------------------------------------------------------------
# /api/v2/feedback
# -----------------------------------------------------------------------------


def test_feedback_writes_to_duckdb(client: TestClient, feedback_db_path: Path):
    r = _post_feedback(client, recipe_id="r_t1", session_id="s_test", score=8.5)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["session_id"] == "s_test"
    assert body["recipe_id"] == "r_t1"
    assert body["n_panel_rows"] == 2
    assert body["recipe_persisted"] is True
    assert body["resolved_ingredients"] >= 1

    # Verify rows in DuckDB
    con = duckdb.connect(str(feedback_db_path))
    n_panel = con.execute(
        "SELECT count(*) FROM panel_score WHERE session_id = 's_test'"
    ).fetchone()[0]
    n_feedback = con.execute(
        "SELECT count(*) FROM feedback WHERE session_id = 's_test' AND recipe_id = 'r_t1'"
    ).fetchone()[0]
    con.close()
    assert n_panel == 2
    assert n_feedback == 1


def test_feedback_auto_generates_ids(client: TestClient):
    r = client.post(
        "/api/v2/feedback",
        json={
            "recipe": {
                "name": "x",
                "ingredients": [{"name": "金萱乌龙", "amount": "200g"}],
            },
            "feedbacks": [{"score": 7.0}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"].startswith("user_")
    assert body["recipe_id"].startswith("user_recipe_")


def test_feedback_score_clamping(client: TestClient, feedback_db_path: Path):
    """0..10 input must land as 1..5 Likert in panel_score."""
    _post_feedback(client, recipe_id="r_low", session_id="s_test", score=0.0)
    _post_feedback(client, recipe_id="r_high", session_id="s_test", score=10.0)
    con = duckdb.connect(str(feedback_db_path))
    lo = con.execute(
        "SELECT min(score), max(score) FROM panel_score WHERE recipe_id = 'r_low'"
    ).fetchone()
    hi = con.execute(
        "SELECT min(score), max(score) FROM panel_score WHERE recipe_id = 'r_high'"
    ).fetchone()
    con.close()
    assert lo[1] == 1  # max likert from a 0.0 input → 1
    assert hi[0] == 5  # min likert from a 10.0/9.0 input → 5


def test_feedback_rejects_empty_feedbacks(client: TestClient):
    r = client.post(
        "/api/v2/feedback",
        json={
            "recipe": {"name": "x", "ingredients": [{"name": "金萱乌龙", "amount": "100g"}]},
            "feedbacks": [],
        },
    )
    assert r.status_code == 422


def test_feedback_handles_unresolved_ingredients(client: TestClient):
    r = client.post(
        "/api/v2/feedback",
        json={
            "recipe": {
                "name": "全是新原料",
                "ingredients": [
                    {"name": "完全没听过的原料1", "amount": "10g"},
                    {"name": "完全没听过的原料2", "amount": "10g"},
                    {"name": "金萱乌龙", "amount": "100g"},  # at least one resolves
                ],
            },
            "feedbacks": [{"score": 7.0}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["unresolved_ingredients"]) == 2
    assert body["resolved_ingredients"] == 1


# -----------------------------------------------------------------------------
# /api/v2/update
# -----------------------------------------------------------------------------


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 30.0):
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/api/v2/update/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.5)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def _seed_synth_session(db_path: Path, vocab, session_id: str = "s_update_test"):
    """Write enough panel + feedback rows that update_from_feedback succeeds."""
    from beverage_ai.feedback.recorder import FeedbackRecorder
    from beverage_ai.recipes.schema import Recipe

    tea_id = next((i.id for i in vocab.all() if i.category == "tea_base"), None)
    sweet_id = next((i.id for i in vocab.all() if i.category == "sweetener"), None)
    aux_id = next((i.id for i in vocab.all() if i.category == "auxiliary"), None)
    assert tea_id and sweet_id and aux_id, "vocab missing baseline ingredients"

    rec = FeedbackRecorder(db_path=str(db_path))
    try:
        for i in range(5):
            recipe = Recipe(
                recipe_id=f"synth_{i:03d}",
                style="奶茶",
                cup_volume_ml=500,
                sugar_level="五分",
                ingredients={
                    tea_id: 200.0 + i * 5,
                    sweet_id: 10.0 + i,
                    aux_id: 80.0,
                },
            )
            rec.record_recipe(
                session_id=session_id,
                recipe=recipe,
                predicted={"preference": 3.5 + 0.1 * i},
            )
            # 4 raters per recipe, scores 3-5 (varying)
            for p, score in enumerate([3, 4, 4, 5]):
                rec.record_panel(
                    session_id=session_id,
                    recipe_id=recipe.recipe_id,
                    panelist_id=f"p{p}",
                    dimension="喜爱度",
                    score=score,
                )
    finally:
        rec.close()
    return session_id


def test_update_lifecycle_skip_stage2(client: TestClient, feedback_db_path: Path, vocab):
    session_id = _seed_synth_session(feedback_db_path, vocab)

    r = client.post(
        "/api/v2/update",
        json={
            "session_id": session_id,
            "feedback_db": str(feedback_db_path),
            "skip_stage2": True,
            "stage2_epochs": 1,
            "note": "smoke",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    job_id = body["job_id"]
    assert body["status"] in ("queued", "running")
    assert body["audit_path"] and body["audit_path"].endswith(
        f"update_session_{session_id}.json"
    )

    final = _wait_for_job(client, job_id, timeout=60.0)
    assert final["status"] == "completed", final
    assert final["returncode"] == 0
    # audit JSON must be parsed and surfaced
    audit = final["audit"]
    assert audit is not None
    assert audit["session"] == session_id
    assert audit["n_recipes"] == 5
    assert "dirichlet_changes" in audit
    assert "typical_serving_changes" in audit
    # stage2 was skipped
    assert audit["stage2"].get("skipped") is True


def test_update_list_endpoint(client: TestClient, feedback_db_path: Path, vocab):
    session_id = _seed_synth_session(feedback_db_path, vocab, "s_list_test")
    client.post(
        "/api/v2/update",
        json={
            "session_id": session_id,
            "feedback_db": str(feedback_db_path),
            "skip_stage2": True,
        },
    )
    r = client.get("/api/v2/update")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert any(j.get("note") for j in body) or len(body) >= 1


def test_update_404_unknown_job(client: TestClient):
    r = client.get("/api/v2/update/job_does_not_exist")
    assert r.status_code == 404
