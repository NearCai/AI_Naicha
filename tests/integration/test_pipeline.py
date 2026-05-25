"""End-to-end pipeline integration test.

Uses MockLLMPlanner + MockSensoryPredictor + MockSalesPredictor so the
test runs without any external dependencies (no torch / lightgbm / API).
"""
from __future__ import annotations

from beverage_ai.pipeline.end_to_end import run_pipeline


def test_pipeline_runs_end_to_end():
    result = run_pipeline(
        user_request="夏季年轻女性低糖, 定价 18-22 元",
        top_k=3,
        n_candidates=40,
        kappa=1.0,
        seed=42,
    )
    assert result.n_generated > 0
    assert len(result.top_recipes) > 0
    assert len(result.top_recipes) <= 3
    for c in result.top_recipes:
        assert "recipe" in c
        assert "means" in c
        assert "nutrition" in c


def test_pipeline_top_k_diversity():
    """Top-K should not all be the same recipe."""
    result = run_pipeline(
        user_request="奶茶, 五分糖, 500ml",
        top_k=5,
        n_candidates=80,
        seed=42,
    )
    ids = {c["recipe"]["recipe_id"] for c in result.top_recipes}
    assert len(ids) == len(result.top_recipes)


def test_pipeline_respects_health_constraints():
    """Healthy request should produce recipes with bounded sugar."""
    result = run_pipeline(
        user_request="无糖控糖, 健康轻负担",
        top_k=3,
        n_candidates=60,
        seed=42,
    )
    # Recipes should have low sugar
    for c in result.top_recipes:
        assert c["nutrition"]["sugar_g"] < 20, c["nutrition"]


def test_pipeline_stats_populated():
    result = run_pipeline(
        user_request="夏季奶茶",
        top_k=2,
        n_candidates=30,
        seed=0,
    )
    d = result.to_dict()
    assert d["stats"]["n_generated"] > 0
    assert d["stats"]["elapsed_sec"] >= 0
    assert "n_pareto" in d["stats"]
