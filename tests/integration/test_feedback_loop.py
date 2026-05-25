"""Closed-loop integration: pipeline → record → posterior update.

Demonstrates that pipeline writes to DuckDB and posterior updates
shift the Dirichlet alpha for subsequent generations (per §3.7 + §E.9.3).
"""
from __future__ import annotations

import numpy as np

from beverage_ai.feedback.recorder import FeedbackRecorder
from beverage_ai.pipeline.end_to_end import run_pipeline


def test_record_and_recall(tmp_path, prior_engine):
    db = tmp_path / "feedback_test.duckdb"
    rec = FeedbackRecorder(db)
    result = run_pipeline(
        user_request="夏季奶茶",
        top_k=3,
        n_candidates=20,
        prior=prior_engine,
        recorder=rec,
        record=True,
    )

    sessions = rec.list_sessions()
    assert result.session_id in sessions

    recipes = rec.get_recipes(result.session_id)
    assert len(recipes) == len(result.top_recipes)
    rec.close()


def test_posterior_update_shifts_alpha(prior_engine):
    """After observing high-scored recipes, alpha for that style should shift."""
    from beverage_ai.recipes.schema import Recipe

    alpha_before = prior_engine.get_dirichlet_alpha("奶茶").copy()
    # Skewed recipes: high milk content, low tea
    recipes = [
        Recipe(
            recipe_id=f"skewed_{i}",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={
                "tea_assam": 100.0,
                "dairy_thick_milk": 250.0,
                "sweet_cane_sugar": 13.0,
                "aux_ice_cube": 100.0,
            },
        )
        for i in range(10)
    ]
    scores = np.array([5, 5, 4, 5, 4, 5, 4, 5, 5, 4], dtype=float)
    new_alpha = prior_engine.update_dirichlet_posterior("奶茶", recipes, scores)

    from beverage_ai.priors.dirichlet import ROLE_ORDER
    milk_idx = ROLE_ORDER.index("milk")
    tea_idx = ROLE_ORDER.index("tea")
    # Milk alpha should increase, tea alpha should not increase as much
    assert new_alpha[milk_idx] > alpha_before[milk_idx]
    assert (new_alpha[milk_idx] - alpha_before[milk_idx]) > (
        new_alpha[tea_idx] - alpha_before[tea_idx]
    )
