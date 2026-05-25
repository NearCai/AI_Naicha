"""Tests for priors/."""
from __future__ import annotations

import numpy as np

from beverage_ai.priors.dirichlet import (
    ROLE_ORDER,
    bayesian_update_alpha,
    partition_of_recipe,
)
from beverage_ai.recipes.schema import Recipe


def test_alpha_for_all_styles(prior_engine):
    for style in prior_engine.all_styles():
        a = prior_engine.get_dirichlet_alpha(style)
        assert len(a) == len(ROLE_ORDER)
        assert np.all(a > 0)


def test_alpha_with_context_shifts_summer(prior_engine):
    base = prior_engine.get_dirichlet_alpha("奶茶")
    shifted = prior_engine.get_dirichlet_alpha("奶茶", {"season": "summer"})
    # Summer: ice up, milk down
    ice_idx = ROLE_ORDER.index("ice")
    milk_idx = ROLE_ORDER.index("milk")
    assert shifted[ice_idx] > base[ice_idx]
    assert shifted[milk_idx] < base[milk_idx]


def test_alpha_with_unknown_context_no_change(prior_engine):
    base = prior_engine.get_dirichlet_alpha("奶茶")
    same = prior_engine.get_dirichlet_alpha("奶茶", {"random_feature": "noop"})
    np.testing.assert_array_equal(base, same)


def test_partition_of_recipe(example_recipe):
    p = partition_of_recipe(example_recipe)
    assert p.shape == (6,)
    assert abs(p.sum() - 1.0) < 1e-9
    # Tea is the largest role in example
    assert p[ROLE_ORDER.index("tea")] > p[ROLE_ORDER.index("milk")]
    # No coffee in example
    assert p[ROLE_ORDER.index("coffee")] == 0


def test_bayesian_update_does_not_diverge(prior_engine):
    """Apply 10 consecutive batches of synthetic feedback; alpha must stay bounded."""
    alpha_init = prior_engine.get_dirichlet_alpha("奶茶")
    alpha = alpha_init.copy()

    # Make a small batch of recipes with similar partitions
    recipes = [
        Recipe(
            recipe_id=f"sim_{i}",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={
                "tea_assam": 250.0,
                "dairy_whole_milk": 100.0,
                "sweet_cane_sugar": 13.0,
                "aux_ice_cube": 100.0,
            },
        )
        for i in range(8)
    ]
    scores = np.array([3, 4, 4, 5, 5, 3, 4, 5], dtype=float)

    for _ in range(10):
        alpha = bayesian_update_alpha(alpha, recipes, scores, learning_rate=0.3)
    # All entries finite and positive
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha > 0)
    # Should not have grown beyond a sane bound
    assert alpha.sum() < 200


def test_bayesian_update_below_min_returns_prior(prior_engine):
    alpha_init = prior_engine.get_dirichlet_alpha("奶茶")
    # Only one recipe → below min_good=3
    recipes = [
        Recipe(
            recipe_id="single",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={"tea_assam": 300.0, "dairy_whole_milk": 100.0},
        )
    ]
    scores = np.array([5.0])
    new_alpha = bayesian_update_alpha(alpha_init, recipes, scores)
    np.testing.assert_array_equal(new_alpha, alpha_init)


def test_update_writes_snapshot(prior_engine, tmp_path):
    """Posterior update should write a JSON snapshot."""
    recipes = [
        Recipe(
            recipe_id=f"snap_{i}",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={
                "tea_assam": 250.0,
                "dairy_whole_milk": 100.0,
                "aux_ice_cube": 100.0,
            },
        )
        for i in range(8)
    ]
    scores = np.array([3, 4, 4, 5, 5, 3, 4, 5], dtype=float)

    files_before = list(prior_engine.history_dir.glob("*.json"))
    prior_engine.update_dirichlet_posterior("奶茶", recipes, scores)
    files_after = list(prior_engine.history_dir.glob("*.json"))
    assert len(files_after) == len(files_before) + 1
