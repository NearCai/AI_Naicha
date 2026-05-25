"""Tests for recipes/reconciler.py."""
from __future__ import annotations

from beverage_ai.recipes.reconciler import reconcile
from beverage_ai.recipes.schema import Recipe


def test_reconcile_passes_through_when_in_range(example_recipe):
    r = reconcile(example_recipe)
    cup = r.cup_volume_ml
    total = r.total_mass_g()
    assert 0.85 * cup <= total <= 1.10 * cup


def test_reconcile_scales_down_overflow():
    # Build a recipe that overflows 500ml cup
    overflow = Recipe(
        recipe_id="overflow",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_assam": 600.0,
            "dairy_whole_milk": 400.0,
            "sweet_cane_sugar": 13.0,    # solid, not scaled
        },
    )
    r = reconcile(overflow)
    total = r.total_mass_g()
    assert total <= 1.10 * 500 + 1, f"total={total}"
    # Sugar (solid) should be unchanged
    assert r.ingredients["sweet_cane_sugar"] == 13.0


def test_reconcile_tops_up_underflow():
    underflow = Recipe(
        recipe_id="under",
        style="纯茶",
        cup_volume_ml=500,
        sugar_level="无糖",
        ingredients={"tea_assam": 100.0},
    )
    r = reconcile(underflow)
    total = r.total_mass_g()
    assert total >= 0.85 * 500


def test_reconcile_no_mutation(example_recipe):
    """Reconcile must not mutate input."""
    snapshot = example_recipe.model_dump()
    _ = reconcile(example_recipe)
    assert example_recipe.model_dump() == snapshot
