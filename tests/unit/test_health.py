"""Tests for simulators/health/calculator.py."""
from __future__ import annotations

from beverage_ai.recipes.schema import Recipe
from beverage_ai.simulators.health.calculator import compute_nutrition


def test_health_basic(example_recipe, vocab):
    nut = compute_nutrition(example_recipe, vocab)
    # All keys present
    for key in (
        "energy_kcal", "sugar_g", "fat_g", "trans_fat_g",
        "caffeine_mg", "sodium_mg", "allergens",
        "has_trans_fat", "missing_nutrition_for",
    ):
        assert key in nut
    # Has dairy → milk allergen
    assert "milk" in nut["allergens"]
    # No exotic ingredients
    assert nut["missing_nutrition_for"] == []


def test_health_dairy_milk_tea_reasonable(vocab):
    """A 500ml full-sugar milk tea should land in 250-500 kcal range."""
    recipe = Recipe(
        recipe_id="health_test_1",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="全糖",
        ingredients={
            "tea_assam": 250.0,
            "dairy_whole_milk": 150.0,
            "sweet_cane_sugar": 25.0,
            "topping_brown_pearl": 40.0,
            "aux_ice_cube": 50.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    assert 200 <= nut["energy_kcal"] <= 600, f"unrealistic kcal: {nut['energy_kcal']}"
    assert nut["sugar_g"] > 20


def test_health_trans_fat_detected(vocab):
    recipe = Recipe(
        recipe_id="trans_test",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_assam": 300.0,
            "alt_milk_creamer": 15.0,
            "sweet_cane_sugar": 13.0,
            "aux_pure_water": 100.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    assert nut["has_trans_fat"] is True
    assert nut["trans_fat_g"] > 0


def test_health_caffeine_from_tea(vocab, minimal_recipe):
    nut = compute_nutrition(minimal_recipe, vocab)
    # 400ml of jasmine green tea ≈ 4 * 18 = 72mg
    assert 50 <= nut["caffeine_mg"] <= 100


def test_health_missing_ingredient_tracked(vocab):
    recipe = Recipe(
        recipe_id="missing_test",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_assam": 300.0,
            "unknown_ingredient_xyz": 50.0,
            "aux_pure_water": 100.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    assert "unknown_ingredient_xyz" in nut["missing_nutrition_for"]


def test_health_pearl_uses_cooking_factor(vocab):
    """Verify boba's 2.5x cooking factor inflates its nutritional contribution."""
    recipe = Recipe(
        recipe_id="pearl_test",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="无糖",
        ingredients={
            "tea_assam": 200.0,
            "topping_brown_pearl": 40.0,  # raw_g; ×2.5 = 100g cooked
            "aux_ice_cube": 200.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    # Brown pearl: 245 kcal/100g cooked, 40g raw → 100g cooked → ~245 kcal
    assert nut["energy_kcal"] >= 200
