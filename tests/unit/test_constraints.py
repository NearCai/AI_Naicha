"""Tests for constraints/checker.py."""
from __future__ import annotations

from beverage_ai.constraints.checker import check_constraints, is_feasible
from beverage_ai.recipes.schema import Recipe
from beverage_ai.simulators.health.calculator import compute_nutrition


def test_example_recipe_within_constraints(example_recipe, vocab):
    nut = compute_nutrition(example_recipe, vocab)
    # Note: 'sugar_limit_g' here is *total* sugar (added + naturally-occurring
    # from dairy lactose + starchy toppings). The example contains 13g added
    # cane sugar but ~37g total sugar.
    v = check_constraints(
        example_recipe, nut,
        targets={"sugar_limit_g": 50, "caffeine_limit_mg": 200},
        vocab=vocab,
    )
    # Should be feasible (no hard violations)
    assert is_feasible(v), [(x.code, x.message) for x in v]


def test_volume_overflow_detected(vocab):
    # Total 1500g in a 500ml cup → overflow
    recipe = Recipe(
        recipe_id="overflow",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={"tea_assam": 500.0, "dairy_whole_milk": 500.0, "aux_ice_cube": 500.0},
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(recipe, nut, None, vocab=vocab)
    codes = {x.code for x in v}
    assert "VOLUME_OVERFLOW" in codes


def test_volume_underflow_detected(vocab):
    # Total 100g in a 500ml cup → underflow (< 85%)
    recipe = Recipe(
        recipe_id="underflow",
        style="纯茶",
        cup_volume_ml=500,
        sugar_level="无糖",
        ingredients={"tea_assam": 100.0},
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(recipe, nut, None, vocab=vocab)
    codes = {x.code for x in v}
    assert "VOLUME_OVERFLOW" in codes


def test_sugar_limit_violation(vocab):
    recipe = Recipe(
        recipe_id="sweet",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="全糖",
        ingredients={"tea_assam": 300.0, "sweet_cane_sugar": 50.0, "aux_pure_water": 150.0},
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(recipe, nut, targets={"sugar_limit_g": 20}, vocab=vocab)
    assert any(x.code == "SUGAR_LIMIT" for x in v)


def test_trans_fat_violation_only_if_strict(vocab):
    recipe = Recipe(
        recipe_id="trans",
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

    # Strict zero-trans target → fail
    v_strict = check_constraints(recipe, nut, targets={"trans_fat_zero": True}, vocab=vocab)
    assert any(x.code == "TRANS_FAT" for x in v_strict)

    # No strict requirement → pass
    v_loose = check_constraints(recipe, nut, targets={"trans_fat_zero": False}, vocab=vocab)
    assert not any(x.code == "TRANS_FAT" for x in v_loose)


def test_soda_dairy_incompatibility(vocab):
    recipe = Recipe(
        recipe_id="curdle",
        style="特调",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "aux_soda_water": 250.0,
            "dairy_whole_milk": 150.0,
            "aux_ice_cube": 100.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(recipe, nut, None, vocab=vocab)
    assert any(x.code == "SODA_DAIRY" for x in v)
    assert not is_feasible(v)


def test_excluded_allergen(vocab):
    recipe = Recipe(
        recipe_id="allergen",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_assam": 250.0,
            "dairy_whole_milk": 150.0,
            "sweet_cane_sugar": 13.0,
            "aux_pure_water": 100.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(
        recipe, nut, targets={"excluded_allergens": ["milk"]}, vocab=vocab
    )
    assert any(x.code == "ALLERGEN" for x in v)


def test_topping_count_soft_warning(vocab):
    recipe = Recipe(
        recipe_id="manytop",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_assam": 200.0,
            "dairy_whole_milk": 80.0,
            "sweet_cane_sugar": 13.0,
            "topping_brown_pearl": 40.0,
            "topping_taro_ball": 35.0,
            "topping_grass_jelly": 50.0,
            "topping_nata_de_coco": 30.0,
            "aux_ice_cube": 50.0,
        },
    )
    nut = compute_nutrition(recipe, vocab)
    v = check_constraints(recipe, nut, None, vocab=vocab)
    soft = [x for x in v if x.code == "TOPPING_COUNT"]
    assert len(soft) == 1
    assert soft[0].severity == "soft"
    # Still feasible since it's soft only (assuming no hard violations)
    hard = [x for x in v if x.severity == "hard"]
    assert is_feasible(v) == (not hard)
