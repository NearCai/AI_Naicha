"""Tests for recipes/schema.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from beverage_ai.recipes.schema import Recipe, sugar_level_to_grams


def test_recipe_roundtrip(example_recipe):
    payload = example_recipe.model_dump()
    rebuilt = Recipe(**payload)
    assert rebuilt == example_recipe


def test_recipe_total_mass(example_recipe):
    expected = 250 + 125 + 13 + 35 + 0.5 + 25 + 85
    assert example_recipe.total_mass_g() == pytest.approx(expected)


def test_recipe_rejects_negative_mass():
    with pytest.raises(ValidationError):
        Recipe(
            recipe_id="x",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={"tea_jasmine_green": -10.0},
        )


def test_recipe_rejects_invalid_cup_size():
    with pytest.raises(ValidationError):
        Recipe(
            recipe_id="x",
            style="奶茶",
            cup_volume_ml=600,    # not in {380, 500, 700}
            sugar_level="五分",
            ingredients={"tea_jasmine_green": 100.0},
        )


def test_recipe_rejects_invalid_style():
    with pytest.raises(ValidationError):
        Recipe(
            recipe_id="x",
            style="不存在的风格",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={"tea_jasmine_green": 100.0},
        )


def test_recipe_must_have_ingredients():
    with pytest.raises(ValidationError):
        Recipe(
            recipe_id="x",
            style="奶茶",
            cup_volume_ml=500,
            sugar_level="五分",
            ingredients={},
        )


def test_sugar_level_to_grams_at_500ml():
    assert sugar_level_to_grams("无糖", 500) == 0
    assert sugar_level_to_grams("三分", 500) == 8
    assert sugar_level_to_grams("五分", 500) == 13
    assert sugar_level_to_grams("七分", 500) == 18
    assert sugar_level_to_grams("全糖", 500) == 25


def test_sugar_level_scales_with_cup():
    s500 = sugar_level_to_grams("七分", 500)
    s700 = sugar_level_to_grams("七分", 700)
    s380 = sugar_level_to_grams("七分", 380)
    assert s700 > s500 > s380
    assert s700 == pytest.approx(s500 * 700 / 500, rel=0.01)


def test_recipe_has_category_helper(example_recipe, vocab):
    assert example_recipe.has_category(vocab, "tea_base")
    assert example_recipe.has_category(vocab, "dairy_base")
    assert not example_recipe.has_category(vocab, "coffee_base")
