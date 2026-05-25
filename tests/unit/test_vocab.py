"""Tests for ingredients/vocab.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from beverage_ai.ingredients.aliases import load_default_aliases
from beverage_ai.ingredients.vocab import Ingredient, IngredientNutrition, Vocab


def test_default_vocab_loads(vocab):
    assert len(vocab) > 0
    # Schema-validated
    for ing in vocab.all():
        assert isinstance(ing, Ingredient)


def test_default_vocab_covers_all_categories(vocab):
    expected_categories = {
        "tea_base", "dairy_base", "alt_milk_base", "coffee_base",
        "sweetener", "fruit", "topping", "flavoring",
        "auxiliary", "gel", "grain",
    }
    present = {i.category for i in vocab.all()}
    missing = expected_categories - present
    assert not missing, f"Missing categories in default vocab: {missing}"


def test_vocab_get_known_id(vocab):
    ing = vocab.get("tea_jinxuan")
    assert ing.name_zh == "金萱乌龙"
    assert "milky" in ing.flavor_descriptors


def test_vocab_get_unknown_raises(vocab):
    with pytest.raises(KeyError, match="tea_nonexistent"):
        vocab.get("tea_nonexistent")


def test_vocab_membership(vocab):
    assert "tea_jasmine_green" in vocab
    assert "xxx_nonsense" not in vocab


def test_vocab_by_category(vocab):
    teas = vocab.by_category("tea_base")
    assert all(i.category == "tea_base" for i in teas)
    assert len(teas) >= 5


def test_ingredient_id_pattern():
    with pytest.raises(ValidationError):
        Ingredient(
            id="BadID",  # uppercase not allowed
            name_zh="x",
            name_en="x",
            category="tea_base",
            default_form="brewed_tea_ml",
            typical_serving_g=10,
            cost_tier="low",
            supply="stable",
            nutrition_per_100g=IngredientNutrition(),
            source="test",
        )


def test_typical_serving_must_be_positive():
    with pytest.raises(ValidationError):
        Ingredient(
            id="tea_x",
            name_zh="x",
            name_en="x",
            category="tea_base",
            default_form="brewed_tea_ml",
            typical_serving_g=0,
            cost_tier="low",
            supply="stable",
            nutrition_per_100g=IngredientNutrition(),
            source="test",
        )


def test_aliases_resolution(vocab):
    resolver = load_default_aliases(vocab)
    assert resolver.resolve("OATLY") == "alt_milk_oat_barista"
    assert resolver.resolve("鲜奶") == "dairy_whole_milk"
    assert resolver.resolve("不存在的原料") is None


def test_aliases_invalid_target_returns_none(vocab):
    from beverage_ai.ingredients.aliases import AliasResolver
    resolver = AliasResolver({"测试": "nonexistent_id"}, vocab=vocab)
    assert resolver.resolve("测试") is None
