"""Unit tests for beverage_ai.api.translator."""
from __future__ import annotations

import pytest

from beverage_ai.api.translator import (
    display_to_vocab_ids,
    frontend_constraints_to_targets,
    merge_targets_into_spec,
    recipe_to_display,
)
from beverage_ai.ingredients.aliases import load_default_aliases


def test_recipe_to_display_uses_name_zh(vocab, example_recipe):
    out = recipe_to_display(example_recipe, vocab)
    assert isinstance(out, list)
    assert len(out) == len(example_recipe.ingredients)
    for item in out:
        assert "name" in item
        assert "amount" in item
        assert item["amount"].endswith("g")
    # Should use Chinese names, not raw ids
    names = [item["name"] for item in out]
    assert all("_" not in n for n in names)


def test_recipe_to_display_accepts_dict(vocab, example_recipe):
    dumped = example_recipe.model_dump()
    out = recipe_to_display(dumped, vocab)
    assert len(out) == len(dumped["ingredients"])


def test_display_to_vocab_ids_roundtrip(vocab, example_recipe):
    aliases = load_default_aliases(vocab)
    display = recipe_to_display(example_recipe, vocab)
    back = display_to_vocab_ids(display, vocab, aliases)
    # Most/all ingredients should round-trip
    assert len(back) >= int(len(example_recipe.ingredients) * 0.7)
    for ing_id in back:
        assert ing_id in vocab


def test_display_to_vocab_ids_drops_unknown(vocab):
    aliases = load_default_aliases(vocab)
    display = [
        {"name": "完全不存在的奇怪原料XYZ", "amount": "10g"},
        {"name": "无量", "amount": "abcde"},
    ]
    back = display_to_vocab_ids(display, vocab, aliases)
    assert back == {}


def test_constraints_health_keywords():
    targets = frontend_constraints_to_targets(
        {"targetAudience": "健康轻负担", "sweetness": "低糖"}
    )
    assert targets["sugar_limit_g"] <= 15.0
    assert targets["calorie_limit_kcal"] == 200.0
    assert targets["trans_fat_zero"] is True


def test_constraints_price_band_parse():
    targets = frontend_constraints_to_targets({"priceBand": "18-22元"})
    assert targets["price_range_cny"] == [18.0, 22.0]


def test_constraints_cost_and_time():
    targets = frontend_constraints_to_targets(
        {"maxIngredientCost": "8元", "maxMakeTime": "60秒"}
    )
    assert targets["cost_cap_cny"] == 8.0
    assert targets["make_time_cap_sec"] == 60.0


def test_constraints_empty():
    assert frontend_constraints_to_targets(None) == {}
    assert frontend_constraints_to_targets({}) == {}


def test_merge_targets_into_spec_overlays_health():
    spec = {
        "style_hint": "奶茶",
        "health": {"sugar_limit_g": 30.0, "caffeine_limit_mg": 200},
    }
    merged = merge_targets_into_spec(spec, {"sugar_limit_g": 12.0})
    assert merged["health"]["sugar_limit_g"] == 12.0
    # untouched keys persist
    assert merged["health"]["caffeine_limit_mg"] == 200
    # original spec is not mutated
    assert spec["health"]["sugar_limit_g"] == 30.0


def test_merge_targets_price_range():
    merged = merge_targets_into_spec({}, {"price_range_cny": [18, 22]})
    assert merged["price_range_cny"] == [18, 22]


@pytest.mark.parametrize(
    "sweet,limit",
    [
        ("无糖", 0.0),
        ("低糖", 8.0),
        ("微糖", 10.0),
        ("七分", 18.0),
        ("全糖", 25.0),
    ],
)
def test_constraints_sweetness_table(sweet: str, limit: float):
    targets = frontend_constraints_to_targets({"sweetness": sweet})
    assert targets["sugar_limit_g"] == limit
