"""Completeness checks for the expanded vocab (219 entries, all 11 categories).

These tests guard against accidental regressions:
  - All 11 categories present
  - Minimum count per category matches 附录 D plan
  - Critical ingredients (used elsewhere in code) still exist
  - Schema integrity holds on every entry
"""
from __future__ import annotations

import pytest


# Minimum count required per category, per 附录 D §D.2
MIN_PER_CATEGORY = {
    "tea_base":      30,
    "dairy_base":    10,
    "alt_milk_base":  8,
    "coffee_base":    6,
    "sweetener":     15,
    "fruit":         35,
    "topping":       25,
    "flavoring":     20,
    "auxiliary":      8,
    "gel":            6,
    "grain":          6,
}


def test_total_count_meets_target(vocab):
    """Should have ≥ 200 entries — close to the 207 target in §附录 D."""
    assert len(vocab) >= 200, f"Expected ≥ 200, got {len(vocab)}"


@pytest.mark.parametrize("category, min_count", MIN_PER_CATEGORY.items())
def test_category_minimum_count(vocab, category, min_count):
    actual = len(vocab.by_category(category))
    assert actual >= min_count, (
        f"Category {category!r} has only {actual} entries, need ≥ {min_count}"
    )


# Critical IDs that other modules / fixtures reference
CRITICAL_IDS = [
    # tests / fixtures
    "tea_jinxuan", "tea_osmanthus_oolong", "tea_jasmine_green", "tea_assam",
    "dairy_thick_milk", "dairy_whole_milk", "alt_milk_creamer",
    "sweet_cane_sugar", "sweet_erythritol", "sweet_sucralose",
    "topping_taro_ball", "topping_brown_pearl",
    "flavor_dried_osmanthus",
    "aux_pure_water", "aux_ice_cube", "aux_soda_water",
    # priors/dirichlet.py role detection
    "fruit_lemon", "coffee_espresso",
]


@pytest.mark.parametrize("id_", CRITICAL_IDS)
def test_critical_ids_present(vocab, id_):
    assert id_ in vocab, f"Critical id {id_!r} missing from vocab"


def test_no_duplicate_ids(vocab):
    ids = [i.id for i in vocab.all()]
    assert len(ids) == len(set(ids)), "Duplicate ids detected"


def test_all_have_nonzero_serving(vocab):
    for ing in vocab.all():
        assert ing.typical_serving_g > 0, f"{ing.id}: zero serving"


def test_id_prefix_matches_category(vocab):
    """Every id's prefix should imply its category."""
    prefix_to_cat = {
        "tea_": "tea_base",
        "dairy_": "dairy_base",
        "alt_milk_": "alt_milk_base",
        "coffee_": "coffee_base",
        "sweet_": "sweetener",
        "fruit_": "fruit",
        "topping_": "topping",
        "flavor_": "flavoring",
        "aux_": "auxiliary",
        "gel_": "gel",
        "grain_": "grain",
    }
    for ing in vocab.all():
        for prefix, expected_cat in prefix_to_cat.items():
            if ing.id.startswith(prefix):
                assert ing.category == expected_cat, (
                    f"{ing.id} has category {ing.category!r}, "
                    f"prefix implies {expected_cat!r}"
                )
                break


def test_trans_fat_only_for_creamer(vocab):
    """Only植脂末 should have non-zero trans fat in this dataset."""
    has_trans = [i.id for i in vocab.all() if i.nutrition_per_100g.trans_fat_g > 0]
    assert has_trans == ["alt_milk_creamer"], f"Unexpected trans-fat entries: {has_trans}"


def test_milk_allergens_consistent(vocab):
    """Every dairy_base entry should declare milk allergen."""
    for ing in vocab.by_category("dairy_base"):
        assert "milk" in ing.allergens, f"{ing.id} missing 'milk' allergen"


def test_nut_allergens_on_almond_cashew(vocab):
    """Almond and cashew alt milks must declare nut allergen."""
    assert "nut" in vocab.get("alt_milk_almond").allergens
    assert "nut" in vocab.get("alt_milk_cashew").allergens


def test_caffeine_in_teas_and_coffee(vocab):
    """All caffeinated teas should declare > 0 mg caffeine; herbal teas should be 0."""
    caffeinated = {"green", "black", "oolong", "puer", "white", "scented", "green_powder"}
    for ing in vocab.by_category("tea_base"):
        sub = ing.subcategory or ""
        caf = ing.nutrition_per_100g.caffeine_mg
        if sub in caffeinated:
            assert caf > 0, f"{ing.id} ({sub}) should have caffeine > 0, got {caf}"
        else:
            # herbal / floral may be 0 or low
            assert caf >= 0


def test_aliases_yaml_resolves_to_valid_ids(vocab):
    """Every alias must resolve to a real vocab id."""
    from beverage_ai.ingredients.aliases import load_default_aliases
    resolver = load_default_aliases(vocab)
    # Walk through the underlying map and verify each target is valid
    for alias, canonical in resolver._map.items():
        assert canonical in vocab, (
            f"Alias {alias!r} points to {canonical!r} which is not in vocab"
        )


def test_compatibility_yaml_references_real_ids(vocab):
    """Every pair in topping_compatibility.yaml must reference real topping ids."""
    import yaml

    from beverage_ai.ingredients.vocab import _default_data_dir

    path = _default_data_dir() / "ingredients" / "topping_compatibility.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    for entry in data:
        a, b = entry["pair"]
        assert a in vocab, f"Compatibility pair references unknown id: {a}"
        assert b in vocab, f"Compatibility pair references unknown id: {b}"
        assert vocab.get(a).category == "topping", f"{a} is not a topping"
        assert vocab.get(b).category == "topping", f"{b} is not a topping"
