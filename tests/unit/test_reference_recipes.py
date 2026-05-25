"""Validate that data/recipes/reference_recipes_v1.yaml is internally consistent."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from beverage_ai.recipes.reconciler import reconcile
from beverage_ai.recipes.schema import Recipe

_PATH = Path(__file__).resolve().parents[2] / "data/recipes/reference_recipes_v1.yaml"


@pytest.fixture(scope="module")
def raw_entries():
    if not _PATH.exists():
        pytest.skip(f"{_PATH} not present")
    with open(_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def test_at_least_100_entries(raw_entries):
    assert len(raw_entries) >= 100, f"Expected >= 100, got {len(raw_entries)}"


def test_all_recipe_ids_unique(raw_entries):
    ids = [r["recipe_id"] for r in raw_entries]
    assert len(ids) == len(set(ids)), f"Duplicate ids: {[k for k, v in __import__('collections').Counter(ids).items() if v > 1]}"


def test_all_entries_parse(raw_entries):
    """Every YAML entry must produce a valid Recipe."""
    for item in raw_entries:
        Recipe(**item)


def test_all_ingredient_ids_in_vocab(vocab, raw_entries):
    bad = []
    for item in raw_entries:
        rid = item["recipe_id"]
        for ing_id in item["ingredients"]:
            if ing_id not in vocab:
                bad.append((rid, ing_id))
    assert not bad, f"Recipes reference unknown ingredient ids: {bad[:10]}"


def test_style_coverage(raw_entries):
    """Every one of the 6 styles should have at least 8 recipes."""
    from collections import Counter
    c = Counter(r["style"] for r in raw_entries)
    expected_styles = {"纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"}
    missing = expected_styles - set(c)
    assert not missing, f"Missing styles: {missing}"
    for style in expected_styles:
        assert c[style] >= 8, f"Style {style!r} only has {c[style]} (want >=8)"


def test_mass_balance_after_reconcile(raw_entries):
    """All recipes must respect cup-volume conservation after reconcile."""
    failed = []
    for item in raw_entries:
        recipe = Recipe(**item)
        r = reconcile(recipe)
        total = r.total_mass_g()
        cup = r.cup_volume_ml
        if total < 0.85 * cup or total > 1.10 * cup:
            failed.append((item["recipe_id"], total, cup))
    assert not failed, f"Recipes that fail mass balance post-reconcile: {failed[:5]}"


def test_provenance_labelled(raw_entries):
    """Every entry should have a metadata.source label from the known set."""
    known = {"well_known_classic", "brand_inspired_typical", "style_coverage_synthetic"}
    bad = []
    for item in raw_entries:
        src = (item.get("metadata") or {}).get("source")
        if src not in known:
            bad.append((item["recipe_id"], src))
    assert not bad, f"Recipes with unknown/missing provenance: {bad[:5]}"


def test_sugar_level_within_reasonable_range(raw_entries):
    """Added-sugar-equivalent grams should be in same ballpark as nominal."""
    from beverage_ai.recipes.generator import EQUIV_SWEETNESS
    from beverage_ai.recipes.schema import sugar_level_to_grams

    bad = []
    for item in raw_entries:
        recipe = Recipe(**item)
        added = sum(
            mass * EQUIV_SWEETNESS.get(ing_id, 1.0)
            for ing_id, mass in recipe.ingredients.items()
            if ing_id.startswith("sweet_")
        )
        expected = sugar_level_to_grams(recipe.sugar_level, recipe.cup_volume_ml)
        # Allow ±60% slack since some recipes use honey/syrup with implicit sugar elsewhere
        if abs(added - expected) > max(10, expected * 0.6):
            bad.append((recipe.recipe_id, recipe.sugar_level, added, expected))
    # Don't fail hard; just bound by a max-count of stragglers
    assert len(bad) < len(raw_entries) // 4, (
        f"Too many recipes with sugar far off nominal: {len(bad)}/{len(raw_entries)}"
    )
