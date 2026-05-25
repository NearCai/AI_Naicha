"""Tests for recipes/generator.py."""
from __future__ import annotations

from beverage_ai.recipes.generator import RecipeGenerator
from beverage_ai.recipes.schema import Recipe


def test_generator_produces_recipes(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    out = gen.generate({"style_hint": "奶茶"}, n_candidates=20)
    assert len(out) >= 5  # dedup may reduce
    assert all(isinstance(r, Recipe) for r in out)


def test_generator_respects_style(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    pure_tea = gen.generate({"style_hint": "纯茶"}, n_candidates=15)
    # Pure tea recipes should overwhelmingly have no dairy
    dairy_count = sum(1 for r in pure_tea if r.has_category(vocab, "dairy_base"))
    assert dairy_count <= 1   # allow rare slip from Dirichlet tail


def test_generator_volume_after_reconcile(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=42)
    recipes = gen.generate({"style_hint": "奶茶"}, n_candidates=30)
    for r in recipes:
        total = r.total_mass_g()
        assert 0.85 * r.cup_volume_ml <= total <= 1.10 * r.cup_volume_ml + 1


def test_generator_dedup(vocab, prior_engine):
    """Same inputs → unique by ingredient set."""
    gen = RecipeGenerator(vocab, prior_engine, seed=1)
    out = gen.generate({"style_hint": "奶茶"}, n_candidates=50)
    keys = [frozenset(r.ingredients.keys()) for r in out]
    assert len(keys) == len(set(keys))


def test_generator_health_strict_prefers_zero_cal_sweetener(vocab, prior_engine):
    """When health_strict=True, sampling tends toward 赤藓糖醇/三氯蔗糖."""
    gen = RecipeGenerator(vocab, prior_engine, seed=7)
    spec = {
        "style_hint": "奶茶",
        "context": {"health_strict": True},
    }
    out = gen.generate(spec, n_candidates=30)
    # Count recipes using low-cal sweeteners
    low_cal_ids = {"sweet_erythritol", "sweet_sucralose"}
    n_low = sum(
        1 for r in out
        if any(i in r.ingredients for i in low_cal_ids)
    )
    # At least some bias should be visible
    assert n_low >= 1
