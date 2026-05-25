"""Nutrition computation — pure lookup + weighted sum.

Corresponds to 技术方案书 §3.3.4 and v1 实现方案 §6.3.
"""
from __future__ import annotations

from collections import defaultdict

from ...ingredients.vocab import Vocab
from ...recipes.schema import Recipe

# Cooking factors: dry mass → cooked/served mass.
# Water-absorbing or hydrating ingredients have factor > 1.
COOKING_FACTOR: dict[str, float] = {
    "topping_brown_pearl": 2.5,
    "topping_taro_ball": 1.4,
    "topping_red_bean": 2.2,
    "topping_grass_jelly": 1.0,    # already in served form
    "topping_nata_de_coco": 1.0,
    "grain_chia": 8.0,
    "grain_basil_seed": 10.0,
    # Default 1.0 for items not listed
}


def compute_nutrition(recipe: Recipe, vocab: Vocab) -> dict:
    """Compute per-serving nutrition for a recipe.

    Returns a dict with energy_kcal, sugar_g, fat_g, trans_fat_g,
    caffeine_mg, sodium_mg, allergens, has_trans_fat,
    missing_nutrition_for.
    """
    totals: dict[str, float] = defaultdict(float)
    allergens: set[str] = set()
    has_trans_fat = False
    missing: list[str] = []

    for ing_id, mass_g in recipe.ingredients.items():
        if ing_id not in vocab:
            missing.append(ing_id)
            continue
        ing = vocab.get(ing_id)
        actual_g = mass_g * COOKING_FACTOR.get(ing_id, 1.0)
        nut = ing.nutrition_per_100g

        # Sum nutrients (skip None values gracefully)
        for nutrient, val in nut.model_dump().items():
            if val is not None:
                totals[nutrient] += val * actual_g / 100

        allergens.update(ing.allergens)
        if nut.trans_fat_g > 0 and actual_g > 0:
            has_trans_fat = True

    return {
        "energy_kcal": round(totals["energy_kcal"], 1),
        "sugar_g": round(totals["sugar_g"], 2),
        "fat_g": round(totals["fat_g"], 2),
        "trans_fat_g": round(totals["trans_fat_g"], 3),
        "caffeine_mg": round(totals["caffeine_mg"], 1),
        "sodium_mg": round(totals["sodium_mg"], 1),
        "allergens": sorted(allergens),
        "has_trans_fat": has_trans_fat,
        "missing_nutrition_for": missing,
    }
