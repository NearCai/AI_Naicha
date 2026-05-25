"""Dirichlet helpers — partition extraction + Bayesian conjugate update.

Corresponds to 技术方案书 §E.9.3.
"""
from __future__ import annotations

import numpy as np

from ..recipes.schema import Recipe

# Order of roles in the alpha vector
ROLE_ORDER = ("tea", "milk", "fruit", "water", "coffee", "ice")


def _role_of(ing_id: str) -> str | None:
    if ing_id.startswith("tea_"):
        return "tea"
    if ing_id.startswith(("dairy_", "alt_milk_")):
        return "milk"
    if ing_id.startswith("fruit_"):
        return "fruit"
    if ing_id.startswith("coffee_"):
        return "coffee"
    if ing_id in ("aux_pure_water", "aux_soda_water", "aux_sparkling_water"):
        return "water"
    if ing_id.startswith("aux_ice") or ing_id == "aux_crushed_ice":
        return "ice"
    return None  # sweetener, topping, flavoring, gel, grain — not part of volume partition


def partition_of_recipe(recipe: Recipe) -> np.ndarray:
    """Extract the (tea, milk, fruit, water, coffee, ice) mass partition.

    Returns a length-6 numpy array summing to 1. If recipe has no
    role ingredients at all, returns a uniform vector.
    """
    role_mass = dict.fromkeys(ROLE_ORDER, 0.0)
    for ing_id, mass in recipe.ingredients.items():
        r = _role_of(ing_id)
        if r is not None:
            role_mass[r] += mass
    total = sum(role_mass.values())
    if total <= 0:
        return np.ones(len(ROLE_ORDER)) / len(ROLE_ORDER)
    return np.array([role_mass[r] / total for r in ROLE_ORDER])


def bayesian_update_alpha(
    alpha_prior: np.ndarray,
    recipes: list[Recipe],
    scores: np.ndarray,
    *,
    learning_rate: float = 0.3,
    top_quantile: float = 0.7,
    min_good: int = 3,
) -> np.ndarray:
    """Conjugate Bayesian update of a Dirichlet prior.

    Uses only the top `top_quantile` fraction of recipes (by score)
    to avoid being dragged down by bad samples.
    Returns the new alpha vector.

    Per 技术方案书 §E.9.3: learning_rate < 1 dampens single-batch
    over-correction.
    """
    if len(recipes) != len(scores):
        raise ValueError("len(recipes) must equal len(scores)")
    if len(recipes) < min_good:
        return np.asarray(alpha_prior, dtype=float).copy()

    scores = np.asarray(scores, dtype=float)
    threshold = float(np.quantile(scores, top_quantile))
    good_mask = scores >= threshold
    if good_mask.sum() < min_good:
        return np.asarray(alpha_prior, dtype=float).copy()

    good_recipes = [r for r, ok in zip(recipes, good_mask, strict=True) if ok]
    partitions = np.stack([partition_of_recipe(r) for r in good_recipes])

    alpha = np.asarray(alpha_prior, dtype=float)
    alpha_obs = len(good_recipes) * partitions.mean(axis=0)
    alpha_new = alpha + learning_rate * alpha_obs
    return np.clip(alpha_new, 0.05, None)
