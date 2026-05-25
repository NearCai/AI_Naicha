"""Compose recipe predictions into 4 objectives + constraint check.

Corresponds to 技术方案书 §3.5.

Used by `pipeline/end_to_end.py` to evaluate a candidate population.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..constraints.checker import check_constraints, is_feasible
from ..ingredients.vocab import Vocab
from ..recipes.schema import Recipe
from ..simulators.health.calculator import compute_nutrition
from ..simulators.repurchase.v1_weighted import RepurchasePredictorV1
from ..simulators.sales.predict import SalesPredictor
from ..simulators.sensory.predict import SensoryPredictor
from .acquisition import lcb, ucb
from .nsga2 import ScoredCandidate


# Rough cost per gram by category (industry estimate, ¥/g)
_COST_PER_G = {
    "tea_base": 0.01,
    "dairy_base": 0.015,
    "alt_milk_base": 0.025,
    "coffee_base": 0.06,
    "sweetener": 0.005,
    "fruit": 0.04,
    "topping": 0.025,
    "flavoring": 0.08,
    "auxiliary": 0.0005,
    "gel": 0.02,
    "grain": 0.05,
}


def estimate_cost_cny(recipe: Recipe, vocab: Vocab) -> float:
    cost = 0.0
    for ing_id, mass in recipe.ingredients.items():
        if ing_id not in vocab:
            continue
        cat = vocab.get(ing_id).category
        cost += mass * _COST_PER_G.get(cat, 0.01)
    return round(cost, 2)


def score_candidates(
    recipes: list[Recipe],
    vocab: Vocab,
    sensory: SensoryPredictor,
    sales: SalesPredictor,
    repurchase: RepurchasePredictorV1,
    targets: dict[str, Any] | None,
    kappa: float = 1.0,
) -> list[ScoredCandidate]:
    """Score every candidate and return ScoredCandidate list.

    Four to-minimize objectives:
        f1 = -(preference_LCB)                  (max preference)
        f2 = -(sales_LCB)                       (max sales)
        f3 = cost_cny                           (min cost)
        f4 = sugar_g                            (min sugar)
    """
    scored: list[ScoredCandidate] = []

    for recipe in recipes:
        sensory_pred = sensory.predict(recipe)
        sales_pred = sales.predict(recipe)
        rep_pred = repurchase.predict(sensory_pred)
        nutrition = compute_nutrition(recipe, vocab)

        pref_mean = sensory_pred.means["喜爱度"]
        pref_sigma = sensory_pred.sigmas["喜爱度"]
        pref_lcb = lcb(pref_mean, pref_sigma, kappa)

        sales_lcb = lcb(sales_pred.mean, sales_pred.sigma, kappa)
        cost = estimate_cost_cny(recipe, vocab)
        sugar = nutrition["sugar_g"]

        objectives = np.array([-pref_lcb, -sales_lcb, cost, sugar])

        violations = check_constraints(recipe, nutrition, targets, vocab)
        feasible = is_feasible(violations)

        scored.append(
            ScoredCandidate(
                recipe=recipe,
                objectives=objectives,
                means={
                    "preference": pref_mean,
                    "sales_proxy": sales_pred.mean,
                    "cost_cny": cost,
                    "sugar_g": sugar,
                    "repurchase": rep_pred.score,
                    "energy_kcal": nutrition["energy_kcal"],
                    "caffeine_mg": nutrition["caffeine_mg"],
                },
                sigmas={
                    "preference": pref_sigma,
                    "sales_proxy": sales_pred.sigma,
                },
                embedding=np.asarray(sensory_pred.embedding),
                nutrition=nutrition,
                feasible=feasible,
            )
        )

    return scored


# Re-export for completeness
__all__ = ["score_candidates", "estimate_cost_cny", "ucb"]
