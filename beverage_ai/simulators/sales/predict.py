"""Sales predictor interface + mock.

Real LightGBM implementation in `.model`. For end-to-end skeleton runs
without trained models, use `MockSalesPredictor`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from ...recipes.schema import Recipe


@dataclass
class SalesPrediction:
    mean: float           # predicted sales-volume proxy (arbitrary unit)
    sigma: float          # uncertainty (std)
    baseline: float
    recipe_contribution: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "sigma": self.sigma,
            "baseline": self.baseline,
            "recipe_contribution": self.recipe_contribution,
        }


class SalesPredictor(Protocol):
    def predict(self, recipe: Recipe) -> SalesPrediction: ...


class MockSalesPredictor:
    """Deterministic mock that prefers familiar/cheap/trendy patterns.

    Recipe scoring (sum of small heuristics):
      + Standard cup size +
      + Sugar level five/seven +
      + Dairy or oat-milk presence +
      + 1-2 toppings +
      - Exotic ingredient combos
    """

    def __init__(self, vocab, seed: int = 0):
        self.vocab = vocab
        self._seed = seed

    def _stable_jitter(self, recipe_id: str, key: str) -> float:
        h = hashlib.md5(f"{recipe_id}|{key}|{self._seed}".encode()).hexdigest()
        return ((int(h[:8], 16) % 1000) / 1000 - 0.5) * 0.8

    def predict(self, recipe: Recipe) -> SalesPrediction:
        baseline = 50.0  # default sales-proxy baseline

        # Recipe contribution
        contrib = 0.0
        topping_count = 0
        unknown_count = 0
        has_dairy_or_oat = False
        sugar_g = 0.0

        for ing_id, mass in recipe.ingredients.items():
            if ing_id not in self.vocab:
                unknown_count += 1
                continue
            ing = self.vocab.get(ing_id)
            if ing.category == "dairy_base" or ing_id.startswith("alt_milk_oat"):
                has_dairy_or_oat = True
            if ing.category == "topping":
                topping_count += 1
            sugar_g += (ing.nutrition_per_100g.sugar_g or 0) * mass / 100

        if has_dairy_or_oat:
            contrib += 8
        if 1 <= topping_count <= 2:
            contrib += 5
        elif topping_count >= 3:
            contrib -= 3
        if recipe.cup_volume_ml == 500:
            contrib += 2
        if recipe.sugar_level in ("五分", "七分"):
            contrib += 3
        elif recipe.sugar_level == "全糖":
            contrib += 1
        contrib -= 2 * unknown_count
        contrib += self._stable_jitter(recipe.recipe_id, "contrib")

        # Sigma — grows with OOD signals
        sigma = 4.0 + 2.5 * unknown_count + 0.5 * max(0, topping_count - 2)
        sigma = min(sigma, 20.0)

        return SalesPrediction(
            mean=round(baseline + contrib, 2),
            sigma=round(sigma, 2),
            baseline=baseline,
            recipe_contribution=round(contrib, 2),
        )
