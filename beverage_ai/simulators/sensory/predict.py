"""Sensory predictor interface + mock implementation.

The real `SensoryGAT` lives in `.model` and requires torch + torch_geometric.
For v1 skeleton / tests / end-to-end demo, the `MockSensoryPredictor`
returns plausible structured predictions derived from recipe content
(so downstream optimizer can run end-to-end without training).

Replace the mock with a trained GNN by implementing the SensoryPredictor
Protocol in `model.py`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from ...ingredients.vocab import Vocab
from ...recipes.schema import Recipe

# Per §3.3.1 double-headed output — core shared with path C, extended path A only
CORE_DIMS = ("甜度", "苦度", "茶香", "奶香", "喜爱度")
EXT_DIMS = ("涩", "酸", "回甘", "顺滑", "果香",
            "咸", "油腻", "清新", "浓郁", "层次")


@dataclass
class SensoryPrediction:
    """Per-dimension mean and sigma (heteroscedastic uncertainty)."""

    means: dict[str, float]
    sigmas: dict[str, float]
    embedding: list[float]  # 16-32d vector for MMR diversity

    def to_dict(self) -> dict:
        return {
            "means": dict(self.means),
            "sigmas": dict(self.sigmas),
            "embedding_dim": len(self.embedding),
        }


class SensoryPredictor(Protocol):
    def predict(self, recipe: Recipe) -> SensoryPrediction: ...


class MockSensoryPredictor:
    """Deterministic, structurally-plausible mock based on recipe content.

    Designed so:
      * sweeter recipes get higher 甜度
      * teas with more caffeine get higher 苦度
      * heavy dairy → higher 奶香
      * total mass + ingredient diversity → some 喜爱度
      * sigma scales with how out-of-distribution the recipe looks
        (large topping count, exotic flavorings)
    """

    def __init__(self, vocab: Vocab, seed: int = 0):
        self.vocab = vocab
        self._seed = seed

    def _stable_jitter(self, recipe: Recipe, key: str) -> float:
        """A small deterministic perturbation per recipe+dim, in [-0.3, 0.3]."""
        h = hashlib.md5(f"{recipe.recipe_id}|{key}|{self._seed}".encode()).hexdigest()
        return ((int(h[:8], 16) % 1000) / 1000 - 0.5) * 0.6

    def predict(self, recipe: Recipe) -> SensoryPrediction:
        sugar = 0.0
        caffeine = 0.0
        dairy = 0.0
        fruit = 0.0
        cream_topping = 0.0
        topping_count = 0
        exotic_flavor_count = 0

        for ing_id, mass in recipe.ingredients.items():
            if ing_id not in self.vocab:
                exotic_flavor_count += 1
                continue
            ing = self.vocab.get(ing_id)
            nut = ing.nutrition_per_100g
            if nut.sugar_g:
                sugar += nut.sugar_g * mass / 100
            if nut.caffeine_mg:
                caffeine += nut.caffeine_mg * mass / 100
            if ing.category == "dairy_base":
                dairy += mass
            if ing.category == "fruit":
                fruit += mass
            if ing.category == "topping":
                topping_count += 1
            if "creamy" in ing.flavor_descriptors or "rich" in ing.flavor_descriptors:
                cream_topping += mass

        # Map to 0-5 Likert-style scale
        def clip5(x: float) -> float:
            return max(1.0, min(5.0, x))

        means = {
            "甜度": clip5(2.0 + sugar / 8 + self._stable_jitter(recipe, "甜度")),
            "苦度": clip5(1.5 + caffeine / 80 + self._stable_jitter(recipe, "苦度")),
            "茶香": clip5(3.0 - caffeine / 200 + self._stable_jitter(recipe, "茶香")),
            "奶香": clip5(1.0 + dairy / 50 + self._stable_jitter(recipe, "奶香")),
            "喜爱度": clip5(
                3.0
                + 0.3 * (dairy > 30)
                + 0.2 * (fruit > 20)
                + 0.2 * (topping_count >= 1)
                - 0.4 * (sugar > 30)
                - 0.4 * (caffeine > 150)
                + self._stable_jitter(recipe, "喜爱度")
            ),
        }
        # Extended dims: scaled placeholders
        ext = {
            d: clip5(2.5 + self._stable_jitter(recipe, d))
            for d in EXT_DIMS
        }
        means.update(ext)

        # Sigma grows with OOD signals
        ood_score = 0.1 + 0.15 * exotic_flavor_count + 0.05 * max(0, topping_count - 2)
        ood_score = min(ood_score, 0.8)
        sigmas = {d: round(ood_score, 3) for d in (*CORE_DIMS, *EXT_DIMS)}

        # Deterministic 16d embedding
        embedding = self._embed(recipe)

        return SensoryPrediction(
            means={k: round(v, 3) for k, v in means.items()},
            sigmas=sigmas,
            embedding=embedding,
        )

    def _embed(self, recipe: Recipe) -> list[float]:
        """Hash-based 16d embedding: stable per ingredient set, used by MMR."""
        vec = [0.0] * 16
        for ing_id, mass in sorted(recipe.ingredients.items()):
            h = hashlib.md5(ing_id.encode()).digest()
            for i in range(16):
                vec[i] += (h[i] / 255) * mass
        total = sum(abs(v) for v in vec) or 1.0
        return [v / total for v in vec]
