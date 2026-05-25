"""9-step hierarchical recipe generator + reconciliation.

Corresponds to 技术方案书 §E.2 and v1 实现方案 §6.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from ..ingredients.vocab import Vocab, _default_data_dir
from .reconciler import reconcile
from .schema import CupSize, Process, Recipe, Style, SugarLevel, sugar_level_to_grams

if TYPE_CHECKING:
    from ..priors.engine import PriorEngine

_ALL_STYLES: tuple[Style, ...] = ("纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调")

# Equivalent-sweetness scaling vs cane sugar (§E.5.2)
EQUIV_SWEETNESS = {
    "sweet_cane_sugar": 1.0,
    "sweet_brown_sugar": 1.0,
    "sweet_dark_brown": 1.0,
    "sweet_fructose_syrup": 1.2,
    "sweet_honey": 0.95,
    "sweet_erythritol": 0.7,
    "sweet_sucralose": 600.0,
}


def _load_compatibility(path: Path | None = None) -> list[dict]:
    p = path or (_default_data_dir() / "ingredients" / "topping_compatibility.yaml")
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _load_keyword_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load Chinese keyword → English flavor_descriptors mapping.

    Used to bridge LLM Planner output (Chinese keywords like '桂花', '厚乳')
    to vocab descriptors (English like 'osmanthus', 'creamy').
    """
    p = path or (_default_data_dir() / "ingredients" / "keyword_aliases.yaml")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k).strip(): [str(d).lower() for d in (v or [])] for k, v in raw.items()}


class RecipeGenerator:
    """Generate candidate recipes for a given (planner output, context)."""

    def __init__(
        self,
        vocab: Vocab,
        prior: PriorEngine,
        seed: int = 42,
        compatibility_path: Path | None = None,
        keyword_aliases_path: Path | None = None,
    ):
        self.vocab = vocab
        self.prior = prior
        self.rng = np.random.default_rng(seed)
        self._compat = _load_compatibility(compatibility_path)
        self._kw_aliases = _load_keyword_aliases(keyword_aliases_path)
        # Precompute candidate id lists per category for fast sampling
        self._by_cat: dict[str, list[str]] = {}
        for cat in (
            "tea_base", "dairy_base", "alt_milk_base", "coffee_base",
            "sweetener", "fruit", "topping", "flavoring",
        ):
            self._by_cat[cat] = [i.id for i in vocab.by_category(cat)]

    # ---------- keyword scoring ----------

    def _score_ingredient_for_keywords(self, ing_id: str, keywords: list[str]) -> int:
        """Score an ingredient against a list of Chinese flavor keywords.

        Hits accumulate:
          - keyword substring in ingredient name_zh:  +3
          - keyword substring in ingredient notes_zh: +2
          - mapped English descriptor in ingredient flavor_descriptors: +1

        Returns total score (0 if no hits). Empty keywords → 0.
        """
        if not keywords or ing_id not in self.vocab:
            return 0
        ing = self.vocab.get(ing_id)
        name = ing.name_zh or ""
        notes = ing.notes_zh or ""
        descriptors = [d.lower() for d in ing.flavor_descriptors]
        score = 0
        for kw in keywords:
            kw = kw.strip()
            if not kw:
                continue
            if kw in name:
                score += 3
            if kw in notes:
                score += 2
            for mapped in self._kw_aliases.get(kw, []):
                if mapped in descriptors:
                    score += 1
        return score

    def _pick_by_keywords(
        self, candidates: list[str], keywords: list[str], top_k: int = 3
    ) -> str | None:
        """Score `candidates`, return random pick from top-tied;
        None if no keyword hits (caller should fall back to random)."""
        if not candidates:
            return None
        if not keywords:
            return None
        scored = [(self._score_ingredient_for_keywords(c, keywords), c) for c in candidates]
        scored.sort(reverse=True)
        best = scored[0][0]
        if best <= 0:
            return None
        top = [c for s, c in scored if s == best][:top_k]
        return str(self.rng.choice(top))

    # ---------- main entry ----------

    def generate(
        self,
        planner_output: dict[str, Any],
        n_candidates: int = 200,
        recipe_id_prefix: str = "gen",
    ) -> list[Recipe]:
        recipes = []
        attempts = 0
        max_attempts = n_candidates * 3
        while len(recipes) < n_candidates and attempts < max_attempts:
            attempts += 1
            r = self._one_recipe(planner_output, idx=len(recipes), prefix=recipe_id_prefix)
            if r is not None:
                recipes.append(r)
        return self._dedupe(recipes)

    # ---------- 9 steps ----------

    def _one_recipe(
        self,
        planner: dict[str, Any],
        idx: int,
        prefix: str,
    ) -> Recipe | None:
        # Step 1: style
        style: Style = planner.get("style_hint") or _ALL_STYLES[
            int(self.rng.integers(0, len(_ALL_STYLES)))
        ]
        if style not in _ALL_STYLES:
            style = "奶茶"

        # Step 2: cup
        cup: CupSize = planner.get("cup_volume_ml", 500)
        if cup not in (380, 500, 700):
            cup = 500

        # Step 3: Dirichlet volume partition
        alpha = self.prior.get_dirichlet_alpha(style, planner.get("context", {}))
        partition = self.rng.dirichlet(alpha)
        tea_v, milk_v, fruit_v, water_v, coffee_v, ice_g = (partition * cup).tolist()

        ingredients: dict[str, float] = {}

        # Step 4: tea
        tea_id = self._pick_tea(style, planner)
        if tea_id and tea_v > 5:
            ingredients[tea_id] = round(tea_v, 2)

        # Step 5: milk
        if milk_v > 5:
            milk_id = self._pick_milk(style, planner)
            if milk_id:
                ingredients[milk_id] = round(milk_v, 2)

        # Coffee
        if coffee_v > 3 and self._by_cat["coffee_base"]:
            coffee_id = str(self.rng.choice(self._by_cat["coffee_base"]))
            ingredients[coffee_id] = round(coffee_v, 2)

        # Fruit
        if fruit_v > 5 and self._by_cat["fruit"]:
            fruit_id = self._pick_fruit(planner)
            if fruit_id:
                ingredients[fruit_id] = round(fruit_v, 2)

        # Water
        if water_v > 5:
            ingredients["aux_pure_water"] = round(water_v, 2)

        # Step 6: sweetener
        sugar_level: SugarLevel = planner.get("sugar_level", "五分")
        sugar_g = sugar_level_to_grams(sugar_level, cup)
        sweetener_id = planner.get("sweetener_id") or self._pick_sweetener(planner)
        if sweetener_id and sugar_g > 0:
            equiv = EQUIV_SWEETNESS.get(sweetener_id, 1.0)
            actual_g = sugar_g / equiv
            ingredients[sweetener_id] = round(max(actual_g, 0.01), 3)

        # Step 7: toppings (0-3, biased by style)
        topping_count = self._sample_topping_count(style)
        toppings = self._sample_compatible_toppings(topping_count)
        for t in toppings:
            base = self.vocab.get(t).typical_serving_g
            actual = max(base * float(self.rng.normal(1.0, 0.2)), 5.0)
            ingredients[t] = round(actual, 2)

        # Step 8: flavoring (0-2)
        flavor_count = int(self.rng.integers(0, 3))
        flavor_ids = self._sample_flavorings(flavor_count, planner)
        for f in flavor_ids:
            base = self.vocab.get(f).typical_serving_g
            ingredients[f] = round(max(base * float(self.rng.normal(1.0, 0.2)), 0.1), 3)

        # Step 9: ice
        if ice_g > 5:
            ingredients["aux_ice_cube"] = round(ice_g, 2)

        if not ingredients:
            return None

        recipe = Recipe(
            recipe_id=f"{prefix}_{idx:04d}_{int(self.rng.integers(1_000_000)):06d}",
            style=style,
            cup_volume_ml=cup,
            ingredients=ingredients,
            sugar_level=sugar_level,
            process=Process(),
            metadata={"planner": planner},
        )
        return reconcile(recipe)

    # ---------- helpers ----------

    def _pick_tea(self, style: Style, planner: dict) -> str | None:
        candidates = self._by_cat["tea_base"]
        if not candidates:
            return None
        keywords = planner.get("flavor_keywords", [])
        biased = self._pick_by_keywords(candidates, keywords)
        return biased if biased is not None else str(self.rng.choice(candidates))

    def _pick_milk(self, style: Style, planner: dict) -> str | None:
        """Pick a milk; respects excluded_ingredients and keyword bias.

        Strategy:
          1. If keywords strongly match any milk (across both pools), pick from those.
          2. Otherwise apply the 60% dairy / 40% alt_milk industry split, random within.
        """
        excluded = set(planner.get("excluded_ingredients", []))
        keywords = planner.get("flavor_keywords", [])

        all_milks = [
            x
            for x in (self._by_cat["dairy_base"] + self._by_cat["alt_milk_base"])
            if x not in excluded
        ]
        if not all_milks:
            return None

        # Step 1: try keyword-biased pick across both pools
        biased = self._pick_by_keywords(all_milks, keywords)
        if biased is not None:
            return biased

        # Step 2: 60/40 split
        dairy_pool = [x for x in self._by_cat["dairy_base"] if x not in excluded]
        alt_pool = [x for x in self._by_cat["alt_milk_base"] if x not in excluded]
        if self.rng.random() < 0.6 and dairy_pool:
            pool = dairy_pool
        elif alt_pool:
            pool = alt_pool
        else:
            pool = dairy_pool or all_milks
        return str(self.rng.choice(pool))

    def _pick_fruit(self, planner: dict) -> str | None:
        candidates = self._by_cat["fruit"]
        if not candidates:
            return None
        keywords = planner.get("flavor_keywords", [])
        biased = self._pick_by_keywords(candidates, keywords)
        return biased if biased is not None else str(self.rng.choice(candidates))

    def _pick_sweetener(self, planner: dict) -> str | None:
        candidates = self._by_cat["sweetener"]
        if not candidates:
            return None
        # If health_strict, prefer low-calorie sweeteners
        health_strict = (planner.get("context", {}) or {}).get("health_strict", False)
        if health_strict:
            preferred = [c for c in candidates
                         if c in ("sweet_erythritol", "sweet_sucralose")]
            if preferred:
                return str(self.rng.choice(preferred))
        return str(self.rng.choice(candidates))

    def _sample_topping_count(self, style: Style) -> int:
        # Cold drinks / 奶茶 / 特调 tend to have more toppings
        if style in ("纯茶", "冰沙"):
            weights = [0.55, 0.35, 0.08, 0.02]
        elif style == "果茶":
            weights = [0.30, 0.50, 0.18, 0.02]
        else:
            weights = [0.15, 0.50, 0.30, 0.05]
        return int(self.rng.choice([0, 1, 2, 3], p=weights))

    def _sample_compatible_toppings(self, count: int) -> list[str]:
        if count <= 0 or not self._by_cat["topping"]:
            return []
        pool = list(self._by_cat["topping"])
        chosen: list[str] = []
        first = str(self.rng.choice(pool))
        chosen.append(first)
        while len(chosen) < count and pool:
            # Score remaining by compatibility with current chosen
            scored = []
            for cand in pool:
                if cand in chosen:
                    continue
                score = self._compat_score(chosen[-1], cand)
                scored.append((score + float(self.rng.normal(0, 0.3)), cand))
            if not scored:
                break
            scored.sort(reverse=True)
            chosen.append(scored[0][1])
        return chosen

    def _compat_score(self, a: str, b: str) -> float:
        for entry in self._compat:
            pair = entry.get("pair", [])
            if {a, b} == set(pair):
                return float(entry.get("score", 0.0))
        return 0.0

    def _sample_flavorings(self, count: int, planner: dict) -> list[str]:
        """Pick `count` flavorings, biased by keywords if any.

        If keywords match strongly, draw from the top 2*count scoring pool.
        Otherwise draw uniformly from all flavorings.
        """
        candidates = self._by_cat["flavoring"]
        if count <= 0 or not candidates:
            return []
        keywords = planner.get("flavor_keywords", [])
        pool: list[str] = candidates
        if keywords:
            scored = [
                (self._score_ingredient_for_keywords(c, keywords), c) for c in candidates
            ]
            scored.sort(reverse=True)
            if scored[0][0] > 0:
                # Keep top 2*count (or 5, whichever bigger) for diversity within bias
                cutoff = max(count * 2, 5)
                pool = [c for s, c in scored[:cutoff]]
        size = min(count, len(pool))
        idxs = self.rng.choice(len(pool), size=size, replace=False)
        return [pool[i] for i in idxs]

    def _dedupe(self, recipes: list[Recipe]) -> list[Recipe]:
        """Drop near-duplicates by ingredient-set hash."""
        seen: set[frozenset[str]] = set()
        out: list[Recipe] = []
        for r in recipes:
            key = frozenset(r.ingredients.keys())
            # Hash by ingredient set; mass differences still preserved
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
