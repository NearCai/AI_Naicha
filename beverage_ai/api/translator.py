"""Shape converters between Python `Recipe`/`ScoredCandidate` and the
frontend `DrinkRecipe` / GenerationConstraints schema.

Used by `beverage_ai.api.main` to:
  - turn `Recipe.ingredients` (vocab_id → grams) into the frontend's
    `ingredients_display` ([{name, amount}]) using vocab.name_zh,
  - turn the frontend's `GenerationConstraints` (Chinese strings like
    "18-22元", "60秒", "低糖") into the Python `health` targets dict
    consumed by `score_candidates`.

Both conversions are best-effort — unknown strings fall through as
None so the pipeline still runs with sensible defaults.
"""
from __future__ import annotations

import re
from typing import Any

from ..ingredients.aliases import AliasResolver
from ..ingredients.vocab import Vocab

# -----------------------------------------------------------------------------
# Recipe → display
# -----------------------------------------------------------------------------


def recipe_to_display(
    recipe: dict[str, Any] | Any,
    vocab: Vocab,
) -> list[dict[str, str]]:
    """Convert a Recipe (model or dump dict) to [{name, amount}] entries.

    Unknown vocab ids fall through with their raw id as the name so the
    caller can still see them — they will simply not survive the round-trip
    back through `display_to_vocab_ids`.
    """
    if hasattr(recipe, "model_dump"):
        ingredients = recipe.model_dump()["ingredients"]
    elif isinstance(recipe, dict):
        ingredients = recipe.get("ingredients", recipe)
    else:
        raise TypeError(f"recipe must be Recipe or dict, got {type(recipe)}")

    out: list[dict[str, str]] = []
    for ing_id, mass in ingredients.items():
        if ing_id in vocab:
            name = vocab.get(ing_id).name_zh
        else:
            name = ing_id
        out.append({"name": name, "amount": f"{float(mass):.0f}g"})
    return out


_MASS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|ml|克|毫升)?", re.IGNORECASE)


def _parse_amount(amount: str) -> float | None:
    m = _MASS_RE.search(amount or "")
    if not m:
        return None
    return float(m.group(1))


def display_to_vocab_ids(
    display: list[dict[str, Any]],
    vocab: Vocab,
    aliases: AliasResolver | None = None,
) -> dict[str, float]:
    """Reverse of recipe_to_display — best-effort alias resolution.

    Returns {vocab_id: float}. Items that cannot be resolved (no alias,
    name not in vocab, malformed amount) are silently dropped.
    """
    out: dict[str, float] = {}
    for item in display:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        mass = _parse_amount(str(item.get("amount", "")))
        if mass is None:
            continue
        canonical: str | None = None
        if aliases is not None:
            canonical = aliases.resolve(name)
        if canonical is None and name in vocab:
            canonical = name
        if canonical is None:
            # Try exact name_zh match
            for ing in vocab.all():
                if ing.name_zh == name:
                    canonical = ing.id
                    break
        if canonical is not None:
            out[canonical] = mass
    return out


# -----------------------------------------------------------------------------
# Frontend constraints → planner health targets
# -----------------------------------------------------------------------------

_SWEET_TO_SUGAR_G: dict[str, float] = {
    "无糖": 0.0,
    "低糖": 8.0,
    "微糖": 10.0,
    "三分": 8.0,
    "五分": 13.0,
    "半糖": 13.0,
    "七分": 18.0,
    "全糖": 25.0,
}

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _extract_first_number(value: str) -> float | None:
    m = _NUM_RE.search(value or "")
    if not m:
        return None
    return float(m.group(1))


def _extract_range(value: str) -> tuple[float, float] | None:
    """'18-22元' / '18~22' / '18 到 22' → (18.0, 22.0)."""
    if not value:
        return None
    nums = _NUM_RE.findall(value)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def frontend_constraints_to_targets(
    constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map the frontend `GenerationConstraints` shape to the planner spec's
    `health` block + a `context.price_range_cny` field if present.

    Frontend keys (all optional, Chinese strings):
      season / targetAudience / priceBand / maxIngredientCost / maxMakeTime
      / sweetness / temperature

    Output is consumed by `score_candidates(..., targets=...)` and by the
    pipeline's planner spec merge.
    """
    if not constraints:
        return {}

    targets: dict[str, Any] = {}
    audience = (constraints.get("targetAudience") or "")
    sweet = (constraints.get("sweetness") or "")
    price_band = (constraints.get("priceBand") or "")
    cost_cap = (constraints.get("maxIngredientCost") or "")
    make_time = (constraints.get("maxMakeTime") or "")

    if sweet in _SWEET_TO_SUGAR_G:
        targets["sugar_limit_g"] = _SWEET_TO_SUGAR_G[sweet]
    else:
        n = _extract_first_number(sweet)
        if n is not None:
            targets["sugar_limit_g"] = n

    if any(k in audience for k in ("健康", "轻负担", "低卡", "控糖", "无负担")):
        targets.setdefault("sugar_limit_g", 15.0)
        targets["calorie_limit_kcal"] = 200.0
        targets["trans_fat_zero"] = True

    cost_n = _extract_first_number(cost_cap)
    if cost_n is not None:
        targets["cost_cap_cny"] = cost_n

    time_n = _extract_first_number(make_time)
    if time_n is not None:
        targets["make_time_cap_sec"] = time_n

    pr = _extract_range(price_band)
    if pr is not None:
        targets["price_range_cny"] = [pr[0], pr[1]]

    return targets


def merge_targets_into_spec(
    spec: dict[str, Any],
    targets: dict[str, Any],
) -> dict[str, Any]:
    """Layer frontend-derived targets onto the planner-produced spec.

    Frontend wins for the keys it explicitly sets; planner defaults stay.
    """
    if not targets:
        return spec
    spec = dict(spec)
    health = dict(spec.get("health") or {})
    for k in ("sugar_limit_g", "calorie_limit_kcal", "trans_fat_zero"):
        if k in targets:
            health[k] = targets[k]
    spec["health"] = health
    if "price_range_cny" in targets:
        spec["price_range_cny"] = targets["price_range_cny"]
    extras = {k: targets[k] for k in ("cost_cap_cny", "make_time_cap_sec") if k in targets}
    if extras:
        ctx = dict(spec.get("context") or {})
        ctx.update(extras)
        spec["context"] = ctx
    return spec
