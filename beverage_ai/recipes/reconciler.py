"""Conservation reconciler — §E.2 Step 10.

If total mass exceeds cup × 1.10, scale down liquid components.
If total mass below cup × 0.85, top up with pure water.
"""
from __future__ import annotations

from .schema import Recipe

# Ingredient ID prefixes considered "liquid" for scaling purposes
_LIQUID_PREFIXES = (
    "tea_",
    "dairy_",
    "alt_milk_",
    "coffee_",
    "fruit_",
    "aux_pure_",
    "aux_soda_",
    "aux_sparkling_",
)


def _is_liquid(ing_id: str) -> bool:
    return ing_id.startswith(_LIQUID_PREFIXES)


def reconcile(recipe: Recipe, *, upper_factor: float = 1.10, lower_factor: float = 0.85) -> Recipe:
    """Return a Recipe whose total mass is within [lower, upper] × cup_volume_ml.

    Does not mutate input; returns a new Recipe.
    """
    upper = recipe.cup_volume_ml * upper_factor
    lower = recipe.cup_volume_ml * lower_factor
    total = recipe.total_mass_g()

    new_ingredients = dict(recipe.ingredients)

    if total > upper:
        liquid_mass = sum(v for k, v in new_ingredients.items() if _is_liquid(k))
        solid_mass = total - liquid_mass
        target_liquid = max(upper - solid_mass, 0.0)
        scale = target_liquid / liquid_mass if liquid_mass > 0 else 1.0
        for k in list(new_ingredients):
            if _is_liquid(k):
                new_ingredients[k] = round(new_ingredients[k] * scale, 2)
    elif total < lower:
        new_ingredients["aux_pure_water"] = round(
            new_ingredients.get("aux_pure_water", 0.0) + (lower - total), 2
        )

    # Drop zero-mass keys
    new_ingredients = {k: v for k, v in new_ingredients.items() if v > 0.01}

    return recipe.model_copy(update={"ingredients": new_ingredients})
