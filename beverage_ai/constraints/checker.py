"""Constraint checker — 技术方案书 §3.4 + §E.3.

Hard constraints (cause infeasibility):
  - Volume conservation (cup capacity)
  - Caffeine GB limit
  - Trans fat (if user requires zero)
  - User sugar limit
  - Soda + dairy chemical incompatibility
  - Excluded allergens

Soft constraints (warning only):
  - Topping count > 3
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..ingredients.vocab import Vocab
from ..recipes.schema import Recipe

Severity = Literal["hard", "soft"]


class ConstraintViolation(BaseModel):
    code: str
    severity: Severity
    message: str


# Default targets (overridden by per-request user targets)
DEFAULT_TARGETS = {
    "sugar_limit_g": 999,
    "calorie_limit_kcal": 999,
    "caffeine_limit_mg": 200,  # GB/T 21733 reference
    "trans_fat_zero": False,
    "excluded_allergens": [],
    "max_toppings": 3,
}


def check_constraints(
    recipe: Recipe,
    nutrition: dict,
    targets: dict | None,
    vocab: Vocab,
) -> list[ConstraintViolation]:
    """Run all constraints against a recipe and return violation list."""
    t = {**DEFAULT_TARGETS, **(targets or {})}
    out: list[ConstraintViolation] = []

    # 1. Volume conservation (1 ml ≈ 1 g approximation)
    total = recipe.total_mass_g()
    cup = recipe.cup_volume_ml
    if total < 0.85 * cup or total > 1.10 * cup:
        out.append(
            ConstraintViolation(
                code="VOLUME_OVERFLOW",
                severity="hard",
                message=(
                    f"总量 {total:.0f}g 不在 [{0.85 * cup:.0f}, {1.10 * cup:.0f}] "
                    f"(杯容 {cup}ml)"
                ),
            )
        )

    # 2. Caffeine GB limit
    if nutrition.get("caffeine_mg", 0) > t["caffeine_limit_mg"]:
        out.append(
            ConstraintViolation(
                code="CAFFEINE_GB",
                severity="hard",
                message=f"咖啡因 {nutrition['caffeine_mg']:.0f}mg 超 {t['caffeine_limit_mg']}mg",
            )
        )

    # 3. Trans fat
    if t.get("trans_fat_zero") and nutrition.get("trans_fat_g", 0) > 0:
        out.append(
            ConstraintViolation(
                code="TRANS_FAT",
                severity="hard",
                message=f"反式脂肪 {nutrition['trans_fat_g']:.2f}g, 违反零反式约束",
            )
        )

    # 4. User sugar limit
    if nutrition.get("sugar_g", 0) > t["sugar_limit_g"]:
        out.append(
            ConstraintViolation(
                code="SUGAR_LIMIT",
                severity="hard",
                message=f"含糖 {nutrition['sugar_g']:.1f}g 超 {t['sugar_limit_g']}g",
            )
        )

    # 5. Calorie limit
    if nutrition.get("energy_kcal", 0) > t["calorie_limit_kcal"]:
        out.append(
            ConstraintViolation(
                code="CALORIE_LIMIT",
                severity="hard",
                message=f"热量 {nutrition['energy_kcal']:.0f}kcal 超 {t['calorie_limit_kcal']}kcal",
            )
        )

    # 6. Soda + dairy chemical incompatibility
    has_soda = any(
        ing_id in recipe.ingredients
        for ing_id in ("aux_soda_water", "aux_sparkling_water")
    )
    has_dairy = recipe.has_category(vocab, "dairy_base")
    if has_soda and has_dairy:
        out.append(
            ConstraintViolation(
                code="SODA_DAIRY",
                severity="hard",
                message="苏打水与乳基同时存在,会凝固",
            )
        )

    # 7. Excluded allergens
    excluded = set(t["excluded_allergens"])
    actual = set(nutrition.get("allergens", []))
    overlap = excluded & actual
    if overlap:
        out.append(
            ConstraintViolation(
                code="ALLERGEN",
                severity="hard",
                message=f"含排除致敏原: {sorted(overlap)}",
            )
        )

    # 8. Topping count (soft)
    n_topping = sum(
        1 for k in recipe.ingredients if k in vocab and vocab.get(k).category == "topping"
    )
    if n_topping > t["max_toppings"]:
        out.append(
            ConstraintViolation(
                code="TOPPING_COUNT",
                severity="soft",
                message=f"配料 {n_topping} 种超过推荐上限 {t['max_toppings']}",
            )
        )

    return out


def is_feasible(violations: list[ConstraintViolation]) -> bool:
    """A recipe is feasible iff it has no hard violations."""
    return not any(v.severity == "hard" for v in violations)
