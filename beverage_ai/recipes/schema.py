"""Recipe data contract — pydantic v2 model.

Corresponds to 技术方案书 §附录 A and v1 实现方案 §6.2.
All modules pass Recipe instances between each other; ingredients dict
maps `vocab_id -> mass_g` (liquids approximated as 1 ml ≈ 1 g).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SugarLevel = Literal["无糖", "三分", "五分", "七分", "全糖"]
Style = Literal["纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"]
CupSize = Literal[380, 500, 700]


# Per 500ml cup, sugar level → grams (industry SOP, §E.5.2).
# Scaled linearly for other cup sizes.
_SUGAR_LEVEL_GRAMS_500 = {
    "无糖": 0.0,
    "三分": 8.0,
    "五分": 13.0,
    "七分": 18.0,
    "全糖": 25.0,
}


def sugar_level_to_grams(level: SugarLevel, cup_volume_ml: int) -> float:
    """Map a sugar level label to absolute sugar mass for a given cup size."""
    base = _SUGAR_LEVEL_GRAMS_500[level]
    return round(base * cup_volume_ml / 500, 2)


class Process(BaseModel):
    """Brewing / preparation parameters."""

    extraction_temp_c: float = 90.0
    extraction_time_s: int = 240
    shake_count: int = 12
    serving_temp_c: float = 4.0


class Recipe(BaseModel):
    """A single beverage recipe."""

    recipe_id: str = Field(min_length=1)
    style: Style
    cup_volume_ml: CupSize
    ingredients: dict[str, float] = Field(min_length=1)
    process: Process = Field(default_factory=Process)
    sugar_level: SugarLevel
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_positive_masses(self) -> Recipe:
        for k, v in self.ingredients.items():
            if v < 0:
                raise ValueError(f"Ingredient mass must be >= 0, got {v} for {k}")
        return self

    def total_mass_g(self) -> float:
        return sum(self.ingredients.values())

    def has_category(self, vocab, category: str) -> bool:
        """Helper: True if any ingredient belongs to the given category."""
        for ing_id in self.ingredients:
            if ing_id in vocab and vocab.get(ing_id).category == category:
                return True
        return False
