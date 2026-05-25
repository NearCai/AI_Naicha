"""Ingredient vocabulary loading and querying.

Corresponds to 技术方案书 §附录 D + §D.4 enrichment schema and
v1 实现方案 §6.1.

Every other module receives a `Vocab` and uses `vocab.get(id)` to look up
nutrition / category / typical_serving etc.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "tea_base",
    "dairy_base",
    "alt_milk_base",
    "coffee_base",
    "sweetener",
    "fruit",
    "topping",
    "flavoring",
    "auxiliary",
    "gel",
    "grain",
]

CostTier = Literal["low", "medium", "high", "premium"]
Supply = Literal["stable", "seasonal", "volatile"]


class IngredientNutrition(BaseModel):
    energy_kcal: float | None = None
    sugar_g: float | None = None
    fat_g: float | None = None
    trans_fat_g: float = 0.0
    caffeine_mg: float = 0.0
    sodium_mg: float | None = None


class Ingredient(BaseModel):
    id: str = Field(pattern=r"^[a-z]+_[a-z0-9_]+$")
    name_zh: str
    name_en: str
    category: Category
    subcategory: str | None = None
    default_form: str
    typical_serving_g: float = Field(gt=0)
    allergens: list[str] = Field(default_factory=list)
    cost_tier: CostTier
    supply: Supply
    shelf_life_days: int | None = None
    nutrition_per_100g: IngredientNutrition
    flavor_descriptors: list[str] = Field(default_factory=list)
    notes_zh: str | None = None
    source: str
    deprecated: bool = False

    @field_validator("typical_serving_g")
    @classmethod
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("typical_serving_g must be > 0")
        return v


class Vocab:
    """Loaded and validated ingredient vocabulary."""

    def __init__(self, items: list[Ingredient]):
        self._items: dict[str, Ingredient] = {it.id: it for it in items if not it.deprecated}
        if len(self._items) != len(items):
            n_dep = sum(1 for it in items if it.deprecated)
            if len(items) - n_dep != len(self._items):
                raise ValueError("Duplicate ingredient ids detected after loading")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Vocab:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        items = [Ingredient(**r) for r in raw]
        return cls(items)

    # ----- query API -----

    def get(self, id_: str) -> Ingredient:
        if id_ not in self._items:
            raise KeyError(f"Ingredient not in vocab: {id_!r}")
        return self._items[id_]

    def __contains__(self, id_: str) -> bool:
        return id_ in self._items

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[Ingredient]:
        return list(self._items.values())

    def by_category(self, cat: Category) -> list[Ingredient]:
        return [i for i in self._items.values() if i.category == cat]

    def ids(self) -> list[str]:
        return list(self._items.keys())

    def search_by_descriptor(self, descriptor: str) -> list[Ingredient]:
        d = descriptor.lower()
        return [i for i in self._items.values() if d in [x.lower() for x in i.flavor_descriptors]]


def _default_data_dir() -> Path:
    """Locate data/ directory relative to package or via env var."""
    env = os.environ.get("BEVERAGE_AI_DATA_DIR")
    if env:
        return Path(env)
    # Walk up from this file to find a sibling `data/` directory
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate data/ directory. Set BEVERAGE_AI_DATA_DIR.")


def load_default_vocab() -> Vocab:
    """Convenience loader for the bundled demo vocabulary."""
    return Vocab.from_yaml(_default_data_dir() / "ingredients" / "ingredient_vocab.yaml")
