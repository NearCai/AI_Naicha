"""Ingredient vocabulary — corresponds to 技术方案书 §附录 D."""

from .aliases import AliasResolver, load_default_aliases
from .vocab import Category, Ingredient, IngredientNutrition, Vocab, load_default_vocab

__all__ = [
    "Category",
    "Ingredient",
    "IngredientNutrition",
    "Vocab",
    "load_default_vocab",
    "AliasResolver",
    "load_default_aliases",
]
