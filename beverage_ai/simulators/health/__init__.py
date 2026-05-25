"""Health calculator — pure lookup, no model. §3.3.4."""

from .calculator import COOKING_FACTOR, compute_nutrition

__all__ = ["compute_nutrition", "COOKING_FACTOR"]
