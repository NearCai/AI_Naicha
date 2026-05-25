"""Validate reference recipes against vocab + Recipe schema.

Checks:
  1. Every entry parses as a valid Recipe (pydantic)
  2. Every ingredient id is in the vocab
  3. Total mass is within [0.6 × cup, 1.4 × cup] (sanity check pre-reconcile)
  4. After reconciliation, total mass is within [0.85 × cup, 1.10 × cup]
  5. Sugar content from `sweet_*` ingredients roughly matches sugar_level
     (with sweetener-equivalence tolerance)
  6. Provenance is a known label

Usage:
    python scripts/validate_recipes.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.recipes.reconciler import reconcile
from beverage_ai.recipes.schema import Recipe, sugar_level_to_grams


KNOWN_SOURCES = {
    "well_known_classic",
    "brand_inspired_typical",
    "style_coverage_synthetic",
}


def main(path: str | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    yaml_path = Path(path) if path else repo / "data/recipes/reference_recipes_v1.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found", file=sys.stderr)
        return 1

    vocab = load_default_vocab()
    print(f"vocab: {len(vocab)} entries loaded")

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    print(f"recipes: {len(raw)} entries loaded")

    errors: list[str] = []
    warnings: list[str] = []

    by_style: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for i, item in enumerate(raw):
        rid = item.get("recipe_id", f"<row {i}>")

        # 1. Schema parse
        try:
            recipe = Recipe(**item)
        except Exception as e:
            errors.append(f"{rid}: schema parse failed — {e}")
            continue

        # 2. Ingredient ids
        unknown = [k for k in recipe.ingredients if k not in vocab]
        if unknown:
            errors.append(f"{rid}: unknown ingredient ids: {unknown}")
            continue

        # 3. Total mass sanity
        total = recipe.total_mass_g()
        if total < 0.6 * recipe.cup_volume_ml or total > 1.4 * recipe.cup_volume_ml:
            errors.append(
                f"{rid}: total {total:.0f}g far outside cup={recipe.cup_volume_ml}ml "
                f"(allowed 0.6× to 1.4×)"
            )
            continue

        # 4. Reconcile + check final mass
        reconciled = reconcile(recipe)
        final = reconciled.total_mass_g()
        upper = 1.10 * recipe.cup_volume_ml
        lower = 0.85 * recipe.cup_volume_ml
        if not (lower <= final <= upper):
            warnings.append(
                f"{rid}: post-reconcile {final:.0f}g outside [{lower:.0f}, {upper:.0f}] "
                f"(pre={total:.0f})"
            )

        # 5. Sugar content rough check
        added_sugar = 0.0
        from beverage_ai.recipes.generator import EQUIV_SWEETNESS
        for ing_id, mass in recipe.ingredients.items():
            if ing_id.startswith("sweet_"):
                # Cane-sugar-equivalent grams
                added_sugar += mass * EQUIV_SWEETNESS.get(ing_id, 1.0)
        expected = sugar_level_to_grams(recipe.sugar_level, recipe.cup_volume_ml)
        if abs(added_sugar - expected) > max(8, expected * 0.6):
            warnings.append(
                f"{rid}: added-sugar-equiv {added_sugar:.1f}g vs expected "
                f"{expected:.1f}g (level={recipe.sugar_level})"
            )

        # 6. Provenance label
        source = (recipe.metadata or {}).get("source")
        if source and source not in KNOWN_SOURCES:
            warnings.append(f"{rid}: unknown provenance label {source!r}")

        by_style[recipe.style] += 1
        if source:
            by_source[source] += 1

    # ----- summary -----
    print(f"\nValidation: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        print("\n--- ERRORS ---")
        for e in errors[:30]:
            print(f"  {e}")
        if len(errors) > 30:
            print(f"  ... +{len(errors) - 30} more")
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings[:30]:
            print(f"  {w}")
        if len(warnings) > 30:
            print(f"  ... +{len(warnings) - 30} more")

    print("\n--- Distribution by style ---")
    for style, n in sorted(by_style.items(), key=lambda kv: -kv[1]):
        print(f"  {style:12s} {n:>4}")
    print("\n--- Distribution by provenance ---")
    for source, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:30s} {n:>4}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
