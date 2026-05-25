"""Generate synthetic SKU records for sales predictor pretraining.

Per 技术方案书 §3.3.2 the sales predictor needs (brand, price, marketing,
season) baseline features + recipe features + a sales proxy label. Since we
have **no real sales data**, this script synthesizes plausible records by:

  1. Take each of the 110 reference recipes (`data/recipes/reference_recipes_v1.yaml`)
  2. Cross-product with brand × season × price × marketing combinations
  3. Compute a hand-coded sales proxy that has:
       - brand prestige effect (multiplicative)
       - inverted-U price sensitivity (sweet spot ~18元)
       - additive marketing boosts (联名/限定/明星/包装)
       - season × style interaction (果茶+夏 +; 奶茶+冬 +)
       - recipe-specific contribution (drives stage-2 residual learning)
       - Gaussian noise

The model trained on this is **NOT for real-market prediction** — it's pipeline
validation. Documented clearly in metadata.source so anyone reading
data/products/synthetic_skus_v1.parquet knows the provenance.

Usage:
    python scripts/generate_synthetic_skus.py --n-per-recipe 18 --seed 42

Output:
    data/products/synthetic_skus_v1.parquet  (~2000 rows)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.recipes.schema import Recipe
from beverage_ai.simulators.health.calculator import compute_nutrition


# Brands with rough prestige effect on baseline sales (higher = more eyeballs)
BRAND_PRESTIGE = {
    "喜茶": 30, "奈雪": 26, "茶颜悦色": 22, "霸王茶姬": 22,
    "古茗": 16, "茶百道": 16, "书亦烧仙草": 12, "蜜雪冰城": 10,
    "一点点": 8, "CoCo都可": 6,
}
SEASONS = ("spring", "summer", "autumn", "winter")
CITY_TIERS = ("一线", "二线", "三线")


def synthetic_sales(row: dict, rng: np.random.Generator) -> float:
    """Hand-coded sales proxy.

    The relationship encodes (a) baseline factors a LightGBM should learn
    in stage 1 and (b) recipe-specific residuals for stage 2.
    """
    base = 50.0
    s = base + BRAND_PRESTIGE.get(row["brand"], 0)

    # Inverted-U price sensitivity (sweet spot 18元)
    price = row["price_cny"]
    s -= 1.4 * abs(price - 18.0)

    # Marketing tags (additive)
    if row["marketing_联名"]: s += 12
    if row["marketing_限定"]: s += 7
    if row["marketing_明星"]: s += 5
    if row["marketing_包装"]: s += 4

    # Season × style interaction
    season, style = row["season"], row["recipe_style"]
    if season == "summer" and style in ("果茶", "冰沙"):
        s += 12
    if season == "winter" and style in ("奶茶", "咖啡奶茶"):
        s += 9
    if season == "summer" and style in ("奶茶",) and "厚乳" not in row["recipe_name"]:
        s += 3
    if season == "winter" and style in ("果茶",):
        s -= 4

    # City-tier baseline
    s += {"一线": 4, "二线": 1, "三线": -2}.get(row["city_tier"], 0)

    # ---- recipe-specific residual signals (stage 2 should pick these up) ----
    # Trendy ingredient bonus: 桂花/油柑/鸭屎香/厚乳/燕麦奶 → currently hot
    text = row["recipe_name"] + " " + str(row["recipe_ingredient_summary"])
    if "桂花" in text: s += 6
    if "油柑" in text: s += 5
    if "鸭屎香" in text: s += 7
    if "厚乳" in text: s += 6
    if "燕麦" in text: s += 4
    if "杨枝甘露" in text: s += 5
    # Heavy items (high sugar) get punished slightly with health-aware market
    if row["recipe_sugar_g"] > 30: s -= 4
    if row["recipe_calorie_kcal"] > 350: s -= 3
    # Ingredient diversity sweet spot (4-7 unique ingredients)
    n_ing = row["recipe_n_ingredients"]
    if 4 <= n_ing <= 7: s += 4
    if n_ing > 10: s -= 3

    # Noise (sigma=8 = realistic variance)
    s += float(rng.normal(0, 8.0))
    return max(0.0, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", default="data/recipes/reference_recipes_v1.yaml")
    parser.add_argument("--out", default="data/products/synthetic_skus_v1.parquet")
    parser.add_argument("--n-per-recipe", type=int, default=18,
                        help="Variations per recipe (brand × season × marketing combos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab = load_default_vocab()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recipes_path = Path(args.recipes)
    with open(recipes_path, encoding="utf-8") as f:
        raw_recipes = yaml.safe_load(f) or []
    print(f"loaded {len(raw_recipes)} reference recipes")

    rows = []
    brand_list = list(BRAND_PRESTIGE.keys())

    for recipe_dict in raw_recipes:
        recipe = Recipe(**recipe_dict)
        nut = compute_nutrition(recipe, vocab)

        # Recipe-derived features (constant across SKU variations)
        recipe_name = (recipe.metadata or {}).get("name_zh", recipe.recipe_id)
        ingredient_summary = ",".join(recipe.ingredients.keys())
        recipe_feat = {
            "recipe_id": recipe.recipe_id,
            "recipe_name": recipe_name,
            "recipe_style": recipe.style,
            "recipe_sugar_level": recipe.sugar_level,
            "recipe_cup_volume_ml": recipe.cup_volume_ml,
            "recipe_n_ingredients": len(recipe.ingredients),
            "recipe_total_mass_g": round(recipe.total_mass_g(), 1),
            "recipe_calorie_kcal": nut["energy_kcal"],
            "recipe_sugar_g": nut["sugar_g"],
            "recipe_fat_g": nut["fat_g"],
            "recipe_caffeine_mg": nut["caffeine_mg"],
            "recipe_sodium_mg": nut["sodium_mg"],
            "recipe_ingredient_summary": ingredient_summary,
            "recipe_n_topping": sum(1 for i in recipe.ingredients
                                    if i in vocab and vocab.get(i).category == "topping"),
            "recipe_has_dairy": recipe.has_category(vocab, "dairy_base"),
            "recipe_has_alt_milk": recipe.has_category(vocab, "alt_milk_base"),
            "recipe_has_coffee": recipe.has_category(vocab, "coffee_base"),
            "recipe_has_fruit": recipe.has_category(vocab, "fruit"),
        }

        for _ in range(args.n_per_recipe):
            brand = rng.choice(brand_list)
            # Brand-dependent price range (蜜雪冰城 cheap, 喜茶 expensive)
            price_base = {"喜茶": 25, "奈雪": 23, "霸王茶姬": 21, "茶颜悦色": 19,
                          "古茗": 16, "茶百道": 17, "书亦烧仙草": 15,
                          "蜜雪冰城": 8, "一点点": 14, "CoCo都可": 14}.get(brand, 18)
            price = round(float(rng.normal(price_base, 2.0)), 0)
            price = max(6.0, min(40.0, price))

            row = {
                **recipe_feat,
                "sku_id": f"{recipe.recipe_id}_{brand}_{int(rng.integers(1e8)):08d}",
                "brand": brand,
                "season": str(rng.choice(SEASONS)),
                "city_tier": str(rng.choice(CITY_TIERS, p=[0.45, 0.35, 0.20])),
                "price_cny": price,
                "launch_year": int(rng.choice([2024, 2025, 2026], p=[0.2, 0.45, 0.35])),
                "marketing_联名": bool(rng.random() < 0.15),
                "marketing_限定": bool(rng.random() < 0.30),
                "marketing_明星": bool(rng.random() < 0.08),
                "marketing_包装": bool(rng.random() < 0.20),
                "metadata_source": "synthetic_v1_handcoded_proxy",
            }
            row["sales_proxy"] = round(synthetic_sales(row, rng), 2)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(df)} SKUs)")

    print("\n=== summary ===")
    print(f"By brand:")
    print(df.groupby("brand")["sales_proxy"].agg(["count", "mean", "std"]).round(1).to_string())
    print(f"\nBy style × season (mean sales):")
    pivot = df.pivot_table(values="sales_proxy", index="recipe_style",
                            columns="season", aggfunc="mean").round(1)
    print(pivot.to_string())
    print(f"\nSales distribution: mean={df.sales_proxy.mean():.1f}  "
          f"std={df.sales_proxy.std():.1f}  "
          f"min={df.sales_proxy.min():.1f}  max={df.sales_proxy.max():.1f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
