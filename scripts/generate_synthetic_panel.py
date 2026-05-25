"""Generate a synthetic panel session so we can validate the closed-loop pipeline.

Per 技术方案书 §6.2 main panel design:
  - 35 panelists × 21 recipes split into 3 groups
  - 5 core sensory dims + overall liking
  - BIBD assignment so each cup gets >= 10 raters
  - Latin square order randomization

This script doesn't try to mimic BIBD/Latin-square — it just produces a
plausible (recipe, panelist, dim, score) feedback table for testing
update_from_feedback.py. Replace with real panel CSV ingestion when a
real session happens.

Workflow:
  1. Sample 21 recipes from reference_recipes_v1.yaml (or pass --recipes)
  2. For each recipe, compute "true" aspect scores from ingredient
     composition (uses the same logic as MockSensoryPredictor)
  3. For each panelist × recipe × dim, add panelist-bias + Gaussian noise
  4. Insert into feedback.duckdb (`feedback` + `panel_score` tables)

Usage:
    python scripts/generate_synthetic_panel.py --session s_synth_w1 --n-panelists 35
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml

from beverage_ai.aspects.schema import CORE_DIMS
from beverage_ai.feedback.recorder import FeedbackRecorder
from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.recipes.schema import Recipe
from beverage_ai.simulators.sensory.predict import MockSensoryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", default="data/recipes/reference_recipes_v1.yaml")
    parser.add_argument("--session", default="s_synth_w1",
                        help="Session id (must be unique per panel run)")
    parser.add_argument("--n-recipes", type=int, default=21,
                        help="Recipes evaluated this session")
    parser.add_argument("--n-panelists", type=int, default=35)
    parser.add_argument("--ratings-per-cup", type=int, default=10,
                        help="How many panelists rate each cup (BIBD ≥10)")
    parser.add_argument("--feedback-db", default="data/feedback.duckdb")
    parser.add_argument("--panelist-bias-sigma", type=float, default=0.4,
                        help="Per-panelist Likert bias (0-5 scale)")
    parser.add_argument("--rating-noise-sigma", type=float, default=0.6,
                        help="Per-rating Likert noise")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab = load_default_vocab()
    sensory = MockSensoryPredictor(vocab, seed=args.seed)

    print("[1/3] Loading recipes ...")
    with open(args.recipes, encoding="utf-8") as f:
        all_recipes = [Recipe(**r) for r in (yaml.safe_load(f) or [])]
    print(f"      {len(all_recipes)} recipes available")
    idx = rng.choice(len(all_recipes), size=min(args.n_recipes, len(all_recipes)),
                     replace=False)
    selected = [all_recipes[i] for i in idx]
    print(f"      sampled {len(selected)} for panel")

    print("\n[2/3] Computing 'true' aspect scores from ingredient composition ...")
    # MockSensoryPredictor returns scores in [0, 1] (after dividing by 1.0 of Likert).
    # Convert to Likert 1-5 by: score = 1 + 4 * x (linear map).
    recipe_truth: dict[str, dict[str, float]] = {}
    for r in selected:
        pred = sensory.predict(r)
        truth = {}
        for d in CORE_DIMS:
            # Predictor mean is on 1-5 scale already (`clip5` in mock).
            mean_15 = pred.means.get(d)
            if mean_15 is None:
                truth[d] = 3.0  # neutral fallback
            else:
                truth[d] = float(mean_15)
        recipe_truth[r.recipe_id] = truth

    print(f"\n[3/3] Simulating panelist ratings and writing to {args.feedback_db} ...")
    # Panelist biases — some are critical, some lenient
    panelist_ids = [f"p_{i:03d}" for i in range(args.n_panelists)]
    panelist_bias = {pid: float(rng.normal(0, args.panelist_bias_sigma))
                     for pid in panelist_ids}

    rec = FeedbackRecorder(args.feedback_db)
    # 1) write each recipe into `feedback` table (so Stage 2 can recover Recipe JSON)
    for recipe in selected:
        rec.record_recipe(
            session_id=args.session,
            recipe=recipe,
            predicted={"mock_truth": recipe_truth[recipe.recipe_id]},
            context={"synthetic_panel": True, "seed": args.seed},
        )

    # 2) write panel_score rows
    total_rows = 0
    rng_panel = np.random.default_rng(args.seed + 1)
    for recipe in selected:
        # Pick which panelists rate this cup (random subset of size ratings_per_cup)
        raters = rng_panel.choice(panelist_ids, size=min(args.ratings_per_cup,
                                                          len(panelist_ids)),
                                   replace=False)
        for cup_order, pid in enumerate(raters, start=1):
            for dim in CORE_DIMS:
                true_mean = recipe_truth[recipe.recipe_id][dim]
                score = true_mean + panelist_bias[pid] + \
                        float(rng_panel.normal(0, args.rating_noise_sigma))
                score = int(round(max(1, min(5, score))))
                rec.record_panel(
                    session_id=args.session,
                    recipe_id=recipe.recipe_id,
                    panelist_id=pid,
                    dimension=dim,
                    score=score,
                    cup_order=cup_order,
                    block=0,
                )
                total_rows += 1

    rec.close()
    print(f"      wrote {total_rows} panel_score rows for session={args.session!r}")
    print(f"      {len(selected)} recipes × {args.ratings_per_cup} panelists × "
          f"{len(CORE_DIMS)} dims")
    print("\nDone. Run `python scripts/update_from_feedback.py --session "
          f"{args.session}` to trigger closed-loop updates.")


if __name__ == "__main__":
    sys.exit(main() or 0)
