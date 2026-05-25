"""End-to-end closed-loop update from a panel session.

Per 技术方案书 §3.7 model-update table, a single panel session should
trigger ALL of these in order:

  1. Sensory GNN Stage 2 fine-tune        — train_sensory_gnn_stage2.py
  2. Dirichlet posterior update           — PriorEngine.update_dirichlet_posterior
  3. typical_serving_g calibration        — this script (§E.9.4)
  4. (optional) Active learning sampler   — TODO Phase 2

Stages 1-3 are concrete; #4 is a stretch goal.

Usage:
    python scripts/update_from_feedback.py --session s_synth_w1

What it does:
  - Reads panel data from data/feedback.duckdb
  - For each recipe with panel rating > threshold (default top 30%):
      * Adds its volume partition to Dirichlet observation pool
      * Adds its ingredient masses to typical_serving running stats
  - Calls Bayesian update on the prior engine (writes prior_history/...json)
  - Updates ingredient_vocab.yaml.v2 with refreshed typical_serving_g
  - Triggers Stage 2 GNN training as a subprocess
  - Writes a single audit-trail JSON: data/feedback/update_session_*.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import yaml

from beverage_ai.aspects.schema import CORE_DIMS
from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.priors.engine import PriorEngine, load_default_engine
from beverage_ai.recipes.schema import Recipe


# =============================================================================
# Step 0: load panel session
# =============================================================================

def _load_session(db_path: str, session_id: str) -> list[tuple[Recipe, float]]:
    """Return list of (recipe, overall_liking_in_0_1) for the session."""
    con = duckdb.connect(db_path)
    rows = con.execute(
        """
        SELECT p.recipe_id, p.dimension, p.score, f.recipe_json
        FROM panel_score p
        LEFT JOIN feedback f
            ON p.session_id = f.session_id AND p.recipe_id = f.recipe_id
        WHERE p.session_id = ? AND p.dimension = '喜爱度'
        """,
        [session_id],
    ).fetchall()
    con.close()

    per_recipe: dict[str, list[float]] = defaultdict(list)
    recipe_json_by_id: dict[str, str] = {}
    for rid, _dim, score, rjson in rows:
        per_recipe[rid].append(float(score))
        if rjson and rid not in recipe_json_by_id:
            recipe_json_by_id[rid] = rjson

    out: list[tuple[Recipe, float]] = []
    for rid, scores in per_recipe.items():
        rjson = recipe_json_by_id.get(rid)
        if not rjson:
            continue
        try:
            recipe = Recipe(**json.loads(rjson))
        except Exception:
            continue
        # mean Likert 1-5 → [0, 1]
        mean_15 = float(np.mean(scores))
        liking_01 = (mean_15 - 1) / 4
        out.append((recipe, round(liking_01, 4)))
    return out


# =============================================================================
# Step 2: Dirichlet posterior update (delegates to PriorEngine)
# =============================================================================

def _update_dirichlet(prior_engine: PriorEngine,
                      recipes_with_liking: list[tuple[Recipe, float]],
                      learning_rate: float = 0.3) -> dict:
    """Group recipes by style, call posterior update for each."""
    by_style: dict[str, list[tuple[Recipe, float]]] = defaultdict(list)
    for r, s in recipes_with_liking:
        by_style[r.style].append((r, s))

    changes: dict[str, dict] = {}
    for style, items in by_style.items():
        if len(items) < 3:
            continue
        recipes = [r for r, _ in items]
        scores = np.array([s for _, s in items])
        before = list(prior_engine.get_dirichlet_alpha(style))
        new_alpha = prior_engine.update_dirichlet_posterior(
            style, recipes, scores, learning_rate=learning_rate,
        )
        changes[style] = {
            "n_observed": len(items),
            "alpha_before": [round(x, 3) for x in before],
            "alpha_after": [round(float(x), 3) for x in new_alpha],
            "delta": [round(float(a - b), 3) for a, b in zip(new_alpha, before)],
        }
    return changes


# =============================================================================
# Step 3: typical_serving_g calibration (§E.9.4)
# =============================================================================

def _update_typical_serving(
    vocab,
    recipes_with_liking: list[tuple[Recipe, float]],
    top_quantile: float = 0.6,
    min_observations: int = 3,
    out_yaml: Path | None = None,
) -> dict:
    """For each ingredient observed in high-liking recipes, update its
    typical_serving_g to a running median of observed masses.

    Returns a dict {ingredient_id: {before, after, n_obs, p25, p75}}.
    """
    scores = [s for _, s in recipes_with_liking]
    if len(scores) < min_observations:
        return {}
    threshold = float(np.quantile(scores, top_quantile))
    good = [r for r, s in recipes_with_liking if s >= threshold]
    if not good:
        return {}

    masses_per_id: dict[str, list[float]] = defaultdict(list)
    for r in good:
        for ing_id, mass in r.ingredients.items():
            if ing_id in vocab:
                masses_per_id[ing_id].append(float(mass))

    changes: dict[str, dict] = {}
    for ing_id, masses in masses_per_id.items():
        if len(masses) < min_observations:
            continue
        before = vocab.get(ing_id).typical_serving_g
        after = float(np.median(masses))
        # Only flag meaningful changes (>20% relative shift)
        if abs(after - before) / max(before, 1e-3) < 0.20:
            continue
        changes[ing_id] = {
            "n_observed": len(masses),
            "before": round(before, 2),
            "after": round(after, 2),
            "p25": round(float(np.percentile(masses, 25)), 2),
            "p75": round(float(np.percentile(masses, 75)), 2),
            "delta_rel": round((after - before) / max(before, 1e-3), 3),
        }

    if out_yaml is not None and changes:
        # Write an "overrides" file (don't clobber the main vocab — let user merge)
        out_yaml.write_text(
            yaml.safe_dump(
                {"typical_serving_g_overrides": changes},
                allow_unicode=True, sort_keys=False,
            ),
            encoding="utf-8",
        )
    return changes


# =============================================================================
# Step 1: trigger GNN Stage 2 fine-tune (delegates to subprocess)
# =============================================================================

def _trigger_stage2(session_id: str, base_model: str, feedback_db: str,
                    epochs: int) -> dict:
    cmd = [
        sys.executable, "scripts/train_sensory_gnn_stage2.py",
        "--base-model", base_model,
        "--feedback-db", feedback_db,
        "--session-id", session_id,
        "--epochs", str(epochs),
    ]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return {"status": "failed", "returncode": proc.returncode,
                "stderr_tail": proc.stderr.splitlines()[-5:]}
    # Read the log
    log_path = Path("models/sensory_gnn_stage2_log.json")
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "best_epoch": log.get("best_epoch"),
            "best_val_loss": log.get("best_val_loss"),
            "initial_zero_shot_pearson": log.get("initial_zero_shot_pearson"),
            "final_val_pearson_after_finetune": log.get("final_val_pearson_after_finetune"),
            "elapsed_sec": log.get("elapsed_sec"),
        }
    return {"status": "ok", "stdout_tail": proc.stdout.splitlines()[-10:]}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True,
                        help="panel session_id to consume")
    parser.add_argument("--feedback-db", default="data/feedback.duckdb")
    parser.add_argument("--base-model", default="models/sensory_gnn_stage1_best.pt")
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--dirichlet-lr", type=float, default=0.3)
    parser.add_argument("--serving-top-quantile", type=float, default=0.6)
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip GNN Stage 2 (only update Dirichlet + typical_serving)")
    parser.add_argument("--audit-dir", default="data/feedback")
    args = parser.parse_args()

    print(f"[0/4] Loading session {args.session!r} ...")
    pairs = _load_session(args.feedback_db, args.session)
    if len(pairs) < 3:
        print(f"ERROR: only {len(pairs)} recipes in session — need ≥ 3"); sys.exit(2)
    likings = [s for _, s in pairs]
    print(f"      {len(pairs)} recipes; liking mean={np.mean(likings):.3f}  "
          f"std={np.std(likings):.3f}  range=[{min(likings):.2f}, {max(likings):.2f}]")

    audit_path = Path(args.audit_dir) / f"update_session_{args.session}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "session": args.session,
        "feedback_db": args.feedback_db,
        "args": vars(args),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_recipes": len(pairs),
        "liking_stats": {"mean": float(np.mean(likings)),
                         "std": float(np.std(likings)),
                         "min": float(min(likings)),
                         "max": float(max(likings))},
    }

    # ----- Stage 1: GNN Stage 2 fine-tune -----
    if args.skip_stage2:
        print(f"\n[1/4] Skipping GNN Stage 2 (--skip-stage2)")
        audit["stage2"] = {"skipped": True}
    else:
        print(f"\n[1/4] GNN Stage 2 fine-tune ...")
        audit["stage2"] = _trigger_stage2(
            args.session, args.base_model, args.feedback_db, args.stage2_epochs,
        )

    # ----- Stage 2: Dirichlet posterior update -----
    print(f"\n[2/4] Dirichlet posterior update (learning_rate={args.dirichlet_lr}) ...")
    prior = load_default_engine()
    dirichlet_changes = _update_dirichlet(prior, pairs, args.dirichlet_lr)
    audit["dirichlet_changes"] = dirichlet_changes
    if not dirichlet_changes:
        print(f"      no style updated (none had ≥3 recipes)")
    else:
        for style, info in dirichlet_changes.items():
            print(f"      {style}: n={info['n_observed']}  Δ={info['delta']}")

    # ----- Stage 3: typical_serving_g calibration -----
    print(f"\n[3/4] typical_serving_g calibration (top {1-args.serving_top_quantile:.0%}) ...")
    vocab = load_default_vocab()
    overrides_path = Path("data/ingredients/typical_serving_overrides.yaml")
    serving_changes = _update_typical_serving(
        vocab, pairs,
        top_quantile=args.serving_top_quantile,
        out_yaml=overrides_path,
    )
    audit["typical_serving_changes"] = serving_changes
    if not serving_changes:
        print(f"      no ingredient crossed 20% shift threshold")
    else:
        print(f"      {len(serving_changes)} ingredients updated:")
        for ing_id, info in list(serving_changes.items())[:10]:
            print(f"        {ing_id:<30s} "
                  f"{info['before']:>6.1f}g → {info['after']:>6.1f}g "
                  f"({info['delta_rel']:+.0%}, n={info['n_observed']})")
        if len(serving_changes) > 10:
            print(f"        ... +{len(serving_changes) - 10} more")
        print(f"      wrote {overrides_path}")

    # ----- Stage 4: audit trail -----
    audit["finished_at"] = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[4/4] Audit trail → {audit_path}")
    print("\nClosed-loop update complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
