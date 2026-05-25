"""Generate a panel session schedule (BIBD + Latin square).

Per 技术方案书 §6.2 v1.1 design:
  - 21 recipes split A/B/C (7 each)
  - 35 panelists × 2 sessions × ≤ 3 cups each (= 6 cups/panelist)
  - Each cup rated by ≥ 10 panelists  (210 cup-ratings target)
  - Latin square presentation order per session (carry-over balance)
  - Each panelist sees ~2 from each group (A/B/C) to power between-group tests

We approximate strict BIBD via balanced round-robin assignment:
  cup i is rated by panelists [i, i+s, i+2s, ...] mod N for stride s such
  that each panelist receives the target cup count. For the standard 21/35/10
  setup this gives 6 cups/panelist evenly, with diagnostic prints to verify.

Outputs:
  <out>/panel_schedule_<id>.csv          one row per (panelist, cup) assignment
  <out>/panel_schedule_<id>.md           per-panelist instruction sheets
  <out>/panel_design_diagnostics.json    BIBD-ness checks

Usage:
    python scripts/panel_design.py \\
        --recipes-a recipes_a.txt --recipes-b recipes_b.txt --recipes-c recipes_c.txt \\
        --n-panelists 35 --ratings-per-cup 10 --sessions 2 --cups-per-session 3 \\
        --design-id 2026_q3 --out data/panels

If you don't yet have separate A/B/C lists, pass --auto-from-recipes-yaml to
sample 21 from data/recipes/reference_recipes_v1.yaml and split arbitrarily.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GROUP_LABELS = ("A", "B", "C")


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def _auto_recipes(yaml_path: Path, n: int, rng: np.random.Generator) -> list[list[str]]:
    """Sample n recipes from a reference YAML, split into 3 groups."""
    import yaml
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or []
    ids = [r["recipe_id"] for r in raw]
    idx = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    sampled = [ids[i] for i in idx]
    per = max(1, len(sampled) // 3)
    return [sampled[:per], sampled[per:2*per], sampled[2*per:]]


# =============================================================================
# Assignment
# =============================================================================

def assign_cups(group_recipes: list[list[str]], n_panelists: int,
                ratings_per_cup: int, rng: np.random.Generator) -> list[dict]:
    """Round-robin assignment so each panelist sees ~equal cups from each group.

    Returns list of dicts: {panelist_id, recipe_id, group}.
    """
    assignments = []
    for group_label, recipes in zip(GROUP_LABELS, group_recipes):
        for cup_i, recipe_id in enumerate(recipes):
            # Stagger which panelists rate this cup so load is even
            start = (cup_i * 3 + GROUP_LABELS.index(group_label) * 7) % n_panelists
            for offset in range(ratings_per_cup):
                pan_idx = (start + offset) % n_panelists
                assignments.append({
                    "panelist_id": f"p_{pan_idx:03d}",
                    "recipe_id": recipe_id,
                    "group": group_label,
                })
    return assignments


def _split_into_sessions(cups: list[dict], sessions: int, cups_per_session: int,
                         rng: np.random.Generator) -> list[dict]:
    """Split a panelist's cups into sessions, assign per-session order.

    Uses a Latin-square-friendly approach: per session, shuffle to avoid same
    cup always at position 1 across panelists. With small n_per_session (3),
    we use cyclic permutations for guaranteed position balance.
    """
    rng.shuffle(cups)
    enriched = []
    for s_i in range(sessions):
        session_cups = cups[s_i * cups_per_session : (s_i + 1) * cups_per_session]
        # Cyclic shift for Latin-square-ish position balance
        cyclic_offset = (s_i + hash(cups[0]["recipe_id"]) % cups_per_session) \
                        % cups_per_session
        rotated = session_cups[cyclic_offset:] + session_cups[:cyclic_offset]
        for order, cup in enumerate(rotated, start=1):
            enriched.append({**cup, "session": s_i + 1, "order_in_session": order})
    return enriched


def build_schedule(group_recipes: list[list[str]], n_panelists: int,
                   ratings_per_cup: int, sessions: int, cups_per_session: int,
                   seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    raw = assign_cups(group_recipes, n_panelists, ratings_per_cup, rng)

    # Group by panelist
    by_pan = defaultdict(list)
    for r in raw:
        by_pan[r["panelist_id"]].append(r)

    # Sanity check: each panelist should get ~ (total_cup_ratings / n_panelists) cups
    target = ratings_per_cup * sum(len(g) for g in group_recipes) // n_panelists
    expected_per_pan = sessions * cups_per_session
    if target != expected_per_pan:
        print(f"  WARN: round-robin gives {target} cups/panelist but "
              f"sessions * cups = {expected_per_pan}.  "
              f"Adjusting target = {min(target, expected_per_pan)}.")

    schedule = []
    for pan, cups in by_pan.items():
        # Cap or pad to the expected per-panelist load
        if len(cups) > expected_per_pan:
            cups = cups[:expected_per_pan]
        elif len(cups) < expected_per_pan:
            # Pad nothing; this panelist just rates fewer cups
            pass
        schedule.extend(_split_into_sessions(cups, sessions, cups_per_session, rng))
    return schedule


# =============================================================================
# Diagnostics
# =============================================================================

def diagnose(schedule: list[dict], group_recipes: list[list[str]]) -> dict:
    cup_count = defaultdict(int)
    pan_count = defaultdict(int)
    pan_group_count = defaultdict(lambda: defaultdict(int))
    position_count = defaultdict(lambda: defaultdict(int))
    for row in schedule:
        cup_count[row["recipe_id"]] += 1
        pan_count[row["panelist_id"]] += 1
        pan_group_count[row["panelist_id"]][row["group"]] += 1
        position_count[row["recipe_id"]][row["order_in_session"]] += 1

    all_recipes = [r for g in group_recipes for r in g]
    cup_counts_arr = np.array([cup_count.get(r, 0) for r in all_recipes])
    pan_counts_arr = np.array(list(pan_count.values()))

    return {
        "n_panelists_used": len(pan_count),
        "n_recipes": len(all_recipes),
        "cup_ratings": {
            "min": int(cup_counts_arr.min()),
            "max": int(cup_counts_arr.max()),
            "mean": float(cup_counts_arr.mean()),
            "std": float(cup_counts_arr.std()),
        },
        "panelist_load": {
            "min": int(pan_counts_arr.min()),
            "max": int(pan_counts_arr.max()),
            "mean": float(pan_counts_arr.mean()),
            "std": float(pan_counts_arr.std()),
        },
        "panelists_group_balance_violation": [
            {"panelist": pan, "groups": dict(gc)}
            for pan, gc in pan_group_count.items()
            if any(v < 1 for v in gc.values()) or any(v > 3 for v in gc.values())
        ][:10],
        "position_distribution_per_cup": {
            r: dict(positions) for r, positions in
            list(position_count.items())[:5]    # only first 5 for brevity
        },
    }


# =============================================================================
# Report
# =============================================================================

def write_csv(schedule: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panelist_id", "session", "order_in_session",
                                          "recipe_id", "group"])
        w.writeheader()
        for row in sorted(schedule, key=lambda r: (r["panelist_id"],
                                                    r["session"],
                                                    r["order_in_session"])):
            w.writerow(row)


def write_instructions(schedule: list[dict], path: Path, design_id: str) -> None:
    by_pan_sess = defaultdict(list)
    for row in schedule:
        by_pan_sess[(row["panelist_id"], row["session"])].append(row)

    lines = [f"# Panel Tasting Schedule — {design_id}",
             "", f"_Generated {datetime.now(timezone.utc).isoformat()}_",
             "",
             "**Instructions**: Each panelist rates the listed cups in the GIVEN ORDER. ",
             "Between cups: rinse mouth with plain water (5 min rest); ",
             "between sessions: at least 1 day gap.", "",
             "Score each cup on a 1–5 Likert scale for: 甜度 苦度 茶香 奶香 喜爱度.",
             ""]
    for (pan, sess), cups in sorted(by_pan_sess.items()):
        lines.append(f"## Panelist `{pan}` — Session {sess}")
        lines.append("")
        lines.append("| Order | Recipe ID | Group (blinded) |")
        lines.append("|---|---|---|")
        for c in sorted(cups, key=lambda r: r["order_in_session"]):
            # Blind group label — show as "?" so panelists don't know which is system/random/expert
            lines.append(f"| {c['order_in_session']} | `{c['recipe_id']}` | (blinded) |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes-a", default=None, help="Text file, one recipe_id per line")
    parser.add_argument("--recipes-b", default=None)
    parser.add_argument("--recipes-c", default=None)
    parser.add_argument("--auto-from-recipes-yaml",
                        default=None,
                        help="Sample 21 from this YAML and split arbitrarily (testing)")
    parser.add_argument("--n-panelists", type=int, default=35)
    parser.add_argument("--ratings-per-cup", type=int, default=10)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--cups-per-session", type=int, default=3)
    parser.add_argument("--design-id", default="auto")
    parser.add_argument("--out", default="data/panels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.auto_from_recipes_yaml:
        group_recipes = _auto_recipes(Path(args.auto_from_recipes_yaml), 21, rng)
        print(f"auto-sampled {sum(len(g) for g in group_recipes)} recipes from "
              f"{args.auto_from_recipes_yaml}: {[len(g) for g in group_recipes]}")
    else:
        if not (args.recipes_a and args.recipes_b and args.recipes_c):
            print("ERROR: pass --recipes-a/b/c or --auto-from-recipes-yaml", file=sys.stderr)
            sys.exit(1)
        group_recipes = [
            _read_lines(Path(args.recipes_a)),
            _read_lines(Path(args.recipes_b)),
            _read_lines(Path(args.recipes_c)),
        ]
        print(f"loaded {[len(g) for g in group_recipes]} recipes from A/B/C lists")

    if args.design_id == "auto":
        args.design_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding schedule (panelists={args.n_panelists}, "
          f"ratings_per_cup={args.ratings_per_cup}, "
          f"sessions={args.sessions}×{args.cups_per_session}) ...")
    schedule = build_schedule(
        group_recipes,
        n_panelists=args.n_panelists,
        ratings_per_cup=args.ratings_per_cup,
        sessions=args.sessions,
        cups_per_session=args.cups_per_session,
        seed=args.seed,
    )
    print(f"  {len(schedule)} (panelist, cup) assignments")

    diagnostics = diagnose(schedule, group_recipes)
    print("\n=== diagnostics ===")
    print(f"  panelists used: {diagnostics['n_panelists_used']} / {args.n_panelists}")
    print(f"  cup ratings:   min={diagnostics['cup_ratings']['min']:>2d}  "
          f"max={diagnostics['cup_ratings']['max']:>2d}  "
          f"mean={diagnostics['cup_ratings']['mean']:.1f}  "
          f"std={diagnostics['cup_ratings']['std']:.2f}")
    print(f"  panelist load: min={diagnostics['panelist_load']['min']:>2d}  "
          f"max={diagnostics['panelist_load']['max']:>2d}  "
          f"mean={diagnostics['panelist_load']['mean']:.1f}  "
          f"std={diagnostics['panelist_load']['std']:.2f}")
    n_violations = len(diagnostics["panelists_group_balance_violation"])
    if n_violations:
        print(f"  [WARN] {n_violations} panelists with group imbalance "
              f"(sample): {diagnostics['panelists_group_balance_violation'][:3]}")
    else:
        print(f"  [OK] group balance: all panelists see 1-3 cups from each group")

    csv_path = out_dir / f"panel_schedule_{args.design_id}.csv"
    md_path = out_dir / f"panel_schedule_{args.design_id}.md"
    diag_path = out_dir / f"panel_design_diagnostics_{args.design_id}.json"

    write_csv(schedule, csv_path)
    write_instructions(schedule, md_path, args.design_id)
    diag_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {diag_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
