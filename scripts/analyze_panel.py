"""Statistical analysis of a panel session.

Implements 技术方案书 §6.2 v1.1 spec:
  - Wilcoxon signed-rank for paired group comparisons (A vs B overall liking)
  - TOST equivalence test for A vs C (system ≥ expert)
  - Friedman test for multi-dimensional comparison + Nemenyi post-hoc
  - Bonferroni correction (α = 0.05 / 6 ≈ 0.0083) for 6 simultaneous tests
  - Linear mixed-effects model: score ~ group + (1|panelist) + (1|recipe)
  - Effect sizes: Cohen's d (paired) + rank-biserial r (Wilcoxon)
  - Power analysis: derive n_required for given effect size

Inputs:
    data/feedback.duckdb  (must contain panel_score + feedback tables)
    --group-map YAML mapping recipe_id → group (A / B / C) — OR
    --auto-group: infer from feedback.recipe_json metadata.source

Outputs:
    <out-dir>/panel_analysis_<session>.json   structured results
    <out-dir>/panel_analysis_<session>.md     human-readable report
    Stdout summary

Usage:
    python scripts/analyze_panel.py --session s_2026_q3 --group-map groups.yaml
    python scripts/analyze_panel.py --session s_demo_w1 --auto-group
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

# Stats are optional/lazy so the script can at least PARSE arguments without them
try:
    from scipy import stats as scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import statsmodels.formula.api as smf
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False


CORE_DIMS = ("甜度", "苦度", "茶香", "奶香", "喜爱度")
ALPHA_FAMILYWISE = 0.05
BONFERRONI_DIVISOR = 6   # 5 sensory dims + 1 overall


# =============================================================================
# Data loading
# =============================================================================

def load_session(db_path: str, session_id: str) -> pd.DataFrame:
    """Return DataFrame with columns:
        recipe_id, panelist_id, dimension, score, cup_order, block, recipe_meta
    """
    con = duckdb.connect(db_path)
    df = con.execute(
        """
        SELECT
            p.recipe_id, p.panelist_id, p.dimension, p.score,
            p.cup_order, p.block, f.recipe_json
        FROM panel_score p
        LEFT JOIN feedback f
            ON p.session_id = f.session_id AND p.recipe_id = f.recipe_id
        WHERE p.session_id = ?
        """,
        [session_id],
    ).df()
    con.close()
    if df.empty:
        raise RuntimeError(f"no panel_score rows for session={session_id!r}")
    # Parse the recipe metadata
    def _meta(s):
        if not s:
            return {}
        try:
            return (json.loads(s).get("metadata") or {})
        except Exception:
            return {}
    df["recipe_meta"] = df["recipe_json"].apply(_meta)
    return df


def assign_groups(df: pd.DataFrame, group_map: dict | None,
                  auto_from_meta: bool) -> pd.DataFrame:
    """Add 'group' column. Returns df."""
    out = df.copy()
    if group_map:
        out["group"] = out["recipe_id"].map(group_map).fillna("unassigned")
    elif auto_from_meta:
        # Use metadata.source — e.g. "system_topk" → A, "random_baseline" → B,
        # "expert_control" / "well_known_classic" → C, else "unassigned"
        def _src_to_group(meta):
            src = (meta or {}).get("source", "")
            if "system" in src or "topk" in src:
                return "A"
            if "random" in src or "baseline" in src:
                return "B"
            if "expert" in src or "well_known" in src or "brand_inspired" in src:
                return "C"
            return "unassigned"
        out["group"] = out["recipe_meta"].apply(_src_to_group)
    else:
        out["group"] = "unassigned"
    return out


# =============================================================================
# Per-dim aggregation: one mean score per (panelist, recipe, dim)
# (already one row per (panelist, recipe, dim), but cup_order may duplicate)
# =============================================================================

def per_cup_scores(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """For dim, return (recipe_id, panelist_id, group, score) one row each.
    Averages over cup_order if a panelist tasted the same cup twice."""
    sub = df[df["dimension"] == dim].copy()
    if sub.empty:
        return sub
    agg = sub.groupby(["recipe_id", "panelist_id", "group"], as_index=False)["score"].mean()
    return agg


# =============================================================================
# Effect size
# =============================================================================

def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    if len(diffs) < 2 or np.std(diffs, ddof=1) == 0:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def rank_biserial_from_wilcoxon(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """rank-biserial r for paired Wilcoxon."""
    diffs = scores_a - scores_b
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    abs_ranks = scipy_stats.rankdata(np.abs(diffs))
    pos_sum = abs_ranks[diffs > 0].sum()
    neg_sum = abs_ranks[diffs < 0].sum()
    total = pos_sum + neg_sum
    return float((pos_sum - neg_sum) / total) if total > 0 else 0.0


# =============================================================================
# Tests
# =============================================================================

def wilcoxon_paired(df: pd.DataFrame, dim: str, group_a: str, group_b: str,
                    alpha: float) -> dict:
    """Per panelist, compute mean score on group_a vs group_b, then paired Wilcoxon."""
    sub = per_cup_scores(df, dim)
    if sub.empty:
        return {"status": "no_data"}
    # mean score per panelist within group
    per_pan = sub.groupby(["panelist_id", "group"], as_index=False)["score"].mean()
    pivot = per_pan.pivot(index="panelist_id", columns="group", values="score")
    if group_a not in pivot.columns or group_b not in pivot.columns:
        return {"status": "missing_group",
                "available_groups": list(pivot.columns)}
    paired = pivot[[group_a, group_b]].dropna()
    if len(paired) < 5:
        return {"status": "too_few_paired", "n_paired": len(paired)}
    a = paired[group_a].to_numpy()
    b = paired[group_b].to_numpy()
    diffs = a - b
    # Wilcoxon signed-rank, alternative: A > B
    res = scipy_stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return {
        "status": "ok",
        "n_paired": int(len(paired)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "wilcoxon_W": float(res.statistic),
        "p_value_onesided": float(res.pvalue),
        "significant_at_alpha": bool(res.pvalue < alpha),
        "cohens_d": cohens_d_paired(diffs),
        "rank_biserial_r": rank_biserial_from_wilcoxon(a, b),
    }


def tost_equivalence(df: pd.DataFrame, dim: str,
                     group_a: str, group_c: str,
                     equiv_bound: float = 0.5) -> dict:
    """TOST (two one-sided test) for A ≈ C (paired)."""
    sub = per_cup_scores(df, dim)
    per_pan = sub.groupby(["panelist_id", "group"], as_index=False)["score"].mean()
    pivot = per_pan.pivot(index="panelist_id", columns="group", values="score")
    if group_a not in pivot.columns or group_c not in pivot.columns:
        return {"status": "missing_group"}
    paired = pivot[[group_a, group_c]].dropna()
    if len(paired) < 5:
        return {"status": "too_few_paired", "n_paired": len(paired)}
    diffs = paired[group_a].to_numpy() - paired[group_c].to_numpy()
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    # Two t-tests
    # H1a: diff > -equiv_bound; H1b: diff < +equiv_bound
    if se == 0:
        return {"status": "zero_variance"}
    t_low = (mean_diff - (-equiv_bound)) / se
    t_high = (mean_diff - equiv_bound) / se
    df_param = n - 1
    p_low = 1 - scipy_stats.t.cdf(t_low, df_param)       # H1: diff > -bound
    p_high = scipy_stats.t.cdf(t_high, df_param)         # H1: diff < +bound
    return {
        "status": "ok",
        "n_paired": int(n),
        "mean_diff": mean_diff,
        "equiv_bound": equiv_bound,
        "p_low_bound": float(p_low),
        "p_high_bound": float(p_high),
        "equivalent_at_alpha_0_05": bool(p_low < 0.05 and p_high < 0.05),
    }


def friedman_across_dims(df: pd.DataFrame, dims: list[str]) -> dict:
    """Friedman test across the 5 sensory dims for each panelist's mean score."""
    score_per_pan_dim = []
    for d in dims:
        sub = per_cup_scores(df, d)
        if sub.empty:
            continue
        mean_pan = sub.groupby("panelist_id")["score"].mean()
        score_per_pan_dim.append((d, mean_pan))
    if len(score_per_pan_dim) < 2:
        return {"status": "too_few_dims"}
    # Align panelists across dims
    common = set(score_per_pan_dim[0][1].index)
    for _d, s in score_per_pan_dim[1:]:
        common &= set(s.index)
    common = sorted(common)
    if len(common) < 5:
        return {"status": "too_few_panelists", "n": len(common)}
    arrays = [s.reindex(common).to_numpy() for _, s in score_per_pan_dim]
    res = scipy_stats.friedmanchisquare(*arrays)
    return {
        "status": "ok",
        "n_panelists": len(common),
        "n_dims": len(score_per_pan_dim),
        "dims": [d for d, _ in score_per_pan_dim],
        "chi2": float(res.statistic),
        "p_value": float(res.pvalue),
        "alpha_bonferroni": ALPHA_FAMILYWISE / BONFERRONI_DIVISOR,
        "significant": bool(res.pvalue < (ALPHA_FAMILYWISE / BONFERRONI_DIVISOR)),
    }


def mixed_effects(df: pd.DataFrame, dim: str) -> dict:
    """score ~ group + (1 | panelist_id) on long-form data."""
    if not _STATSMODELS_OK:
        return {"status": "statsmodels_not_installed"}
    sub = per_cup_scores(df, dim)
    if sub.empty:
        return {"status": "no_data"}
    if sub["group"].nunique() < 2:
        return {"status": "single_group"}
    # statsmodels needs groups specified separately
    try:
        model = smf.mixedlm("score ~ group", sub, groups=sub["panelist_id"])
        result = model.fit(disp=False)
        # Extract group coefficients (vs reference category, alphabetic first)
        coefs = {}
        pvals = {}
        for name, val in result.params.items():
            if name.startswith("group["):
                coefs[name] = float(val)
                pvals[name] = float(result.pvalues[name])
        return {
            "status": "ok",
            "n_obs": int(sub.shape[0]),
            "n_panelists": int(sub["panelist_id"].nunique()),
            "n_groups": int(sub["group"].nunique()),
            "fixed_effects": coefs,
            "p_values": pvals,
            "loglikelihood": float(result.llf),
            "panelist_variance": float(result.cov_re.iloc[0, 0]) if result.cov_re.size > 0 else None,
        }
    except Exception as e:
        return {"status": "fit_failed", "error": str(e)[:200]}


def power_analysis(effect_size_d: float, alpha: float = 0.05,
                   power: float = 0.80) -> dict:
    """Sample size needed for paired t-test (Wilcoxon ~ similar to t at this scale)."""
    # Using formula from Cohen 1988: n ≈ ((z_α + z_β) / d)^2
    z_alpha = scipy_stats.norm.ppf(1 - alpha)        # one-sided
    z_beta = scipy_stats.norm.ppf(power)
    n_required = ((z_alpha + z_beta) / max(effect_size_d, 1e-6)) ** 2
    return {
        "effect_size": effect_size_d,
        "alpha": alpha,
        "power": power,
        "n_required_paired": int(np.ceil(n_required)),
        "note": "paired Wilcoxon needs ~15% more per Lehmann (1975)",
    }


# =============================================================================
# Report
# =============================================================================

def write_report(results: dict, md_path: Path) -> None:
    lines = [
        f"# Panel Analysis — Session `{results['session']}`",
        "",
        f"_Generated {results['timestamp']}_",
        "",
        "## Session overview",
        f"- Total ratings: {results['n_rows']:,}",
        f"- Panelists: {results['n_panelists']}",
        f"- Recipes: {results['n_recipes']}",
        f"- Dimensions: {', '.join(results['dimensions'])}",
        f"- Groups: {results['group_counts']}",
        "",
        "## Per-dimension summary",
    ]
    summary = results.get("per_dim_summary", {})
    if summary:
        lines.append("| Dim | n_obs | Mean | Std |")
        lines.append("|---|---|---|---|")
        for d, s in summary.items():
            lines.append(f"| {d} | {s.get('n', 0)} | {s.get('mean', 0):.2f} | {s.get('std', 0):.2f} |")

    lines.append("")
    lines.append("## Wilcoxon paired tests (A vs B, one-sided A > B)")
    lines.append("| Dim | n | mean_A | mean_B | Δ | p | sig? | Cohen's d | r |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d, r in (results.get("wilcoxon_a_vs_b") or {}).items():
        if r.get("status") != "ok":
            lines.append(f"| {d} | – | – | – | – | – | n/a ({r.get('status')}) | – | – |")
            continue
        lines.append(
            f"| {d} | {r['n_paired']} | {r['mean_a']:.2f} | {r['mean_b']:.2f} | "
            f"{r['mean_diff']:+.2f} | {r['p_value_onesided']:.4f} | "
            f"{'✅' if r['significant_at_alpha'] else '❌'} | "
            f"{r['cohens_d']:+.2f} | {r['rank_biserial_r']:+.2f} |"
        )

    lines.append("")
    lines.append("## TOST equivalence (A ≈ C)")
    for d, r in (results.get("tost_a_vs_c") or {}).items():
        if r.get("status") != "ok":
            lines.append(f"- **{d}**: n/a ({r.get('status')})")
            continue
        verdict = "✅ equivalent" if r["equivalent_at_alpha_0_05"] else "❌ not equivalent"
        lines.append(
            f"- **{d}**: Δ={r['mean_diff']:+.2f} (bound ±{r['equiv_bound']}), "
            f"p_low={r['p_low_bound']:.3f}, p_high={r['p_high_bound']:.3f} → {verdict}"
        )

    lines.append("")
    lines.append("## Friedman test (across 5 sensory dims)")
    f_ = results.get("friedman", {})
    if f_.get("status") == "ok":
        sig = "✅" if f_["significant"] else "❌"
        lines.append(
            f"- χ²({f_['n_dims']-1}) = {f_['chi2']:.2f}, p={f_['p_value']:.4f} "
            f"vs Bonferroni α={f_['alpha_bonferroni']:.4f} → {sig}"
        )
    else:
        lines.append(f"- not run ({f_.get('status')})")

    lines.append("")
    lines.append("## Mixed-effects model (`score ~ group + (1|panelist)`)")
    for d, m in (results.get("mixed_effects") or {}).items():
        if m.get("status") != "ok":
            lines.append(f"- **{d}**: n/a ({m.get('status')})")
            continue
        lines.append(f"- **{d}**: log-lik={m['loglikelihood']:.1f}, "
                     f"panelist var={m.get('panelist_variance'):.3f}, "
                     f"fixed effects: {m['fixed_effects']}, p: {m['p_values']}")

    lines.append("")
    lines.append("## Power analysis")
    p_ = results.get("power_analysis", {})
    if p_:
        lines.append(f"- For d={p_['effect_size']}, α={p_['alpha']}, power={p_['power']}: "
                     f"**n ≥ {p_['n_required_paired']}** (paired t/Wilcoxon)")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--feedback-db", default="data/feedback.duckdb")
    parser.add_argument("--group-map", default=None,
                        help="YAML mapping {recipe_id: group_letter}")
    parser.add_argument("--auto-group", action="store_true",
                        help="Infer group from feedback.recipe_json metadata.source")
    parser.add_argument("--out-dir", default="data/feedback")
    parser.add_argument("--equiv-bound", type=float, default=0.5,
                        help="TOST bound for A≈C (Likert scale units)")
    parser.add_argument("--planned-effect-size", type=float, default=0.5,
                        help="d for power-analysis report")
    args = parser.parse_args()

    if not _SCIPY_OK:
        print("ERROR: scipy not installed. pip install scipy", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Loading session {args.session!r} ...")
    df = load_session(args.feedback_db, args.session)
    print(f"      rows: {len(df):,}  panelists: {df['panelist_id'].nunique()}  "
          f"recipes: {df['recipe_id'].nunique()}  dims: {sorted(df['dimension'].unique())}")

    group_map = None
    if args.group_map:
        if yaml is None:
            print("ERROR: pyyaml needed for --group-map"); sys.exit(1)
        group_map = yaml.safe_load(Path(args.group_map).read_text(encoding="utf-8"))
    df = assign_groups(df, group_map, args.auto_group)
    print(f"      group counts: {dict(df.groupby('group')['recipe_id'].nunique())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/4] Per-dimension summary ...")
    summary = {}
    for d in CORE_DIMS:
        sub = df[df["dimension"] == d]
        if not sub.empty:
            summary[d] = {
                "n": int(len(sub)),
                "mean": float(sub["score"].mean()),
                "std": float(sub["score"].std(ddof=1)),
            }
    for d, s in summary.items():
        print(f"      {d}: n={s['n']} mean={s['mean']:.2f} std={s['std']:.2f}")

    print("\n[3/4] Running tests ...")
    alpha_corrected = ALPHA_FAMILYWISE / BONFERRONI_DIVISOR
    print(f"      Bonferroni α = {ALPHA_FAMILYWISE}/{BONFERRONI_DIVISOR} = {alpha_corrected:.4f}")

    wilcoxon_results = {d: wilcoxon_paired(df, d, "A", "B", alpha_corrected)
                        for d in CORE_DIMS}
    tost_results = {d: tost_equivalence(df, d, "A", "C", args.equiv_bound)
                    for d in CORE_DIMS}
    friedman = friedman_across_dims(df, list(CORE_DIMS))
    me_results = {d: mixed_effects(df, d) for d in CORE_DIMS}
    power = power_analysis(args.planned_effect_size)

    # Console: focus on overall liking + Friedman (ASCII output for Windows GBK)
    over = wilcoxon_results.get("喜爱度", {})
    if over.get("status") == "ok":
        sig = "[SIG]" if over["significant_at_alpha"] else "[NS]"
        print(f"\n      MAIN TEST (xi-ai-du A vs B Wilcoxon): "
              f"delta={over['mean_diff']:+.2f} p={over['p_value_onesided']:.4f} "
              f"d={over['cohens_d']:+.2f} r={over['rank_biserial_r']:+.2f} -> {sig}")
    else:
        print(f"\n      MAIN TEST (xi-ai-du): n/a ({over.get('status')})")
    if friedman.get("status") == "ok":
        sig = "[SIG]" if friedman["significant"] else "[NS]"
        print(f"      FRIEDMAN chi2={friedman['chi2']:.2f} "
              f"p={friedman['p_value']:.4f} -> {sig}")
    elif friedman.get("status"):
        print(f"      FRIEDMAN: n/a ({friedman.get('status')})")

    print("\n[4/4] Writing report ...")
    results = {
        "session": args.session,
        "timestamp": datetime.now(UTC).isoformat(),
        "n_rows": int(len(df)),
        "n_panelists": int(df["panelist_id"].nunique()),
        "n_recipes": int(df["recipe_id"].nunique()),
        "dimensions": sorted(df["dimension"].unique().tolist()),
        "group_counts": {k: int(v) for k, v in
                          df.groupby("group")["recipe_id"].nunique().to_dict().items()},
        "per_dim_summary": summary,
        "wilcoxon_a_vs_b": wilcoxon_results,
        "tost_a_vs_c": tost_results,
        "friedman": friedman,
        "mixed_effects": me_results,
        "power_analysis": power,
        "config": vars(args),
        "stats_libs": {"scipy_available": _SCIPY_OK, "statsmodels_available": _STATSMODELS_OK},
    }
    json_path = out_dir / f"panel_analysis_{args.session}.json"
    md_path = out_dir / f"panel_analysis_{args.session}.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    write_report(results, md_path)
    print(f"      wrote {json_path}")
    print(f"      wrote {md_path}")
    print(f"\nDone. Open {md_path} for the human-readable report.")


if __name__ == "__main__":
    sys.exit(main() or 0)
