"""Path A data-acquisition dashboard.

Shows what's been collected per source, available capacity per route, and
the recommended next step. Doesn't itself fetch — it inspects current state
and tells you which ingest script to run.

Per 技术方案书 §3.3.1 路径 A target: 50,000–100,000 reviews for GNN Stage 1
predicting pretraining.

Usage:
    python scripts/build_path_a_dataset.py
    python scripts/build_path_a_dataset.py --target 80000
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import typer

from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


def _hf_available() -> bool:
    """Importing `datasets` can fail with OSError if a transitive torch DLL
    is broken; treat any import failure as 'not available'."""
    try:
        importlib.import_module("datasets")
        return True
    except Exception:
        return False


def _weibo_token_available() -> bool:
    return bool(os.environ.get("WEIBO_ACCESS_TOKEN"))


def _httpx_available() -> bool:
    try:
        importlib.import_module("httpx")
        return True
    except Exception:
        return False


@app.command()
def main(
    target: int = typer.Option(50000, help="Total reviews target (§3.3.1)"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
):
    store = RawReviewStore(raw_dir)
    typer.echo("=" * 70)
    typer.echo("Path A — Review Data Acquisition Dashboard")
    typer.echo("=" * 70)

    # ----- current state -----
    shards = store.list_shards()
    if not shards:
        typer.echo("\n(no shards yet)")
        total = 0
        by_source: dict[str, int] = {}
    else:
        df = store.read()
        total = len(df)
        by_source = dict(df["source"].value_counts())
        typer.echo(f"\nShards: {len(shards)}  Total reviews: {total:,}")
        typer.echo("\nBy source:")
        for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
            tag = _categorize_source(src)
            typer.echo(f"  {src:<30s}  {n:>8,d}   ({tag})")

    progress = min(100.0, 100.0 * total / target)
    bar_len = 40
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    typer.echo(f"\nProgress toward {target:,}: |{bar}| {progress:.1f}%")

    # ----- source availability -----
    typer.echo("\n--- Source availability ---")
    rows = [
        ("Mock (synthetic)",       True,                    "beverage-ai scrape ingest --source mock --shard mock_x"),
        ("Local file (partner)",   True,                    "python scripts/ingest_partner.py --path X.csv --shard partner_x"),
        ("Manual REPL",            True,                    "python scripts/collect_manual.py --shard manual_x"),
        ("HuggingFace datasets",   _hf_available(),         "python scripts/ingest_hf_reviews.py --recipe scripts/hf_recipe.yaml"),
        ("Weibo Open API",         _weibo_token_available() and _httpx_available(),
                                                            "python scripts/ingest_weibo.py --keywords '奶茶,茶饮' --max-records 500"),
    ]
    for name, available, cmd in rows:
        flag = "✓" if available else "✗"
        typer.echo(f"  [{flag}] {name:<24s}  {cmd}")
        if not available:
            if name.startswith("HuggingFace"):
                typer.echo("       → pip install -e '.[hf]'")
            elif name.startswith("Weibo"):
                typer.echo("       → set WEIBO_ACCESS_TOKEN env var; pip install -e '.[scrape]'")

    # ----- gap analysis + recommendation -----
    remaining = max(target - total, 0)
    typer.echo("\n--- Gap analysis ---")
    typer.echo(f"  Current: {total:,}")
    typer.echo(f"  Target:  {target:,}")
    typer.echo(f"  Gap:     {remaining:,}")

    if remaining <= 0:
        typer.echo("\n✅ Target reached. Proceed to:")
        typer.echo("  beverage-ai aspects extract     # LLM aspect extraction")
        typer.echo("  python scripts/train_sensory_gnn_stage1.py   # (TODO)")
        return

    plan = _recommend(remaining, by_source)
    typer.echo("\n--- Recommended plan to close gap ---")
    for step in plan:
        typer.echo(f"  {step}")


def _categorize_source(src: str) -> str:
    if src == "mock":             return "synthetic, not for training"
    if src.startswith("hf:"):     return "HuggingFace dataset, OK"
    if src == "weibo_api":        return "Weibo, OK"
    if src.startswith("partner"): return "partner / brand, OK"
    if src.startswith("manual"):  return "manual, OK"
    if src == "local_file":       return "local / custom"
    return "unknown"


def _recommend(gap: int, by_source: dict[str, int]) -> list[str]:
    hf_have = sum(v for k, v in by_source.items() if k.startswith("hf:"))
    weibo_have = by_source.get("weibo_api", 0)
    manual_have = sum(v for k, v in by_source.items() if k.startswith("manual"))
    partner_have = sum(v for k, v in by_source.items() if k.startswith("partner"))

    plan: list[str] = []
    # Step 1: HF if we have capacity
    if hf_have < 30000:
        plan.append(
            f"1. HF datasets: run `python scripts/ingest_hf_reviews.py --recipe scripts/hf_recipe.yaml` "
            f"to add ~5-30k reviews (currently {hf_have:,}). Realistic: aim for 20-30k from 2-3 datasets."
        )
    # Step 2: Weibo
    if weibo_have < 5000:
        plan.append(
            f"2. Weibo API: register at https://open.weibo.com, get access token, run "
            f"`python scripts/ingest_weibo.py --keywords '奶茶,茶饮新品,鲜奶茶' --max-records 2000` "
            f"(currently {weibo_have:,}). Realistic: 1-3k/day depending on quota."
        )
    # Step 3: Partner
    plan.append(
        f"3. Partner data: reach out to brands or use existing exports — "
        f"`python scripts/ingest_partner.py --path data/inbox/X.csv --shard partner_X` "
        f"(currently {partner_have:,})."
    )
    # Step 4: Manual
    plan.append(
        f"4. Manual: `python scripts/collect_manual.py --shard manual_q3` for the "
        f"last 100-500 high-quality records (currently {manual_have:,}). "
        f"NOT a scalable route to 50k."
    )
    plan.append(
        "5. Scraper (last resort): see docs/SCRAPING_NOTICE.md for legal warnings."
    )
    return plan


if __name__ == "__main__":
    app()
