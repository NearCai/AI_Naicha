"""Spot-check LLM aspect extractions.

Prints a random sample of (raw review, extracted aspects, customization)
so a human can verify the extractor is reasonable.
Per 技术方案书 §4.3 数据质量控制.
"""
from __future__ import annotations

import random
from pathlib import Path

import typer

from beverage_ai.aspects.cache import AspectCache
from beverage_ai.aspects.pipeline import aspects_to_dataframe
from beverage_ai.aspects.schema import ALL_DIMS
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    cache_db: Path = typer.Option(Path("data/reviews/aspects_cache.duckdb")),
    extractor_version: str = typer.Option(None, help="Filter by version"),
    n: int = typer.Option(20, help="Sample size"),
    seed: int = typer.Option(42),
):
    store = RawReviewStore(raw_dir)
    raw_df = store.read()
    if raw_df.empty:
        typer.echo("No raw reviews found.", err=True)
        raise typer.Exit(1)

    with AspectCache(cache_db) as cache:
        aspects_df = aspects_to_dataframe(cache, extractor_version)
    if aspects_df.empty:
        typer.echo("No cached aspects found.", err=True)
        raise typer.Exit(1)

    # Join on review_id
    merged = aspects_df.merge(
        raw_df[["review_id", "brand", "sku", "text"]],
        on="review_id", how="inner",
    )
    if merged.empty:
        typer.echo("No overlap between cache and raw store.", err=True)
        raise typer.Exit(1)

    rng = random.Random(seed)
    sample = merged.sample(min(n, len(merged)), random_state=seed)

    typer.echo(f"=== Audit sample ({len(sample)} of {len(merged)}) ===\n")
    for i, row in enumerate(sample.itertuples(index=False), 1):
        typer.echo(f"--- #{i} review_id={row.review_id} ({row.brand or '?'}) ---")
        typer.echo(f"  text: {row.text}")
        typer.echo(f"  extractor: {row.extractor_version}  conf={row.confidence:.2f}")
        nonnull = [
            (dim, getattr(row, f"aspect_{dim}"))
            for dim in ALL_DIMS
            if getattr(row, f"aspect_{dim}") is not None
        ]
        if nonnull:
            typer.echo("  aspects:")
            for dim, score in nonnull:
                typer.echo(f"    {dim}: {score:.2f}")
        typer.echo(
            f"  sugar={row.sugar_level or '-'} ice={row.ice_level or '-'} "
            f"toppings=[{row.toppings or ''}] size={row.size or '-'}"
        )
        typer.echo("")

    # Aggregate distribution
    typer.echo("=== Aspect coverage (% non-null) ===")
    for dim in ALL_DIMS:
        col = f"aspect_{dim}"
        if col in aspects_df.columns:
            pct = 100 * aspects_df[col].notna().mean()
            typer.echo(f"  {dim}: {pct:.1f}%")


if __name__ == "__main__":
    app()
