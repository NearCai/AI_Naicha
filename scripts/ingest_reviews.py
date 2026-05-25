"""End-to-end review ingestion script.

  scrape (mock/local) → store raw parquet → extract aspects → cache

Usage:
    python scripts/ingest_reviews.py --source mock --shard mock_w1 --n 200
    python scripts/ingest_reviews.py --source local --path data/seed.csv --shard seed
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from beverage_ai.aspects.cache import AspectCache
from beverage_ai.aspects.extractor import get_default_extractor
from beverage_ai.aspects.pipeline import AspectExtractionPipeline
from beverage_ai.scrapers.adapters.local_file import LocalFileScraper
from beverage_ai.scrapers.adapters.mock import MockScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: str = typer.Option("mock", help="mock | local"),
    shard: str = typer.Option("default", help="Logical partition under data/reviews/raw/"),
    n: int = typer.Option(100, help="Max number of reviews to ingest"),
    path: Path = typer.Option(None, help="Required for source=local"),
    keywords: str = typer.Option(None, help="Comma-separated keyword filter"),
    brand: str = typer.Option(None, help="Brand filter"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw"), help="Raw store dir"),
    cache_db: Path = typer.Option(
        Path("data/reviews/aspects_cache.duckdb"), help="Aspect cache DB"
    ),
    self_consistency: int = typer.Option(1, help="LLM self-consistency votes (>=1)"),
    cost_ceiling_usd: float = typer.Option(
        None, help="Stop extraction when cost exceeds this"
    ),
    seed: int = typer.Option(42, help="Mock scraper seed"),
):
    # ----- pick scraper -----
    if source == "mock":
        scraper = MockScraper(seed=seed)
    elif source == "local":
        if not path:
            typer.echo("ERROR: --path required for source=local", err=True)
            raise typer.Exit(1)
        scraper = LocalFileScraper(path=path)
    else:
        typer.echo(f"ERROR: unknown source {source!r}", err=True)
        raise typer.Exit(1)

    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None

    # ----- step 1: scrape + persist raw -----
    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)
    scrape_stats = runner.run(
        scraper, shard=shard, keywords=kw_list, brand=brand, max_records=n
    )
    typer.echo(
        f"[scrape] source={scrape_stats.source} requested={scrape_stats.requested} "
        f"written_new={scrape_stats.written}"
    )

    # ----- step 2: extract aspects -----
    extractor = get_default_extractor()
    typer.echo(f"[extract] extractor={extractor.version}")
    with AspectCache(cache_db) as cache:
        pipeline = AspectExtractionPipeline(
            extractor=extractor, cache=cache, self_consistency=self_consistency
        )
        stats = pipeline.run_on_store(
            store, shard=shard, cost_ceiling_usd=cost_ceiling_usd
        )
        typer.echo("[extract] stats:")
        for k, v in stats.to_dict().items():
            typer.echo(f"  {k}: {v}")
        typer.echo(
            f"[cache] total cached across versions: {cache.count()}; "
            f"versions: {cache.list_versions()}"
        )

    sys.exit(0 if stats.errors == 0 else 2)


if __name__ == "__main__":
    app()
