"""Ingest reviews from Weibo Open API.

Requires:
    pip install -e '.[scrape]'
    export WEIBO_ACCESS_TOKEN=...   (obtain via open.weibo.com OAuth2)

Usage:
    python scripts/ingest_weibo.py --keywords "奶茶,茶饮新品" --max-records 500 --shard weibo_w1
    python scripts/ingest_weibo.py --keywords "喜茶,奈雪" --brand "" --max-records 1000

See `beverage_ai/scrapers/adapters/weibo_api.py` docstring for auth setup
and the realistic-yield warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from beverage_ai.scrapers.adapters.weibo_api import WeiboAPIError, WeiboAPIScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    keywords: str = typer.Option("奶茶", help="Comma-separated search terms"),
    brand: str = typer.Option(None, help="Tag all records with this brand"),
    max_records: int = typer.Option(500, help="Cap on records (mind 150 req/h quota)"),
    shard: str = typer.Option("weibo_default"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    max_per_hour: int = typer.Option(150, help="Rate limit (Weibo free tier ≈ 150)"),
):
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    try:
        scraper = WeiboAPIScraper(max_per_hour=max_per_hour)
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)
    try:
        stats = runner.run(
            scraper, shard=shard, keywords=kw_list, brand=brand, max_records=max_records
        )
    except WeiboAPIError as e:
        typer.echo(f"WEIBO API ERROR: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(
        f"[weibo] shard={shard}  written_new={stats.written}  "
        f"total_in_shard={store.read(shard).shape[0]}"
    )


if __name__ == "__main__":
    sys.exit(app() or 0)
