"""Ingest a partner / manual export file (csv / jsonl / json) into the raw store.

Thin wrapper around `LocalFileScraper` with validation + reporting.
See data/reviews/templates/README.md for the expected schema.

Usage:
    python scripts/ingest_partner.py \
        --path data/inbox/2026_q2_export.csv \
        --shard partner_q2 \
        --source-tag partner_brandX
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from beverage_ai.scrapers.adapters.local_file import LocalFileScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: Path = typer.Option(..., help="Path to .csv / .tsv / .jsonl / .json"),
    shard: str = typer.Option(..., help="Shard name under data/reviews/raw/"),
    source_tag: str = typer.Option(
        "partner", help="default `source` value if file rows omit it"
    ),
    brand: str = typer.Option(None, help="Filter to one brand"),
    keywords: str = typer.Option(None, help="Comma-separated keyword filter"),
    max_records: int = typer.Option(100000),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
):
    if not path.exists():
        typer.echo(f"ERROR: {path} not found", err=True)
        raise typer.Exit(1)

    scraper = LocalFileScraper(path=path, default_source=source_tag)
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)

    stats = runner.run(
        scraper, shard=shard, keywords=kw_list, brand=brand, max_records=max_records
    )
    total = store.read(shard).shape[0]
    typer.echo(
        f"[partner] shard={shard}  source_tag={source_tag}  "
        f"written_new={stats.written}  total_in_shard={total}"
    )


if __name__ == "__main__":
    sys.exit(app() or 0)
