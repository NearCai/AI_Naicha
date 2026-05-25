"""Generate synthetic tea-drink reviews via Claude API.

Requires:
    pip install -e '.[llm]'
    export ANTHROPIC_API_KEY=...

Usage:
    python scripts/ingest_llm_synthetic.py --n 1000 --shard llm_synth_w1
    python scripts/ingest_llm_synthetic.py --n 5000 --cost-ceiling-usd 15 --shard llm_synth_bulk

Cost estimate:
    5000 reviews ÷ 8 per batch × $0.002 ≈ $1.25 (Haiku)
    NOTE: synthetic data IS clearly labeled in `source` field
    (e.g. `llm_synthetic:claude-haiku-4-5-...`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from beverage_ai.scrapers.adapters.llm_synthetic import LLMSyntheticScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(500, help="Target number of reviews"),
    shard: str = typer.Option("llm_synth_default"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    model: str = typer.Option("claude-haiku-4-5-20251001"),
    batch_size: int = typer.Option(8, help="Reviews per API call"),
    cost_ceiling_usd: float = typer.Option(
        5.0, help="Hard cap; stops generating when exceeded"
    ),
    brand: str = typer.Option(None, help="Only generate for this brand"),
    seed: int = typer.Option(42),
):
    try:
        scraper = LLMSyntheticScraper(
            model=model,
            batch_size=batch_size,
            cost_ceiling_usd=cost_ceiling_usd,
            seed=seed,
        )
    except ImportError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)
    stats = runner.run(scraper, shard=shard, brand=brand, max_records=n)
    typer.echo(
        f"[llm_synthetic] shard={shard}  generated={stats.requested}  "
        f"written_new={stats.written}  estimated_cost_usd=${scraper.total_cost_usd:.4f}"
    )


if __name__ == "__main__":
    sys.exit(app() or 0)
