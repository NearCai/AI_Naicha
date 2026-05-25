"""Ingest reviews from a HuggingFace dataset.

Requires: pip install -e '.[hf]'

Usage examples:
    # Food-delivery reviews (most relevant — covers 奶茶/咖啡 discussion):
    python scripts/ingest_hf_reviews.py \
        --dataset XiangPan/waimai_10k \
        --shard hf_waimai \
        --keywords "奶茶,茶饮,拿铁,珍珠,芋圆,厚乳" \
        --max-records 5000

    # Generic Chinese sentiment for language transfer:
    python scripts/ingest_hf_reviews.py \
        --dataset seamew/ChnSentiCorp \
        --shard hf_chnsenticorp \
        --keywords "奶茶,茶,咖啡,饮料" \
        --max-records 3000

    # Multi-dataset batch:
    python scripts/ingest_hf_reviews.py --recipe scripts/hf_recipe.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml

from beverage_ai.scrapers.adapters.hf_dataset import HFDatasetScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option(None, help="HF dataset name, e.g. XiangPan/waimai_10k"),
    shard: str = typer.Option(None, help="Shard name under data/reviews/raw/"),
    split: str = typer.Option("train"),
    text_column: str = typer.Option(None, help="Auto-detected if omitted"),
    rating_column: str = typer.Option(None),
    keywords: str = typer.Option(
        "奶茶,茶饮,拿铁,鲜茶,珍珠,芋圆,厚乳,茶咖,果茶,鲜奶",
        help="Comma-separated; row must contain at least one to be kept",
    ),
    brand: str = typer.Option(None, help="Tag all records with this brand"),
    max_records: int = typer.Option(10000, help="Cap on records emitted"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    cache_dir: str = typer.Option(None, help="HF cache directory"),
    recipe: Path = typer.Option(
        None, help="YAML file listing multiple datasets to ingest in one run"
    ),
    no_streaming: bool = typer.Option(
        False, help="Disable streaming (downloads full dataset first)"
    ),
):
    """Ingest from one or more HuggingFace datasets."""
    if recipe is not None:
        sys.exit(_run_recipe(recipe, raw_dir, cache_dir))

    if not dataset or not shard:
        typer.echo("ERROR: --dataset and --shard required (or use --recipe)", err=True)
        raise typer.Exit(1)

    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    _ingest_one(
        dataset=dataset,
        shard=shard,
        split=split,
        text_column=text_column,
        rating_column=rating_column,
        keywords=kw_list,
        brand=brand,
        max_records=max_records,
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        streaming=not no_streaming,
    )


def _ingest_one(
    *, dataset, shard, split, text_column, rating_column,
    keywords, brand, max_records, raw_dir, cache_dir, streaming,
):
    scraper = HFDatasetScraper(
        dataset_name=dataset,
        split=split,
        text_column=text_column,
        rating_column=rating_column,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)
    stats = runner.run(
        scraper, shard=shard, keywords=keywords, brand=brand, max_records=max_records
    )
    typer.echo(
        f"[hf:{dataset}] shard={shard}  written_new={stats.written}  "
        f"requested={stats.requested}  total_in_shard={store.read(shard).shape[0]}"
    )


def _run_recipe(recipe_path: Path, raw_dir: Path, cache_dir: str | None) -> int:
    """Run a batch of datasets defined in YAML.

    Recipe format:
        - dataset: XiangPan/waimai_10k
          shard: hf_waimai
          split: train
          text_column: review
          rating_column: label
          keywords: [奶茶, 茶饮, 拿铁]
          max_records: 5000
        - dataset: seamew/ChnSentiCorp
          ...
    """
    with open(recipe_path, encoding="utf-8") as f:
        items = yaml.safe_load(f) or []
    typer.echo(f"running batch of {len(items)} datasets from {recipe_path}")
    failures = 0
    for item in items:
        try:
            _ingest_one(
                dataset=item["dataset"],
                shard=item["shard"],
                split=item.get("split", "train"),
                text_column=item.get("text_column"),
                rating_column=item.get("rating_column"),
                keywords=item.get("keywords"),
                brand=item.get("brand"),
                max_records=item.get("max_records", 10000),
                raw_dir=raw_dir,
                cache_dir=cache_dir,
                streaming=item.get("streaming", True),
            )
        except Exception as e:
            typer.echo(f"FAILED {item.get('dataset')}: {e}", err=True)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    app()
