"""Interactive review collection CLI.

For Path A source #4: manual collection (realistic for 100-500 entries).
You sit with a notebook / screenshots and feed them to this tool one at a time.

Usage:
    python scripts/collect_manual.py --shard manual_2026_q2
    python scripts/collect_manual.py --shard manual_xhs --resume

Each session appends to `data/reviews/raw/<shard>/raw_reviews.parquet`.
`Ctrl-C` saves what you've entered so far and exits cleanly.

Tips:
  - Keep brand / SKU consistent (use vocab aliases where possible)
  - Customization free-text is fine ("三分糖去冰加芋圆")
  - Source URL recommended if pasted from a public post
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import typer

from beverage_ai.scrapers.base import ReviewRecord, make_review_id, normalize_text
from beverage_ai.scrapers.store import RawReviewStore

app = typer.Typer(add_completion=False)


KNOWN_BRANDS = [
    "喜茶", "奈雪", "茶颜悦色", "蜜雪冰城", "书亦烧仙草",
    "一点点", "古茗", "CoCo都可", "霸王茶姬", "茶百道",
    "瑞幸", "Manner", "M Stand", "Tims",
]


@app.command()
def main(
    shard: str = typer.Option(..., help="Shard name under data/reviews/raw/"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    default_source: str = typer.Option(
        "manual", help="What to put in record.source if you don't override per-entry"
    ),
    resume: bool = typer.Option(False, help="Skip startup banner; just start collecting"),
):
    """Interactive collection loop."""
    store = RawReviewStore(raw_dir)
    existing = store.read(shard).shape[0] if shard in store.list_shards() else 0

    if not resume:
        typer.echo("=" * 60)
        typer.echo(f"Manual review collection — shard={shard}")
        typer.echo(f"Existing in shard: {existing}")
        typer.echo("Common brands: " + " / ".join(KNOWN_BRANDS[:10]))
        typer.echo("Press Ctrl-C anytime to save and quit.")
        typer.echo("=" * 60)

    pending: list[ReviewRecord] = []
    session_n = 0

    try:
        while True:
            session_n += 1
            typer.echo(f"\n--- Entry #{existing + session_n} ---")
            try:
                record = _collect_one(default_source=default_source)
            except _SkipEntry:
                continue
            if record is None:
                break

            # Preview
            typer.echo("\n  Preview:")
            typer.echo(f"    source: {record.source}")
            typer.echo(f"    brand:  {record.brand or '(none)'}")
            typer.echo(f"    sku:    {record.sku or '(none)'}")
            typer.echo(f"    rating: {record.rating if record.rating is not None else '(none)'}")
            typer.echo(f"    custom: {record.customization_raw or '(none)'}")
            typer.echo(f"    text:   {record.text[:80]}{'...' if len(record.text) > 80 else ''}")

            confirm = typer.prompt("  Save? (y/n/edit)", default="y")
            if confirm.strip().lower().startswith("y"):
                pending.append(record)
                typer.echo(f"  ✓ queued ({len(pending)} pending)")
            elif confirm.strip().lower().startswith("e"):
                typer.echo("  edit not supported yet, discarded")
                session_n -= 1
            else:
                typer.echo("  discarded")
                session_n -= 1

            if len(pending) >= 5:
                _flush(store, shard, pending)
                pending.clear()

            again = typer.prompt("\nAnother? (y/n)", default="y")
            if not again.strip().lower().startswith("y"):
                break

    except (KeyboardInterrupt, EOFError):
        typer.echo("\n\n[interrupted] saving pending entries...")

    if pending:
        _flush(store, shard, pending)
    final = store.read(shard).shape[0]
    typer.echo(f"\nDone. Shard '{shard}' now has {final} records (was {existing}).")


class _SkipEntry(Exception):
    pass


def _collect_one(*, default_source: str) -> ReviewRecord | None:
    """Prompt for one record. Returns None if user wants to exit."""
    text_raw = typer.prompt(
        "Review text (empty to finish)", default="", show_default=False
    )
    text_raw = text_raw.strip()
    if not text_raw:
        return None
    text = normalize_text(text_raw)

    brand = typer.prompt("Brand (e.g. 喜茶)", default="", show_default=False).strip() or None
    sku = typer.prompt("SKU / product name", default="", show_default=False).strip() or None
    customization = typer.prompt(
        "Customization (e.g. 三分糖去冰加芋圆)", default="", show_default=False
    ).strip() or None

    rating_raw = typer.prompt("Rating 1-5", default="", show_default=False).strip()
    rating: float | None = None
    if rating_raw:
        try:
            v = float(rating_raw)
            if 1 <= v <= 5:
                rating = round(v, 1)
        except ValueError:
            typer.echo("  (couldn't parse rating, skipping field)", err=True)

    source = typer.prompt(
        "Source tag", default=default_source, show_default=True
    ).strip() or default_source
    source_url = typer.prompt(
        "Source URL (optional)", default="", show_default=False
    ).strip() or None

    return ReviewRecord(
        review_id=make_review_id(source, brand, text),
        source=source,
        brand=brand,
        sku=sku,
        text=text,
        customization_raw=customization,
        rating=rating,
        source_url=source_url,
        scraped_at=datetime.now(UTC),
        metadata={"manual": True},
    )


def _flush(store: RawReviewStore, shard: str, records: list[ReviewRecord]) -> None:
    n = store.append(shard, records)
    typer.echo(f"  [flushed {n} new to shard {shard}]")


if __name__ == "__main__":
    sys.exit(app() or 0)
