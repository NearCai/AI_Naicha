"""Command-line interface for beverage_ai.

Usage:
    beverage-ai health <recipe.yaml>
    beverage-ai generate --request "夏季年轻女性低糖" --n 10
    beverage-ai pipeline --request "夏季年轻女性低糖" --top-k 5
    beverage-ai vocab list [--category tea_base]
    beverage-ai scrape ingest --source mock --shard mock_w1 --n 200
    beverage-ai aspects extract --shard mock_w1
    beverage-ai aspects audit --n 10
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .aspects.cache import AspectCache
from .aspects.extractor import get_default_extractor
from .aspects.pipeline import AspectExtractionPipeline, aspects_to_dataframe
from .aspects.schema import ALL_DIMS
from .constraints.checker import check_constraints, is_feasible
from .ingredients.vocab import load_default_vocab
from .pipeline.end_to_end import run_pipeline
from .planner.llm_planner import get_default_planner
from .priors.engine import load_default_engine
from .recipes.generator import RecipeGenerator
from .recipes.schema import Recipe
from .scrapers.adapters.local_file import LocalFileScraper
from .scrapers.adapters.mock import MockScraper
from .scrapers.runner import ScrapeRunner
from .scrapers.store import RawReviewStore
from .simulators.health.calculator import compute_nutrition

app = typer.Typer(
    no_args_is_help=True,
    help="beverage_ai — closed-loop AI for tea drink R&D (v1)",
)

scrape_app = typer.Typer(no_args_is_help=True, help="Review scraping commands")
aspects_app = typer.Typer(no_args_is_help=True, help="LLM aspect extraction commands")
app.add_typer(scrape_app, name="scrape")
app.add_typer(aspects_app, name="aspects")


@app.command()
def health(
    recipe_path: Path = typer.Argument(..., help="YAML/JSON file describing a Recipe"),
    sugar_limit: float = typer.Option(30.0, help="Override sugar limit (g)"),
):
    """Compute nutrition + run constraints on a single recipe."""
    with open(recipe_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) if str(recipe_path).endswith(("yaml", "yml")) else json.load(f)
    recipe = Recipe(**data)
    vocab = load_default_vocab()
    nutrition = compute_nutrition(recipe, vocab)
    violations = check_constraints(
        recipe, nutrition,
        targets={"sugar_limit_g": sugar_limit, "trans_fat_zero": False},
        vocab=vocab,
    )
    typer.echo("=== Nutrition ===")
    typer.echo(json.dumps(nutrition, ensure_ascii=False, indent=2))
    typer.echo(f"\n=== Constraints ({'OK' if is_feasible(violations) else 'INFEASIBLE'}) ===")
    if not violations:
        typer.echo("(no violations)")
    else:
        for v in violations:
            typer.echo(f"  [{v.severity.upper():4s}] {v.code}: {v.message}")


@app.command()
def generate(
    request: str = typer.Option(..., "--request", "-r"),
    n: int = typer.Option(10, "--n", help="Number of candidates"),
    seed: int = typer.Option(42, "--seed"),
):
    """Run planner + generator and dump candidate recipes (no scoring)."""
    planner = get_default_planner()
    spec = planner.plan(request)
    typer.echo("=== Planner spec ===")
    typer.echo(json.dumps(spec, ensure_ascii=False, indent=2))

    vocab = load_default_vocab()
    prior = load_default_engine()
    gen = RecipeGenerator(vocab, prior, seed=seed)
    candidates = gen.generate(spec, n_candidates=n)
    typer.echo(f"\n=== Generated {len(candidates)} unique candidates ===")
    for i, r in enumerate(candidates):
        typer.echo(f"\n--- Candidate {i+1} ({r.recipe_id}) ---")
        typer.echo(f"  style: {r.style}  cup: {r.cup_volume_ml}ml  sugar: {r.sugar_level}")
        for ing_id, mass in r.ingredients.items():
            name = vocab.get(ing_id).name_zh if ing_id in vocab else ing_id
            typer.echo(f"  {ing_id} ({name}): {mass}g")


@app.command()
def pipeline(
    request: str = typer.Option(..., "--request", "-r"),
    top_k: int = typer.Option(5, "--top-k"),
    n_candidates: int = typer.Option(200, "--n-candidates"),
    kappa: float = typer.Option(1.0, "--kappa", help="LCB conservativeness"),
    seed: int = typer.Option(42, "--seed"),
    record: bool = typer.Option(False, "--record", help="Persist to DuckDB"),
):
    """End-to-end pipeline: request → Top-K candidate recipes."""
    result = run_pipeline(
        user_request=request,
        top_k=top_k,
        n_candidates=n_candidates,
        kappa=kappa,
        seed=seed,
        record=record,
    )
    typer.echo("=== Spec ===")
    typer.echo(json.dumps(result.spec, ensure_ascii=False, indent=2))
    typer.echo("\n=== Stats ===")
    typer.echo(f"  generated: {result.n_generated}")
    typer.echo(f"  feasible: {result.n_feasible}")
    typer.echo(f"  Pareto front: {result.n_pareto}")
    typer.echo(f"  elapsed: {result.elapsed_sec:.2f}s")

    typer.echo(f"\n=== Top {len(result.top_recipes)} ===")
    vocab = load_default_vocab()
    for i, c in enumerate(result.top_recipes, start=1):
        r = c["recipe"]
        means = c["means"]
        nut = c["nutrition"]
        typer.echo(f"\n--- #{i}  {r['style']} {r['cup_volume_ml']}ml {r['sugar_level']} ---")
        typer.echo(f"  Recipe: {r['recipe_id']}")
        typer.echo(f"  喜爱度: {means['preference']:.2f} ± {c['sigmas']['preference']:.2f}")
        typer.echo(f"  销量分: {means['sales_proxy']:.1f} ± {c['sigmas']['sales_proxy']:.1f}")
        typer.echo(f"  复购分: {means['repurchase']:.3f}")
        typer.echo(f"  成本: ¥{means['cost_cny']:.2f}")
        typer.echo(f"  含糖: {nut['sugar_g']:.1f}g  热量: {nut['energy_kcal']:.0f}kcal")
        typer.echo(f"  咖啡因: {nut['caffeine_mg']:.0f}mg")
        if nut["allergens"]:
            typer.echo(f"  致敏原: {', '.join(nut['allergens'])}")
        typer.echo("  原料:")
        for ing_id, mass in r["ingredients"].items():
            name = vocab.get(ing_id).name_zh if ing_id in vocab else ing_id
            typer.echo(f"    - {name}: {mass}g")


@app.command("vocab")
def vocab_cmd(
    action: str = typer.Argument("list", help="list | get | count"),
    target: str = typer.Argument(None, help="ingredient id (for 'get')"),
    category: str = typer.Option(None, "--category", "-c"),
):
    """Inspect the ingredient vocabulary."""
    vocab = load_default_vocab()
    if action == "count":
        typer.echo(f"Total: {len(vocab)} ingredients")
        from collections import Counter
        c = Counter(i.category for i in vocab.all())
        for cat, n in sorted(c.items()):
            typer.echo(f"  {cat}: {n}")
    elif action == "list":
        items = vocab.by_category(category) if category else vocab.all()
        for i in items:
            typer.echo(f"  {i.id}  -  {i.name_zh} ({i.category})")
    elif action == "get":
        if not target:
            typer.echo("provide id, e.g.: beverage-ai vocab get tea_jinxuan", err=True)
            raise typer.Exit(1)
        ing = vocab.get(target)
        typer.echo(ing.model_dump_json(indent=2))
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(1)


# =============================================================================
# scrape commands
# =============================================================================

@scrape_app.command("ingest")
def scrape_ingest(
    source: str = typer.Option("mock", help="mock | local"),
    shard: str = typer.Option("default", help="Logical shard under data/reviews/raw/"),
    n: int = typer.Option(100, help="Max number of records"),
    path: Path = typer.Option(None, help="Required for source=local"),
    keywords: str = typer.Option(None, help="Comma-separated keyword filter"),
    brand: str = typer.Option(None, help="Brand filter"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    seed: int = typer.Option(42),
):
    """Scrape reviews and persist to data/reviews/raw/<shard>/."""
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
    store = RawReviewStore(raw_dir)
    runner = ScrapeRunner(store)
    stats = runner.run(
        scraper, shard=shard, keywords=kw_list, brand=brand, max_records=n
    )
    typer.echo(
        f"source={stats.source}  requested={stats.requested}  "
        f"written_new={stats.written}  shard={shard}"
    )


@scrape_app.command("stats")
def scrape_stats(raw_dir: Path = typer.Option(Path("data/reviews/raw"))):
    """Show counts by shard / source / brand."""
    store = RawReviewStore(raw_dir)
    shards = store.list_shards()
    if not shards:
        typer.echo("No raw shards found.")
        return
    typer.echo(f"Total: {store.count()} reviews across {len(shards)} shard(s)")
    df = store.read()
    typer.echo("\nBy source:")
    typer.echo(df.groupby("source").size().to_string())
    if "brand" in df.columns:
        typer.echo("\nTop 10 brands:")
        typer.echo(df["brand"].value_counts().head(10).to_string())


# =============================================================================
# aspects commands
# =============================================================================

@aspects_app.command("extract")
def aspects_extract(
    shard: str = typer.Option(None, help="Limit to a specific shard"),
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    cache_db: Path = typer.Option(Path("data/reviews/aspects_cache.duckdb")),
    self_consistency: int = typer.Option(1, help=">=1; 3 recommended for production"),
    cost_ceiling_usd: float = typer.Option(None),
    limit: int = typer.Option(None, help="Cap number of reviews to process"),
):
    """Extract aspects for all uncached reviews."""
    store = RawReviewStore(raw_dir)
    extractor = get_default_extractor()
    typer.echo(f"extractor: {extractor.version}")
    with AspectCache(cache_db) as cache:
        pipeline = AspectExtractionPipeline(
            extractor=extractor, cache=cache, self_consistency=self_consistency
        )
        stats = pipeline.run_on_store(
            store, shard=shard, limit=limit, cost_ceiling_usd=cost_ceiling_usd
        )
        for k, v in stats.to_dict().items():
            typer.echo(f"  {k}: {v}")
        if stats.error_messages:
            typer.echo("First few errors:")
            for e in stats.error_messages[:5]:
                typer.echo(f"  {e}")


@aspects_app.command("audit")
def aspects_audit(
    raw_dir: Path = typer.Option(Path("data/reviews/raw")),
    cache_db: Path = typer.Option(Path("data/reviews/aspects_cache.duckdb")),
    extractor_version: str = typer.Option(None),
    n: int = typer.Option(10),
    seed: int = typer.Option(42),
):
    """Sample N (raw, extracted) pairs for human review."""
    store = RawReviewStore(raw_dir)
    raw_df = store.read()
    if raw_df.empty:
        typer.echo("No raw reviews.", err=True); raise typer.Exit(1)

    with AspectCache(cache_db) as cache:
        aspects_df = aspects_to_dataframe(cache, extractor_version)
    if aspects_df.empty:
        typer.echo("No cached aspects.", err=True); raise typer.Exit(1)

    merged = aspects_df.merge(
        raw_df[["review_id", "brand", "sku", "text"]], on="review_id", how="inner"
    )
    sample = merged.sample(min(n, len(merged)), random_state=seed)

    import math
    typer.echo(f"=== Audit ({len(sample)} of {len(merged)}) ===\n")
    for i, row in enumerate(sample.itertuples(index=False), 1):
        typer.echo(f"--- #{i}  {row.brand or '?'} / {row.sku or '?'}  conf={row.confidence:.2f}")
        typer.echo(f"  text: {row.text}")
        ns = []
        for d in ALL_DIMS:
            v = getattr(row, f"aspect_{d}")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ns.append((d, v))
        if ns:
            typer.echo(f"  aspects: {', '.join(f'{d}={s:.2f}' for d, s in ns)}")
        else:
            typer.echo("  aspects: (none mentioned)")
        typer.echo(
            f"  custom: sugar={row.sugar_level or '-'}  ice={row.ice_level or '-'}  "
            f"toppings=[{row.toppings or ''}]  size={row.size or '-'}\n"
        )


@aspects_app.command("stats")
def aspects_stats(
    cache_db: Path = typer.Option(Path("data/reviews/aspects_cache.duckdb")),
):
    """Cache size + cost + per-version count."""
    with AspectCache(cache_db) as cache:
        typer.echo(f"Total cached: {cache.count()}")
        typer.echo(f"Total estimated cost: ${cache.total_cost_usd():.4f}")
        typer.echo("\nBy version:")
        for ver, n in cache.list_versions():
            typer.echo(f"  {ver}: {n} entries (${cache.total_cost_usd(ver):.4f})")


if __name__ == "__main__":
    app()
