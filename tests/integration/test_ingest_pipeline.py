"""End-to-end ingest test: mock scrape → store → extract → cache → query."""
from __future__ import annotations

from beverage_ai.aspects.cache import AspectCache
from beverage_ai.aspects.extractor import MockAspectExtractor
from beverage_ai.aspects.pipeline import AspectExtractionPipeline, aspects_to_dataframe
from beverage_ai.scrapers.adapters.mock import MockScraper
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore


def test_full_ingest_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_db = tmp_path / "cache.duckdb"

    # 1. scrape
    store = RawReviewStore(raw_dir)
    stats_scrape = ScrapeRunner(store).run(
        MockScraper(seed=0), shard="t1", max_records=30
    )
    assert stats_scrape.written == 30

    # 2. extract
    with AspectCache(cache_db) as cache:
        pipeline = AspectExtractionPipeline(
            extractor=MockAspectExtractor(), cache=cache, self_consistency=1
        )
        stats = pipeline.run_on_store(store, shard="t1")
        assert stats.total_reviews == 30
        assert stats.extracted == 30
        assert stats.cache_hits == 0
        assert stats.errors == 0
        assert stats.cost_usd == 0.0   # mock is free

        # 3. second run should hit cache entirely
        stats2 = pipeline.run_on_store(store, shard="t1")
        assert stats2.cache_hits == 30
        assert stats2.extracted == 0

        # 4. materialize
        df = aspects_to_dataframe(cache)
        assert len(df) == 30
        # All 喜爱度 should be filled (mock cues are common in fixtures)
        n_with_pref = df["aspect_喜爱度"].notna().sum()
        assert n_with_pref >= 20


def test_self_consistency_aggregates(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_db = tmp_path / "cache.duckdb"

    store = RawReviewStore(raw_dir)
    ScrapeRunner(store).run(MockScraper(seed=1), shard="sc", max_records=5)

    with AspectCache(cache_db) as cache:
        pipeline = AspectExtractionPipeline(
            extractor=MockAspectExtractor(), cache=cache, self_consistency=3
        )
        stats = pipeline.run_on_store(store, shard="sc")
        assert stats.extracted == 5
        # Version should be tagged with |sc
        versions = cache.list_versions()
        assert any(v.endswith("|sc") for v, _ in versions)
