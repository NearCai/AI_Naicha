"""Tests for MockScraper + store layer."""
from __future__ import annotations

from beverage_ai.scrapers.adapters.mock import MockScraper
from beverage_ai.scrapers.base import ReviewRecord, make_review_id
from beverage_ai.scrapers.runner import ScrapeRunner
from beverage_ai.scrapers.store import RawReviewStore


def test_mock_scraper_yields_requested_count():
    scraper = MockScraper(seed=0)
    records = list(scraper.scrape(max_records=25))
    assert len(records) == 25
    for r in records:
        assert isinstance(r, ReviewRecord)
        assert r.source == "mock"
        assert r.brand and r.sku and r.text
        assert r.text.startswith(("今天", "种草", "下午", "排了", "喜茶", "奈雪",
                                  "茶颜悦色", "蜜雪冰城", "书亦烧仙草", "一点点",
                                  "古茗", "CoCo都可")) or "{" not in r.text


def test_mock_scraper_seed_deterministic():
    a = list(MockScraper(seed=42).scrape(max_records=10))
    b = list(MockScraper(seed=42).scrape(max_records=10))
    assert [r.text for r in a] == [r.text for r in b]


def test_review_id_deterministic():
    """Same source + brand + text → same id."""
    a = make_review_id("mock", "喜茶", "好喝")
    b = make_review_id("mock", "喜茶", "好喝")
    c = make_review_id("mock", "奈雪", "好喝")
    assert a == b
    assert a != c
    assert a.startswith("rv_")


def test_store_dedup_on_append(tmp_path):
    store = RawReviewStore(tmp_path)
    scraper = MockScraper(seed=0)
    records = list(scraper.scrape(max_records=20))

    n1 = store.append("w1", records)
    n2 = store.append("w1", records)  # same again
    assert n1 == 20
    assert n2 == 0


def test_store_read_back(tmp_path):
    store = RawReviewStore(tmp_path)
    records = list(MockScraper(seed=0).scrape(max_records=30))
    store.append("w1", records)
    df = store.read("w1")
    assert len(df) == 30
    assert "review_id" in df.columns
    assert "text" in df.columns
    assert df["review_id"].is_unique


def test_store_query(tmp_path):
    store = RawReviewStore(tmp_path)
    store.append("w1", list(MockScraper(seed=0).scrape(max_records=20)))
    df = store.query("SELECT COUNT(*) as n FROM raw WHERE source = 'mock'")
    assert int(df["n"][0]) == 20


def test_runner_orchestration(tmp_path):
    store = RawReviewStore(tmp_path)
    runner = ScrapeRunner(store)
    stats = runner.run(MockScraper(seed=0), shard="r1", max_records=15)
    assert stats.requested == 15
    assert stats.written == 15
    assert store.count() == 15
