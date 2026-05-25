"""Tests for LLMSyntheticScraper using mocked anthropic client."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from beverage_ai.scrapers.adapters.llm_synthetic import (
    LLMSyntheticScraper,
    _strip_fences,
)


def test_strip_fences_removes_markdown_fences():
    assert _strip_fences('```json\n{"x": 1}\n```') == '{"x": 1}'
    assert _strip_fences('{"x": 1}') == '{"x": 1}'


def test_requires_anthropic(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"\[llm\]"):
        LLMSyntheticScraper(api_key="x")


def _fake_anthropic_module(reviews_payload: list[dict]) -> MagicMock:
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text=json.dumps({"reviews": reviews_payload},
                                                       ensure_ascii=False))]
    fake_messages = MagicMock()
    fake_messages.create.return_value = fake_response
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)
    return fake_anthropic


def test_scrape_yields_records(monkeypatch):
    fake_reviews = [
        {"brand": "喜茶", "sku": "多肉葡萄", "sentiment": "positive_strong",
         "text": "今天的多肉葡萄真的太好喝了,葡萄味很浓厚,茶味也压得住,会回购的",
         "customization": "五分糖少冰", "rating": 4.8},
        {"brand": "奈雪", "sku": "宝藏茶", "sentiment": "negative",
         "text": "宝藏茶有点甜得发齁,茶味淡,有点失望",
         "customization": "七分糖正常冰", "rating": 2.5},
    ]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(fake_reviews))

    scraper = LLMSyntheticScraper(api_key="dummy", batch_size=2, seed=1)
    records = list(scraper.scrape(max_records=2))
    assert len(records) == 2
    assert records[0].source.startswith("llm_synthetic:")
    assert records[0].brand == "喜茶"
    assert records[0].rating == 4.8
    assert records[0].metadata["synthetic"] is True


def test_scrape_rejects_short_text(monkeypatch):
    """Short text is rejected by the >=15-char filter; safety-guard bails
    out after MAX_EMPTY_BATCHES so the loop doesn't run forever."""
    fake_reviews = [{"brand": "喜茶", "sku": "X", "text": "好"}]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(fake_reviews))
    scraper = LLMSyntheticScraper(api_key="dummy", batch_size=1, seed=1)
    records = list(scraper.scrape(max_records=5))
    assert len(records) == 0


def test_cost_ceiling_stops_generation(monkeypatch):
    fake_reviews = [
        {"brand": "喜茶", "sku": "X", "text": f"好喝的茶饮 #{i}, 茶味浓厚回甘"}
        for i in range(20)
    ]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(fake_reviews))
    scraper = LLMSyntheticScraper(
        api_key="dummy", batch_size=20, cost_ceiling_usd=0.003, seed=1
    )
    # cost_ceiling is checked BEFORE each batch call (in scrape()'s while loop):
    #   iter 1: cost 0.0   < 0.003 → call (cost → 0.002), yield 20 → emitted=20
    #   iter 2: cost 0.002 < 0.003 → call (cost → 0.004), yield 20 → emitted=40
    #   iter 3: cost 0.004 >= 0.003 → stop
    # → upper bound is 40 records (max_records=100 not reached)
    records = list(scraper.scrape(max_records=100))
    assert len(records) < 100, "should have stopped well before max_records"
    assert len(records) <= 40
    assert scraper.total_cost_usd > 0.003


def test_rating_clamps(monkeypatch):
    fake_reviews = [
        {"brand": "喜茶", "sku": "X",
         "text": "茶味浓郁层次丰富, 推荐购买这款奶茶, 真的非常好喝", "rating": 10},
        {"brand": "喜茶", "sku": "X",
         "text": "另一条评论, 桂花香气足, 整体感受不错值得回购", "rating": -5},
    ]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(fake_reviews))
    scraper = LLMSyntheticScraper(api_key="dummy", batch_size=2, seed=2)
    records = list(scraper.scrape(max_records=2))
    assert records[0].rating == 5.0
    assert records[1].rating == 1.0
