"""Tests for HFDatasetScraper using mocked `datasets` import."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from beverage_ai.scrapers.adapters.hf_dataset import (
    HFDatasetScraper,
    _autodetect_column,
    _normalize_rating,
)


def test_autodetect_picks_text_column():
    row = {"review": "好喝", "label": 1}
    col = _autodetect_column(row, ("text", "review", "content"))
    assert col == "review"


def test_autodetect_returns_none_when_no_match():
    row = {"foo": "bar"}
    assert _autodetect_column(row, ("text", "review")) is None


def test_autodetect_case_insensitive():
    row = {"Review": "x"}
    assert _autodetect_column(row, ("review",)) == "Review"


@pytest.mark.parametrize("v, scale, expected", [
    (0, (0, 1), 1.5),
    (1, (0, 1), 4.5),
    (3.5, (1, 5), 3.5),
    (5, (1, 5), 5.0),
    (0.8, (0, 1), 4.2),     # 1 + 4*0.8 = 4.2
    (None, (0, 1), None),
    ("xx", (0, 1), None),
])
def test_normalize_rating(v, scale, expected):
    out = _normalize_rating(v, scale)
    if expected is None:
        assert out is None
    else:
        assert abs(out - expected) < 0.05


def test_scrape_filters_by_keyword(monkeypatch):
    """Mock the datasets library and verify keyword filter."""
    fake_dataset = [
        {"review": "今天的奶茶很好喝", "label": 1, "user_id": "u1"},
        {"review": "这家手机店服务很差", "label": 0, "user_id": "u2"},   # off-topic
        {"review": "桂花乌龙拿铁香气浓郁", "label": 1, "user_id": "u3"},
        {"review": "完全没关系的评论", "label": 1, "user_id": "u4"},
    ]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    scraper = HFDatasetScraper(dataset_name="fake/dataset")
    records = list(scraper.scrape(keywords=["奶茶", "拿铁"], max_records=10))
    assert len(records) == 2
    assert records[0].source == "hf:fake/dataset"
    assert "奶茶" in records[0].text or "拿铁" in records[0].text
    assert records[0].rating == 4.5            # label=1 with binary scale → 4.5
    assert records[0].user_id_hashed is not None


def test_scrape_respects_max_records(monkeypatch):
    fake_dataset = [{"review": f"好喝 #{i}"} for i in range(20)]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    scraper = HFDatasetScraper(dataset_name="fake/d")
    records = list(scraper.scrape(max_records=5))
    assert len(records) == 5


def test_scrape_raises_when_no_text_column(monkeypatch):
    fake_dataset = [{"foo": "bar", "baz": "qux"}]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    scraper = HFDatasetScraper(dataset_name="fake/d")
    with pytest.raises(ValueError, match="text column"):
        list(scraper.scrape())


def test_scrape_explicit_text_column(monkeypatch):
    fake_dataset = [{"custom_field": "好喝的奶茶"}]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    scraper = HFDatasetScraper(dataset_name="fake/d", text_column="custom_field")
    records = list(scraper.scrape())
    assert len(records) == 1


def test_source_name_includes_dataset():
    scraper = HFDatasetScraper(dataset_name="seamew/ChnSentiCorp")
    assert scraper.source_name == "hf:seamew/ChnSentiCorp"


def test_raises_clear_error_without_datasets(monkeypatch):
    """If `datasets` package is not installed, error message must guide install."""
    monkeypatch.setitem(sys.modules, "datasets", None)   # simulate ImportError
    scraper = HFDatasetScraper(dataset_name="fake/d")
    with pytest.raises(ImportError, match=r"\[hf\]"):
        list(scraper.scrape())
