"""Tests for LocalFileScraper."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from beverage_ai.scrapers.adapters.local_file import LocalFileScraper


def _write_csv(tmp_path, rows):
    p = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_local_csv(tmp_path):
    p = _write_csv(tmp_path, [
        {"source": "manual", "brand": "喜茶", "sku": "多肉葡萄", "text": "好喝", "rating": 5},
        {"source": "manual", "brand": "奈雪", "sku": "霸气葡萄", "text": "一般", "rating": 3},
    ])
    records = list(LocalFileScraper(p).scrape(max_records=10))
    assert len(records) == 2
    assert records[0].brand == "喜茶"
    assert records[0].rating == 5.0


def test_local_csv_filters_by_keyword(tmp_path):
    p = _write_csv(tmp_path, [
        {"text": "桂花乌龙真的好喝", "brand": "喜茶"},
        {"text": "普通奶茶", "brand": "奈雪"},
        {"text": "另一杯桂花", "brand": "古茗"},
    ])
    records = list(LocalFileScraper(p).scrape(keywords=["桂花"]))
    assert len(records) == 2


def test_local_csv_filters_by_brand(tmp_path):
    p = _write_csv(tmp_path, [
        {"text": "x", "brand": "喜茶"},
        {"text": "y", "brand": "奈雪"},
    ])
    records = list(LocalFileScraper(p).scrape(brand="喜茶"))
    assert len(records) == 1
    assert records[0].brand == "喜茶"


def test_local_jsonl(tmp_path):
    p = tmp_path / "reviews.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in [
            {"text": "好喝", "brand": "喜茶"},
            {"text": "一般", "brand": "奈雪"},
        ]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    records = list(LocalFileScraper(p).scrape())
    assert len(records) == 2


def test_local_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalFileScraper(tmp_path / "doesnotexist.csv")


def test_local_skips_empty_text(tmp_path):
    p = _write_csv(tmp_path, [
        {"text": "好喝"},
        {"text": ""},
        {"text": "   "},
    ])
    records = list(LocalFileScraper(p).scrape())
    assert len(records) == 1
