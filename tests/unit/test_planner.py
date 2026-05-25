"""Tests for planner/llm_planner.py — MockLLMPlanner only (no API needed)."""
from __future__ import annotations

from beverage_ai.planner.llm_planner import MockLLMPlanner


def test_mock_basic_request():
    p = MockLLMPlanner()
    spec = p.plan("夏季年轻女性低糖轻负担, 定价 18-22 元")
    assert spec["style_hint"] in ("纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调")
    assert spec["sugar_level"] in ("无糖", "三分", "五分", "七分", "全糖")
    assert spec["context"]["season"] == "summer"
    assert spec["context"]["target_age"] == "youth"
    assert spec["context"]["health_strict"] is True
    assert spec["health"]["sugar_limit_g"] <= 20
    assert spec["price_range_cny"] == [18.0, 22.0]


def test_mock_extracts_cup_volume():
    p = MockLLMPlanner()
    spec = p.plan("我要一杯 700ml 大杯奶茶")
    assert spec["cup_volume_ml"] == 700
    assert spec["style_hint"] == "奶茶"


def test_mock_winter_mature_strict():
    p = MockLLMPlanner()
    spec = p.plan("冬天上班族暖身, 控糖, 30 元以内")
    assert spec["context"].get("season") == "winter"
    assert spec["context"].get("target_age") == "mature"
    assert spec["context"].get("health_strict") is True


def test_mock_no_kw_fallbacks():
    p = MockLLMPlanner()
    spec = p.plan("一杯东西")
    # Has all required fields
    assert "style_hint" in spec
    assert "cup_volume_ml" in spec
    assert "sugar_level" in spec
    assert "health" in spec


def test_mock_flavor_keywords_extracted():
    p = MockLLMPlanner()
    spec = p.plan("想要桂花茶香配厚乳, 微甜花香")
    keywords = spec.get("flavor_keywords", [])
    assert "桂花" in keywords
    assert "厚乳" in keywords
    assert "花香" in keywords
