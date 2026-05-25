"""Tests for AspectExtractor (mock + payload parsing)."""
from __future__ import annotations

from beverage_ai.aspects.extractor import (
    MockAspectExtractor,
    parse_llm_payload,
    payload_to_aspects,
)
from beverage_ai.aspects.schema import ALL_DIMS, CORE_DIMS
from beverage_ai.scrapers.base import ReviewRecord, make_review_id


def _rec(text: str, brand: str = "喜茶") -> ReviewRecord:
    return ReviewRecord(
        review_id=make_review_id("test", brand, text),
        source="test", brand=brand, text=text,
    )


def test_mock_extracts_high_sweet():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("齁甜啊！甜得发齁，太腻"))
    assert r.aspects["甜度"] is not None and r.aspects["甜度"] > 0.8


def test_mock_extracts_high_caffeine_bitter():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("茶味很浓厚，但是有点苦"))
    assert r.aspects["茶香"] is not None and r.aspects["茶香"] > 0.7
    assert r.aspects["苦度"] is not None and r.aspects["苦度"] > 0.5


def test_mock_extracts_low_preference():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("踩雷，不会回购"))
    assert r.aspects["喜爱度"] is not None and r.aspects["喜爱度"] < 0.3


def test_mock_extracts_high_preference():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("无脑回购，强烈推荐"))
    assert r.aspects["喜爱度"] is not None and r.aspects["喜爱度"] > 0.9


def test_mock_unmentioned_dims_are_none():
    """If review doesn't mention a dim, it should be None — never fabricated."""
    ext = MockAspectExtractor()
    r = ext.extract(_rec("只说一句好喝"))
    # 喜爱度 is mentioned ("好喝"); but most others should be None
    none_count = sum(1 for d in ALL_DIMS if r.aspects[d] is None)
    assert none_count > len(ALL_DIMS) // 2


def test_mock_customization_picked_up():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("我点的是三分糖去冰加芋圆"))
    assert r.customization.sugar_level == "三分"
    assert r.customization.ice_level == "无冰"
    assert "芋圆" in r.customization.toppings


def test_mock_zero_cost():
    ext = MockAspectExtractor()
    r = ext.extract(_rec("好喝"))
    assert r.cost_estimate_usd == 0.0


def test_mock_extractor_version_consistent():
    """Version string must be stable for cache invalidation logic."""
    a = MockAspectExtractor().version
    b = MockAspectExtractor().version
    assert a == b == "mock|kw_v1"


# ---------------------------------------------------------- payload parsing


def test_parse_llm_payload_clean_json():
    raw = '{"aspects": {"甜度": 0.8}, "confidence": 0.7}'
    out = parse_llm_payload(raw)
    assert out["aspects"]["甜度"] == 0.8


def test_parse_llm_payload_strips_markdown_fence():
    raw = '```json\n{"aspects": {"甜度": 0.5}}\n```'
    out = parse_llm_payload(raw)
    assert out["aspects"]["甜度"] == 0.5


def test_parse_llm_payload_with_stray_prose():
    raw = '抱歉, 输出如下:\n{"aspects": {"甜度": 0.6}}\n谢谢'
    out = parse_llm_payload(raw)
    assert out["aspects"]["甜度"] == 0.6


def test_payload_to_aspects_clamps_scores():
    payload = {"aspects": {"甜度": 1.5, "苦度": -0.2, "茶香": "0.4"}, "confidence": 0.8}
    r = payload_to_aspects(payload, "rv_test", "v1", None, 0.0)
    assert r.aspects["甜度"] == 1.0    # clamped from 1.5
    assert r.aspects["苦度"] == 0.0    # clamped from -0.2
    assert r.aspects["茶香"] == 0.4    # coerced from string


def test_payload_handles_missing_aspects():
    """If LLM returns customization but no aspects, all aspects → None."""
    payload = {"customization": {"sugar_level": "五分"}, "confidence": 0.5}
    r = payload_to_aspects(payload, "rv_test", "v1", None, 0.0)
    for dim in CORE_DIMS:
        assert r.aspects[dim] is None
    assert r.customization.sugar_level == "五分"


def test_payload_invalid_score_becomes_none():
    payload = {"aspects": {"甜度": "not_a_number"}, "confidence": 0.5}
    r = payload_to_aspects(payload, "rv_test", "v1", None, 0.0)
    assert r.aspects["甜度"] is None
