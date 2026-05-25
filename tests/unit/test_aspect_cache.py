"""Tests for AspectCache (DuckDB-backed)."""
from __future__ import annotations

from beverage_ai.aspects.cache import AspectCache
from beverage_ai.aspects.schema import Customization, ExtractedAspects


def _aspects(review_id="rv1", version="v1") -> ExtractedAspects:
    return ExtractedAspects(
        review_id=review_id,
        extractor_version=version,
        aspects={"甜度": 0.8, "苦度": 0.2, "茶香": None},
        customization=Customization(sugar_level="五分", toppings=["珍珠"]),
        confidence=0.85,
        cost_estimate_usd=0.002,
        raw_response="raw text",
    )


def test_put_get_roundtrip(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        r = _aspects()
        c.put(r)
        got = c.get(r.review_id, r.extractor_version)
        assert got is not None
        assert got.review_id == r.review_id
        assert got.aspects["甜度"] == 0.8
        assert got.aspects["茶香"] is None
        assert got.customization.sugar_level == "五分"
        assert got.confidence == 0.85


def test_has_returns_correctly(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        assert not c.has("rv1", "v1")
        c.put(_aspects())
        assert c.has("rv1", "v1")
        assert not c.has("rv1", "v2")    # version-specific


def test_put_replaces_on_conflict(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        c.put(_aspects())
        new = _aspects().model_copy(update={"confidence": 0.42})
        c.put(new)
        assert c.count() == 1
        assert c.get("rv1", "v1").confidence == 0.42


def test_count_by_version(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        c.put(_aspects(review_id="r1", version="v1"))
        c.put(_aspects(review_id="r2", version="v1"))
        c.put(_aspects(review_id="r3", version="v2"))
        assert c.count() == 3
        assert c.count("v1") == 2
        assert c.count("v2") == 1


def test_total_cost(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        c.put(_aspects(review_id="r1"))
        c.put(_aspects(review_id="r2"))
        assert c.total_cost_usd() == 0.004


def test_list_versions_sorted_desc(tmp_path):
    with AspectCache(tmp_path / "c.duckdb") as c:
        for i in range(3):
            c.put(_aspects(review_id=f"r_{i}", version="v_popular"))
        c.put(_aspects(review_id="r_rare", version="v_rare"))
        versions = c.list_versions()
        assert versions[0] == ("v_popular", 3)
        assert versions[1] == ("v_rare", 1)
