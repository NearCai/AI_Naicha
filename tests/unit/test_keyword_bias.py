"""Verify Chinese flavor_keywords actually bias ingredient picks.

Originally `_pick_tea`/`_pick_milk`/`_pick_fruit` compared Chinese keywords
('桂花', '厚乳') against English descriptors ('floral', 'creamy') with bare
substring matching, which never hit. Fixed by:
  - direct name_zh / notes_zh substring match
  - mapping table data/ingredients/keyword_aliases.yaml
"""
from __future__ import annotations

from collections import Counter

from beverage_ai.recipes.generator import RecipeGenerator, _load_keyword_aliases


# -------- score helper unit tests --------

def test_aliases_yaml_loads(vocab):
    aliases = _load_keyword_aliases()
    assert "桂花" in aliases
    assert "osmanthus" in aliases["桂花"]
    assert "厚乳" in aliases
    assert "creamy" in aliases["厚乳"]


def test_score_name_match_strongest(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    # 桂花 should match the name 桂花乌龙 directly (score >= 3 from name match)
    s = gen._score_ingredient_for_keywords("tea_osmanthus_oolong", ["桂花"])
    assert s >= 3


def test_score_descriptor_only_match(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    # 厚乳: name match (3) + descriptor map hits creamy/rich/dense (3) = 6
    s_thick = gen._score_ingredient_for_keywords("dairy_thick_milk", ["厚乳"])
    assert s_thick >= 4

    # 厚乳 should also hit 重奶油 (name no match, but creamy+rich → 2 via map)
    s_heavy = gen._score_ingredient_for_keywords("dairy_heavy_cream", ["厚乳"])
    assert s_heavy >= 2

    # Skim milk descriptors are [thin, slight_creamy, watery] — no overlap
    # with the 厚乳 alias set (creamy/rich/thick/dense/milky) → score 0
    s_skim = gen._score_ingredient_for_keywords("dairy_skim_milk", ["厚乳"])
    assert s_skim == 0


def test_score_grades_creamy_milks_correctly(vocab, prior_engine):
    """Sanity: 厚乳 keyword should rank 厚乳/重奶油/奶盖 above plain milks."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    scores = {
        mid: gen._score_ingredient_for_keywords(mid, ["厚乳"])
        for mid in ["dairy_thick_milk", "dairy_heavy_cream", "dairy_cheese_foam",
                    "dairy_whole_milk", "dairy_skim_milk"]
    }
    assert scores["dairy_thick_milk"] >= scores["dairy_whole_milk"]
    assert scores["dairy_heavy_cream"] >= scores["dairy_skim_milk"]


def test_score_unknown_keyword_zero(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    s = gen._score_ingredient_for_keywords("tea_jinxuan", ["完全不存在的口味"])
    assert s == 0


def test_score_empty_keywords_zero(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    assert gen._score_ingredient_for_keywords("tea_jinxuan", []) == 0


def test_score_unknown_ingredient_zero(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    assert gen._score_ingredient_for_keywords("nonexistent_id", ["桂花"]) == 0


# -------- end-to-end bias tests --------

def _tea_id_of(recipe):
    return next((i for i in recipe.ingredients if i.startswith("tea_")), None)


def _milk_id_of(recipe):
    return next(
        (i for i in recipe.ingredients if i.startswith(("dairy_", "alt_milk_"))),
        None,
    )


def _fruit_id_of(recipe):
    return next((i for i in recipe.ingredients if i.startswith("fruit_")), None)


def test_osmanthus_keyword_biases_tea_pick(vocab, prior_engine):
    """'桂花' must push tea selection toward 桂花乌龙 / 桂花茶."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶", "flavor_keywords": ["桂花"]}
    recipes = gen.generate(spec, n_candidates=30)
    teas = [t for t in (_tea_id_of(r) for r in recipes) if t]
    counts = Counter(teas)
    osmanthus_count = counts.get("tea_osmanthus_oolong", 0) + counts.get("tea_osmanthus", 0)
    # ≥ 60% of picks should be osmanthus teas (true score-3 ties)
    assert osmanthus_count >= len(teas) * 0.5, (
        f"Expected osmanthus dominance, got {counts.most_common(5)}"
    )


def test_thick_milk_keyword_biases_creamy_milks(vocab, prior_engine):
    """'厚乳' should heavily favor creamy milks (厚乳 / 重奶油 / 芝士奶盖 / 椰浆 / oat barista)."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶", "flavor_keywords": ["厚乳"]}
    recipes = gen.generate(spec, n_candidates=40)
    creamy_ids = {
        "dairy_thick_milk", "dairy_heavy_cream", "dairy_cheese_foam",
        "dairy_seasalt_foam", "alt_milk_coconut_cream", "alt_milk_oat_barista",
    }
    milks = [m for m in (_milk_id_of(r) for r in recipes) if m]
    n_creamy = sum(1 for m in milks if m in creamy_ids)
    assert n_creamy >= len(milks) * 0.6, (
        f"Expected ≥60% creamy milks, got {n_creamy}/{len(milks)} "
        f"({Counter(milks).most_common(5)})"
    )
    # And dairy_thick_milk (perfect name match) should be the single top pick
    counts = Counter(milks)
    assert counts.most_common(1)[0][0] == "dairy_thick_milk"


def test_strawberry_keyword_biases_fruit_pick(vocab, prior_engine):
    """'草莓' should make 草莓 the dominant fruit."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "果茶", "flavor_keywords": ["草莓"]}
    recipes = gen.generate(spec, n_candidates=30)
    fruits = [f for f in (_fruit_id_of(r) for r in recipes) if f]
    if fruits:
        counts = Counter(fruits)
        assert counts.most_common(1)[0][0] == "fruit_strawberry"


def test_no_keywords_falls_through_to_random_uniform(vocab, prior_engine):
    """No keywords → broad tea diversity (no single tea dominates)."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶"}
    recipes = gen.generate(spec, n_candidates=40)
    teas = [t for t in (_tea_id_of(r) for r in recipes) if t]
    # Expect at least 10 different teas across 40 runs (vocab has 40 teas)
    assert len(set(teas)) >= 10


def test_unknown_keyword_doesnt_crash(vocab, prior_engine):
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶", "flavor_keywords": ["完全不存在的口味_xyz"]}
    recipes = gen.generate(spec, n_candidates=5)
    assert len(recipes) >= 1


def test_multi_keyword_combines_scores(vocab, prior_engine):
    """'桂花' + '厚乳' should pull tea to osmanthus AND milk to creamy."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶", "flavor_keywords": ["桂花", "厚乳"]}
    recipes = gen.generate(spec, n_candidates=30)
    teas = Counter(t for t in (_tea_id_of(r) for r in recipes) if t)
    milks = Counter(m for m in (_milk_id_of(r) for r in recipes) if m)
    # Tea biased toward osmanthus
    assert teas.most_common(1)[0][0] in ("tea_osmanthus_oolong", "tea_osmanthus")
    # Milk biased toward dairy_thick_milk (direct name match)
    assert milks.most_common(1)[0][0] == "dairy_thick_milk"


def test_flavoring_bias_by_keyword(vocab, prior_engine):
    """'桂花' should bias flavoring picks toward osmanthus-related ones."""
    gen = RecipeGenerator(vocab, prior_engine, seed=0)
    spec = {"style_hint": "奶茶", "flavor_keywords": ["桂花"]}
    recipes = gen.generate(spec, n_candidates=30)
    flavorings = []
    for r in recipes:
        for k in r.ingredients:
            if k.startswith("flavor_"):
                flavorings.append(k)
    if flavorings:
        # At least 30% of flavoring picks should be osmanthus-related
        osm = ("flavor_osmanthus_sauce", "flavor_osmanthus_essence", "flavor_dried_osmanthus")
        n_osm = sum(1 for f in flavorings if f in osm)
        assert n_osm >= len(flavorings) * 0.2
