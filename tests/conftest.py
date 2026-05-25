"""Pytest fixtures shared across tests."""
from __future__ import annotations

import pytest

from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.priors.engine import load_default_engine
from beverage_ai.recipes.schema import Process, Recipe


@pytest.fixture(scope="session")
def vocab():
    return load_default_vocab()


@pytest.fixture(scope="session")
def prior_engine(tmp_path_factory):
    """Use a temp history dir so tests don't pollute real snapshots."""
    import yaml

    from beverage_ai.ingredients.vocab import _default_data_dir
    from beverage_ai.priors.engine import PriorEngine

    data = _default_data_dir()
    history = tmp_path_factory.mktemp("prior_history")
    with open(data / "priors" / "dirichlet_alpha_v1.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(data / "priors" / "context_deltas.yaml", encoding="utf-8") as f:
        deltas = yaml.safe_load(f) or {}
    return PriorEngine(base, deltas, history_dir=history)


@pytest.fixture
def example_recipe() -> Recipe:
    """Matches the worked example in 技术方案书 §E.7."""
    return Recipe(
        recipe_id="fixture_001",
        style="奶茶",
        cup_volume_ml=500,
        sugar_level="五分",
        ingredients={
            "tea_osmanthus_oolong": 250.0,
            "dairy_thick_milk": 125.0,
            "sweet_cane_sugar": 13.0,
            "topping_taro_ball": 35.0,
            "flavor_dried_osmanthus": 0.5,
            "aux_pure_water": 25.0,
            "aux_ice_cube": 85.0,
        },
        process=Process(),
        metadata={"source": "test fixture"},
    )


@pytest.fixture
def minimal_recipe() -> Recipe:
    return Recipe(
        recipe_id="minimal_001",
        style="纯茶",
        cup_volume_ml=500,
        sugar_level="无糖",
        ingredients={"tea_jasmine_green": 400.0, "aux_ice_cube": 100.0},
    )
