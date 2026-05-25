"""Tests for optimizer/acquisition.py and optimizer/mmr.py."""
from __future__ import annotations

import numpy as np

from beverage_ai.optimizer.acquisition import lcb, ucb
from beverage_ai.optimizer.mmr import mmr_select


def test_lcb_formula():
    assert lcb(5.0, 1.0, kappa=1.0) == 4.0
    assert lcb(5.0, 0.5, kappa=2.0) == 4.0
    assert lcb(5.0, 0.0, kappa=1.0) == 5.0


def test_ucb_formula():
    assert ucb(5.0, 1.0, kappa=1.0) == 6.0
    assert ucb(5.0, 0.0, kappa=1.0) == 5.0


def test_lcb_penalizes_uncertainty():
    """A high-mean / high-sigma point is worse than slightly-lower-mean / low-sigma."""
    a = lcb(5.0, 2.0, kappa=1.0)   # 3.0
    b = lcb(4.5, 0.5, kappa=1.0)   # 4.0
    assert b > a


def test_mmr_k_equals_one_picks_argmax():
    scores = np.array([1.0, 5.0, 3.0, 2.0])
    embs = np.eye(4)
    out = mmr_select(scores, embs, k=1)
    assert out == [1]


def test_mmr_k_equals_n_returns_all():
    scores = np.array([1.0, 2.0, 3.0])
    embs = np.eye(3)
    out = mmr_select(scores, embs, k=3)
    assert set(out) == {0, 1, 2}


def test_mmr_zero_k_returns_empty():
    scores = np.array([1.0, 2.0, 3.0])
    embs = np.eye(3)
    assert mmr_select(scores, embs, k=0) == []


def test_mmr_empty_input():
    assert mmr_select(np.array([]), np.zeros((0, 4)), k=5) == []


def test_mmr_diversity_beats_pure_relevance():
    """With λ=0, MMR picks for diversity, not relevance."""
    scores = np.array([10.0, 9.9, 1.0])
    # First two near-identical, third is far away
    embs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.01, 0.0],
        [0.0, 0.0, 5.0],
    ])
    out = mmr_select(scores, embs, k=2, lam=0.0)
    # Pure diversity → pick item 0 first, then item 2 (farther)
    assert 2 in out


def test_mmr_relevance_dominates_when_lam_one():
    """λ=1 → pure relevance → top-K by score."""
    scores = np.array([1.0, 9.0, 8.0, 3.0])
    embs = np.eye(4)
    out = mmr_select(scores, embs, k=2, lam=1.0)
    assert set(out) == {1, 2}


def test_mmr_score_embedding_size_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        mmr_select(np.array([1.0, 2.0]), np.eye(3), k=2)
