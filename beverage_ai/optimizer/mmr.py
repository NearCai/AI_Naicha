"""Maximal Marginal Relevance — Top-K diversity selection.

Per 技术方案书 §3.5 (R3 diversity fix) and v1 实现方案 §6.9.
"""
from __future__ import annotations

import numpy as np


def mmr_select(
    scores: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lam: float = 0.6,
) -> list[int]:
    """Greedy MMR.

    Args:
        scores: 1-D array of relevance scores (higher = better).
        embeddings: (N, D) array of item embeddings, used for diversity distance.
        k: number of items to select.
        lam: trade-off, lam=1 → pure relevance, lam=0 → pure diversity.

    Returns:
        List of selected indices, length min(k, N).
    """
    scores = np.asarray(scores, dtype=float)
    embeddings = np.asarray(embeddings, dtype=float)
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    if embeddings.shape[0] != n:
        raise ValueError("scores and embeddings must have same length")
    k = min(k, n)

    # Normalize scores to [0, 1] so lam comparison is meaningful
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm = (scores - s_min) / (s_max - s_min)
    else:
        norm = np.full_like(scores, 0.5)

    selected: list[int] = [int(np.argmax(norm))]
    remaining = set(range(n)) - set(selected)

    while len(selected) < k and remaining:
        best_i, best_score = None, -np.inf
        for i in remaining:
            div = min(
                float(np.linalg.norm(embeddings[i] - embeddings[j]))
                for j in selected
            )
            s = lam * norm[i] + (1 - lam) * div
            if s > best_score:
                best_score = s
                best_i = i
        selected.append(int(best_i))
        remaining.remove(best_i)

    return selected
