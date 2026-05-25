"""NSGA-II wrapper for multi-objective recipe optimization.

Per 技术方案书 §3.5 + v1 实现方案 §6.9 (A3 mixed-space fix).

NOTE: In v1 we don't run pymoo's MixedVariableGA inside the optimizer
loop — instead, we use the RecipeGenerator (warm start) to produce a
diverse seed population, then **score and rank** them with NSGA-II
non-dominated sorting. This is simpler than encoding the variable-length
ingredient dict as a pymoo Problem, gives the same Pareto front for v1
purposes, and avoids pymoo categorical-encoding pitfalls.

For v2, replace this with a true MixedVariableGA evolution loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..recipes.schema import Recipe


@dataclass
class ScoredCandidate:
    recipe: Recipe
    objectives: np.ndarray              # all to-minimize
    means: dict[str, float]             # for reporting
    sigmas: dict[str, float]
    embedding: np.ndarray
    nutrition: dict
    feasible: bool


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Sort indices into Pareto fronts. objectives is (N, M), all to-minimize."""
    n = len(objectives)
    domination_count = [0] * n
    dominated = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominated[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

    fronts: list[list[int]] = [[]]
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    while fronts[-1]:
        next_front: list[int] = []
        for i in fronts[-1]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
    return fronts[:-1]


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    feas = [c for c in candidates if c.feasible]
    if not feas:
        return []
    objectives = np.stack([c.objectives for c in feas])
    fronts = non_dominated_sort(objectives)
    if not fronts:
        return []
    return [feas[i] for i in fronts[0]]
