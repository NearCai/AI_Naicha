"""LCB / UCB acquisition — 技术方案书 §3.5 (R9 fix)."""
from __future__ import annotations


def lcb(mean: float, sigma: float, kappa: float = 1.0) -> float:
    """Lower Confidence Bound for maximization-side objectives.

    The optimizer maximizes (mean - kappa * sigma), penalising OOD regions
    where sigma is large.
    """
    return mean - kappa * sigma


def ucb(mean: float, sigma: float, kappa: float = 1.0) -> float:
    """Upper Confidence Bound for minimization-side objectives.

    The optimizer minimizes (mean + kappa * sigma).
    """
    return mean + kappa * sigma
