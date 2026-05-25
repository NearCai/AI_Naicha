"""Multi-objective optimizer — NSGA-II + LCB + MMR.

Corresponds to 技术方案书 §3.5.
"""

from .acquisition import lcb, ucb
from .mmr import mmr_select

__all__ = ["lcb", "ucb", "mmr_select"]
