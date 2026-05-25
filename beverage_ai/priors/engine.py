"""PriorEngine — unified access to all quantity priors.

Wraps:
  - Conditional Dirichlet alpha (§E.9.2)
  - Bayesian posterior update (§E.9.3)
  - typical_serving_g calibration hooks (§E.9.4)

Snapshot files are written to data/priors/prior_history/ so that
every update is auditable and reversible.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from ..recipes.schema import Recipe
from .dirichlet import ROLE_ORDER, bayesian_update_alpha

ROLES = ROLE_ORDER  # public alias


class PriorEngine:
    def __init__(
        self,
        base_alpha: dict[str, list[float]],
        context_deltas: dict[str, dict[str, dict[str, float]]],
        history_dir: Path | str | None = None,
    ):
        self.base_alpha = {k: list(v) for k, v in base_alpha.items()}
        self.context_deltas = context_deltas
        self.history_dir = Path(history_dir) if history_dir else None
        if self.history_dir is not None:
            self.history_dir.mkdir(parents=True, exist_ok=True)

    # ----- alpha access -----

    def get_dirichlet_alpha(self, style: str, context: dict | None = None) -> np.ndarray:
        if style not in self.base_alpha:
            raise KeyError(f"Unknown style: {style!r}")
        a = np.array(self.base_alpha[style], dtype=float)
        if context:
            for feature, value in context.items():
                if feature not in self.context_deltas:
                    continue
                value_key = str(value)  # YAML normalises booleans to strings here
                deltas_for_feature = self.context_deltas[feature]
                delta = deltas_for_feature.get(value_key) or deltas_for_feature.get(value)
                if delta is None:
                    continue
                a = a + np.array([delta.get(r, 0.0) for r in ROLE_ORDER])
        return np.clip(a, 0.05, None)

    def all_styles(self) -> list[str]:
        return list(self.base_alpha.keys())

    # ----- posterior update -----

    def update_dirichlet_posterior(
        self,
        style: str,
        observed_recipes: list[Recipe],
        scores: np.ndarray,
        learning_rate: float = 0.3,
    ) -> np.ndarray:
        if style not in self.base_alpha:
            raise KeyError(f"Unknown style: {style!r}")
        alpha_prior = np.array(self.base_alpha[style], dtype=float)
        alpha_new = bayesian_update_alpha(
            alpha_prior=alpha_prior,
            recipes=observed_recipes,
            scores=scores,
            learning_rate=learning_rate,
        )

        # Persist snapshot
        if self.history_dir is not None:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            snap = {
                "style": style,
                "timestamp_utc": ts,
                "n_observed": len(observed_recipes),
                "learning_rate": learning_rate,
                "alpha_prior": alpha_prior.tolist(),
                "alpha_new": alpha_new.tolist(),
                "roles": list(ROLE_ORDER),
            }
            (self.history_dir / f"{style}_{ts}.json").write_text(
                json.dumps(snap, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Update in-memory state for next call
        self.base_alpha[style] = alpha_new.tolist()
        return alpha_new

    # ----- IO -----

    @classmethod
    def from_yaml(
        cls,
        alpha_path: str | Path,
        deltas_path: str | Path,
        history_dir: Path | str | None = None,
    ) -> PriorEngine:
        with open(alpha_path, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        with open(deltas_path, encoding="utf-8") as f:
            deltas = yaml.safe_load(f) or {}
        return cls(base, deltas, history_dir)


def load_default_engine() -> PriorEngine:
    """Convenience loader using bundled YAMLs."""
    from ..ingredients.vocab import _default_data_dir

    data = _default_data_dir()
    return PriorEngine.from_yaml(
        alpha_path=data / "priors" / "dirichlet_alpha_v1.yaml",
        deltas_path=data / "priors" / "context_deltas.yaml",
        history_dir=data / "priors" / "prior_history",
    )
