"""Real sales predictor — LightGBM + brand fixed effects + K-fold residual fitting.

Corresponds to 技术方案书 §3.3.2 and v1 实现方案 §6.7.

Requires lightgbm + scikit-learn. Training is offline; this file exposes
SalesPredictorLGB which loads pickled model bundle and serves SalesPrediction.
"""
from __future__ import annotations

import pickle
from pathlib import Path

try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    _ML_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERR = e
    _ML_AVAILABLE = False

from ...recipes.schema import Recipe
from .predict import SalesPrediction


def _require_ml() -> None:
    if not _ML_AVAILABLE:
        raise ImportError(
            "lightgbm + scikit-learn required. Install with: pip install -e .[ml]"
        ) from _IMPORT_ERR  # type: ignore[name-defined]


class SalesPredictorLGB:
    """Two-stage LightGBM with K-fold cross-fitted baseline.

    Training flow (offline, see scripts/train_sales_model.py):
        1. baseline = f(brand, price, season, marketing)  (with K-fold OOF residuals)
        2. recipe_model = g(recipe_embedding) → OOF residuals
        3. q05_model, q95_model → quantile predictors for sigma estimate
    """

    def __init__(
        self,
        baseline_models: list,
        recipe_model,
        q05_model,
        q95_model,
        feature_columns: dict,
    ):
        self.baseline_models = baseline_models
        self.recipe_model = recipe_model
        self.q05_model = q05_model
        self.q95_model = q95_model
        self.feature_columns = feature_columns

    @classmethod
    def fit(
        cls,
        baseline_df: "pd.DataFrame",
        recipe_embed_df: "pd.DataFrame",
        y: "np.ndarray",
        k: int = 5,
        seed: int = 42,
    ) -> SalesPredictorLGB:
        _require_ml()
        residuals = np.zeros_like(y, dtype=float)
        baseline_models = []
        for tr, va in KFold(k, shuffle=True, random_state=seed).split(baseline_df):
            m = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, verbose=-1, random_state=seed
            )
            m.fit(baseline_df.iloc[tr], y[tr])
            residuals[va] = y[va] - m.predict(baseline_df.iloc[va])
            baseline_models.append(m)

        recipe_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03,
            objective="quantile", alpha=0.5, verbose=-1, random_state=seed,
        )
        recipe_model.fit(recipe_embed_df, residuals)

        q05 = lgb.LGBMRegressor(
            n_estimators=500, objective="quantile", alpha=0.05,
            verbose=-1, random_state=seed,
        ).fit(recipe_embed_df, residuals)
        q95 = lgb.LGBMRegressor(
            n_estimators=500, objective="quantile", alpha=0.95,
            verbose=-1, random_state=seed,
        ).fit(recipe_embed_df, residuals)

        return cls(
            baseline_models=baseline_models,
            recipe_model=recipe_model,
            q05_model=q05, q95_model=q95,
            feature_columns={
                "baseline": list(baseline_df.columns),
                "recipe":   list(recipe_embed_df.columns),
            },
        )

    def predict(self, baseline_feat: "np.ndarray", recipe_embed: "np.ndarray") -> SalesPrediction:
        _require_ml()
        base = float(np.mean([m.predict(baseline_feat) for m in self.baseline_models], axis=0))
        contrib = float(self.recipe_model.predict(recipe_embed))
        q05 = float(self.q05_model.predict(recipe_embed))
        q95 = float(self.q95_model.predict(recipe_embed))
        sigma = max((q95 - q05) / 3.29, 0.1)
        return SalesPrediction(
            mean=round(base + contrib, 2),
            sigma=round(sigma, 2),
            baseline=round(base, 2),
            recipe_contribution=round(contrib, 2),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> SalesPredictorLGB:
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_from_recipe(self, recipe: Recipe) -> SalesPrediction:
        """Convenience wrapper — caller must transform Recipe → features upstream.
        Not used directly in v1 since we need brand/price metadata."""
        raise NotImplementedError(
            "Need brand/price/marketing features alongside recipe; "
            "use predict() with explicit feature arrays."
        )
