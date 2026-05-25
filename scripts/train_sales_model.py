"""Train sales predictor (LightGBM + brand fixed effects + K-fold residuals).

Implements 技术方案书 §3.3.2 the way it's specified:
  Stage 1: baseline = f(brand, price, season, marketing, city) trained K-fold
           → OOF residuals = y - baseline_predict
  Stage 2: recipe_model = g(recipe_features) → residuals
  + Quantile models (0.05 / 0.95) for sigma estimate

Inputs:  data/products/synthetic_skus_v1.parquet  (from generate_synthetic_skus.py)
Outputs: models/sales_v1.pkl  +  models/sales_v1_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split
    _ML_OK = True
    _ML_ERR = None
except Exception as e:
    _ML_OK = False
    _ML_ERR = e


# Feature partitions per §3.3.2
BASELINE_CATEGORICAL = ["brand", "season", "city_tier", "recipe_style"]
BASELINE_NUMERIC = ["price_cny", "launch_year",
                    "marketing_联名", "marketing_限定", "marketing_明星", "marketing_包装"]

RECIPE_NUMERIC = [
    "recipe_cup_volume_ml", "recipe_n_ingredients", "recipe_total_mass_g",
    "recipe_calorie_kcal", "recipe_sugar_g", "recipe_fat_g",
    "recipe_caffeine_mg", "recipe_sodium_mg", "recipe_n_topping",
    "recipe_has_dairy", "recipe_has_alt_milk",
    "recipe_has_coffee", "recipe_has_fruit",
]


def _onehot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode + cast bool → int."""
    encoded = pd.get_dummies(df[cols], dummy_na=False)
    return encoded.astype(np.float32)


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    cat = _onehot(df, BASELINE_CATEGORICAL)
    num = df[BASELINE_NUMERIC].astype(np.float32)
    return pd.concat([cat, num], axis=1)


def build_recipe_features(df: pd.DataFrame) -> pd.DataFrame:
    # recipe_sugar_level as ordinal
    level_map = {"无糖": 0, "三分": 1, "五分": 2, "七分": 3, "全糖": 4}
    df = df.copy()
    df["recipe_sugar_level_ord"] = df["recipe_sugar_level"].map(level_map).fillna(2).astype(np.float32)
    feats = df[RECIPE_NUMERIC + ["recipe_sugar_level_ord"]].astype(np.float32)
    return feats


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Normalized DCG @ k."""
    order_pred = np.argsort(-y_pred)[:k]
    order_true = np.argsort(-y_true)[:k]
    gains_pred = y_true[order_pred]
    gains_ideal = y_true[order_true]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains_pred * discount).sum()
    idcg = (gains_ideal * discount).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    if not _ML_OK:
        print(f"ERROR: lightgbm / scikit-learn missing ({_ML_ERR})")
        print("Install:  pip install lightgbm scikit-learn scipy")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/products/synthetic_skus_v1.parquet")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators-baseline", type=int, default=300)
    parser.add_argument("--n-estimators-recipe", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[1/5] Loading synthetic SKUs ...")
    df = pd.read_parquet(args.data)
    print(f"      total: {len(df)} SKUs, {df['brand'].nunique()} brands, "
          f"{df['recipe_id'].nunique()} unique recipes")
    print(f"      sales range: {df['sales_proxy'].min():.1f} – {df['sales_proxy'].max():.1f}")

    print(f"\n[2/5] Train/test split (test_frac={args.test_frac}) ...")
    train_df, test_df = train_test_split(df, test_size=args.test_frac,
                                          random_state=args.seed, shuffle=True)
    print(f"      train: {len(train_df)}   test: {len(test_df)}")

    y_train = train_df["sales_proxy"].to_numpy(dtype=np.float32)
    y_test = test_df["sales_proxy"].to_numpy(dtype=np.float32)

    print(f"\n[3/5] Stage 1: baseline model with K={args.k_folds}-fold OOF residuals ...")
    baseline_train = build_baseline_features(train_df)
    baseline_test = build_baseline_features(test_df)
    baseline_test = baseline_test.reindex(columns=baseline_train.columns, fill_value=0.0)
    print(f"      baseline features: {baseline_train.shape[1]} (one-hot + numeric)")

    residuals_train = np.zeros_like(y_train)
    baseline_models = []
    fold_t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(
        KFold(args.k_folds, shuffle=True, random_state=args.seed).split(baseline_train)
    ):
        m = lgb.LGBMRegressor(
            n_estimators=args.n_estimators_baseline,
            learning_rate=0.05,
            verbose=-1,
            random_state=args.seed,
        )
        m.fit(baseline_train.iloc[tr_idx], y_train[tr_idx])
        residuals_train[va_idx] = y_train[va_idx] - m.predict(baseline_train.iloc[va_idx])
        baseline_models.append(m)
    print(f"      done in {time.time() - fold_t0:.1f}s  "
          f"|residual| mean={np.abs(residuals_train).mean():.2f}")

    print(f"\n[4/5] Stage 2: recipe model fits residuals ...")
    recipe_train = build_recipe_features(train_df)
    recipe_test = build_recipe_features(test_df)
    print(f"      recipe features: {recipe_train.shape[1]}")

    recipe_model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators_recipe, learning_rate=0.03,
        objective="quantile", alpha=0.5, verbose=-1, random_state=args.seed,
    )
    recipe_model.fit(recipe_train, residuals_train)

    q05_model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators_recipe, objective="quantile", alpha=0.05,
        verbose=-1, random_state=args.seed,
    ).fit(recipe_train, residuals_train)
    q95_model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators_recipe, objective="quantile", alpha=0.95,
        verbose=-1, random_state=args.seed,
    ).fit(recipe_train, residuals_train)
    print(f"      recipe model + q05/q95 trained")

    print(f"\n[5/5] Evaluating on held-out test set ...")
    # Mean ensemble of K baselines on test
    baseline_pred_test = np.mean([m.predict(baseline_test) for m in baseline_models], axis=0)
    recipe_contrib_test = recipe_model.predict(recipe_test)
    y_pred_test = baseline_pred_test + recipe_contrib_test

    # Sigma from quantiles
    q05_test = q05_model.predict(recipe_test)
    q95_test = q95_model.predict(recipe_test)
    sigma_test = np.maximum((q95_test - q05_test) / 3.29, 0.1)

    spearman_baseline = float(spearmanr(y_test, baseline_pred_test).statistic)
    spearman_full = float(spearmanr(y_test, y_pred_test).statistic)
    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    ndcg10 = ndcg_at_k(y_test, y_pred_test, k=10)
    ndcg50 = ndcg_at_k(y_test, y_pred_test, k=50)

    print(f"      Spearman ρ  (baseline only): {spearman_baseline:+.3f}")
    print(f"      Spearman ρ  (baseline + recipe): {spearman_full:+.3f}")
    print(f"      Recipe contribution: +{spearman_full - spearman_baseline:.3f}")
    print(f"      MAE:  {mae:.2f}")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      NDCG@10: {ndcg10:.3f}")
    print(f"      NDCG@50: {ndcg50:.3f}")
    print(f"      sigma mean: {sigma_test.mean():.2f}  (median: {np.median(sigma_test):.2f})")

    # Feature importance
    fi_baseline = pd.DataFrame({
        "feature": baseline_train.columns,
        "importance": baseline_models[0].feature_importances_,
    }).sort_values("importance", ascending=False).head(10)
    fi_recipe = pd.DataFrame({
        "feature": recipe_train.columns,
        "importance": recipe_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(10)
    print(f"\n      Top baseline features:")
    for _, r in fi_baseline.iterrows():
        print(f"        {r['feature']:<35s} {int(r['importance'])}")
    print(f"      Top recipe features:")
    for _, r in fi_recipe.iterrows():
        print(f"        {r['feature']:<35s} {int(r['importance'])}")

    print(f"\n[6/6] Saving model + log ...")
    import pickle
    bundle = {
        "baseline_models": baseline_models,
        "recipe_model": recipe_model,
        "q05_model": q05_model,
        "q95_model": q95_model,
        "feature_columns": {
            "baseline": list(baseline_train.columns),
            "recipe": list(recipe_train.columns),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "test_metrics": {
            "spearman_baseline": round(spearman_baseline, 3),
            "spearman_full": round(spearman_full, 3),
            "recipe_contrib_delta": round(spearman_full - spearman_baseline, 3),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "ndcg_at_10": round(ndcg10, 3),
            "ndcg_at_50": round(ndcg50, 3),
        },
    }
    model_path = out_dir / "sales_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"      wrote {model_path}")

    log_path = out_dir / "sales_v1_log.json"
    log = {k: v for k, v in bundle.items() if k not in ("baseline_models", "recipe_model",
                                                         "q05_model", "q95_model")}
    log["top_features"] = {
        "baseline": fi_baseline.to_dict(orient="records"),
        "recipe": fi_recipe.to_dict(orient="records"),
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      wrote {log_path}")
    print("\nDone. Sales predictor trained on synthetic data — pipeline validated.")


if __name__ == "__main__":
    sys.exit(main() or 0)
