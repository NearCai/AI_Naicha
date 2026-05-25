"""Quick inspection of what's on disk."""
import os
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

print("=" * 60)
print("STATIC DATA (committed schemas + curated content)")
print("=" * 60)

for p in (
    "data/ingredients/ingredient_vocab.yaml",
    "data/ingredients/aliases.yaml",
    "data/ingredients/topping_compatibility.yaml",
    "data/priors/dirichlet_alpha_v1.yaml",
    "data/priors/context_deltas.yaml",
):
    full = ROOT / p
    if full.exists():
        size_kb = os.path.getsize(full) / 1024
        with open(full, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        n = (len(content) if isinstance(content, (list, dict)) else 1)
        print(f"  {p:<55s}  {size_kb:6.1f} KB  {n:>4} entries")

print()
print("=" * 60)
print("DYNAMIC DATA (scraped / extracted at runtime)")
print("=" * 60)

raw_dir = ROOT / "data/reviews/raw"
if raw_dir.exists():
    for shard in sorted(raw_dir.iterdir()):
        p = shard / "raw_reviews.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            size_kb = os.path.getsize(p) / 1024
            print(f"\n  Shard '{shard.name}'  ({size_kb:.1f} KB, {len(df)} rows)")
            print(f"    sources: {dict(df['source'].value_counts())}")
            if "brand" in df.columns:
                print(f"    brands:  {dict(df['brand'].value_counts())}")
            synthetic = df["metadata"].apply(
                lambda m: '"synthetic": true' in (m or "")
            ).sum() if "metadata" in df.columns else 0
            print(f"    synthetic flag: {synthetic}/{len(df)} rows")

cache = ROOT / "data/reviews/aspects_cache.duckdb"
if cache.exists():
    size_kb = os.path.getsize(cache) / 1024
    con = duckdb.connect(str(cache))
    n = con.execute("SELECT COUNT(*) FROM aspects_cache").fetchone()[0]
    by_ver = con.execute(
        "SELECT extractor_version, COUNT(*) FROM aspects_cache GROUP BY 1"
    ).fetchall()
    cost = con.execute(
        "SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM aspects_cache"
    ).fetchone()[0]
    print(f"\n  aspects_cache.duckdb  ({size_kb:.1f} KB, {n} entries)")
    print(f"    versions: {by_ver}")
    print(f"    estimated cost spent: ${cost:.4f}")
    con.close()

print()
print("=" * 60)
print("EXPECTED-BUT-MISSING (per technical proposal §4.1)")
print("=" * 60)
missing = [
    ("data/reviews/raw/*  REAL reviews from dianping/xiaohongshu",
     "0", "50,000+"),
    ("data/recipes/reference_recipes_v1.yaml",
     "0", "100+"),
    ("data/recipes/reverse_engineered_v1.yaml",
     "0", "30"),
    ("data/products/sku_features_v1.parquet",
     "0", "1,000+"),
    ("data/panel/tasting_v1.parquet",
     "0", "200-300"),
    ("models/sensory_gnn_stage1.pt",
     "0", "trained"),
    ("models/sensory_gnn_v1.pt  (after Stage 2)",
     "0", "trained"),
    ("models/sales_v1.pkl",
     "0", "trained"),
]
print(f"  {'path':<60s} {'current':<10s} {'needed'}")
for path, current, needed in missing:
    print(f"  {path:<60s} {current:<10s} {needed}")
