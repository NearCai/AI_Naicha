"""GNN Stage 1 prototype training on review-derived graphs.

Per 技术方案书 §3.3.1 路径 A pretraining: use LLM-extracted aspect labels
from reviews as weak supervision for a GAT that maps ingredient graphs to
sensory scores.

Strategy (this prototype is a "validate the pipeline runs" stage):

  reviews (~15K)
    └─ aspect extractor → 15-dim aspect labels (cached in DuckDB)
    └─ ingredient mention extractor → pseudo-recipe (ingredient_ids mentioned in text)
       └─ build graph: nodes = ingredients, edges = co-occurrence
          └─ Y = aspect labels (core 5 dims)
             └─ train SensoryGAT
                └─ validate per-dim Pearson r

Reality check:
- The "recipe" used as GNN input is INFERRED from review text via name matching,
  NOT a real product recipe. Production needs SKU-level base recipes.
- Aspect labels come from MockAspectExtractor by default (keyword-based);
  swap to ClaudeAspectExtractor + ANTHROPIC_API_KEY for real quality.
- The goal here is end-to-end pipeline validation: graph construction works,
  forward+backward runs, gradient flows, val loss drops over epochs.

Usage:
    python scripts/train_sensory_gnn_stage1.py --epochs 20
    python scripts/train_sensory_gnn_stage1.py --max-reviews 2000 --epochs 30
    python scripts/train_sensory_gnn_stage1.py --extractor mock      # fast, default
    python scripts/train_sensory_gnn_stage1.py --extractor claude    # higher quality, needs API key
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# ----- imports that may fail without [ml] extras -----
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    _TORCH_OK = True
    _TORCH_ERR = None
except Exception as e:  # ImportError or OSError (broken DLL)
    _TORCH_OK = False
    _TORCH_ERR = e

try:
    from scipy.stats import pearsonr
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# ----- imports from our package (always available) -----
from beverage_ai.aspects.cache import AspectCache
from beverage_ai.aspects.extractor import (
    ClaudeAspectExtractor,
    MockAspectExtractor,
)
from beverage_ai.aspects.pipeline import AspectExtractionPipeline
from beverage_ai.aspects.schema import CORE_DIMS
from beverage_ai.ingredients.aliases import load_default_aliases
from beverage_ai.ingredients.vocab import Vocab, load_default_vocab
from beverage_ai.scrapers.base import ReviewRecord
from beverage_ai.scrapers.store import RawReviewStore

CATEGORIES = (
    "tea_base", "dairy_base", "alt_milk_base", "coffee_base",
    "sweetener", "fruit", "topping", "flavoring",
    "auxiliary", "gel", "grain",
)
NODE_FEAT_DIM = 5 + len(CATEGORIES)   # 5 nutrition + 11 category one-hot = 16


# =============================================================================
# Ingredient mention extraction (review text → pseudo-recipe)
# =============================================================================

class IngredientMentionExtractor:
    """Find vocab ingredients mentioned in review text via name + alias match."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        aliases_resolver = load_default_aliases(vocab)
        # Build inverse index of (Chinese name or alias) → ingredient_id
        self.name_to_id: dict[str, str] = {}
        for ing in vocab.all():
            self.name_to_id[ing.name_zh] = ing.id
        for alias, canonical in aliases_resolver._map.items():
            if canonical in vocab:
                self.name_to_id.setdefault(alias, canonical)
        # Sort longest first so substring matches prefer the more specific name
        self._sorted_names = sorted(self.name_to_id.keys(), key=len, reverse=True)

    def extract(self, text: str) -> list[str]:
        ids: set[str] = set()
        for name in self._sorted_names:
            if name and name in text:
                ids.add(self.name_to_id[name])
        return sorted(ids)


# =============================================================================
# Graph construction
# =============================================================================

def ingredient_node_features(ing) -> torch.Tensor:
    """16-dim node feature vector: 5 nutrition + 11 category one-hot."""
    nut = ing.nutrition_per_100g
    feat = [
        (nut.energy_kcal or 0) / 500.0,
        (nut.sugar_g or 0) / 100.0,
        (nut.fat_g or 0) / 50.0,
        (nut.caffeine_mg or 0) / 200.0,
        (nut.sodium_mg or 0) / 1000.0,
    ]
    # category one-hot
    cat_idx = CATEGORIES.index(ing.category) if ing.category in CATEGORIES else -1
    cat_oh = [0.0] * len(CATEGORIES)
    if cat_idx >= 0:
        cat_oh[cat_idx] = 1.0
    return torch.tensor(feat + cat_oh, dtype=torch.float32)


def build_graph(
    ingredient_ids: list[str],
    vocab: Vocab,
    aspect_labels: dict[str, float | None],
) -> Data:
    """Build a torch_geometric Data with complete graph + node features + label."""
    n = len(ingredient_ids)
    x = torch.stack([ingredient_node_features(vocab.get(i)) for i in ingredient_ids])
    # Complete graph (no self loops)
    edge_src = []
    edge_dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    # Label: core 5 dims as float (NaN where label missing)
    y = torch.tensor(
        [(aspect_labels.get(d) if aspect_labels.get(d) is not None else float("nan"))
         for d in CORE_DIMS],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, y=y.unsqueeze(0))


# =============================================================================
# Model — local GAT (mirrors simulators/sensory/model.py without import cycle)
# =============================================================================

class SensoryGATProto(torch.nn.Module):
    """Minimal GAT for the prototype.

    Two GATv2 layers → mean+max pool → MLP → (mean, logvar) for 5 core dims.
    """

    def __init__(self, node_in_dim: int = NODE_FEAT_DIM, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden, heads=heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads)
        self.proj = torch.nn.Linear(hidden * heads * 2, 128)
        self.head_mean = torch.nn.Linear(128, len(CORE_DIMS))
        self.head_logvar = torch.nn.Linear(128, len(CORE_DIMS))

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.elu(self.conv2(x, ei))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        g = F.elu(self.proj(g))
        mean = self.head_mean(g)
        logvar = self.head_logvar(g).clamp(-6, 6)   # numerical stability
        return mean, logvar


def heteroscedastic_nll(target, mean, logvar, mask):
    """Gaussian NLL ignoring NaN targets via mask."""
    sq = (target - mean) ** 2
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (inv_var * sq + logvar)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# =============================================================================
# Main training pipeline
# =============================================================================

def main():
    if not _TORCH_OK:
        print(f"ERROR: torch / torch_geometric not installed ({_TORCH_ERR}).")
        print("Install via:  pip install -e '.[ml]'")
        sys.exit(1)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/reviews/raw")
    parser.add_argument("--cache-db", default="data/reviews/aspects_cache.duckdb")
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Cap on number of reviews to use (default: all)")
    parser.add_argument("--min-mentions", type=int, default=2,
                        help="Drop reviews with fewer than this many ingredient mentions")
    parser.add_argument("--extractor", choices=["mock", "claude"], default="mock")
    parser.add_argument("--cost-ceiling-usd", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = auto (32 on CPU, 128 on GPU)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip aspect extraction step (use whatever is in cache)")
    # ----- GPU-specific options -----
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (CUDA only)")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers; -1 = auto (0 CPU, 4 GPU)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = off)")
    # ----- experiment tracking -----
    parser.add_argument("--wandb-project", default=None,
                        help="Enable WandB logging under this project name")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--tag", default=None,
                        help="Free-form tag stored in log + WandB (e.g. 'autodl_rtx3090')")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading vocab + reviews ...")
    vocab = load_default_vocab()
    store = RawReviewStore(raw_dir)
    df = store.read()
    print(f"      vocab: {len(vocab)} ingredients")
    print(f"      reviews on disk: {len(df)} across {len(store.list_shards())} shards")
    if args.max_reviews:
        df = df.head(args.max_reviews)
        print(f"      capped to {len(df)}")

    print("\n[2/6] Aspect extraction ...")
    if args.extractor == "claude":
        try:
            extractor = ClaudeAspectExtractor()
        except Exception as e:
            print(f"      Claude unavailable ({e}); falling back to mock")
            extractor = MockAspectExtractor()
    else:
        extractor = MockAspectExtractor()
    print(f"      using {extractor.version}")

    cache = AspectCache(args.cache_db)
    if not args.no_extract:
        records_for_extraction = [
            ReviewRecord(
                review_id=row["review_id"], source=row["source"],
                brand=row.get("brand"), sku=row.get("sku"), text=row["text"],
                customization_raw=row.get("customization_raw"),
                rating=row.get("rating"), source_url=row.get("source_url"),
            )
            for row in df.to_dict(orient="records")
        ]
        pipeline = AspectExtractionPipeline(extractor=extractor, cache=cache)
        stats = pipeline.run_on_records(
            records_for_extraction, cost_ceiling_usd=args.cost_ceiling_usd
        )
        print(f"      extracted={stats.extracted}  cache_hits={stats.cache_hits}  "
              f"errors={stats.errors}  cost=${stats.cost_usd:.4f}")
    else:
        print("      (skipped; using existing cache)")

    print("\n[3/6] Loading aspect labels ...")
    labelled: list[tuple[str, str, dict]] = []   # (review_id, text, aspects)
    for row in df.to_dict(orient="records"):
        result = cache.get(row["review_id"], extractor.version)
        if result is None:
            continue
        # Only keep if at least one CORE_DIM is labelled
        core = {d: result.aspects.get(d) for d in CORE_DIMS}
        if all(v is None for v in core.values()):
            continue
        labelled.append((row["review_id"], row["text"], core))
    print(f"      labelled reviews: {len(labelled)} / {len(df)}")

    print("\n[4/6] Extracting ingredient mentions + building graphs ...")
    mention_extractor = IngredientMentionExtractor(vocab)
    graphs: list[Data] = []
    mention_counts: list[int] = []
    for review_id, text, aspects in labelled:
        ings = mention_extractor.extract(text)
        if len(ings) < args.min_mentions:
            continue
        if len(ings) > 30:
            ings = ings[:30]      # bound graph size
        graph = build_graph(ings, vocab, aspects)
        graphs.append(graph)
        mention_counts.append(len(ings))
    print(f"      graphs built: {len(graphs)} (min-mentions={args.min_mentions})")
    if graphs:
        c = Counter(mention_counts)
        top = sorted(c.items())[:8]
        print(f"      mention-count distribution (sample): {top}")

    if len(graphs) < 20:
        print(f"\nERROR: only {len(graphs)} valid graphs — too few to train.\n"
              f"Suggestions:\n"
              f"  - Lower --min-mentions (currently {args.min_mentions})\n"
              f"  - Add more reviews (run ingest scripts)\n"
              f"  - Use Claude extractor for richer aspect coverage")
        cache.close()
        sys.exit(2)

    print("\n[5/6] Train/val split + training ...")
    rng.shuffle(graphs)
    n_val = max(int(len(graphs) * args.val_frac), 10)
    train_graphs = graphs[n_val:]
    val_graphs = graphs[:n_val]
    print(f"      train: {len(train_graphs)}   val: {len(val_graphs)}")

    # ----- device selection -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    is_gpu = device.type == "cuda"
    if is_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"      GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        print("      device: CPU (slow; for production scale use GPU via AutoDL)")
        if args.amp:
            print("      WARN: --amp ignored on CPU")
            args.amp = False

    # ----- auto-scale batch size / workers when GPU available -----
    if args.batch_size == 0:
        args.batch_size = 128 if is_gpu else 32
        print(f"      auto batch_size: {args.batch_size}")
    if args.num_workers < 0:
        args.num_workers = 4 if is_gpu else 0
    print(f"      batch_size={args.batch_size}  workers={args.num_workers}  amp={args.amp}")

    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=is_gpu,
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=is_gpu,
    )

    model = SensoryGATProto(node_in_dim=NODE_FEAT_DIM, hidden=64, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and is_gpu) else None

    # ----- optional WandB -----
    wandb = None
    if args.wandb_project:
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       config=vars(args), tags=[args.tag] if args.tag else None)
            print(f"      WandB: project={args.wandb_project}  run={wandb.run.name}")
        except ImportError:
            print("      WARN: --wandb-project set but `wandb` not installed; skipping")

    log = {
        "args": vars(args), "epochs": [], "device": str(device),
        "gpu": gpu_name if is_gpu else None,
        "n_train": len(train_graphs), "n_val": len(val_graphs),
        "tag": args.tag, "started_at": datetime.now(UTC).isoformat(),
    }
    t0 = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=is_gpu)
            target = batch.y.view(-1, len(CORE_DIMS))
            mask = (~torch.isnan(target)).float()
            target_filled = torch.where(torch.isnan(target),
                                        torch.zeros_like(target), target)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    mean, logvar = model(batch)
                    loss = heteroscedastic_nll(target_filled, mean, logvar, mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                mean, logvar = model(batch)
                loss = heteroscedastic_nll(target_filled, mean, logvar, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss_sum += float(loss) * batch.num_graphs
            train_n += batch.num_graphs

        model.eval()
        val_preds = {d: [] for d in CORE_DIMS}
        val_targets = {d: [] for d in CORE_DIMS}
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = batch.y.view(-1, len(CORE_DIMS))
                mask = (~torch.isnan(target)).float()
                target_filled = torch.where(torch.isnan(target),
                                            torch.zeros_like(target), target)
                mean, logvar = model(batch)
                loss = heteroscedastic_nll(target_filled, mean, logvar, mask)
                val_loss_sum += float(loss) * batch.num_graphs
                val_n += batch.num_graphs
                for i, d in enumerate(CORE_DIMS):
                    m = mask[:, i].bool()
                    val_preds[d].extend(mean[m, i].cpu().tolist())
                    val_targets[d].extend(target[m, i].cpu().tolist())

        per_dim_r = {}
        for d in CORE_DIMS:
            if len(val_preds[d]) >= 3 and _SCIPY_OK:
                p = np.array(val_preds[d])
                t = np.array(val_targets[d])
                if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                    r, _ = pearsonr(p, t)
                    per_dim_r[d] = round(float(r), 3)
                else:
                    per_dim_r[d] = None
            else:
                per_dim_r[d] = None

        ep_train_loss = train_loss_sum / max(train_n, 1)
        ep_val_loss = val_loss_sum / max(val_n, 1)
        log["epochs"].append({
            "epoch": epoch, "train_loss": round(ep_train_loss, 4),
            "val_loss": round(ep_val_loss, 4), "val_pearson": per_dim_r,
        })
        # Compact one-line per-epoch status
        r_str = " ".join(f"{d}={per_dim_r[d] if per_dim_r[d] is not None else 'n/a':<5}"
                         for d in CORE_DIMS)
        improved = ep_val_loss < best_val_loss - 1e-4
        flag = " *" if improved else ""
        print(f"      epoch {epoch:>3d}  "
              f"train_loss={ep_train_loss:7.4f}  val_loss={ep_val_loss:7.4f}  {r_str}{flag}")

        # ----- WandB logging -----
        if wandb is not None:
            wandb_metrics = {
                "train/loss": ep_train_loss, "val/loss": ep_val_loss, "epoch": epoch,
            }
            for d, r in per_dim_r.items():
                if r is not None:
                    wandb_metrics[f"val/pearson_{d}"] = r
            wandb.log(wandb_metrics, step=epoch)

        # ----- best-val checkpoint + early stop -----
        if improved:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_improve += 1
            if args.patience > 0 and epochs_since_improve >= args.patience:
                print(f"      early stop at epoch {epoch} (no improvement for "
                      f"{args.patience} epochs; best epoch={best_epoch})")
                break

    elapsed = time.time() - t0
    print(f"      trained in {elapsed:.1f}s")
    log["elapsed_sec"] = round(elapsed, 1)
    log["best_epoch"] = best_epoch
    log["best_val_loss"] = round(best_val_loss, 4)

    print("\n[6/6] Saving model + log ...")
    model_path = out_dir / "sensory_gnn_stage1_prototype.pt"
    best_path = out_dir / "sensory_gnn_stage1_best.pt"
    log_path = out_dir / "sensory_gnn_stage1_log.json"
    # Save FINAL state
    torch.save({
        "state_dict": model.state_dict(),
        "model_arch": {"node_in_dim": NODE_FEAT_DIM, "hidden": 64, "heads": 4},
        "core_dims": list(CORE_DIMS),
        "categories": list(CATEGORIES),
        "saved_at": datetime.now(UTC).isoformat(),
        "tag": args.tag,
    }, model_path)
    print(f"      wrote {model_path}  (final state)")
    # Save BEST-val checkpoint separately
    if best_state is not None:
        torch.save({
            "state_dict": best_state,
            "model_arch": {"node_in_dim": NODE_FEAT_DIM, "hidden": 64, "heads": 4},
            "core_dims": list(CORE_DIMS),
            "categories": list(CATEGORIES),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "saved_at": datetime.now(UTC).isoformat(),
            "tag": args.tag,
        }, best_path)
        print(f"      wrote {best_path}  (best @ epoch {best_epoch}, val={best_val_loss:.4f})")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      wrote {log_path}")

    if wandb is not None:
        wandb.save(str(model_path))
        if best_state is not None:
            wandb.save(str(best_path))
        wandb.finish()

    cache.close()
    print(f"\nDone. Stage 1 prototype trained on {device.type.upper()} — pipeline validated.")


if __name__ == "__main__":
    main()
