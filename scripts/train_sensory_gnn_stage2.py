"""Sensory GNN Stage 2 — fine-tune on panel scores.

Per 技术方案书 §3.3.1 fine-tune strategy:
  - Load Stage 1 checkpoint (weak-label pretrained on path A reviews)
  - Freeze backbone (conv1/2 + proj) AND extended head (10 path-A-only dims)
  - Update ONLY core 5-dim heads (head_mean / head_logvar)
  - Lower LR (5e-4), fewer epochs (20-30), heteroscedastic NLL

Input: panel ratings from feedback.duckdb (panel_score JOIN feedback table).
       Recipes are recovered from the `feedback.recipe_json` column so the
       script works for ANY recipe id — reference recipes OR pipeline-generated
       ones from a real session.

Output: models/sensory_gnn_stage2_best.pt + log JSON.

CLI:
    python scripts/train_sensory_gnn_stage2.py \
        --base-model models/sensory_gnn_stage1_best.pt \
        --feedback-db data/feedback.duckdb \
        --session-id s_synth_w1 \
        --epochs 30 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import pearsonr
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    _TORCH_OK = True
    _TORCH_ERR = None
except Exception as e:
    _TORCH_OK = False
    _TORCH_ERR = e

import duckdb

from beverage_ai.aspects.schema import CORE_DIMS
from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.recipes.schema import Recipe

# These MUST match what train_sensory_gnn_stage1.py uses
CATEGORIES = (
    "tea_base", "dairy_base", "alt_milk_base", "coffee_base",
    "sweetener", "fruit", "topping", "flavoring",
    "auxiliary", "gel", "grain",
)
NODE_FEAT_DIM = 5 + len(CATEGORIES)


# =============================================================================
# Same model arch as Stage 1 prototype
# =============================================================================

class SensoryGATProto(torch.nn.Module if _TORCH_OK else object):
    def __init__(self, node_in_dim: int = NODE_FEAT_DIM, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden, heads=heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads)
        self.proj = torch.nn.Linear(hidden * heads * 2, 128)
        self.head_mean = torch.nn.Linear(128, len(CORE_DIMS))
        self.head_logvar = torch.nn.Linear(128, len(CORE_DIMS))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.elu(self.conv2(x, ei))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        g = F.elu(self.proj(g))
        return self.head_mean(g), self.head_logvar(g).clamp(-6, 6)


def heteroscedastic_nll(target, mean, logvar, mask):
    sq = (target - mean) ** 2
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (inv_var * sq + logvar)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def ingredient_node_features(ing):
    nut = ing.nutrition_per_100g
    feat = [
        (nut.energy_kcal or 0) / 500.0,
        (nut.sugar_g or 0) / 100.0,
        (nut.fat_g or 0) / 50.0,
        (nut.caffeine_mg or 0) / 200.0,
        (nut.sodium_mg or 0) / 1000.0,
    ]
    cat_oh = [0.0] * len(CATEGORIES)
    if ing.category in CATEGORIES:
        cat_oh[CATEGORIES.index(ing.category)] = 1.0
    return torch.tensor(feat + cat_oh, dtype=torch.float32)


def build_recipe_graph(recipe: Recipe, vocab, aspect_targets: dict):
    """Build a graph directly from a Recipe object's ingredient ids.

    Unlike Stage 1 where graphs come from review keyword matching, here the
    recipe is *real* (from panel), so the graph faithfully represents it.
    """
    ids = [k for k in recipe.ingredients if k in vocab]
    if len(ids) < 2:
        return None
    x = torch.stack([ingredient_node_features(vocab.get(i)) for i in ids])
    n = len(ids)
    edge_src, edge_dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_src.append(i); edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor(
        [aspect_targets.get(d, float("nan")) for d in CORE_DIMS],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, y=y.unsqueeze(0))


# =============================================================================
# Panel data loading
# =============================================================================

def load_panel_session(db_path: str, session_id: str) -> list[tuple[Recipe, dict]]:
    """Read panel scores and aggregate to (recipe, {dim: mean_score in [0,1]}).

    Likert 1-5 → normalised to [0, 1] via (score-1)/4 to match the model's
    sigmoid-friendly output range.
    """
    con = duckdb.connect(db_path)
    rows = con.execute(
        """
        SELECT p.recipe_id, p.dimension, p.score, f.recipe_json
        FROM panel_score p
        LEFT JOIN feedback f
            ON p.session_id = f.session_id AND p.recipe_id = f.recipe_id
        WHERE p.session_id = ?
        """, [session_id],
    ).fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"no panel_score rows for session={session_id!r}")

    # Group: (recipe_id) → {dim: [scores]}
    per_recipe_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    recipe_json_by_id: dict[str, str] = {}
    for recipe_id, dimension, score, recipe_json in rows:
        per_recipe_scores[recipe_id][dimension].append(float(score))
        if recipe_json and recipe_id not in recipe_json_by_id:
            recipe_json_by_id[recipe_id] = recipe_json

    out: list[tuple[Recipe, dict[str, float]]] = []
    skipped = 0
    for rid, dim_scores in per_recipe_scores.items():
        rjson = recipe_json_by_id.get(rid)
        if not rjson:
            skipped += 1
            continue
        try:
            recipe = Recipe(**json.loads(rjson))
        except Exception:
            skipped += 1
            continue
        targets = {d: round((float(np.mean(s)) - 1) / 4, 4)
                   for d, s in dim_scores.items()}
        out.append((recipe, targets))
    print(f"      loaded {len(out)} recipes ({skipped} skipped: missing recipe_json)")
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    if not _TORCH_OK:
        print(f"ERROR: torch / torch_geometric missing ({_TORCH_ERR})")
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/sensory_gnn_stage1_best.pt")
    parser.add_argument("--feedback-db", default="data/feedback.duckdb")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading Stage 1 checkpoint: {args.base_model}")
    ckpt = torch.load(args.base_model, map_location="cpu", weights_only=False)
    arch = ckpt.get("model_arch", {"node_in_dim": NODE_FEAT_DIM, "hidden": 64, "heads": 4})
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"      device: {device}")

    model = SensoryGATProto(**arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    stage1_epoch = ckpt.get("best_epoch")
    print(f"      loaded weights (best @ epoch {stage1_epoch})")

    print(f"\n[2/5] Loading panel session {args.session_id!r} ...")
    pairs = load_panel_session(args.feedback_db, args.session_id)
    if len(pairs) < 5:
        print(f"ERROR: only {len(pairs)} (recipe, targets) pairs — need ≥ 5 for Stage 2")
        sys.exit(2)

    print("\n[3/5] Building graphs from panel recipes ...")
    vocab = load_default_vocab()
    graphs: list[Data] = []
    for recipe, targets in pairs:
        g = build_recipe_graph(recipe, vocab, targets)
        if g is not None:
            graphs.append(g)
    print(f"      built {len(graphs)} graphs")

    # Train/val split — by RECIPE (not by rating), so val is truly out-of-sample
    rng.shuffle(graphs)
    n_val = max(int(len(graphs) * args.val_frac), 2)
    train_graphs = graphs[n_val:]
    val_graphs = graphs[:n_val]
    print(f"      train: {len(train_graphs)}   val: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    print(f"\n[4/5] Freezing backbone + tuning core heads (lr={args.lr}) ...")
    # Freeze everything, then unfreeze just the heads
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head_mean.parameters():
        p.requires_grad = True
    for p in model.head_logvar.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"      trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.1f}%)")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # Track val Pearson r BEFORE fine-tune (i.e. Stage 1 zero-shot)
    initial_r = evaluate(model, val_loader, device)
    print("      [zero-shot Stage 1 on panel val]:")
    for d, v in initial_r.items():
        print(f"        {d}: {v if v is None else f'{v:+.3f}'}")

    log = {
        "args": vars(args),
        "device": str(device),
        "n_train": len(train_graphs), "n_val": len(val_graphs),
        "stage1_base_epoch": stage1_epoch,
        "trainable_params": n_trainable,
        "initial_zero_shot_pearson": initial_r,
        "epochs": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.y.view(-1, len(CORE_DIMS))
            mask = (~torch.isnan(target)).float()
            target_filled = torch.where(torch.isnan(target),
                                        torch.zeros_like(target), target)
            mean, logvar = model(batch)
            loss = heteroscedastic_nll(target_filled, mean, logvar, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            train_loss += float(loss) * batch.num_graphs
            train_n += batch.num_graphs

        val_loss, val_n, val_r = _eval_loss(model, val_loader, device)
        ep_train_loss = train_loss / max(train_n, 1)
        ep_val_loss = val_loss / max(val_n, 1)
        improved = ep_val_loss < best_val_loss - 1e-4
        log["epochs"].append({
            "epoch": epoch, "train_loss": round(ep_train_loss, 4),
            "val_loss": round(ep_val_loss, 4), "val_pearson": val_r,
        })
        r_str = " ".join(f"{d}={(val_r.get(d) if val_r.get(d) is not None else 'n/a'):<6}"
                         for d in CORE_DIMS)
        print(f"      epoch {epoch:>3d}  train={ep_train_loss:7.4f}  val={ep_val_loss:7.4f}  "
              f"{r_str}{'  *' if improved else ''}")
        if improved:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    log["elapsed_sec"] = round(elapsed, 1)
    log["best_epoch"] = best_epoch
    log["best_val_loss"] = round(best_val_loss, 4)
    log["final_val_pearson_after_finetune"] = val_r

    # Eval gains
    print("\n      ZERO-SHOT (Stage 1) vs FINE-TUNED (Stage 2):")
    for d in CORE_DIMS:
        z = initial_r.get(d)
        f_ = val_r.get(d)
        z_s = "n/a" if z is None else f"{z:+.3f}"
        f_s = "n/a" if f_ is None else f"{f_:+.3f}"
        delta = "" if (z is None or f_ is None) else (
            f"  Δ={f_ - z:+.3f}")
        print(f"        {d:<6}: {z_s} → {f_s}{delta}")

    print("\n[5/5] Saving Stage 2 checkpoints ...")
    final_path = out_dir / "sensory_gnn_stage2_final.pt"
    best_path = out_dir / "sensory_gnn_stage2_best.pt"
    log_path = out_dir / "sensory_gnn_stage2_log.json"

    torch.save({
        "state_dict": model.state_dict(),
        "model_arch": arch,
        "core_dims": list(CORE_DIMS),
        "categories": list(CATEGORIES),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage2_final",
        "session_id": args.session_id,
        "base_model": args.base_model,
    }, final_path)
    if best_state is not None:
        torch.save({
            "state_dict": best_state,
            "model_arch": arch,
            "core_dims": list(CORE_DIMS),
            "categories": list(CATEGORIES),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "stage": "stage2_best",
            "session_id": args.session_id,
            "base_model": args.base_model,
        }, best_path)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      wrote {final_path}")
    print(f"      wrote {best_path}  (best @ epoch {best_epoch}, val={best_val_loss:.4f})")
    print(f"      wrote {log_path}")
    print(f"\nDone. Stage 2 fine-tune complete in {elapsed:.1f}s.")


def evaluate(model, loader, device) -> dict[str, float | None]:
    model.eval()
    preds = {d: [] for d in CORE_DIMS}
    targets = {d: [] for d in CORE_DIMS}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, len(CORE_DIMS))
            mask = (~torch.isnan(target))
            mean, _ = model(batch)
            for i, d in enumerate(CORE_DIMS):
                m = mask[:, i]
                preds[d].extend(mean[m, i].cpu().tolist())
                targets[d].extend(target[m, i].cpu().tolist())
    out: dict[str, float | None] = {}
    for d in CORE_DIMS:
        if len(preds[d]) >= 3 and np.std(preds[d]) > 1e-6 and np.std(targets[d]) > 1e-6:
            r, _ = pearsonr(np.array(preds[d]), np.array(targets[d]))
            out[d] = round(float(r), 3)
        else:
            out[d] = None
    return out


def _eval_loss(model, loader, device) -> tuple[float, int, dict]:
    model.eval()
    val_loss = val_n = 0
    preds = {d: [] for d in CORE_DIMS}
    targets = {d: [] for d in CORE_DIMS}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, len(CORE_DIMS))
            mask = (~torch.isnan(target)).float()
            target_filled = torch.where(torch.isnan(target),
                                        torch.zeros_like(target), target)
            mean, logvar = model(batch)
            loss = heteroscedastic_nll(target_filled, mean, logvar, mask)
            val_loss += float(loss) * batch.num_graphs
            val_n += batch.num_graphs
            mb = mask.bool()
            for i, d in enumerate(CORE_DIMS):
                m = mb[:, i]
                preds[d].extend(mean[m, i].cpu().tolist())
                targets[d].extend(target[m, i].cpu().tolist())
    r_out: dict[str, float | None] = {}
    for d in CORE_DIMS:
        if len(preds[d]) >= 3 and np.std(preds[d]) > 1e-6 and np.std(targets[d]) > 1e-6:
            r, _ = pearsonr(np.array(preds[d]), np.array(targets[d]))
            r_out[d] = round(float(r), 3)
        else:
            r_out[d] = None
    return val_loss, val_n, r_out


if __name__ == "__main__":
    main()
