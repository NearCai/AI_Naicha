"""Real sensory GNN — GATv2 + dual output heads.

Corresponds to v1 实现方案 §6.6.

This module requires torch + torch_geometric. If those are not installed,
importing this file raises ImportError; the rest of beverage_ai still
works via `MockSensoryPredictor`.

Training scripts live in `scripts/train_sensory_gnn_stage1.py` and
`scripts/train_sensory_gnn_stage2.py` (TBD — see implementation plan §6.6
training-loop notes).
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    _TORCH_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERR = e
    _TORCH_AVAILABLE = False

from .predict import CORE_DIMS, EXT_DIMS


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch + torch_geometric required for SensoryGAT. "
            "Install with: pip install -e .[ml]"
        ) from _IMPORT_ERR  # type: ignore[name-defined]


def build_sensory_gat(
    node_in_dim: int,
    custom_feat_dim: int,
    hidden: int = 128,
    heads: int = 4,
):
    """Factory that returns a SensoryGAT module."""
    _require_torch()

    class SensoryGAT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GATv2Conv(node_in_dim, hidden, heads=heads)
            self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads)
            self.proj = nn.Linear(hidden * heads * 2, 256)
            # Dual heads (§3.3.1)
            inp = 256 + custom_feat_dim
            self.head_core_mean = nn.Linear(inp, len(CORE_DIMS))
            self.head_core_logvar = nn.Linear(inp, len(CORE_DIMS))
            self.head_ext_mean = nn.Linear(inp, len(EXT_DIMS))
            self.head_ext_logvar = nn.Linear(inp, len(EXT_DIMS))

        def forward(self, data: "Batch", customization: "torch.Tensor") -> dict:  # noqa: UP037
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, 0.2, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            g = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
            )
            g = F.elu(self.proj(g))
            g = torch.cat([g, customization], dim=-1)
            return {
                "core_mean": self.head_core_mean(g),
                "core_logvar": self.head_core_logvar(g),
                "ext_mean": self.head_ext_mean(g),
                "ext_logvar": self.head_ext_logvar(g),
            }

    return SensoryGAT()


def nll_gaussian_heteroscedastic(target, mean, logvar):
    """Heteroscedastic Gaussian NLL — used in Stage 2 fine-tune."""
    _require_torch()
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (target - mean) ** 2 + logvar).mean()
