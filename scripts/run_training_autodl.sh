#!/usr/bin/env bash
# Run GNN Stage 1 training on AutoDL with sensible GPU defaults.
#
# Usage:
#   bash scripts/run_training_autodl.sh
#   EPOCHS=50 bash scripts/run_training_autodl.sh
#   EXTRACTOR=claude COST=30 bash scripts/run_training_autodl.sh   (needs ANTHROPIC_API_KEY)

set -euo pipefail
cd "$(dirname "$0")/.."

# venv check
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# Defaults
EPOCHS="${EPOCHS:-50}"
EXTRACTOR="${EXTRACTOR:-mock}"
COST="${COST:-5.0}"
PATIENCE="${PATIENCE:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
TAG="${TAG:-autodl}"

echo "Running training: epochs=$EPOCHS  extractor=$EXTRACTOR  patience=$PATIENCE  tag=$TAG"

WANDB_ARGS=""
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS="--wandb-project $WANDB_PROJECT"
fi

python scripts/train_sensory_gnn_stage1.py \
    --epochs "$EPOCHS" \
    --extractor "$EXTRACTOR" \
    --cost-ceiling-usd "$COST" \
    --device auto \
    --amp \
    --patience "$PATIENCE" \
    --tag "$TAG" \
    $WANDB_ARGS

echo
echo "Trained model saved to models/sensory_gnn_stage1_*.pt"
echo "Log: models/sensory_gnn_stage1_log.json"
echo "Pull back to local with rsync:"
echo "  rsync -avz <autodl_ssh>:/root/beverage_ai/models/ ./models/"
