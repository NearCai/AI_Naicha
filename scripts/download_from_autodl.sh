#!/usr/bin/env bash
# Pull trained model + log + (optionally) aspects cache back from AutoDL.
#
# Usage:
#   bash scripts/download_from_autodl.sh <ssh_url>
#   bash scripts/download_from_autodl.sh root@region-X.seetacloud.com:25731

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 root@region-X.seetacloud.com:PORT"
    exit 1
fi

SSH_URL="$1"
USER_HOST="${SSH_URL%:*}"
PORT="${SSH_URL##*:}"
REMOTE_DIR="${REMOTE_DIR:-/root/beverage_ai}"

mkdir -p models data/reviews

echo "--- pulling models/ ---"
rsync -avz --progress -e "ssh -p $PORT" \
    "$USER_HOST:$REMOTE_DIR/models/" models/

echo "--- pulling aspect cache (optional) ---"
rsync -avz --progress -e "ssh -p $PORT" \
    "$USER_HOST:$REMOTE_DIR/data/reviews/aspects_cache.duckdb" \
    data/reviews/ || echo "no aspect cache found on remote"

echo
echo "Download complete. Inspect with:"
echo "  python scripts/report_training.py"
