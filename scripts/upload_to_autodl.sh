#!/usr/bin/env bash
# Sync local code + data → AutoDL instance.
# Run from PROJECT ROOT on your LOCAL machine (Git Bash / WSL / mac / Linux).
#
# Usage:
#   bash scripts/upload_to_autodl.sh <ssh_url>
#   bash scripts/upload_to_autodl.sh root@region-X.seetacloud.com:25731
#
# Notes:
#   - Excludes .venv, __pycache__, .pytest_cache, large model checkpoints
#   - Uses AutoDL's fast scratch path /root/autodl-tmp/ if available, else /root/
#   - Run twice safely; rsync skips unchanged files

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 root@region-X.seetacloud.com:PORT"
    exit 1
fi

SSH_URL="$1"
# Split user@host:port
USER_HOST="${SSH_URL%:*}"
PORT="${SSH_URL##*:}"
REMOTE_DIR="${REMOTE_DIR:-/root/beverage_ai}"

echo "Uploading to $USER_HOST (port $PORT) → $REMOTE_DIR"

# Ensure remote dir exists
ssh -p "$PORT" "$USER_HOST" "mkdir -p $REMOTE_DIR/data/reviews/raw $REMOTE_DIR/data/ingredients $REMOTE_DIR/data/priors $REMOTE_DIR/data/recipes"

# 1. Code (small)
echo "--- syncing code ---"
rsync -avz --progress -e "ssh -p $PORT" \
    --exclude '.venv*' --exclude '__pycache__' --exclude '.pytest_cache' \
    --exclude '*.duckdb' --exclude 'models/*.pt' \
    --exclude 'data/reviews/raw' --exclude 'data/reviews/aspects_cache.duckdb' \
    --exclude '.git' --exclude 'node_modules' \
    beverage_ai/ scripts/ tests/ demo/ docs/ \
    pyproject.toml README.md .python-version .env.example \
    "$USER_HOST:$REMOTE_DIR/"

# 2. Static data (vocab, priors, recipes)
echo "--- syncing static data ---"
rsync -avz --progress -e "ssh -p $PORT" \
    data/ingredients/ "$USER_HOST:$REMOTE_DIR/data/ingredients/"
rsync -avz --progress -e "ssh -p $PORT" \
    data/priors/ "$USER_HOST:$REMOTE_DIR/data/priors/"
rsync -avz --progress -e "ssh -p $PORT" \
    data/recipes/ "$USER_HOST:$REMOTE_DIR/data/recipes/"

# 3. Reviews (likely larger — only sync if local has them)
if [[ -d data/reviews/raw ]] && [[ "$(ls -A data/reviews/raw 2>/dev/null)" ]]; then
    echo "--- syncing scraped reviews ---"
    rsync -avz --progress -e "ssh -p $PORT" \
        data/reviews/raw/ "$USER_HOST:$REMOTE_DIR/data/reviews/raw/"
else
    echo "--- skipping reviews (none locally; ingest on AutoDL instead) ---"
fi

# 4. Aspect cache (if exists, skip re-extraction on AutoDL)
if [[ -f data/reviews/aspects_cache.duckdb ]]; then
    SIZE_MB=$(du -m data/reviews/aspects_cache.duckdb | cut -f1)
    if [[ "$SIZE_MB" -lt 500 ]]; then
        echo "--- syncing aspect cache (${SIZE_MB}MB) ---"
        rsync -avz --progress -e "ssh -p $PORT" \
            data/reviews/aspects_cache.duckdb "$USER_HOST:$REMOTE_DIR/data/reviews/"
    else
        echo "--- aspect cache is ${SIZE_MB}MB; skipping (re-extract on AutoDL) ---"
    fi
fi

echo
echo "Upload complete."
echo "Now SSH in and run:"
echo "  ssh -p $PORT $USER_HOST"
echo "  cd $REMOTE_DIR && bash scripts/setup_autodl.sh"
