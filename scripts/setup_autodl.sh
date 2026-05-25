#!/usr/bin/env bash
# Setup script for an AutoDL GPU instance.
#
# Run this AFTER you SSH into your AutoDL instance and your project code
# is in `/root/beverage_ai/` (via git clone or rsync).
#
# Usage:
#   bash scripts/setup_autodl.sh
#
# What it does:
#   - Sets HuggingFace mirror env (国内访问)
#   - Creates a Python venv (Python 3.10/3.11 already on most AutoDL images)
#   - Installs our package + [ml] + [hf] + [llm] extras
#   - Picks the right torch wheel for the installed CUDA
#   - Verifies GPU is visible

set -euo pipefail

cd "$(dirname "$0")/.."   # project root

# ----- AutoDL Python: usually at /root/miniconda3/bin/python, not on PATH -----
for cand in /root/miniconda3/bin /opt/conda/bin /root/miniconda/bin; do
    if [[ -d "$cand" ]]; then
        export PATH="$cand:$PATH"
        if ! grep -q "$cand" ~/.bashrc 2>/dev/null; then
            echo "export PATH=$cand:\$PATH" >> ~/.bashrc
        fi
        break
    fi
done

# ----- environment for HF mirror (国内访问加速) -----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF_ENDPOINT=$HF_ENDPOINT"

# Persist for future SSH sessions
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
    echo "Added HF_ENDPOINT to ~/.bashrc"
fi

# ----- detect CUDA version -----
if command -v nvidia-smi &> /dev/null; then
    CUDA_RAW=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "Detected CUDA driver: $CUDA_RAW"
    # Map driver CUDA to torch wheel index
    case "$CUDA_RAW" in
        12.4|12.5|12.6|12.7|12.8) TORCH_IDX="cu124" ;;
        12.1|12.2|12.3)            TORCH_IDX="cu121" ;;
        11.8)                       TORCH_IDX="cu118" ;;
        *)                          TORCH_IDX="cu121" ;;
    esac
    echo "Will install torch with index: $TORCH_IDX"
else
    echo "WARN: nvidia-smi not found; assuming no GPU. Will install CPU torch."
    TORCH_IDX="cpu"
fi

# ----- pick python -----
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_BIN ($PY_VER)"
if [[ "$PY_VER" != "3.11" && "$PY_VER" != "3.10" ]]; then
    echo "WARN: project tested on 3.10/3.11. Found $PY_VER. Continuing anyway."
fi

# ----- venv -----
if [[ ! -d .venv ]]; then
    echo "Creating .venv ..."
    $PYTHON_BIN -m venv .venv
fi
source .venv/bin/activate
echo "venv active: $(which python)"

pip install --upgrade pip -q

# ----- install torch with right CUDA wheel -----
echo "Installing torch (${TORCH_IDX}) ..."
if [[ "$TORCH_IDX" == "cpu" ]]; then
    pip install -q "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu
else
    pip install -q "torch>=2.1" --index-url "https://download.pytorch.org/whl/${TORCH_IDX}"
fi

# ----- install our package + extras -----
echo "Installing beverage_ai + extras ..."
pip install -q -e ".[ml,hf,llm]"

# ----- verify -----
echo
echo "===== verification ====="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
import torch_geometric
print(f'torch_geometric: {torch_geometric.__version__}')
import beverage_ai
print(f'beverage_ai: {beverage_ai.__version__}')
import datasets
print(f'datasets: {datasets.__version__}')
"
echo
echo "Setup complete."
echo "Next: bash scripts/run_training_autodl.sh    (or run train script directly)"
