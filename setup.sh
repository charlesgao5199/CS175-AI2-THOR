#!/usr/bin/env bash
# Setup script for the AI2-THOR Object Goal Navigation project.
# Creates a conda env 'objectnav' with python 3.10, installs deps,
# editable-installs this package, and verifies AI2-THOR + torch MPS.

set -euo pipefail

ENV_NAME="objectnav"
PY_VERSION="3.10"

# Locate conda and make `conda activate` usable inside this script
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Create env (idempotent)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[1/5] conda env '${ENV_NAME}' already exists — skipping creation."
else
    echo "[1/5] Creating conda env '${ENV_NAME}' with python ${PY_VERSION}..."
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# 2. Install requirements
echo "[2/5] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Editable install
echo "[3/5] Installing project (editable)..."
pip install -e .

# 4. Verify AI2-THOR
echo "[4/5] Verifying AI2-THOR launches a controller..."
python - <<'PY'
import sys
try:
    from ai2thor.controller import Controller
    # On macOS the default OSX build opens a small window briefly; this is fine.
    c = Controller(scene="FloorPlan1", width=300, height=300)
    pos = c.last_event.metadata["agent"]["position"]
    print(f"  AI2-THOR OK — agent at {pos}")
    c.stop()
except Exception as e:
    print(f"  AI2-THOR check FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY

# 5. Verify torch MPS
echo "[5/5] Verifying torch MPS backend..."
python - <<'PY'
import sys
import torch
ok = torch.backends.mps.is_available()
built = torch.backends.mps.is_built()
print(f"  torch {torch.__version__}  mps.is_available={ok}  mps.is_built={built}")
if not ok:
    print("  ERROR: MPS not available. Are you on Apple Silicon with a recent macOS + torch?", file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "Setup complete. Activate the env with:"
echo "  conda activate ${ENV_NAME}"
