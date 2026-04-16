#!/bin/bash
# Full evaluation pipeline: benchmark -> JSON -> figures
# Usage: bash benchmarks/run_eval.sh
#
# Options (passed through to runner):
#   --baselines cupy jax    Run specific baselines
#   --all                   Run all baselines
#   --scaling 1 2 3 4 5     Multi-GPU scaling study

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== cuTile Stencil DSL — Evaluation Pipeline ==="
echo ""

# 1. Run benchmarks
echo "[1/2] Running benchmarks..."
python -m benchmarks.runner \
    --baselines cupy jax handwritten cuda cuda_smem \
    --scaling 1 2 4 \
    --warmup 30 --iters 100 \
    --output benchmarks/results/results.json \
    "$@"

# 2. Generate figures
echo ""
echo "[2/2] Generating figures..."
mkdir -p paper/figures
python -m benchmarks.plotting benchmarks/results/results.json --outdir paper/figures/

echo ""
echo "Done!"
echo "  Results: benchmarks/results/results.json"
echo "  Figures: paper/figures/"
