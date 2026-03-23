#!/usr/bin/env bash
# run_all.sh — End-to-end TANNS-C benchmark pipeline
# Usage: bash run_all.sh [--skip-download] [--data-dir DATA_DIR]
set -euo pipefail

DATA_DIR="data"
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo " TANNS-C Benchmark Pipeline"
echo "========================================"
echo "Repo root : $REPO_ROOT"
echo "Data dir  : $DATA_DIR"
echo ""

# Step 1 — Download dataset
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo "[1/6] Downloading arxiv-for-FANNS dataset..."
  python download_data.py --out-dir "$DATA_DIR"
else
  echo "[1/6] Skipping download (--skip-download)"
fi

# Step 2 — Compute ground truth
echo ""
echo "[2/6] Computing ground truth..."
python -m benchmarks.compute_ground_truth --data-dir "$DATA_DIR" --output-dir results

# Step 3 — Evaluate all baselines (PostFilter + PreFilter)
echo ""
echo "[3/6] Evaluating baselines..."
python -m benchmarks.evaluate_all --data-dir "$DATA_DIR" --output-dir results

# Step 4 — Run ablation study (TANNS-C variants)
echo ""
echo "[4/6] Running ablation study..."
python -m benchmarks.run_ablation --data-dir "$DATA_DIR" --results-dir results

# Step 5 — Compute selectivity recall & construction costs
echo ""
echo "[5/6] Computing selectivity and construction costs..."
python -m benchmarks.compute_selectivity --data-dir "$DATA_DIR" --results-dir results
python -m benchmarks.measure_construction --data-dir "$DATA_DIR" --results-dir results

# Step 6 — Generate figures
echo ""
echo "[6/6] Generating publication figures..."
python -m benchmarks.generate_figures --results-dir results --output-dir figures

echo ""
echo "========================================"
echo " Pipeline complete!"
echo " Results : results/"
echo " Figures : figures/"
echo "========================================"
