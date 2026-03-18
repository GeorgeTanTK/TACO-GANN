#!/usr/bin/env python3
"""
measure_construction.py — Measure construction time and memory for all methods.

Usage:
    python benchmarks/measure_construction.py
    python benchmarks/measure_construction.py --data-dir data/ --results-dir results/
"""

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
import numpy as np
import hnswlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_fvecs, load_metadata, TOP10_CATEGORIES

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Measure construction time and memory")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Directory containing dataset files")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Auto-detect: try small split first, then medium
    for vec_name in ["database_vectors_small.fvecs", "database_vectors.fvecs"]:
        vec_path = os.path.join(args.data_dir, vec_name)
        if os.path.exists(vec_path):
            break
    for attr_name in ["database_attributes_small.jsonl", "database_attributes.jsonl"]:
        attr_path = os.path.join(args.data_dir, attr_name)
        if os.path.exists(attr_path):
            break

    V = load_fvecs(vec_path)
    cats, udays = load_metadata(attr_path)
    N, D = V.shape

    results = {}

    # 1. PostFilter-HNSW (single global HNSW)
    logger.info("Building PostFilter-HNSW...")
    tracemalloc.start()
    t0 = time.time()
    hnsw = hnswlib.Index(space="cosine", dim=D)
    hnsw.init_index(max_elements=N, M=32, ef_construction=200)
    hnsw.add_items(V, ids=np.arange(N))
    build_time = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["PostFilter"] = {"build_time_s": round(build_time, 3), "peak_mem_mb": round(peak / 1024**2, 2)}
    del hnsw

    # 2. ACORN-1 (HNSW + adjacency extraction)
    logger.info("Building ACORN-1...")
    tracemalloc.start()
    t0 = time.time()
    from src.baselines.acorn1 import ACORN1Baseline
    acorn = ACORN1Baseline(V, cats, udays)
    build_time = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["ACORN-1"] = {"build_time_s": round(build_time, 3), "peak_mem_mb": round(peak / 1024**2, 2)}
    del acorn

    # 3. TANNS (per-year HNSW)
    logger.info("Building TANNS...")
    tracemalloc.start()
    t0 = time.time()
    from src.baselines.tanns import TimestampGraphBaseline
    tanns = TimestampGraphBaseline(V, cats, udays)
    build_time = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["TANNS+Post"] = {"build_time_s": round(build_time, 3), "peak_mem_mb": round(peak / 1024**2, 2)}
    del tanns

    # 4. FDiskANN (per-category HNSW)
    logger.info("Building FDiskANN...")
    tracemalloc.start()
    t0 = time.time()
    from src.baselines.filtered_diskann import FilteredDiskANNBaseline
    fdann = FilteredDiskANNBaseline(V, cats, udays, TOP10_CATEGORIES)
    build_time = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["FDiskANN+Post"] = {"build_time_s": round(build_time, 3), "peak_mem_mb": round(peak / 1024**2, 2)}
    del fdann

    # 5. TANNS-C (category-aware graph + snapshots)
    logger.info("Building TANNS-C...")
    tracemalloc.start()
    t0 = time.time()
    from src.tanns_c import TANNSC
    tannsc = TANNSC(V, cats, udays, TOP10_CATEGORIES)
    build_time = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["TANNS-C"] = {"build_time_s": round(build_time, 3), "peak_mem_mb": round(peak / 1024**2, 2)}
    del tannsc

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "_construction_costs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved to {out_path}")

    logger.info(f"\n{'Method':<20} {'Build(s)':>10} {'Memory(MB)':>12}")
    logger.info("=" * 45)
    for m, d in results.items():
        logger.info(f"{m:<20} {d['build_time_s']:>10.3f} {d['peak_mem_mb']:>12.2f}")


if __name__ == "__main__":
    main()
