#!/usr/bin/env python3
"""
run_ablation.py — Ablation study for TANNS-C.

Evaluates 3 ablation variants at ef_search=200 (in-process, no subprocess):
  - P1:   Pillar 1 only (category-aware graph, no snapshot, no fallback)
  - P2:   Pillar 2 only (temporal snapshots, no category graph, no fallback)
  - P1P2: Pillars 1+2 (no ACORN fallback)

Loads pre-computed ground truth from results/_state.pkl (run
compute_ground_truth.py first).

Usage:
    python benchmarks/run_ablation.py
    python benchmarks/run_ablation.py --data-dir data/ --results-dir results/
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import (
    load_fvecs, load_metadata, generate_queries, TOP10_CATEGORIES,
)
from src.tanns_c import TANNSC

logger = logging.getLogger(__name__)


def compute_metrics(all_result_ids, gt, k_values=(10, 100)):
    """Compute mean recall@k for each k, skipping empty ground truths."""
    recalls = {k: [] for k in k_values}
    for qi in range(len(all_result_ids)):
        gt_arr = gt[qi]
        if len(gt_arr) == 0:
            continue
        result_ids = all_result_ids[qi]
        for k in k_values:
            gt_topk = set(int(x) for x in gt_arr[:k])
            ret_topk = set(int(x) for x in result_ids[:k])
            denom = min(k, len(gt_topk))
            if denom > 0:
                recalls[k].append(len(ret_topk & gt_topk) / denom)
            else:
                recalls[k].append(1.0)
    return {k: float(np.mean(v)) for k, v in recalls.items()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="TANNS-C Ablation Study")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Directory containing dataset files")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Directory with _state.pkl and for output")
    parser.add_argument("--ef-search", type=int, default=200,
                        help="ef_search value for ablation runs")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load pre-computed state ──────────────────────────────────────
    state_path = os.path.join(args.results_dir, "_state.pkl")
    logger.info(f"Loading state from {state_path}")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    gt = state["gt"]
    NQ = state["NQ"]

    # ── Load data and build TANNS-C index ────────────────────────────
    # Auto-detect: try small split first, then medium
    for vec_name in ["database_vectors_small.fvecs", "database_vectors.fvecs"]:
        vec_path = os.path.join(args.data_dir, vec_name)
        if os.path.exists(vec_path):
            break
    for attr_name in ["database_attributes_small.jsonl", "database_attributes.jsonl"]:
        attr_path = os.path.join(args.data_dir, attr_name)
        if os.path.exists(attr_path):
            break

    logger.info(f"Loading dataset from {args.data_dir}")
    V = load_fvecs(vec_path)
    cats, udays = load_metadata(attr_path)

    logger.info("Generating queries (same seed as ground truth)...")
    queries = generate_queries(V, cats, udays, n_queries=NQ, seed=42)

    logger.info("Building TANNS-C index (one-time)...")
    tannsc = TANNSC(V, cats, udays, TOP10_CATEGORIES)

    # ── Ablation configs ─────────────────────────────────────────────
    ef = args.ef_search
    configs = [
        ("P1",   f"TANNS-C (P1 only)"),
        ("P2",   f"TANNS-C (P2 only)"),
        ("P1P2", f"TANNS-C (P1+P2)"),
    ]

    all_results = []

    for mode, method_name in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {method_name} (mode={mode}, ef={ef})")
        logger.info(f"{'='*60}")

        all_ids = []
        t0 = time.perf_counter()
        for qi, q in enumerate(queries):
            ids = tannsc.query(
                query_vector=q["query_vector"],
                target_category=q["target_category"],
                t_start=q["t_start"],
                t_end=q["t_end"],
                k=10,
                ef_search=ef,
                mode=mode,
            )
            all_ids.append(ids)
            if qi % 200 == 0:
                logger.info(f"  [{mode}] processed {qi}/{NQ}")
        total_time = time.perf_counter() - t0

        logger.info(f"  [{mode}] {NQ} queries in {total_time:.3f}s")

        recalls = compute_metrics(all_ids, gt)
        qps = NQ / total_time if total_time > 0 else 0
        latency = (total_time / NQ) * 1000

        row = {
            "method": method_name,
            "expansion_factor": ef,
            "recall@10": round(recalls[10], 4),
            "recall@100": round(recalls[100], 4),
            "QPS": round(qps, 1),
            "latency_ms": round(latency, 2),
        }
        all_results.append(row)
        logger.info(f"  R@10={row['recall@10']}, R@100={row['recall@100']}, "
              f"QPS={row['QPS']}, latency={row['latency_ms']}ms")

    # ── Save ablation results ────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "_ablation_partial.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved {len(all_results)} ablation rows to {out_path}")

    # ── Summary table ────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Method':<25} {'ef':>5} {'R@10':>8} {'R@100':>8} {'QPS':>10} {'Lat(ms)':>10}")
    logger.info(f"{'='*80}")
    for r in all_results:
        logger.info(f"{r['method']:<25} {r['expansion_factor']:>5} {r['recall@10']:>8.4f} "
              f"{r['recall@100']:>8.4f} {r['QPS']:>10.1f} {r['latency_ms']:>10.2f}")


if __name__ == "__main__":
    main()
