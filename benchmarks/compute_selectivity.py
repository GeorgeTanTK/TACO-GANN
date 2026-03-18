#!/usr/bin/env python3
"""
compute_selectivity.py — Compute per-selectivity-bin recall for Figure 5.

Bins queries by filter selectivity (% of dataset matching), then computes
recall@10 for each method within each bin.

Usage:
    python benchmarks/compute_selectivity.py
    python benchmarks/compute_selectivity.py --data-dir data/ --results-dir results/
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import numpy as np
import hnswlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_fvecs, load_metadata, generate_queries, TOP10_CATEGORIES
from src.baselines.acorn1 import ACORN1Baseline
from src.baselines.tanns import TimestampGraphBaseline
from src.baselines.filtered_diskann import FilteredDiskANNBaseline
from src.tanns_c import TANNSC

logger = logging.getLogger(__name__)


def recall_at_k_single(retrieved, gt_arr, k):
    """Compute recall@k for a single query. Returns None if GT is empty."""
    if len(gt_arr) == 0:
        return None  # skip
    gt_topk = set(int(x) for x in gt_arr[:k])
    ret_topk = set(int(x) for x in retrieved[:k])
    denom = min(k, len(gt_topk))
    return len(ret_topk & gt_topk) / denom if denom > 0 else 1.0


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compute per-selectivity-bin recall")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Directory containing dataset files")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Directory with _state.pkl and for output")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load state ───────────────────────────────────────────────────
    state_path = os.path.join(args.results_dir, "_state.pkl")
    logger.info(f"Loading state from {state_path}")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    gt = state["gt"]
    Ms = state["Ms"]
    Vn = state["Vn"]
    Qn = state["Qn"]
    N = state["N"]
    NQ = state["NQ"]

    # ── Load data for methods that need re-evaluation per bin ────────
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
    queries = generate_queries(V, cats, udays, n_queries=NQ, seed=42)

    # ── Compute per-query selectivity ────────────────────────────────
    selectivities = np.array([int(m.sum()) for m in Ms])
    selectivity_pct = selectivities / N * 100  # as percentage

    # Define bins (percentage of dataset)
    bin_edges = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    bin_labels = ["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%"]

    # Assign each query to a bin
    query_bins = np.digitize(selectivity_pct, bin_edges) - 1  # 0-indexed

    # ── Build indices ────────────────────────────────────────────────
    logger.info("Building indices...")

    # 1. PostFilter-HNSW
    hnsw = hnswlib.Index(space="cosine", dim=V.shape[1])
    hnsw.init_index(max_elements=N, M=32, ef_construction=200)
    hnsw.add_items(V, ids=np.arange(N))

    # 2. ACORN-1
    acorn = ACORN1Baseline(V, cats, udays)

    # 3. TANNS baseline
    tanns = TimestampGraphBaseline(V, cats, udays)

    # 4. FDiskANN
    fdann = FilteredDiskANNBaseline(V, cats, udays, TOP10_CATEGORIES)

    # 5. TANNS-C
    tannsc = TANNSC(V, cats, udays, TOP10_CATEGORIES)

    # ── Evaluate each method per query ───────────────────────────────
    logger.info("Running per-query evaluation...")

    methods_config = {
        "PostFilter": {"ef": 50},
        "ACORN-1": {"ef": 200},
        "TANNS+Post": {"ef": 200},
        "FDiskANN+Post": {"ef": 200},
        "TANNS-C": {"ef": 200},
        "PreFilter": {},
    }

    per_query_recall = {m: [None]*NQ for m in methods_config}

    for qi in range(NQ):
        q = queries[qi]
        qv = q["query_vector"]
        cat = q["target_category"]
        ts, te = q["t_start"], q["t_end"]

        if len(gt[qi]) == 0:
            continue

        # PostFilter ef=50
        k10e = min(10 * 50, N)
        hnsw.set_ef(max(k10e, 50))
        labels, _ = hnsw.knn_query(qv.reshape(1, -1), k=k10e)
        keep = [int(x) for x in labels[0] if Ms[qi][int(x)]][:10]
        per_query_recall["PostFilter"][qi] = recall_at_k_single(keep, gt[qi], 10)

        # PreFilter (brute force)
        fi = np.where(Ms[qi])[0]
        if len(fi) > 0:
            dists = 1.0 - Vn[fi] @ Qn[qi]
            k = min(10, len(fi))
            order = np.argsort(dists)[:k]
            per_query_recall["PreFilter"][qi] = recall_at_k_single(fi[order], gt[qi], 10)
        else:
            per_query_recall["PreFilter"][qi] = None

        # ACORN-1 ef=200
        ids_a, _ = acorn.query(qv, cat, ts, te, k=10, ef_search=200)
        per_query_recall["ACORN-1"][qi] = recall_at_k_single(ids_a, gt[qi], 10)

        # TANNS+PostFilter ef=200
        ids_t = tanns.query_postfilter(qv, cat, ts, te, k=10, ef_search=200)
        per_query_recall["TANNS+Post"][qi] = recall_at_k_single(ids_t, gt[qi], 10)

        # FDiskANN+PostFilter ef=200
        ids_f = fdann.query_postfilter(qv, cat, ts, te, k=10, ef_search=200)
        per_query_recall["FDiskANN+Post"][qi] = recall_at_k_single(ids_f, gt[qi], 10)

        # TANNS-C ef=200
        ids_c = tannsc.query(qv, cat, ts, te, k=10, ef_search=200, mode="full")
        per_query_recall["TANNS-C"][qi] = recall_at_k_single(ids_c, gt[qi], 10)

        if qi % 100 == 0:
            logger.info(f"  processed {qi}/{NQ}")

    # ── Bin results ──────────────────────────────────────────────────
    logger.info("Binning results...")

    results = {}
    for method in methods_config:
        results[method] = {}
        for bin_idx in range(len(bin_labels)):
            query_indices = np.where(query_bins == bin_idx)[0]
            recalls = [per_query_recall[method][qi] for qi in query_indices
                       if per_query_recall[method][qi] is not None]
            if len(recalls) > 0:
                results[method][bin_labels[bin_idx]] = {
                    "mean_recall": round(float(np.mean(recalls)), 4),
                    "count": len(recalls),
                }

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "_selectivity_recall.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved to {out_path}")

    # ── Print summary ────────────────────────────────────────────────
    header = f"{'Bin':<10}" + "".join(f"{m:>15}" for m in methods_config)
    logger.info(f"\n{header}")
    for bl in bin_labels:
        row = f"{bl:<10}"
        for m in methods_config:
            if bl in results[m]:
                row += f"{results[m][bl]['mean_recall']:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        logger.info(row)


if __name__ == "__main__":
    main()
