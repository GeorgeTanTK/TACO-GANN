#!/usr/bin/env python3
"""
compute_selectivity.py — Per-selectivity-bin recall for TACO-GANN baselines.

Methods:
  - PostFilter-HNSW: HNSW over all vectors + dual post-filter (category AND time)
  - TANNS+Post:      TANNS timestamp graph + category post-filter
  - TACO-GANN:         Full TACO-GANN (category-aware + temporal HNT)

Bins queries by filter selectivity (% of dataset matching C × [t_start,t_end]),
then computes recall@10 for each method within each bin.

Prerequisite:
  - results/_state.pkl must exist, containing at least:
      gt      : list/array of ground-truth ids per query
      Ms      : list/array of boolean masks (N) per query for C × [t_start,t_end]
      Vn, Qn  : normalized vectors (optional; used only for brute-force baseline)
      N, NQ   : corpus size and number of queries

Usage:
  python benchmarks/compute_selectivity.py
  python benchmarks/compute_selectivity.py --data-dir data --results-dir results
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

from src.data_loader import load_fvecs, load_metadata, generate_queries
from src.baselines.tanns_post_filtering import TANNS
from src.taco_gann import TACOGANN

logger = logging.getLogger(__name__)


def recall_at_k_single(retrieved, gt_arr, k=10):
    """Compute recall@k for a single query. Returns None if GT is empty."""
    if len(gt_arr) == 0:
        return None
    gt_topk = set(int(x) for x in gt_arr[:k])
    ret_topk = set(int(x) for x in retrieved[:k])
    denom = min(k, len(gt_topk))
    return len(ret_topk & gt_topk) / denom if denom > 0 else 1.0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compute per-selectivity-bin recall for TACO-GANN baselines"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(REPO_ROOT, "data"),
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(REPO_ROOT, "results"),
        help="Directory with _state.pkl and for output",
    )
    parser.add_argument(
        "--ef-postfilter",
        type=int,
        default=50,
        help="ef_search for HNSW PostFilter baseline",
    )
    parser.add_argument(
        "--ef-tanns",
        type=int,
        default=200,
        help="ef_search for TANNS+Post baseline",
    )
    parser.add_argument(
        "--ef-tacogann",
        type=int,
        default=200,
        help="ef_search for TACO-GANN baseline",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load state ───────────────────────────────────────────────────
    state_path = os.path.join(args.results_dir, "_state.pkl")
    logger.info(f"Loading state from {state_path}")
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    gt = state["gt"]        # list/array of ground-truth ids per query
    Ms = state["Ms"]        # boolean masks per query (N,)
    N = state["N"]
    NQ = state["NQ"]

    # ── Load data ────────────────────────────────────────────────────
    # Auto-detect: try small split first, then full
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

    # Prefer queries saved in state (avoids seed / version mismatch)
    if "qs" in state:
        queries = state["qs"]
    else:
        queries = generate_queries(V, cats, udays, n_queries=NQ, seed=42)

    assert V.shape[0] == N, f"V.shape[0]={V.shape[0]} but N={N}"

    # Normalize udays for TANNS / TACO-GANN builds (numpy int64 → Python int)
    udays_list = [int(d) for d in udays]

    # ── Compute per-query selectivity ────────────────────────────────
    # |valid_set_q| = Ms[q].sum(), selectivity as percentage of dataset
    selectivities = np.array([int(m.sum()) for m in Ms])
    selectivity_pct = selectivities / N * 100.0

    # Define bins (percentage of dataset)
    bin_edges = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    bin_labels = ["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%"]
    query_bins = np.digitize(selectivity_pct, bin_edges) - 1  # 0-indexed

    # ── Build indices ────────────────────────────────────────────────
    logger.info("Building indices...")

    # 1. PostFilter-HNSW (no metadata awareness, dual post-filter)
    hnsw = hnswlib.Index(space="cosine", dim=V.shape[1])
    hnsw.init_index(max_elements=N, M=32, ef_construction=200)
    hnsw.add_items(V, ids=np.arange(N))

    # 2. TANNS (timestamp graph, category as post-filter)
    tanns = TANNS()
    tanns.build(V, cats, udays_list)

    # 3. TACO-GANN (full temporal + category-aware)
    tacogann = TACOGANN()
    tacogann.build(V, cats, udays_list)

    # ── Evaluate per query ───────────────────────────────────────────
    methods = ["PostFilter", "TANNS+Post", "TACO-GANN"]
    per_query_recall = {m: [None] * NQ for m in methods}

    logger.info("Running per-query evaluation...")
    t0 = time.perf_counter()

    for qi in range(NQ):
        q = queries[qi]
        qv = q["query_vector"]
        cat = q["target_category"]
        ts, te = q["t_start"], q["t_end"]
        mask = Ms[qi]

        if len(gt[qi]) == 0:
            continue

        # 1) PostFilter-HNSW
        #   - search k' = 10 * ef (clipped at N)
        #   - then filter candidates by Ms[qi]
        k10e = min(10 * args.ef_postfilter, N)
        hnsw.set_ef(max(k10e, args.ef_postfilter))
        labels, _ = hnsw.knn_query(qv.reshape(1, -1), k=k10e)
        filtered = [int(x) for x in labels[0] if mask[int(x)]][:10]
        per_query_recall["PostFilter"][qi] = recall_at_k_single(filtered, gt[qi], 10)

        # 2) TANNS + post-filter
        #    TANNS is a point-in-time index.  Query at t_end so its
        #    valid_set ({v: start_days[v] <= t_end}) is a superset of the
        #    GT valid_set ({v: t_start <= udays[v] <= t_end}), then
        #    post-filter with the range mask.
        ids_t, _visited_t = tanns.query(
            qv, cat, te, k=10, ef=args.ef_tanns
        )
        ids_t_filtered = [int(x) for x in ids_t if mask[int(x)]][:10]
        per_query_recall["TANNS+Post"][qi] = recall_at_k_single(ids_t_filtered, gt[qi], 10)

        # 3) TACO-GANN
        ids_c, _visited_c = tacogann.query(
            qv, cat, ts, te, k=10, ef=args.ef_tacogann
        )
        ids_c_filtered = [int(x) for x in ids_c if mask[int(x)]][:10]
        per_query_recall["TACO-GANN"][qi] = recall_at_k_single(ids_c_filtered, gt[qi], 10)

        if qi % 200 == 0:
            logger.info(f" processed {qi}/{NQ}")

    total_time = time.perf_counter() - t0
    logger.info(f"Finished evaluation of {NQ} queries in {total_time:.2f}s")

    # ── Bin results and aggregate ────────────────────────────────────
    logger.info("Binning results by selectivity...")

    results = {m: {} for m in methods}

    for bin_idx in range(len(bin_labels)):
        q_idx = np.where(query_bins == bin_idx)[0]
        for m in methods:
            recalls = [
                per_query_recall[m][qi]
                for qi in q_idx
                if per_query_recall[m][qi] is not None
            ]
            if len(recalls) > 0:
                results[m][bin_labels[bin_idx]] = {
                    "mean_recall": round(float(np.mean(recalls)), 4),
                    "count": len(recalls),
                }

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "_selectivity_recall.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "bins": bin_labels,
                "methods": methods,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved per-selectivity recall to {out_path}")

    # ── Print summary table ─────────────────────────────────────────
    header = f"{'Bin':<10}" + "".join(f"{m:>15}" for m in methods)
    logger.info("\n" + header)
    for bl in bin_labels:
        row = f"{bl:<10}"
        for m in methods:
            if bl in results[m]:
                row += f"{results[m][bl]['mean_recall']:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        logger.info(row)


if __name__ == "__main__":
    main()
