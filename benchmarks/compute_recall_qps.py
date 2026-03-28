#!/usr/bin/env python3
"""
compute_recall_qps.py — Per-method recall@k and QPS for TACO-GANN baselines.

Methods:
- PostFilter-HNSW: HNSW over all vectors + dual post-filter (category AND time)
- TANNS+Post: TANNS timestamp graph + category post-filter
- TACO-GANN: Full TACO-GANN (category-aware + temporal HNT)

Outputs:
- results/_recall_qps.json with mean recall@k and QPS per method.
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

from src.data_loader import load_fvecs, load_metadata, generate_queries  # noqa: E402
from src.baselines.tanns_post_filtering import TANNS  # noqa: E402
from src.taco_gann import TACOGANN  # noqa: E402

logger = logging.getLogger(__name__)


def recall_at_k_single(retrieved, gt_arr, k: int) -> float | None:
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
        description="Compute recall@k and QPS for TACO-GANN baselines"
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
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 100],
        help="List of k values to compute recall@k for (e.g. 10 100)",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load state ─────────────────────────────────────────────────
    state_path = os.path.join(args.results_dir, "_state.pkl")
    logger.info("Loading state from %s", state_path)
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    gt = state["gt"]   # list/array of ground-truth ids per query
    Ms = state["Ms"]   # boolean masks per query (N,)
    N = state["N"]
    NQ = state["NQ"]

    # ── Load data ─────────────────────────────────────────────────
    # Auto-detect: try small split first, then full
    for vec_name in ["database_vectors_small.fvecs", "database_vectors.fvecs"]:
        vec_path = os.path.join(args.data_dir, vec_name)
        if os.path.exists(vec_path):
            break
    for attr_name in [
        "database_attributes_small.jsonl",
        "database_attributes.jsonl",
    ]:
        attr_path = os.path.join(args.data_dir, attr_name)
        if os.path.exists(attr_path):
            break

    logger.info("Loading dataset from %s", args.data_dir)
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

    # ── Build indices ─────────────────────────────────────────────
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

    methods = ["PostFilter", "TANNS+Post", "TACO-GANN"]
    ks = sorted(set(args.ks))

    # Per-query recalls per method and per k
    per_query_recall = {
        m: {k: [None] * NQ for k in ks} for m in methods
    }

    # Timing info for QPS
    total_query_time = {m: 0.0 for m in methods}
    total_queries = 0

    logger.info("Running per-query evaluation...")
    for qi in range(NQ):
        q = queries[qi]
        qv = q["query_vector"]
        cat = q["target_category"]
        ts, te = q["t_start"], q["t_end"]
        mask = Ms[qi]

        if len(gt[qi]) == 0:
            continue

        total_queries += 1

        # 1) PostFilter-HNSW
        k_max = max(ks)
        k_search = min(10 * args.ef_postfilter, N)
        hnsw.set_ef(max(k_search, args.ef_postfilter))
        t0 = time.perf_counter()
        labels, _ = hnsw.knn_query(qv.reshape(1, -1), k=k_search)
        elapsed = time.perf_counter() - t0
        total_query_time["PostFilter"] += elapsed
        filtered = [int(x) for x in labels[0] if mask[int(x)]][:k_max]
        for k in ks:
            per_query_recall["PostFilter"][k][qi] = recall_at_k_single(
                filtered, gt[qi], k
            )

        # 2) TANNS + post-filter (query at t_end)
        t0 = time.perf_counter()
        ids_t, _visited_t = tanns.query(
            qv, cat, te, k=k_max, ef=args.ef_tanns
        )
        elapsed = time.perf_counter() - t0
        total_query_time["TANNS+Post"] += elapsed
        ids_t_filtered = [int(x) for x in ids_t if mask[int(x)]][:k_max]
        for k in ks:
            per_query_recall["TANNS+Post"][k][qi] = recall_at_k_single(
                ids_t_filtered, gt[qi], k
            )

        # 3) TACO-GANN (full temporal range)
        t0 = time.perf_counter()
        ids_c, _visited_c = tacogann.query(
            qv, cat, ts, te, k=k_max, ef=args.ef_tacogann
        )
        elapsed = time.perf_counter() - t0
        total_query_time["TACO-GANN"] += elapsed
        ids_c_filtered = [int(x) for x in ids_c if mask[int(x)]][:k_max]
        for k in ks:
            per_query_recall["TACO-GANN"][k][qi] = recall_at_k_single(
                ids_c_filtered, gt[qi], k
            )

        if qi % 200 == 0:
            logger.info(" processed %d/%d", qi, NQ)

    # ── Aggregate recall and QPS ──────────────────────────────────
    logger.info("Aggregating results...")

    out = {}
    for m in methods:
        out[m] = {"recall": {}, "QPS": None}
        for k in ks:
            recalls = [
                r
                for r in per_query_recall[m][k]
                if r is not None
            ]
            if len(recalls) == 0:
                mean_r = None
            else:
                mean_r = float(np.mean(recalls))
            out[m]["recall"][f"@{k}"] = mean_r

        if total_query_time[m] > 0 and total_queries > 0:
            out[m]["QPS"] = float(total_queries / total_query_time[m])
        else:
            out[m]["QPS"] = None

    out_path = os.path.join(args.results_dir, "_recall_qps.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote results to %s", out_path)


if __name__ == "__main__":
    main()