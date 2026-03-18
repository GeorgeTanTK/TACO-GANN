#!/usr/bin/env python3
"""
evaluate_all.py — Full evaluation harness for TANNS-C baselines.

Runs both PostFilter (HNSW) and PreFilter (brute-force) baselines on the
arxiv-for-FANNS dataset, measuring recall@10, recall@100, QPS, and latency.

Ground truth: exact brute-force search on the filtered subset.

Usage:
    python benchmarks/evaluate_all.py                          # SMALL split
    python benchmarks/evaluate_all.py --data-dir /path/to/data # custom path
    python benchmarks/evaluate_all.py --n-queries 5000         # More queries

Output:
    baseline_results.csv   — Tabular results
    baseline_results.json  — Same, as JSON (for chart generation)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import numpy as np
import hnswlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import (
    load_fvecs, load_metadata, generate_queries,
    build_filter_mask, epoch_day_to_year, TOP10_CATEGORIES,
)

logger = logging.getLogger(__name__)


# ── Recall computation ───────────────────────────────────────────────

def recall_at_k(retrieved_list, gt_list, k):
    """Compute mean recall@k across all queries (skip empty GTs)."""
    recalls = []
    for ret, gt in zip(retrieved_list, gt_list):
        if len(gt) == 0:
            continue
        gt_set = set(int(x) for x in gt[:k])
        ret_set = set(int(x) for x in ret[:k])
        denom = min(k, len(gt_set))
        if denom > 0:
            recalls.append(len(ret_set & gt_set) / denom)
    return np.mean(recalls) if recalls else 0.0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="TANNS-C Baseline Evaluation")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Directory containing dataset files")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hnsw-M", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Batch size for HNSW queries")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    NQ = args.n_queries
    CHUNK = args.chunk_size
    EFS = [2, 5, 10, 20, 50]

    # ── Load data ────────────────────────────────────────────────────
    data_dir = args.data_dir

    # Auto-detect: try small split first, then medium
    for vec_name in ["database_vectors_small.fvecs", "database_vectors.fvecs"]:
        vec_path = os.path.join(data_dir, vec_name)
        if os.path.exists(vec_path):
            break
    for attr_name in ["database_attributes_small.jsonl", "database_attributes.jsonl"]:
        attr_path = os.path.join(data_dir, attr_name)
        if os.path.exists(attr_path):
            break

    logger.info(f"Loading vectors: {vec_path}")
    V = load_fvecs(vec_path)
    N, D = V.shape
    logger.info(f"  {N:,} vectors × {D} dimensions")

    logger.info(f"Loading metadata: {attr_path}")
    categories, update_days = load_metadata(attr_path)
    logger.info(f"  Year range: {epoch_day_to_year(update_days.min())} – "
                f"{epoch_day_to_year(update_days.max())}")

    # Normalize for cosine similarity
    norms = np.linalg.norm(V, axis=1, keepdims=True).clip(1e-10)
    Vn = (V / norms).astype(np.float32)

    # Pre-build per-category boolean arrays
    cat_masks = {c: np.array([c in categories[i] for i in range(N)], dtype=bool)
                 for c in TOP10_CATEGORIES}

    # ── Generate queries ─────────────────────────────────────────────
    logger.info(f"Generating {NQ} queries...")
    queries = generate_queries(V, categories, update_days, n_queries=NQ, seed=args.seed)
    Q = np.stack([q["query_vector"] for q in queries]).astype(np.float32)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-10)

    # Pre-compute filter masks (vectorized)
    masks = [cat_masks[q["target_category"]] &
             (update_days >= q["t_start"]) & (update_days <= q["t_end"])
             for q in queries]

    sel = [m.sum() for m in masks]
    logger.info(f"  Filter selectivity: mean={np.mean(sel):.1f}, median={np.median(sel):.0f}, "
                f"min={min(sel)}, max={max(sel)}")

    # ── Ground truth ─────────────────────────────────────────────────
    logger.info("Computing ground truth (brute-force)...")
    gt = []
    for qi in range(NQ):
        fi = np.where(masks[qi])[0]
        if len(fi) == 0:
            gt.append(np.array([], dtype=np.int64))
            continue
        dists = 1.0 - Vn[fi] @ Qn[qi]
        k = min(100, len(fi))
        if k >= len(fi):
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, k)[:k]
            order = order[np.argsort(dists[order])]
        gt.append(fi[order])

    empty = sum(1 for g in gt if len(g) == 0)
    logger.info(f"  Empty queries (no matching vectors): {empty}/{NQ}")

    # ── Build HNSW index ─────────────────────────────────────────────
    logger.info(f"Building HNSW index (M={args.hnsw_M}, ef_c={args.hnsw_ef_construction})...")
    t0 = time.time()
    hnsw = hnswlib.Index(space="cosine", dim=D)
    hnsw.init_index(max_elements=N, M=args.hnsw_M, ef_construction=args.hnsw_ef_construction)
    hnsw.add_items(V, ids=np.arange(N))
    logger.info(f"  Built in {time.time()-t0:.2f}s")

    # ── Evaluate PostFilter ──────────────────────────────────────────
    logger.info("--- PostFilter-HNSW ---")
    results = []

    for ef in EFS:
        k10e = min(10 * ef, N)

        # k=10 retrieval pass (chunked batch)
        r10_all = []
        t_total = 0
        hnsw.set_ef(max(k10e, 50))
        for s in range(0, NQ, CHUNK):
            e = min(s + CHUNK, NQ)
            t0 = time.perf_counter()
            labels, _ = hnsw.knn_query(Q[s:e], k=k10e)
            t_total += time.perf_counter() - t0
            for i, qi in enumerate(range(s, e)):
                keep = [int(x) for x in labels[i] if masks[qi][int(x)]][:10]
                r10_all.append(np.array(keep, dtype=np.int64))

        # recall@100: if 100*ef >= N, post-filter gets all vectors → perfect recall
        if 100 * ef >= N:
            r100_val = 1.0
        else:
            k100e = 100 * ef
            r100_all = []
            hnsw.set_ef(max(k100e, 50))
            for s in range(0, NQ, CHUNK):
                e = min(s + CHUNK, NQ)
                labels, _ = hnsw.knn_query(Q[s:e], k=k100e)
                for i, qi in enumerate(range(s, e)):
                    keep = [int(x) for x in labels[i] if masks[qi][int(x)]][:100]
                    r100_all.append(np.array(keep, dtype=np.int64))
            r100_val = recall_at_k(r100_all, gt, 100)

        r10_val = recall_at_k(r10_all, gt, 10)
        qps = NQ / t_total
        lat = t_total / NQ * 1000

        results.append({
            "method": "PostFilter-HNSW",
            "expansion_factor": ef,
            "recall@10": round(r10_val, 4),
            "recall@100": round(r100_val, 4),
            "QPS": round(qps, 1),
            "latency_ms": round(lat, 2),
        })
        logger.info(f"  ef={ef:3d}: R@10={r10_val:.4f}  R@100={r100_val:.4f}  "
                     f"QPS={qps:,.0f}  lat={lat:.2f}ms")

    # ── Evaluate PreFilter ───────────────────────────────────────────
    logger.info("--- PreFilter-BruteForce ---")
    r10p, r100p = [], []
    t_total = 0
    for qi in range(NQ):
        fi = np.where(masks[qi])[0]
        t0 = time.perf_counter()
        if len(fi) == 0:
            r10p.append(np.array([], dtype=np.int64))
            r100p.append(np.array([], dtype=np.int64))
            t_total += time.perf_counter() - t0
            continue
        dists = 1.0 - Vn[fi] @ Qn[qi]
        k = min(100, len(fi))
        if k >= len(fi):
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, k)[:k]
            order = order[np.argsort(dists[order])]
        t_total += time.perf_counter() - t0
        r100p.append(fi[order])
        r10p.append(fi[order[:min(10, len(fi))]])

    r10pv = recall_at_k(r10p, gt, 10)
    r100pv = recall_at_k(r100p, gt, 100)
    qpsp = NQ / t_total
    latp = t_total / NQ * 1000

    results.append({
        "method": "PreFilter-BruteForce",
        "expansion_factor": "N/A",
        "recall@10": round(r10pv, 4),
        "recall@100": round(r100pv, 4),
        "QPS": round(qpsp, 1),
        "latency_ms": round(latp, 2),
    })
    logger.info(f"  R@10={r10pv:.4f}  R@100={r100pv:.4f}  "
                f"QPS={qpsp:,.0f}  lat={latp:.2f}ms")

    # ── Save results ─────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "baseline_results.csv")
    json_path = os.path.join(args.output_dir, "baseline_results.json")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "expansion_factor", "recall@10", "recall@100", "QPS", "latency_ms"
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 72}")
    logger.info(f"  Dataset: N={N:,}, D={D}, Queries={NQ}")
    logger.info(f"  {'Method':<25} {'EF':>5} {'R@10':>8} {'R@100':>8} {'QPS':>10} {'Lat(ms)':>10}")
    logger.info(f"  {'─' * 67}")
    for r in results:
        logger.info(f"  {r['method']:<25} {str(r['expansion_factor']):>5} "
              f"{r['recall@10']:>8.4f} {r['recall@100']:>8.4f} "
              f"{r['QPS']:>10.1f} {r['latency_ms']:>10.2f}")
    logger.info(f"{'=' * 72}")
    logger.info(f"  Results: {csv_path}")
    logger.info(f"           {json_path}")


if __name__ == "__main__":
    main()
