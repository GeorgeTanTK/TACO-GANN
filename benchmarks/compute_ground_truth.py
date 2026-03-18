#!/usr/bin/env python3
"""
compute_ground_truth.py — Setup + ground truth computation + save intermediate state.

Loads the dataset, generates queries, computes brute-force ground truth for
recall evaluation, and saves the complete state to a pickle file for
downstream benchmark scripts.

Usage:
    python benchmarks/compute_ground_truth.py
    python benchmarks/compute_ground_truth.py --data-dir data/ --output-dir results/
"""

import argparse
import logging
import os
import pickle
import sys
import time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_fvecs, load_metadata, generate_queries, TOP10_CATEGORIES

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compute ground truth for TANNS-C benchmarks")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Directory containing dataset files")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Directory to save state pickle")
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    NQ = args.n_queries

    # ── Load data ────────────────────────────────────────────────────
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

    norms = np.linalg.norm(V, axis=1, keepdims=True).clip(1e-10)
    Vn = (V / norms).astype(np.float32)
    cm = {c: np.array([c in cats[i] for i in range(N)], dtype=bool) for c in TOP10_CATEGORIES}

    qs = generate_queries(V, cats, udays, n_queries=NQ, seed=args.seed)
    Q = np.stack([q["query_vector"] for q in qs]).astype(np.float32)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-10)
    Ms = [cm[q["target_category"]] & (udays >= q["t_start"]) & (udays <= q["t_end"]) for q in qs]

    logger.info(f"N={N} D={D} Q={NQ} sel_mean={np.mean([m.sum() for m in Ms]):.1f}")

    # ── Ground truth ─────────────────────────────────────────────────
    gt = []
    for qi in range(NQ):
        fi = np.where(Ms[qi])[0]
        if len(fi) == 0:
            gt.append(np.array([], dtype=np.int64))
            continue
        d = 1.0 - Vn[fi] @ Qn[qi]
        k = min(100, len(fi))
        o = np.argsort(d)[:k] if k >= len(fi) else np.argpartition(d, k)[:k]
        if k < len(fi):
            o = o[np.argsort(d[o])]
        gt.append(fi[o])

    logger.info(f"GT done. Empty={sum(1 for g in gt if len(g) == 0)}")

    # ── Save state ───────────────────────────────────────────────────
    state = {"N": N, "D": D, "NQ": NQ, "V": V, "Vn": Vn, "Q": Q, "Qn": Qn, "Ms": Ms, "gt": gt}
    out_path = os.path.join(args.output_dir, "_state.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(state, f)
    logger.info(f"State saved to {out_path}")


if __name__ == "__main__":
    main()
