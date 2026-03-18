#!/usr/bin/env python3
"""
filtered_diskann.py — Simplified Filtered-DiskANN Baseline for TANNS-C.

Approximates the Filtered-DiskANN approach (Gollapudi et al., NeurIPS 2023)
using per-category HNSW indices. This is analogous to StitchedVamana without
the stitching step: each category label gets its own navigable graph built
only on vectors carrying that label.

Architecture:
  - For each of the top-10 arXiv categories, build a separate HNSW index
    containing only vectors tagged with that category.
  - Compute per-category medoids (vector closest to the category centroid)
    as conceptual entry points (used internally by HNSW).
  - At query time with filter (category=C, time=[t_start, t_end]):
    search C's dedicated HNSW index, then apply temporal filtering.

Two temporal filter variants:

  Variant A — FDiskANN + Post-filter on time:
    1. Search the category-C HNSW index (category is already satisfied)
    2. Post-filter results by time window [t_start, t_end]
    Tradeoff: fast HNSW search, but may lose results outside time window

  Variant B — FDiskANN + Pre-filter on time:
    1. Identify vectors in category C that also satisfy the time window
    2. Brute-force cosine search on that subset
    Tradeoff: exact recall on the category-temporal intersection

Why per-category indices instead of filtered beam search:
  hnswlib does not expose internal graph traversal or custom neighbor
  expansion. Building per-label indices achieves the same effect as
  Filtered-DiskANN's label-restricted graph navigation — each index
  only contains same-category vectors, so every graph edge respects
  the category filter by construction.

Parameters:
  - Per-category HNSW: M=16, ef_construction=200
  - ef_search ∈ {50, 100, 200, 500, 1000} for Variant A
  - Variant B is brute-force (no ef_search tuning)

References:
  Gollapudi et al., "Filtered-DiskANN: Graph Algorithms for Approximate
  Nearest Neighbor Search with Filters", NeurIPS 2023.

  Shen et al., "StitchedVamana: Label-specific Index Construction for
  Filtered ANN Search", 2024.
"""

import logging
import time
import numpy as np
import hnswlib
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class FilteredDiskANNBaseline:
    """Per-category HNSW indices for category-aware filtered ANN search."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
        target_categories: List[str],
        M: int = 16,
        ef_construction: int = 200,
    ):
        """
        Build per-category HNSW indices and compute medoids.

        Args:
            vectors: (N, D) float32 database embeddings.
            categories: list of main categories per paper.
            update_days: (N,) int32 epoch-day timestamps.
            target_categories: list of category labels to index.
            M: HNSW graph out-degree per index.
            ef_construction: HNSW build-time exploration factor.
        """
        self.N, self.D = vectors.shape
        self.vectors = vectors
        self.categories = categories
        self.update_days = update_days

        # Pre-normalize for cosine distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normed = (vectors / norms).astype(np.float32)

        logger.info(f"[FDiskANN] Building per-category HNSW indices...")
        t0 = time.time()

        # category → global IDs of vectors with that category
        self.cat_to_ids: Dict[str, np.ndarray] = {}
        # category → HNSW index (local IDs map via cat_to_ids)
        self.cat_hnsw: Dict[str, hnswlib.Index] = {}
        # category → normalized vectors for brute-force variant
        self.cat_normed: Dict[str, np.ndarray] = {}
        # category → medoid global ID
        self.cat_medoid: Dict[str, int] = {}

        for cat in target_categories:
            # Find all vectors with this category
            global_ids = np.array(
                [i for i in range(self.N) if cat in categories[i]],
                dtype=np.int32
            )
            if len(global_ids) == 0:
                continue

            self.cat_to_ids[cat] = global_ids
            cat_vecs = vectors[global_ids]
            cat_normed = self.normed[global_ids]
            self.cat_normed[cat] = cat_normed

            # Compute medoid: vector closest to the centroid
            centroid = cat_normed.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            sims = cat_normed @ centroid
            medoid_local = int(np.argmax(sims))
            self.cat_medoid[cat] = int(global_ids[medoid_local])

            # Build HNSW index for this category
            n_cat = len(global_ids)
            m_eff = min(M, max(2, n_cat // 2))
            idx = hnswlib.Index(space="cosine", dim=self.D)
            idx.init_index(max_elements=n_cat, M=m_eff,
                           ef_construction=ef_construction)
            idx.add_items(cat_vecs, ids=np.arange(n_cat))
            self.cat_hnsw[cat] = idx

        elapsed = time.time() - t0
        logger.info(f"[FDiskANN] Built {len(self.cat_hnsw)} category indices "
                     f"in {elapsed:.2f}s")
        for cat in target_categories:
            if cat in self.cat_to_ids:
                n = len(self.cat_to_ids[cat])
                med = self.cat_medoid[cat]
                logger.info(f"  {cat:>10s}: {n:4d} vectors, medoid={med}")

    # ── Variant A: FDiskANN + Post-filter on time ────────────────────

    def query_postfilter(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
        ef_search: int = 100,
    ) -> np.ndarray:
        """
        Variant A — Search category HNSW, post-filter by time window.

        1. Search the category-specific HNSW index
        2. Map local IDs → global IDs
        3. Post-filter by time window [t_start, t_end]
        4. Return top-k by cosine distance

        Returns:
            ids: array of global database indices (up to k).
        """
        if target_category not in self.cat_hnsw:
            return np.array([], dtype=np.int64)

        qvec = query_vector.astype(np.float32).reshape(1, -1)
        hnsw = self.cat_hnsw[target_category]
        gids = self.cat_to_ids[target_category]
        n_cat = len(gids)

        # Search with ef_search (capped to category size)
        search_k = min(ef_search, n_cat)
        if search_k == 0:
            return np.array([], dtype=np.int64)

        hnsw.set_ef(max(search_k, 10))
        labels, dists = hnsw.knn_query(qvec, k=search_k)

        # Map to global IDs and post-filter by time
        filtered = []
        for local_id, dist in zip(labels[0], dists[0]):
            global_id = int(gids[int(local_id)])
            if t_start <= self.update_days[global_id] <= t_end:
                filtered.append((float(dist), global_id))

        if not filtered:
            return np.array([], dtype=np.int64)

        filtered.sort(key=lambda x: x[0])
        return np.array([f[1] for f in filtered[:k]], dtype=np.int64)

    # ── Variant B: FDiskANN + Pre-filter on time ─────────────────────

    def query_prefilter(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
    ) -> np.ndarray:
        """
        Variant B — Pre-filter by time within category, brute-force search.

        1. Within the category's vectors, select those in [t_start, t_end]
        2. Brute-force cosine distance on the subset
        3. Return top-k

        Returns:
            ids: array of global database indices (up to k).
        """
        if target_category not in self.cat_to_ids:
            return np.array([], dtype=np.int64)

        qvec = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qnormed = qvec / qnorm
        else:
            qnormed = qvec

        gids = self.cat_to_ids[target_category]
        cat_normed = self.cat_normed[target_category]

        # Pre-filter: category vectors within time window
        time_mask = np.array([
            t_start <= self.update_days[gid] <= t_end
            for gid in gids
        ], dtype=bool)

        if not np.any(time_mask):
            return np.array([], dtype=np.int64)

        subset_normed = cat_normed[time_mask]
        subset_gids = gids[time_mask]

        # Brute-force cosine distance
        sims = subset_normed @ qnormed
        dists = 1.0 - sims

        # Top-k
        if len(dists) <= k:
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, k)[:k]
            order = order[np.argsort(dists[order])]

        return np.array([int(subset_gids[i]) for i in order], dtype=np.int64)


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data_loader import (load_fvecs, load_metadata, generate_queries,
                                 TOP10_CATEGORIES)

    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    parser = argparse.ArgumentParser(description="Filtered-DiskANN baseline demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vectors = load_fvecs(os.path.join(data_dir, "database_vectors_small.fvecs"))
    categories, update_days = load_metadata(
        os.path.join(data_dir, "database_attributes_small.jsonl"))

    fdann = FilteredDiskANNBaseline(
        vectors, categories, update_days, TOP10_CATEGORIES)

    queries = generate_queries(vectors, categories, update_days,
                               n_queries=10, seed=42)

    logger.info("\n--- Variant A: FDiskANN + Post-filter ---")
    for ef in [50, 100, 200]:
        results_a = []
        t0 = time.perf_counter()
        for q in queries:
            ids = fdann.query_postfilter(
                q["query_vector"], q["target_category"],
                q["t_start"], q["t_end"], k=10, ef_search=ef)
            results_a.append(ids)
        elapsed = time.perf_counter() - t0
        logger.info(f"  ef={ef}: avg_returned={np.mean([len(r) for r in results_a]):.1f}, "
              f"time={elapsed:.3f}s, QPS={len(queries)/elapsed:.0f}")

    logger.info("\n--- Variant B: FDiskANN + Pre-filter ---")
    results_b = []
    t0 = time.perf_counter()
    for q in queries:
        ids = fdann.query_prefilter(
            q["query_vector"], q["target_category"],
            q["t_start"], q["t_end"], k=10)
        results_b.append(ids)
    elapsed = time.perf_counter() - t0
    logger.info(f"  avg_returned={np.mean([len(r) for r in results_b]):.1f}, "
          f"time={elapsed:.3f}s, QPS={len(queries)/elapsed:.0f}")
