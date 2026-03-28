#!/usr/bin/env python3
"""
postfilter.py — Post-Filtering Baseline for TACO-GANN Evaluation.

Strategy:
  1. Build a single HNSW index on ALL database vectors (no metadata awareness).
  2. At query time, retrieve k * expansion_factor candidates.
  3. Post-filter: keep only candidates matching the target category AND
     falling within the time window [t_start, t_end].
  4. Return top-k from the filtered set.

If fewer than k results survive filtering, return whatever is available.

Parameters:
  - M: HNSW graph out-degree (default 32)
  - ef_construction: construction-time search width (default 200)
  - ef_search: query-time search width (auto-scaled with expansion_factor)
  - expansion_factor: {2, 5, 10, 20, 50} — controls over-retrieval ratio
"""

import logging
import time
import numpy as np
import hnswlib
from typing import List, Dict, Tuple, Optional

from src.data_loader import build_filter_mask

logger = logging.getLogger(__name__)


class PostFilterBaseline:
    """HNSW index with post-filtering on category + time window."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
        M: int = 32,
        ef_construction: int = 200,
        space: str = "cosine",
    ):
        """
        Build the HNSW index on all database vectors.

        Args:
            vectors: (N, D) float32 array of database embeddings.
            categories: list of list of main categories per vector.
            update_days: (N,) int32 array of epoch-day timestamps.
            M: HNSW out-degree parameter.
            ef_construction: construction-time ef parameter.
            space: distance metric — 'cosine', 'l2', or 'ip'.
        """
        self.N, self.D = vectors.shape
        self.categories = categories
        self.update_days = update_days
        self.space = space

        logger.info(f"[PostFilter] Building HNSW index: {self.N} vectors, dim={self.D}, M={M}, ef_c={ef_construction}")
        t0 = time.time()

        self.index = hnswlib.Index(space=space, dim=self.D)
        self.index.init_index(max_elements=self.N, M=M, ef_construction=ef_construction)
        self.index.add_items(vectors, ids=np.arange(self.N))

        build_time = time.time() - t0
        logger.info(f"[PostFilter] Index built in {build_time:.2f}s")

    def query(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
        expansion_factor: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with post-filtering.

        Args:
            query_vector: (D,) query embedding.
            target_category: required arXiv main category.
            t_start: inclusive lower bound on update_date (epoch day).
            t_end: inclusive upper bound on update_date (epoch day).
            k: number of results to return.
            expansion_factor: over-retrieval multiplier.

        Returns:
            ids: (k',) array of database indices (k' <= k).
            distances: (k',) array of distances.
        """
        # Over-retrieve
        k_expanded = min(k * expansion_factor, self.N)
        self.index.set_ef(max(k_expanded, 50))  # ef must be >= k

        labels, distances = self.index.knn_query(
            query_vector.reshape(1, -1), k=k_expanded
        )
        labels = labels[0]
        distances = distances[0]

        # Post-filter: keep only matching category + time window
        keep = []
        for i, idx in enumerate(labels):
            idx = int(idx)
            if (t_start <= self.update_days[idx] <= t_end) and \
               (target_category in self.categories[idx]):
                keep.append(i)
            if len(keep) >= k:
                break

        if len(keep) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        keep = np.array(keep)
        return labels[keep], distances[keep]

    def batch_query(
        self,
        queries: List[Dict],
        k: int = 10,
        expansion_factor: int = 10,
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Run batch queries and return results + timing.

        Args:
            queries: list of query dicts from data_loader.generate_queries().
            k: number of results per query.
            expansion_factor: over-retrieval multiplier.

        Returns:
            results: list of (ids,) arrays, one per query.
            qps: queries per second.
            avg_latency_ms: average latency in milliseconds.
        """
        results = []
        latencies = []

        for q in queries:
            t0 = time.perf_counter()
            ids, dists = self.query(
                query_vector=q["query_vector"],
                target_category=q["target_category"],
                t_start=q["t_start"],
                t_end=q["t_end"],
                k=k,
                expansion_factor=expansion_factor,
            )
            latency = time.perf_counter() - t0
            latencies.append(latency)
            results.append(ids)

        total_time = sum(latencies)
        qps = len(queries) / total_time if total_time > 0 else 0
        avg_latency_ms = (total_time / len(queries)) * 1000

        return results, qps, avg_latency_ms


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data_loader import load_fvecs, load_metadata, generate_queries, epoch_day_to_year

    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    parser = argparse.ArgumentParser(description="Post-filter baseline demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vec_path = os.path.join(data_dir, "database_vectors_small.fvecs")
    attr_path = os.path.join(data_dir, "database_attributes_small.jsonl")

    vectors = load_fvecs(vec_path)
    categories, update_days = load_metadata(attr_path)

    baseline = PostFilterBaseline(vectors, categories, update_days)

    queries = generate_queries(vectors, categories, update_days, n_queries=10)

    for ef in [2, 5, 10, 20, 50]:
        results, qps, latency = baseline.batch_query(queries, k=10, expansion_factor=ef)
        avg_results = np.mean([len(r) for r in results])
        logger.info(f"  ef={ef:3d}: QPS={qps:.1f}, latency={latency:.2f}ms, avg_results={avg_results:.1f}")
