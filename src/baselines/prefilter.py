#!/usr/bin/env python3
"""
prefilter.py — Pre-Filtering Baseline for TANNS-C Evaluation.

Strategy:
  1. At query time, first filter the database to keep only vectors matching
     the target category AND falling within [t_start, t_end].
  2. Perform exact (brute-force) k-NN search on the filtered subset.
  3. Return exact top-k results.

This gives perfect recall by definition — it serves as the accuracy ceiling.
The trade-off is low QPS because every query builds a fresh brute-force search
over the filtered subset.

Distance metric: cosine similarity (inner product on L2-normalized vectors).
"""

import logging
import time
import numpy as np
from typing import List, Dict, Tuple

from src.data_loader import build_filter_mask

logger = logging.getLogger(__name__)


class PreFilterBaseline:
    """Brute-force exact search with pre-filtering on category + time window."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
    ):
        """
        Store the full database (no index is pre-built).

        Args:
            vectors: (N, D) float32 array of database embeddings.
            categories: list of list of main categories per vector.
            update_days: (N,) int32 array of epoch-day timestamps.
        """
        self.vectors = vectors
        self.N, self.D = vectors.shape
        self.categories = categories
        self.update_days = update_days

        # Pre-normalize vectors for cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normed_vectors = vectors / norms

        logger.info(f"[PreFilter] Loaded {self.N} vectors, dim={self.D} (brute-force mode)")

    def query(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact k-NN search on the filtered subset.

        Args:
            query_vector: (D,) query embedding.
            target_category: required arXiv main category.
            t_start: inclusive lower bound on update_date (epoch day).
            t_end: inclusive upper bound on update_date (epoch day).
            k: number of results to return.

        Returns:
            ids: (k',) array of original database indices (k' <= k).
            distances: (k',) cosine distances (1 - similarity).
        """
        # Step 1: Build filter mask
        mask = build_filter_mask(
            self.categories, self.update_days, target_category, t_start, t_end
        )
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Step 2: Brute-force search on filtered subset
        subset = self.normed_vectors[filtered_indices]  # (M, D)

        # Normalize query
        qvec = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        # Cosine similarity = dot product on normalized vectors
        similarities = subset @ qvec  # (M,)
        distances = 1.0 - similarities  # cosine distance

        # Top-k
        k_actual = min(k, len(filtered_indices))
        if k_actual >= len(filtered_indices):
            # Return all, sorted by distance
            top_k_local = np.argsort(distances)
        else:
            top_k_local = np.argpartition(distances, k_actual)[:k_actual]
            top_k_local = top_k_local[np.argsort(distances[top_k_local])]

        return filtered_indices[top_k_local], distances[top_k_local]

    def batch_query(
        self,
        queries: List[Dict],
        k: int = 10,
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Run batch queries and return results + timing.

        Args:
            queries: list of query dicts from data_loader.generate_queries().
            k: number of results per query.

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

    parser = argparse.ArgumentParser(description="Pre-filter baseline demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vec_path = os.path.join(data_dir, "database_vectors_small.fvecs")
    attr_path = os.path.join(data_dir, "database_attributes_small.jsonl")

    vectors = load_fvecs(vec_path)
    categories, update_days = load_metadata(attr_path)

    baseline = PreFilterBaseline(vectors, categories, update_days)

    queries = generate_queries(vectors, categories, update_days, n_queries=10)

    results, qps, latency = baseline.batch_query(queries, k=10)
    avg_results = np.mean([len(r) for r in results])
    logger.info(f"  PreFilter: QPS={qps:.1f}, latency={latency:.2f}ms, avg_results={avg_results:.1f}")
