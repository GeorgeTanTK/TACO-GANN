#!/usr/bin/env python3
"""
acorn1.py — ACORN-1 (Two-Hop Expansion) Baseline for TANNS-C.

ACORN-1 Algorithm (adapted from the ACORN paper):
  1. Build an HNSW index on ALL database vectors.
  2. Extract the approximate graph adjacency from the index.
  3. At query time with filter predicate P = (category=C ∧ time ∈ [t_start, t_end]):
     a. Run a modified beam search starting from the HNSW entry point.
     b. For each candidate node n examined:
        - If n satisfies P: add n to the result set.
        - If n does NOT satisfy P: perform two-hop expansion —
          check all of n's graph neighbors m. If m satisfies P,
          add m as a search candidate.
     c. The beam search maintains a priority queue (min-heap by distance)
        with beam width = ef_search.
     d. Return top-k from candidates that satisfy P.

Key insight: even when direct neighbors don't match the filter, their
neighbors might — this maintains graph navigability under restrictive filters.

Reference:
  Patel et al., "ACORN: Performant and Predicate-Agnostic Search Over Vector
  Embeddings and Structured Data", SIGMOD 2024.
"""

import logging
import time
import heapq
import numpy as np
import hnswlib
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ACORN1Baseline:
    """HNSW with two-hop expansion for filtered ANN search."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        n_neighbors: int = 32,
    ):
        """
        Build the HNSW index and extract graph adjacency.

        Args:
            vectors: (N, D) float32 database embeddings.
            categories: list of main categories per paper.
            update_days: (N,) int32 epoch-day timestamps.
            M: HNSW graph out-degree.
            ef_construction: HNSW construction parameter.
            n_neighbors: number of neighbors per node in adjacency list.
        """
        self.N, self.D = vectors.shape
        self.vectors = vectors
        self.categories = categories
        self.update_days = update_days

        # Pre-normalize for cosine distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normed = (vectors / norms).astype(np.float32)

        # Step 1: Build HNSW index
        logger.info(f"[ACORN-1] Building HNSW: N={self.N}, D={self.D}, M={M}, ef_c={ef_construction}")
        t0 = time.time()
        self.hnsw = hnswlib.Index(space="cosine", dim=self.D)
        self.hnsw.init_index(max_elements=self.N, M=M, ef_construction=ef_construction)
        self.hnsw.add_items(vectors, ids=np.arange(self.N))
        logger.info(f"[ACORN-1] HNSW built in {time.time()-t0:.2f}s")

        # Step 2: Extract adjacency list by querying each node
        logger.info(f"[ACORN-1] Extracting adjacency ({n_neighbors} neighbors/node)...")
        t0 = time.time()
        self.adj = self._build_adjacency(n_neighbors)
        logger.info(f"[ACORN-1] Adjacency built in {time.time()-t0:.2f}s")

    def _build_adjacency(self, n_neighbors: int) -> List[List[int]]:
        """Extract approximate graph neighbors via KNN queries."""
        k = min(n_neighbors + 1, self.N)
        self.hnsw.set_ef(max(k + 10, 100))

        adj = [None] * self.N
        CHUNK = 200
        for start in range(0, self.N, CHUNK):
            end = min(start + CHUNK, self.N)
            labels, _ = self.hnsw.knn_query(self.vectors[start:end], k=k)
            for i, node_id in enumerate(range(start, end)):
                neighbors = [int(l) for l in labels[i] if int(l) != node_id]
                adj[node_id] = neighbors[:n_neighbors]

        return adj

    def _cosine_dist(self, query_normed: np.ndarray, node_id: int) -> float:
        """Compute cosine distance between normalized query and a node."""
        return 1.0 - float(np.dot(query_normed, self.normed[node_id]))

    def _satisfies_filter(self, node_id: int, target_cat: str, t_start: int, t_end: int) -> bool:
        """Check if a node matches the category + time predicate."""
        return (t_start <= self.update_days[node_id] <= t_end) and \
               (target_cat in self.categories[node_id])

    def query(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
        ef_search: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ACORN-1 two-hop beam search with filter predicate.

        Algorithm:
          1. Seed the search from HNSW's approximate nearest neighbors
             (using a small initial KNN to find good entry points).
          2. Run beam search with two-hop expansion:
             - Visited nodes that pass the filter go into the result set.
             - Visited nodes that fail the filter contribute their
               neighbors as new candidates (two-hop).
          3. Return top-k from the filtered result set.

        Args:
            query_vector: (D,) query embedding.
            target_category: required arXiv category.
            t_start, t_end: time window (epoch days, inclusive).
            k: number of results to return.
            ef_search: beam width for the search.

        Returns:
            ids: (k',) array of database indices.
            distances: (k',) array of cosine distances.
        """
        # Normalize query
        qvec = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        # === Phase 1: Seed candidates from HNSW ===
        # Get a small set of initial neighbors to seed the beam search
        seed_k = min(max(10, ef_search // 5), self.N)
        self.hnsw.set_ef(max(seed_k, 50))
        seed_labels, seed_dists = self.hnsw.knn_query(
            query_vector.reshape(1, -1), k=seed_k
        )

        # === Phase 2: Two-hop beam search ===
        # Priority queue: (distance, node_id)
        # We use a min-heap so closest candidates are explored first
        visited = set()
        candidates = []  # min-heap: (dist, node_id)
        results = []     # min-heap: (dist, node_id) — only nodes passing filter

        # Seed from HNSW results
        for lbl, dist in zip(seed_labels[0], seed_dists[0]):
            node_id = int(lbl)
            if node_id not in visited:
                visited.add(node_id)
                d = float(dist)  # hnswlib cosine returns distance
                heapq.heappush(candidates, (d, node_id))

                if self._satisfies_filter(node_id, target_category, t_start, t_end):
                    heapq.heappush(results, (d, node_id))

        # Beam search with two-hop expansion
        steps = 0
        max_steps = ef_search * 3  # safety limit

        while candidates and steps < max_steps:
            steps += 1
            dist_n, n = heapq.heappop(candidates)

            # If we have enough filtered results and this candidate is farther
            # than our worst result, we can stop early
            if len(results) >= ef_search:
                worst_result_dist = max(r[0] for r in results) if results else float('inf')
                if dist_n > worst_result_dist:
                    break

            # Explore n's neighbors
            for neighbor in self.adj[n]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                d_neighbor = self._cosine_dist(qvec, neighbor)

                if self._satisfies_filter(neighbor, target_category, t_start, t_end):
                    # Neighbor passes filter → add to results and candidates
                    heapq.heappush(results, (d_neighbor, neighbor))
                    heapq.heappush(candidates, (d_neighbor, neighbor))
                else:
                    # === TWO-HOP EXPANSION ===
                    # Neighbor fails filter → check ITS neighbors
                    for hop2 in self.adj[neighbor]:
                        if hop2 in visited:
                            continue
                        visited.add(hop2)

                        if self._satisfies_filter(hop2, target_category, t_start, t_end):
                            d_hop2 = self._cosine_dist(qvec, hop2)
                            heapq.heappush(results, (d_hop2, hop2))
                            heapq.heappush(candidates, (d_hop2, hop2))

        # === Phase 3: Return top-k from filtered results ===
        if len(results) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Sort results by distance, take top-k
        results.sort()
        top_k = results[:k]
        ids = np.array([r[1] for r in top_k], dtype=np.int64)
        dists = np.array([r[0] for r in top_k], dtype=np.float32)

        return ids, dists

    def batch_query(
        self,
        queries: List[Dict],
        k: int = 10,
        ef_search: int = 100,
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Run batch evaluation.

        Returns:
            results: list of id arrays.
            qps: queries per second.
            avg_latency_ms: average latency in milliseconds.
        """
        results = []
        total_time = 0

        for q in queries:
            t0 = time.perf_counter()
            ids, dists = self.query(
                query_vector=q["query_vector"],
                target_category=q["target_category"],
                t_start=q["t_start"],
                t_end=q["t_end"],
                k=k,
                ef_search=ef_search,
            )
            total_time += time.perf_counter() - t0
            results.append(ids)

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

    parser = argparse.ArgumentParser(description="ACORN-1 baseline demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vec_path = os.path.join(data_dir, "database_vectors_small.fvecs")
    attr_path = os.path.join(data_dir, "database_attributes_small.jsonl")

    vectors = load_fvecs(vec_path)
    categories, update_days = load_metadata(attr_path)

    acorn = ACORN1Baseline(vectors, categories, update_days)
    queries = generate_queries(vectors, categories, update_days, n_queries=10)

    for ef in [50, 100, 200]:
        results, qps, latency = acorn.batch_query(queries, k=10, ef_search=ef)
        avg_r = np.mean([len(r) for r in results])
        logger.info(f"  ef_search={ef}: QPS={qps:.1f}, latency={latency:.2f}ms, avg_returned={avg_r:.1f}")
