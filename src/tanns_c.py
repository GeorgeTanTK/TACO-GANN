#!/usr/bin/env python3
"""
tanns_c.py — TANNS-C: Temporal- and Category-Aware ANN Index.

Combines three pillars into a unified filtered ANN search system:

PILLAR 1 — Category-Aware Graph (from Filtered-DiskANN):
  - Per-category medoids as entry points.
  - Graph construction with category-aware neighbor selection:
    edges prefer neighbors sharing the inserting node's category.
    Combined score = alpha * label_affinity + (1-alpha) * (1 - cosine_dist).
  - Stores category labels per node.

PILLAR 2 — Temporal Snapshots:
  - Yearly temporal intervals partition the timeline.
  - Per-interval neighbor list snapshots: for each interval, only edges
    where BOTH endpoints have timestamps within or before that interval.
  - At query time, loads the snapshot matching the query's time window.

PILLAR 3 — ACORN-Style Fallback:
  - During filtered beam search, if the filtered candidate pool is thin
    (< 2*k), activates two-hop expansion: for candidates that don't
    match the filter, check THEIR neighbors for valid matches.
  - If selectivity is extremely low (< 1% of N), falls back entirely
    to brute-force pre-filtering.

QUERY PIPELINE for (q, category, t_start, t_end, k):
  1. Select temporal snapshot(s) covering [t_start, t_end].
  2. Entry point = medoid of target category.
  3. Filtered beam search on the snapshot's graph:
     - Only traverse neighbors matching category.
     - Only consider vectors with timestamp in [t_start, t_end].
  4. If candidate pool thin → two-hop expansion (Pillar 3).
  5. Return top-k results.

Ablation modes:
  - "full"    : all three pillars (default)
  - "P1"      : Pillar 1 only (category-aware graph, no temporal snapshots, no fallback)
  - "P2"      : Pillar 2 only (temporal snapshots, no category-aware graph, no fallback)
  - "P1P2"    : Pillars 1+2 (no ACORN fallback)
"""

import logging
import time
import heapq
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from src.data_loader import epoch_day_to_year, year_to_epoch_day_range

logger = logging.getLogger(__name__)


class TANNSC:
    """TANNS-C: Temporal- and Category-Aware ANN Index."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
        target_categories: List[str],
        M: int = 16,
        ef_construction: int = 200,
        alpha: float = 0.3,
        n_neighbors: int = 32,
    ):
        """
        Build TANNS-C index with all three pillars.

        Args:
            vectors: (N, D) float32 database vectors.
            categories: per-vector list of category labels.
            update_days: (N,) int32 epoch-day timestamps.
            target_categories: category labels to index (top-10).
            M: graph out-degree (max neighbors per node).
            ef_construction: construction-time beam width.
            alpha: weight for label affinity in neighbor selection.
                   score = alpha * same_cat + (1-alpha) * similarity
            n_neighbors: target neighbors per node in the graph.
        """
        self.N, self.D = vectors.shape
        self.vectors = vectors
        self.categories = categories
        self.update_days = update_days
        self.target_categories = target_categories
        self.M = M
        self.alpha = alpha
        self.n_neighbors = min(n_neighbors, self.N - 1)

        # Pre-normalize for cosine
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normed = (vectors / norms).astype(np.float32)

        # Category set per vector (for fast lookup)
        self.cat_sets: List[Set[str]] = [set(c) for c in categories]

        # Map from category → set of global IDs
        self.cat_to_ids: Dict[str, np.ndarray] = {}
        for cat in target_categories:
            ids = np.array([i for i in range(self.N) if cat in self.cat_sets[i]], dtype=np.int32)
            if len(ids) > 0:
                self.cat_to_ids[cat] = ids

        # Year info
        self.years = np.array([epoch_day_to_year(int(d)) for d in update_days], dtype=np.int32)
        self.unique_years = sorted(set(self.years))

        # ── Pillar 1: Category-aware graph + medoids ──
        logger.info(f"[TANNS-C] Building category-aware graph (N={self.N}, M={M}, alpha={alpha})...")
        t0 = time.time()
        self.adj = self._build_category_aware_graph(M, ef_construction)
        logger.info(f"[TANNS-C] Graph built in {time.time()-t0:.2f}s")

        # Per-category medoids
        self.cat_medoid: Dict[str, int] = {}
        for cat, gids in self.cat_to_ids.items():
            cat_normed = self.normed[gids]
            centroid = cat_normed.mean(axis=0)
            cn = np.linalg.norm(centroid)
            if cn > 0:
                centroid /= cn
            sims = cat_normed @ centroid
            medoid_local = int(np.argmax(sims))
            self.cat_medoid[cat] = int(gids[medoid_local])
        logger.info(f"[TANNS-C] Computed medoids for {len(self.cat_medoid)} categories")

        # ── Pillar 2: Temporal snapshots ──
        logger.info(f"[TANNS-C] Building temporal snapshots ({len(self.unique_years)} years)...")
        t0 = time.time()
        self.snapshots = self._build_temporal_snapshots()
        logger.info(f"[TANNS-C] Snapshots built in {time.time()-t0:.2f}s")

        logger.info(f"[TANNS-C] Index ready. {self.N} vectors, {len(self.unique_years)} years, "
                     f"{len(self.cat_medoid)} categories.")

    # ─── Pillar 1: Category-Aware Graph Construction ─────────────────

    def _build_category_aware_graph(
        self, M: int, ef_construction: int
    ) -> List[List[int]]:
        """
        Build adjacency list with category-aware neighbor selection.

        Uses a two-phase approach:
        1. Build a standard HNSW to get approximate neighbors.
        2. Re-rank neighbors with category-aware scoring:
           combined_score = alpha * same_category + (1-alpha) * cosine_sim

        This favors neighbors that share the inserting node's categories
        while still maintaining proximity-based navigability.
        """
        import hnswlib

        # Phase 1: Build standard HNSW for initial neighbor candidates
        hnsw = hnswlib.Index(space="cosine", dim=self.D)
        hnsw.init_index(max_elements=self.N, M=M, ef_construction=ef_construction)
        hnsw.add_items(self.vectors, ids=np.arange(self.N))

        # Phase 2: Extract and re-rank neighbors with category awareness
        k = min(self.n_neighbors * 2, self.N)  # get extra candidates for re-ranking
        hnsw.set_ef(max(k + 20, 200))

        adj = [[] for _ in range(self.N)]
        CHUNK = 200

        for start in range(0, self.N, CHUNK):
            end = min(start + CHUNK, self.N)
            batch = self.vectors[start:end]
            labels, dists = hnsw.knn_query(batch, k=min(k, self.N))

            for i, node_id in enumerate(range(start, end)):
                node_cats = self.cat_sets[node_id]
                candidates = []

                for j in range(labels.shape[1]):
                    nbr_id = int(labels[i][j])
                    if nbr_id == node_id:
                        continue
                    cos_dist = float(dists[i][j])
                    cos_sim = 1.0 - cos_dist  # similarity in [0, 1]

                    # Category affinity: 1 if any shared category, 0 otherwise
                    nbr_cats = self.cat_sets[nbr_id]
                    shared = 1.0 if len(node_cats & nbr_cats) > 0 else 0.0

                    # Combined score (higher = better neighbor)
                    score = self.alpha * shared + (1.0 - self.alpha) * cos_sim
                    candidates.append((-score, nbr_id))  # negate for min-heap

                # Take top-M neighbors by combined score
                heapq.heapify(candidates)
                neighbors = []
                seen = set()
                while candidates and len(neighbors) < self.n_neighbors:
                    _, nbr_id = heapq.heappop(candidates)
                    if nbr_id not in seen:
                        seen.add(nbr_id)
                        neighbors.append(nbr_id)

                adj[node_id] = neighbors

        return adj

    # ─── Pillar 2: Temporal Snapshots ────────────────────────────────

    def _build_temporal_snapshots(self) -> Dict[int, List[List[int]]]:
        """
        Build per-year neighbor list snapshots.

        For each year Y, the snapshot contains the adjacency list pruned to
        only include edges where BOTH endpoints have timestamps within year Y
        or earlier. This means searching the year-Y snapshot only traverses
        vectors that existed by year Y.
        """
        snapshots = {}

        for year in self.unique_years:
            # Find the epoch day for end of this year
            _, year_end_day = year_to_epoch_day_range(year)

            # Valid nodes: those with timestamp <= year_end_day
            valid_nodes = set(np.where(self.update_days <= year_end_day)[0].tolist())

            # Prune adjacency: keep only edges where BOTH endpoints are valid
            year_adj = [[] for _ in range(self.N)]
            for node_id in valid_nodes:
                year_adj[node_id] = [
                    nbr for nbr in self.adj[node_id] if nbr in valid_nodes
                ]

            snapshots[year] = year_adj

        return snapshots

    def _get_snapshot_adj(self, t_start: int, t_end: int) -> List[List[int]]:
        """
        Get the best temporal snapshot for the query time window.

        Strategy: use the snapshot for the latest year that overlaps the
        query window. This snapshot contains all vectors up to that year.
        For multi-year windows, merge neighbor lists from all overlapping
        year snapshots.
        """
        y_start = epoch_day_to_year(int(t_start))
        y_end = epoch_day_to_year(int(t_end))
        overlap_years = [y for y in self.unique_years if y_start <= y <= y_end]

        if not overlap_years:
            # Fallback: use the last year's snapshot (most complete)
            return self.snapshots.get(self.unique_years[-1], self.adj)

        # Use the latest year snapshot (most complete within the window)
        latest_year = max(overlap_years)
        return self.snapshots.get(latest_year, self.adj)

    # ─── Core: Filtered Beam Search ──────────────────────────────────

    def _cosine_dist(self, q_normed: np.ndarray, node_id: int) -> float:
        """Cosine distance between normalized query and a node."""
        return 1.0 - float(np.dot(q_normed, self.normed[node_id]))

    def _satisfies_filter(
        self, node_id: int, target_cat: str, t_start: int, t_end: int
    ) -> bool:
        """Check if node matches (category AND time window)."""
        return (
            t_start <= self.update_days[node_id] <= t_end
            and target_cat in self.cat_sets[node_id]
        )

    def query(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
        ef_search: int = 100,
        mode: str = "full",
    ) -> np.ndarray:
        """
        TANNS-C query pipeline.

        Args:
            query_vector: (D,) query embedding.
            target_category: required category label.
            t_start, t_end: time window (epoch days, inclusive).
            k: number of results.
            ef_search: beam width.
            mode: ablation mode —
                "full"  : all 3 pillars
                "P1"    : category-aware graph only (no snapshot, no fallback)
                "P2"    : temporal snapshots only (no category graph, no fallback)
                "P1P2"  : pillars 1+2, no ACORN fallback

        Returns:
            ids: array of global database indices (up to k).
        """
        # Normalize query
        qvec = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        use_fallback = mode == "full"
        use_snapshot = mode in ("full", "P2", "P1P2")
        use_cat_entry = mode in ("full", "P1", "P1P2")

        if target_category not in self.cat_to_ids:
            return np.array([], dtype=np.int64)

        cat_ids = self.cat_to_ids[target_category]

        # Quick selectivity check
        time_mask = (self.update_days[cat_ids] >= t_start) & (self.update_days[cat_ids] <= t_end)
        n_match = int(np.sum(time_mask))

        # Pillar 3: extremely low selectivity → brute-force pre-filter
        if use_fallback and n_match > 0 and (n_match / self.N) < 0.01:
            return self._brute_force_prefilter(qvec, target_category, t_start, t_end, k)

        if n_match == 0:
            return np.array([], dtype=np.int64)

        # Step 1: Select adjacency list
        if use_snapshot:
            adj = self._get_snapshot_adj(t_start, t_end)
        else:
            adj = self.adj

        # Step 2: Entry points — use multiple seeds for robustness
        visited: Set[int] = set()
        candidates = []  # min-heap: (distance, node_id)
        results = []     # (distance, node_id) — only filter-passing nodes

        # Seed strategy: combine medoid with nearest matching vectors
        seeds = []

        # (a) Category medoid if it's in the snapshot
        if use_cat_entry and target_category in self.cat_medoid:
            medoid = self.cat_medoid[target_category]
            seeds.append(medoid)

        # (b) Find a few matching vectors closest to query as seeds
        #     This is critical when the medoid is outside the time window
        matching_ids = cat_ids[time_mask]
        matching_normed = self.normed[matching_ids]
        sims = matching_normed @ qvec
        n_seeds = min(5, len(matching_ids))
        top_seed_local = np.argpartition(-sims, min(n_seeds, len(sims) - 1))[:n_seeds]
        for li in top_seed_local:
            seeds.append(int(matching_ids[li]))

        # Seed the search from all entry points
        for seed in seeds:
            if seed in visited:
                continue
            visited.add(seed)
            d = self._cosine_dist(qvec, seed)
            heapq.heappush(candidates, (d, seed))
            if self._satisfies_filter(seed, target_category, t_start, t_end):
                heapq.heappush(results, (d, seed))

            # Also add seed's neighbors
            for nbr in adj[seed]:
                if nbr not in visited:
                    visited.add(nbr)
                    d_nbr = self._cosine_dist(qvec, nbr)
                    heapq.heappush(candidates, (d_nbr, nbr))
                    if self._satisfies_filter(nbr, target_category, t_start, t_end):
                        heapq.heappush(results, (d_nbr, nbr))

        # Step 3: Filtered beam search
        max_steps = ef_search * 4
        steps = 0
        two_hop_activated = False

        while candidates and steps < max_steps:
            steps += 1
            dist_n, n = heapq.heappop(candidates)

            # Early termination
            if len(results) >= ef_search:
                worst_dist = max(r[0] for r in results)
                if dist_n > worst_dist:
                    break

            # Explore neighbors
            for nbr in adj[n]:
                if nbr in visited:
                    continue
                visited.add(nbr)

                d_nbr = self._cosine_dist(qvec, nbr)

                if self._satisfies_filter(nbr, target_category, t_start, t_end):
                    heapq.heappush(results, (d_nbr, nbr))
                    heapq.heappush(candidates, (d_nbr, nbr))
                else:
                    heapq.heappush(candidates, (d_nbr, nbr))

            # Pillar 3: thin pool check
            if use_fallback and not two_hop_activated:
                if len(results) < 2 * k and steps > ef_search // 2:
                    two_hop_activated = True

        # Pillar 3: Two-hop expansion
        if use_fallback and two_hop_activated:
            self._two_hop_expand(
                qvec, target_category, t_start, t_end,
                adj, visited, results, candidates,
                max_extra_steps=ef_search * 2,
            )

        if not results:
            return np.array([], dtype=np.int64)

        results.sort()
        return np.array([r[1] for r in results[:k]], dtype=np.int64)

    def _two_hop_expand(
        self,
        qvec: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        adj: List[List[int]],
        visited: Set[int],
        results: list,
        candidates: list,
        max_extra_steps: int,
    ):
        """
        ACORN-style two-hop expansion for thin candidate pools.

        For non-matching nodes in the candidate queue, check THEIR
        neighbors for filter-satisfying vectors.
        """
        extra_steps = 0
        while candidates and extra_steps < max_extra_steps:
            extra_steps += 1
            dist_n, n = heapq.heappop(candidates)

            for nbr in adj[n]:
                if nbr in visited:
                    continue
                visited.add(nbr)

                if self._satisfies_filter(nbr, target_category, t_start, t_end):
                    d_nbr = self._cosine_dist(qvec, nbr)
                    heapq.heappush(results, (d_nbr, nbr))
                    heapq.heappush(candidates, (d_nbr, nbr))
                else:
                    # Two-hop: check nbr's neighbors
                    for hop2 in adj[nbr]:
                        if hop2 in visited:
                            continue
                        visited.add(hop2)
                        if self._satisfies_filter(hop2, target_category, t_start, t_end):
                            d_hop2 = self._cosine_dist(qvec, hop2)
                            heapq.heappush(results, (d_hop2, hop2))
                            heapq.heappush(candidates, (d_hop2, hop2))

    def _brute_force_prefilter(
        self,
        q_normed: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int,
    ) -> np.ndarray:
        """
        Brute-force pre-filter fallback for extremely low selectivity.
        """
        if target_category not in self.cat_to_ids:
            return np.array([], dtype=np.int64)

        cat_ids = self.cat_to_ids[target_category]
        time_mask = (self.update_days[cat_ids] >= t_start) & (self.update_days[cat_ids] <= t_end)

        if not np.any(time_mask):
            return np.array([], dtype=np.int64)

        valid_ids = cat_ids[time_mask]
        valid_normed = self.normed[valid_ids]

        sims = valid_normed @ q_normed
        dists = 1.0 - sims

        if len(dists) <= k:
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, k)[:k]
            order = order[np.argsort(dists[order])]

        return np.array([int(valid_ids[i]) for i in order], dtype=np.int64)


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data_loader import (load_fvecs, load_metadata, generate_queries,
                                 TOP10_CATEGORIES)

    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    parser = argparse.ArgumentParser(description="TANNS-C demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vectors = load_fvecs(os.path.join(data_dir, "database_vectors_small.fvecs"))
    categories, update_days = load_metadata(
        os.path.join(data_dir, "database_attributes_small.jsonl"))

    tc = TANNSC(vectors, categories, update_days, TOP10_CATEGORIES)

    queries = generate_queries(vectors, categories, update_days, n_queries=10, seed=42)

    logger.info("\n--- TANNS-C (full) ---")
    for ef in [50, 100, 200]:
        t0 = time.perf_counter()
        results = []
        for q in queries:
            ids = tc.query(
                q["query_vector"], q["target_category"],
                q["t_start"], q["t_end"], k=10, ef_search=ef, mode="full")
            results.append(ids)
        elapsed = time.perf_counter() - t0
        avg_r = np.mean([len(r) for r in results])
        logger.info(f"  ef={ef}: avg_returned={avg_r:.1f}, QPS={len(queries)/elapsed:.0f}")

    logger.info("\nDone.")
