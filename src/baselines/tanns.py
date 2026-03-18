#!/usr/bin/env python3
"""
tanns.py — Simplified Timestamp Graph Baseline for TANNS-C.

Inspired by TANNS (Temporal-Aware ANN Search), this baseline partitions
the database by yearly time intervals and builds a separate HNSW index
per interval. At query time, only the HNSW indices whose year intervals
overlap [t_start, t_end] are searched; results are merged across years.

Two category-filter variants:

  Variant A — TANNS + Post-filter:
    1. Search temporal HNSW indices (no category awareness)
    2. Post-filter results by target category C
    Tradeoff: fast search but many results discarded if category is rare

  Variant B — TANNS + Pre-filter:
    1. For each overlapping year, identify vectors matching (year ∧ category)
    2. Brute-force cosine search on that small subset
    Tradeoff: exact recall within temporal scope, but slower for large subsets

Parameters:
  - Interval granularity: yearly (2007–2025 for SMALL split)
  - Per-year HNSW: M=16, ef_construction=200 (same as other baselines)
  - ef_search ∈ {50, 100, 200, 500, 1000} for Variant A
  - Variant B has no ef_search tuning (brute-force)

Reference:
  Chen et al., "TANNS: Timestamp-Aware Nearest Neighbor Search",
  arXiv:2504.02028, 2025.
"""

import logging
import time
import numpy as np
import hnswlib
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.data_loader import epoch_day_to_year, year_to_epoch_day_range

logger = logging.getLogger(__name__)


class TimestampGraphBaseline:
    """Yearly-partitioned HNSW indices for temporal ANN search."""

    def __init__(
        self,
        vectors: np.ndarray,
        categories: List[List[str]],
        update_days: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
    ):
        """
        Build per-year HNSW indices.

        Args:
            vectors: (N, D) float32 database embeddings.
            categories: list of main categories per paper.
            update_days: (N,) int32 epoch-day timestamps.
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

        # Convert days → years
        self.years = np.array(
            [epoch_day_to_year(int(d)) for d in update_days], dtype=np.int32
        )
        unique_years = sorted(set(self.years))
        self.unique_years = unique_years

        # Build per-year structures
        logger.info(f"[TANNS] Building per-year HNSW indices for {len(unique_years)} years...")
        t0 = time.time()

        # year → list of global indices
        self.year_to_ids: Dict[int, np.ndarray] = {}
        # year → HNSW index (maps local idx → global idx via year_to_ids)
        self.year_hnsw: Dict[int, hnswlib.Index] = {}
        # year → normalized vectors for that year (for brute-force variant)
        self.year_normed: Dict[int, np.ndarray] = {}

        for year in unique_years:
            global_ids = np.where(self.years == year)[0]
            self.year_to_ids[year] = global_ids
            year_vecs = vectors[global_ids]
            self.year_normed[year] = self.normed[global_ids]

            n_year = len(global_ids)
            if n_year == 0:
                continue

            # Build HNSW for this year's vectors
            idx = hnswlib.Index(space="cosine", dim=self.D)
            # M is clamped to allow small indices
            m_eff = min(M, max(2, n_year // 2))
            idx.init_index(max_elements=n_year, M=m_eff, ef_construction=ef_construction)
            idx.add_items(year_vecs, ids=np.arange(n_year))
            self.year_hnsw[year] = idx

        elapsed = time.time() - t0
        logger.info(f"[TANNS] Built {len(unique_years)} indices in {elapsed:.2f}s")
        for y in unique_years:
            logger.info(f"  {y}: {len(self.year_to_ids[y])} vectors")

    def _overlapping_years(self, t_start: int, t_end: int) -> List[int]:
        """Return list of years whose range overlaps [t_start, t_end]."""
        y_start = epoch_day_to_year(int(t_start))
        y_end = epoch_day_to_year(int(t_end))
        return [y for y in self.unique_years if y_start <= y <= y_end]

    # ── Variant A: TANNS + Post-filter ───────────────────────────────

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
        Variant A — Search temporal HNSW indices, then post-filter by category.

        1. Find overlapping year indices
        2. Search each year's HNSW with ef_search
        3. Map local IDs → global IDs
        4. Filter by (category ∧ time range)
        5. Re-rank by cosine distance, return top-k

        Returns:
            ids: array of global database indices (up to k).
        """
        qvec = query_vector.astype(np.float32).reshape(1, -1)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qnormed = (qvec / qnorm).ravel()
        else:
            qnormed = qvec.ravel()

        overlap_years = self._overlapping_years(t_start, t_end)
        if not overlap_years:
            return np.array([], dtype=np.int64)

        # Gather candidates from all overlapping year indices
        candidates = {}  # global_id → cosine_distance

        for year in overlap_years:
            if year not in self.year_hnsw:
                continue
            hnsw = self.year_hnsw[year]
            gids = self.year_to_ids[year]
            n_year = len(gids)

            search_k = min(ef_search, n_year)
            if search_k == 0:
                continue

            hnsw.set_ef(max(search_k, 10))
            labels, dists = hnsw.knn_query(qvec, k=search_k)

            for local_id, dist in zip(labels[0], dists[0]):
                global_id = int(gids[int(local_id)])
                # Only keep if within exact time range
                if t_start <= self.update_days[global_id] <= t_end:
                    candidates[global_id] = float(dist)

        # Post-filter by category
        filtered = {}
        for gid, dist in candidates.items():
            if target_category in self.categories[gid]:
                filtered[gid] = dist

        if not filtered:
            return np.array([], dtype=np.int64)

        # Sort by distance, return top-k
        sorted_results = sorted(filtered.items(), key=lambda x: x[1])
        return np.array([r[0] for r in sorted_results[:k]], dtype=np.int64)

    # ── Variant B: TANNS + Pre-filter ────────────────────────────────

    def query_prefilter(
        self,
        query_vector: np.ndarray,
        target_category: str,
        t_start: int,
        t_end: int,
        k: int = 10,
    ) -> np.ndarray:
        """
        Variant B — Pre-filter each temporal partition by category, brute-force.

        1. Find overlapping year indices
        2. For each year, select vectors matching (year ∧ category ∧ time)
        3. Brute-force cosine search on that subset
        4. Merge across years, return top-k

        Returns:
            ids: array of global database indices (up to k).
        """
        qvec = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qnormed = (qvec / qnorm).astype(np.float32)
        else:
            qnormed = qvec

        overlap_years = self._overlapping_years(t_start, t_end)
        if not overlap_years:
            return np.array([], dtype=np.int64)

        all_candidates = []  # (cosine_dist, global_id)

        for year in overlap_years:
            gids = self.year_to_ids[year]
            normed_year = self.year_normed[year]

            # Filter within this year: category + exact time range
            local_mask = []
            for i, gid in enumerate(gids):
                if (target_category in self.categories[gid] and
                        t_start <= self.update_days[gid] <= t_end):
                    local_mask.append(i)

            if not local_mask:
                continue

            local_mask = np.array(local_mask, dtype=np.int32)
            subset_normed = normed_year[local_mask]
            subset_gids = gids[local_mask]

            # Brute-force cosine distance
            sims = subset_normed @ qnormed  # shape (n_subset,)
            dists = 1.0 - sims

            for i in range(len(subset_gids)):
                all_candidates.append((float(dists[i]), int(subset_gids[i])))

        if not all_candidates:
            return np.array([], dtype=np.int64)

        all_candidates.sort(key=lambda x: x[0])
        return np.array([c[1] for c in all_candidates[:k]], dtype=np.int64)


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data_loader import load_fvecs, load_metadata, generate_queries

    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    parser = argparse.ArgumentParser(description="TANNS baseline demo")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    vectors = load_fvecs(os.path.join(data_dir, "database_vectors_small.fvecs"))
    categories, update_days = load_metadata(os.path.join(data_dir, "database_attributes_small.jsonl"))

    tanns = TimestampGraphBaseline(vectors, categories, update_days)

    queries = generate_queries(vectors, categories, update_days, n_queries=10, seed=42)

    logger.info("\n--- Variant A: TANNS + Post-filter ---")
    for ef in [50, 100, 200]:
        results_a = []
        t0 = time.perf_counter()
        for q in queries:
            ids = tanns.query_postfilter(
                q["query_vector"], q["target_category"],
                q["t_start"], q["t_end"], k=10, ef_search=ef
            )
            results_a.append(ids)
        elapsed = time.perf_counter() - t0
        logger.info(f"  ef={ef}: avg_returned={np.mean([len(r) for r in results_a]):.1f}, "
              f"time={elapsed:.3f}s, QPS={len(queries)/elapsed:.0f}")

    logger.info("\n--- Variant B: TANNS + Pre-filter ---")
    results_b = []
    t0 = time.perf_counter()
    for q in queries:
        ids = tanns.query_prefilter(
            q["query_vector"], q["target_category"],
            q["t_start"], q["t_end"], k=10
        )
        results_b.append(ids)
    elapsed = time.perf_counter() - t0
    logger.info(f"  avg_returned={np.mean([len(r) for r in results_b]):.1f}, "
          f"time={elapsed:.3f}s, QPS={len(queries)/elapsed:.0f}")
