#!/usr/bin/env python3
"""
data_loader.py — Shared utilities for loading arxiv-for-FANNS dataset.

Loads .fvecs vectors and .jsonl metadata, extracts category + temporal
attributes, and generates synthetic queries when real query vectors are
unavailable.

Compatible with both SMALL (1K) and MEDIUM (100K) splits.
"""

import json
import logging
import struct
import os
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
EPOCH = date(1970, 1, 1)

TOP10_CATEGORIES = [
    "cs", "math", "cond-mat", "astro-ph", "physics",
    "hep-ph", "hep-th", "quant-ph", "stat", "gr-qc",
]

# ── .fvecs reader ────────────────────────────────────────────────────

def load_fvecs(path: str) -> np.ndarray:
    """
    Read an .fvecs file into a float32 numpy array of shape (N, D).

    .fvecs format: each vector is [dim (int32)] [dim × float32 values].
    All vectors share the same dimensionality.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]

    vec_bytes = 4 + dim * 4  # 4 bytes for dim + dim*4 bytes for data
    n_vecs = file_size // vec_bytes

    vectors = np.zeros((n_vecs, dim), dtype=np.float32)

    with open(path, "rb") as f:
        for i in range(n_vecs):
            d = struct.unpack("i", f.read(4))[0]
            assert d == dim, f"Dimension mismatch at vector {i}: expected {dim}, got {d}"
            vectors[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

    return vectors


# ── Metadata loader ──────────────────────────────────────────────────

def load_metadata(path: str) -> Tuple[List[List[str]], np.ndarray]:
    """
    Load database_attributes.jsonl and extract:
      - categories: list of list of main categories per paper
      - update_days: int array of epoch-day values (update_date field)

    Returns:
        categories: List[List[str]] — main_categories for each paper
        update_days: np.ndarray of int32 — epoch day for each paper
    """
    categories = []
    update_days = []

    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            categories.append(rec["main_categories"])
            update_days.append(rec["update_date"])

    return categories, np.array(update_days, dtype=np.int32)


def epoch_day_to_year(day: int) -> int:
    """Convert epoch day to calendar year."""
    return (EPOCH + timedelta(days=int(day))).year


def year_to_epoch_day_range(year: int) -> Tuple[int, int]:
    """Return (start_day, end_day) for a given calendar year (inclusive)."""
    start = (date(year, 1, 1) - EPOCH).days
    end = (date(year, 12, 31) - EPOCH).days
    return start, end


# ── Query generator ──────────────────────────────────────────────────

def generate_queries(
    db_vectors: np.ndarray,
    categories: List[List[str]],
    update_days: np.ndarray,
    n_queries: int = 1000,
    k: int = 100,
    window_years: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate evaluation queries with random category + time window filters.

    For each query:
      1. Pick a random database vector, add small Gaussian noise -> query vector
      2. Randomly assign a target category from TOP10_CATEGORIES
      3. Pick a random center year, create a 5-year window [center-2, center+2]

    Returns list of dicts:
      {
        "query_vector": np.ndarray (D,),
        "target_category": str,
        "t_start": int (epoch day),
        "t_end": int (epoch day),
        "source_idx": int  (index of source DB vector),
      }
    """
    rng = np.random.RandomState(seed)
    N, D = db_vectors.shape

    # Determine year range from data
    years = sorted(set(epoch_day_to_year(d) for d in update_days))
    min_year, max_year = years[0], years[-1]

    queries = []
    for _ in range(n_queries):
        # Pick source vector and add noise
        src_idx = rng.randint(0, N)
        noise = rng.randn(D).astype(np.float32) * 0.01
        qvec = db_vectors[src_idx] + noise
        # Normalize to unit length (cosine similarity)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        # Random category from top-10
        cat = TOP10_CATEGORIES[rng.randint(0, len(TOP10_CATEGORIES))]

        # Random center year -> 5-year window
        half = window_years // 2
        center_year = rng.randint(min_year + half, max_year - half + 1)
        t_start, _ = year_to_epoch_day_range(center_year - half)
        _, t_end = year_to_epoch_day_range(center_year + half)

        queries.append({
            "query_vector": qvec,
            "target_category": cat,
            "t_start": t_start,
            "t_end": t_end,
            "source_idx": src_idx,
        })

    return queries


# ── Filter mask builder ──────────────────────────────────────────────

def build_filter_mask(
    categories: List[List[str]],
    update_days: np.ndarray,
    target_category: str,
    t_start: int,
    t_end: int,
) -> np.ndarray:
    """
    Build a boolean mask over the database: True if paper matches
    the target category AND falls within [t_start, t_end].

    Returns: np.ndarray of bool, shape (N,)
    """
    N = len(categories)
    mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if (t_start <= update_days[i] <= t_end) and (target_category in categories[i]):
            mask[i] = True

    return mask


# ── CLI test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    parser = argparse.ArgumentParser(description="Test data loader utilities")
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir

    # Try small split first, fall back to medium
    vec_path = os.path.join(data_dir, "database_vectors_small.fvecs")
    if not os.path.exists(vec_path):
        vec_path = os.path.join(data_dir, "database_vectors.fvecs")

    attr_path = os.path.join(data_dir, "database_attributes_small.jsonl")
    if not os.path.exists(attr_path):
        attr_path = os.path.join(data_dir, "database_attributes.jsonl")

    logger.info(f"Loading vectors from {vec_path}...")
    vectors = load_fvecs(vec_path)
    logger.info(f"  Shape: {vectors.shape}")

    logger.info(f"Loading metadata from {attr_path}...")
    categories, update_days = load_metadata(attr_path)
    logger.info(f"  Records: {len(categories)}")
    logger.info(f"  Year range: {epoch_day_to_year(update_days.min())} - {epoch_day_to_year(update_days.max())}")

    logger.info("Generating 1000 queries...")
    queries = generate_queries(vectors, categories, update_days, n_queries=1000)
    logger.info(f"  Generated {len(queries)} queries")
    logger.info(f"  Sample: cat={queries[0]['target_category']}, "
          f"window=[{epoch_day_to_year(queries[0]['t_start'])}-{epoch_day_to_year(queries[0]['t_end'])}]")
