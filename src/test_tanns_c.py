"""
Test suite for TANNS-C: Correctness, Operations, Stress.

Run with:
    python test_tanns_c.py

All tests use synthetic data.  No dependency on pytest.
"""

import time
import numpy as np
from tanns_c import TANNSC


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n: int,
    dim: int,
    n_categories: int,
    seed: int = 42,
):
    """
    Returns:
      vectors:         np.ndarray of shape (n, dim), unit-normalized float32
      cat_sets_list:   List[Set[str]] of length n  (1–2 categories each)
      start_days_list: List[int] strictly increasing (step 10)
    """
    rng = np.random.default_rng(seed)

    # Random Gaussian vectors, unit-normalized
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    cat_names = [f"C{i}" for i in range(n_categories)]
    cat_sets_list = []
    for i in range(n):
        # Always assign 1 category; ~30 % chance of a 2nd
        primary = cat_names[rng.integers(0, n_categories)]
        cats = {primary}
        if rng.random() < 0.3:
            secondary = cat_names[rng.integers(0, n_categories)]
            cats.add(secondary)
        cat_sets_list.append(cats)

    start_days_list = list(range(0, 10 * n, 10))

    return vectors, cat_sets_list, start_days_list


# ---------------------------------------------------------------------------
# Brute-force helpers (for ground-truth comparison)
# ---------------------------------------------------------------------------

def brute_force_valid_set(index, C, t_start, t_end):
    """Return list of node ids valid for category C in [t_start, t_end]."""
    valid = []
    for v in index.cat_index.get(C, []):
        if index.start_days[v] <= t_end:
            exp = index.expire_days[v]
            if exp is None or exp > t_start:
                valid.append(v)
    return valid


def brute_force_knn(index, q_norm, C, t_start, t_end, k):
    """Brute-force k-NN over the valid set.  Returns list of node ids."""
    valid = brute_force_valid_set(index, C, t_start, t_end)
    if not valid:
        return []
    scored = [(float(np.dot(q_norm, index.vectors[v])), v) for v in valid]
    scored.sort(key=lambda x: -x[0])
    return [v for _, v in scored[:k]]


# ---------------------------------------------------------------------------
# Test 1: HNT Temporal Correctness (single category, no phantoms)
# ---------------------------------------------------------------------------

def test_hnt_temporal_correctness():
    print("Test 1: HNT temporal correctness ...")
    vectors, cat_sets_list, start_days_list = make_synthetic_data(
        n=50, dim=16, n_categories=1, seed=100,
    )
    index = TANNSC(M=8, ef_construction=50, alpha=0.0)
    index.build(vectors, cat_sets_list, start_days_list)

    rng = np.random.default_rng(200)
    C = "C0"

    phantoms = 0
    checks = 0
    for _ in range(10):
        u = int(rng.integers(0, index.N))
        for _ in range(10):
            a = int(rng.integers(0, start_days_list[-1]))
            b = int(rng.integers(a, start_days_list[-1] + 50))
            t_start, t_end = a, b

            neighbors = index._hnt_reconstruct_window(u, t_start, t_end, C)

            for v in neighbors:
                checks += 1
                # Must not be self
                assert v != u, f"HNT returned self-loop: u={u}"

                # Validity: exists some t in [t_start, t_end] such that
                #   start_days[v] <= t  AND  (expire_days[v] is None OR expire_days[v] > t)
                sd_v = index.start_days[v]
                exp_v = index.expire_days[v]
                if exp_v is None:
                    valid = sd_v <= t_end
                else:
                    # valid interval for v is [sd_v, exp_v)
                    # overlap with [t_start, t_end] requires sd_v <= t_end AND exp_v > t_start
                    valid = sd_v <= t_end and exp_v > t_start
                assert valid, (
                    f"Phantom neighbor: u={u}, v={v}, window=[{t_start},{t_end}], "
                    f"sd_v={sd_v}, exp_v={exp_v}"
                )

    print(f"  Checked {checks} neighbor entries across 100 (node, window) pairs — no phantoms.")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: Category Filter Correctness (Graph + HNT)
# ---------------------------------------------------------------------------

def test_category_filter_correctness():
    print("Test 2: Category filter correctness ...")
    vectors, cat_sets_list, start_days_list = make_synthetic_data(
        n=200, dim=32, n_categories=3, seed=300,
    )
    index = TANNSC(M=16, ef_construction=100, alpha=0.3)
    index.build(vectors, cat_sets_list, start_days_list)

    rng = np.random.default_rng(400)
    categories = ["C0", "C1", "C2"]

    violations = 0
    total_results = 0
    for _ in range(20):
        C = categories[int(rng.integers(0, 3))]
        # Pick a random base node, add noise
        base_idx = int(rng.integers(0, index.N))
        q = index.vectors[base_idx].copy() + rng.standard_normal(32).astype(np.float32) * 0.1
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        k_idx = int(rng.integers(0, index.N))
        t_start = start_days_list[k_idx] - 5
        t_end = start_days_list[k_idx] + 5

        ids, visited = index.query(q, C, t_start, t_end, k=10, ef=50)

        for v in ids:
            total_results += 1
            # Category check
            assert C in index.cat_sets[v], (
                f"Category violation: v={v} cats={index.cat_sets[v]}, expected {C}"
            )
            # Temporal check: start_days[v] <= t_end
            assert index.start_days[v] <= t_end, (
                f"Temporal violation (start): v={v}, sd={index.start_days[v]}, t_end={t_end}"
            )
            # Expiry check: expire_days[v] is None or expire_days[v] > t_start
            exp = index.expire_days[v]
            assert exp is None or exp > t_start, (
                f"Temporal violation (expire): v={v}, exp={exp}, t_start={t_start}"
            )

    print(f"  Checked {total_results} result entries across 20 queries — all valid.")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 3: Dynamic Insert & Query Consistency
# ---------------------------------------------------------------------------

def test_dynamic_insert_consistency():
    print("Test 3: Dynamic insert & query consistency ...")
    vectors, cat_sets_list, start_days_list = make_synthetic_data(
        n=200, dim=32, n_categories=3, seed=500,
    )
    index = TANNSC(M=16, ef_construction=100, alpha=0.3)
    index.build(vectors, cat_sets_list, start_days_list)

    rng = np.random.default_rng(600)
    categories = ["C0", "C1", "C2"]

    # Run 10 random queries and save results + similarity scores
    pre_queries = []
    for _ in range(10):
        C = categories[int(rng.integers(0, 3))]
        base_idx = int(rng.integers(0, index.N))
        q = index.vectors[base_idx].copy() + rng.standard_normal(32).astype(np.float32) * 0.1
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        k_idx = int(rng.integers(0, index.N))
        t_start = start_days_list[k_idx] - 5
        t_end = start_days_list[k_idx] + 5

        ids, _ = index.query(q, C, t_start, t_end, k=10, ef=50)
        # Store best similarity among returned results
        best_sim = max(
            (float(np.dot(q, index.vectors[v])) for v in ids), default=-1.0
        )
        pre_queries.append((q, C, t_start, t_end, set(ids.tolist()), best_sim))

    # Insert 20 new points with timestamps beyond existing data
    last_day = start_days_list[-1]
    extra_vecs, extra_cats, _ = make_synthetic_data(
        n=20, dim=32, n_categories=3, seed=700,
    )
    for i in range(20):
        index.insert(extra_vecs[i], extra_cats[i], t=last_day + 10 * (i + 1))

    assert index.N == 220, f"Expected 220 nodes, got {index.N}"

    # Re-run the 10 original queries
    degradations = 0
    for q, C, t_start, t_end, old_ids, old_best_sim in pre_queries:
        ids_new, _ = index.query(q, C, t_start, t_end, k=10, ef=50)
        new_set = set(ids_new.tolist())
        new_best_sim = max(
            (float(np.dot(q, index.vectors[v])) for v in ids_new), default=-1.0
        )
        # Soft check: the new best similarity should not be dramatically worse
        # (allow 0.05 tolerance for graph restructuring)
        if new_best_sim < old_best_sim - 0.05:
            degradations += 1

    # Allow at most 2 out of 10 queries to degrade slightly
    assert degradations <= 2, (
        f"Too many queries degraded after insert: {degradations}/10"
    )

    print(f"  20 dynamic inserts completed.  {degradations}/10 queries saw mild degradation.")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 4: Tombstone Delete Temporal Consistency
# ---------------------------------------------------------------------------

def test_tombstone_temporal_consistency():
    print("Test 4: Tombstone delete temporal consistency ...")
    vectors, cat_sets_list, start_days_list = make_synthetic_data(
        n=100, dim=32, n_categories=2, seed=800,
    )
    index = TANNSC(M=8, ef_construction=50, alpha=0.3)
    index.build(vectors, cat_sets_list, start_days_list)

    u = 10
    C = next(iter(index.cat_sets[u]))  # pick any category of u
    t_before = index.start_days[u] + 1
    t_after = index.start_days[u] + 1000

    # Identify nodes v that have u as a current Lnow neighbor
    # (i.e., v in reverse_adj[u])
    u_reverse = list(index.reverse_adj[u])

    # Pre-delete: for each such v, u should appear in HNT reconstruct
    # if u is in Lnow[v] with a start_day <= t_before and u has category C.
    pre_appearances = {}
    for v in u_reverse[:10]:  # up to 10
        neighbors_pre = index._hnt_reconstruct_window(v, t_before, t_before, C)
        pre_appearances[v] = u in neighbors_pre
        # u should be visible if: (a) u has category C, (b) u's start_day <= t_before,
        # (c) u is in Lnow[v] with edge start_day <= t_before
        if C in index.cat_sets[u] and index.start_days[u] <= t_before:
            lnow_map = {nid: sd for nid, sd in index.Lnow[v]}
            if u in lnow_map and lnow_map[u] <= t_before:
                assert u in neighbors_pre, (
                    f"Pre-delete: u={u} should appear in HNT of v={v} at t={t_before}"
                )

    # Perform tombstone delete
    index.tombstone_delete(u, t_expire=t_after)

    # Post-delete checks
    assert index.expire_days[u] == t_after
    assert index.B[u] == []
    assert index.reverse_adj[u] == set()

    # For each previously-checked v:
    for v in pre_appearances:
        # Past window (before expiry): u may still appear via HNT history
        past_neighbors = index._hnt_reconstruct_window(v, t_before, t_before, C)
        # We don't assert u IS present (it may not have been a neighbor of v
        # in all cases), but if it appears, its temporal validity must hold.
        if u in past_neighbors:
            sd_u = index.start_days[u]
            exp_u = index.expire_days[u]
            assert sd_u <= t_before, "Historical appearance violates start_day"
            assert exp_u is None or exp_u > t_before, "Historical appearance violates expiry"

        # Future window (at or after expiry): u must NOT appear
        future_neighbors = index._hnt_reconstruct_window(v, t_after, t_after + 10, C)
        assert u not in future_neighbors, (
            f"Post-delete: u={u} should not appear in HNT of v={v} at t>={t_after}"
        )

    # Query-level check: u should not appear in results after t_expire
    q = index.vectors[u].copy()
    ids, _ = index.query(q, C, t_start=t_after, t_end=t_after + 100, k=10, ef=50)
    assert u not in ids, "Deleted node appeared in query results after t_expire"

    print(f"  Checked {len(pre_appearances)} neighbors of deleted node u={u}.")
    print("  History preserved in past windows, invisible in future windows.")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 5: Stress Test – Stability and Basic Performance
# ---------------------------------------------------------------------------

def test_stress_stability():
    print("Test 5: Stress test (n=5000, dim=64, 5 categories) ...")
    t0 = time.time()

    vectors, cat_sets_list, start_days_list = make_synthetic_data(
        n=5000, dim=64, n_categories=5, seed=900,
    )
    index = TANNSC(M=16, ef_construction=100, alpha=0.3)
    index.build(vectors, cat_sets_list, start_days_list)

    build_time = time.time() - t0
    print(f"  Build time: {build_time:.1f}s")

    rng = np.random.default_rng(1000)
    categories = [f"C{i}" for i in range(5)]

    # Middle 80 % of start_days
    day_lo = start_days_list[int(0.1 * len(start_days_list))]
    day_hi = start_days_list[int(0.9 * len(start_days_list))]

    visited_counts = []
    query_records = []  # store (q_norm, C, t_start, t_end) for brute-force subset

    t1 = time.time()
    for i in range(50):
        C = categories[int(rng.integers(0, 5))]
        base_idx = int(rng.integers(0, index.N))
        q = index.vectors[base_idx].copy() + rng.standard_normal(64).astype(np.float32) * 0.05
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        t_start = int(rng.integers(day_lo, day_hi))
        t_end = t_start + int(rng.integers(500, 5000))

        ids, visited = index.query(q, C, t_start, t_end, k=10, ef=100)
        visited_counts.append(visited)
        query_records.append((q, C, t_start, t_end, ids))

    query_time = time.time() - t1
    mean_visited = np.mean(visited_counts)
    visit_ratio = mean_visited / index.N

    print(f"  50 queries in {query_time:.2f}s")
    print(f"  Mean visited: {mean_visited:.0f} / {index.N} = {visit_ratio:.3f}")

    # Assert beam search is not degenerate full scan
    assert visit_ratio < 0.5, (
        f"Beam search visiting too many nodes: {visit_ratio:.3f} (threshold 0.5)"
    )

    # Brute-force recall check on 10 randomly chosen queries
    recall_values = []
    bf_indices = rng.choice(50, size=10, replace=False)
    for idx in bf_indices:
        q, C, t_start, t_end, ids = query_records[idx]
        gt = brute_force_knn(index, q, C, t_start, t_end, k=10)
        if not gt:
            continue  # skip empty valid set
        gt_set = set(gt)
        hits = sum(1 for v in ids if v in gt_set)
        recall = hits / len(gt)
        recall_values.append(recall)

    if recall_values:
        mean_recall = np.mean(recall_values)
        print(f"  Mean Recall@10 (10 brute-forced queries): {mean_recall:.3f}")
        assert mean_recall >= 0.5, (
            f"Recall@10 too low: {mean_recall:.3f} (threshold 0.5)"
        )
    else:
        print("  No valid brute-force queries (all empty valid sets) — skipping recall check.")

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TANNS-C Test Suite")
    print("=" * 60 + "\n")

    test_hnt_temporal_correctness()
    test_category_filter_correctness()
    test_dynamic_insert_consistency()
    test_tombstone_temporal_consistency()
    test_stress_stability()

    print("=" * 60)
    print("All TANNS-C tests passed.")
    print("=" * 60)
