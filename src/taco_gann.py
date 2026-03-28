"""
TACO-GANN: Temporal- and Category-Aware Approximate Nearest Neighbor Search Index

A single navigable graph built with Filtered-Vamana (category-aware,
ST-connectivity guaranteed per label) and per-node Historic Neighbor Trees
(HNT) for temporal reconstruction at query time.  Supports dynamic insert
and tombstone delete.

Architecture
------------
Category → handled at graph construction time via Filtered-Vamana
           (FilteredGreedySearch + ST-connectivity invariant per label).
Temporal → handled at query time via per-node HNT flat sorted list
           reconstruction.

This is NOT TANNS + post-filtering, NOT stitched-Vamana, NOT HNSW-based.
"""

import struct
import json
import bisect
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_fvecs(path: str, max_n: int = None) -> np.ndarray:
    """Load .fvecs file -> (N, D) float32 array, unit-normalized.
    If max_n is given, only load the first max_n vectors (memory efficient)."""
    vectors = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            dim = struct.unpack("<i", buf)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            vectors.append(vec)
            if max_n is not None and len(vectors) >= max_n:
                break
    out = np.vstack(vectors).astype(np.float32)
    # Unit-normalize
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def load_ivecs(path: str) -> List[List[int]]:
    """Load .ivecs file -> list of int lists (ground-truth id lists)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    result = []
    while offset < len(data):
        dim = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        ids = list(struct.unpack_from(f"<{dim}i", data, offset))
        offset += dim * 4
        result.append(ids)
    return result


def load_jsonl(path: str) -> List[dict]:
    """Load .jsonl file -> list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# HNT Entry
# ---------------------------------------------------------------------------

@dataclass
class HNTEntry:
    """One historical neighbor record: neighbor `node_id` was in Lnow from
    `start_day` (inclusive) to `end_day` (exclusive).
    Sorted by end_day for bisect acceleration in _hnt_reconstruct_window."""
    end_day: int
    start_day: int
    node_id: int

    def __lt__(self, other):
        """Order by end_day for bisect.insort."""
        if isinstance(other, HNTEntry):
            return self.end_day < other.end_day
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, HNTEntry):
            return self.end_day <= other.end_day
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, HNTEntry):
            return self.end_day > other.end_day
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, HNTEntry):
            return self.end_day >= other.end_day
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, HNTEntry):
            return (self.end_day == other.end_day and
                    self.start_day == other.start_day and
                    self.node_id == other.node_id)
        return NotImplemented


# ---------------------------------------------------------------------------
# TACO-GANN Index
# ---------------------------------------------------------------------------

# DiskANN-style pruning scaling factor (separate from category alpha)
PRUNE_ALPHA = 1.2


class TACOGANN:
    def __init__(self, M: int = 16, ef_construction: int = 100, alpha: float = 0.3):
        self.M = M
        self.ef_construction = ef_construction
        self.alpha = alpha
        self._rng = np.random.default_rng(42)

        # Per-node parallel arrays
        self.vectors: List[np.ndarray] = []          # (D,) float32 normalized
        self.cat_sets: List[Set[str]] = []
        self.start_days: List[int] = []              # epoch-day of insertion
        self.expire_days: List[Optional[int]] = []   # None = alive
        self.Lnow: List[List[Tuple[int, int]]] = []  # [(node_id, start_day), ...]
        self.B: List[List[int]] = []                 # backup neighbors
        self.hnt: List[List[HNTEntry]] = []          # per-node HNT, sorted by end_day
        self.cat_index: Dict[str, List[int]] = {}    # category -> node_ids
        self.cat_medoid: Dict[str, int] = {}         # category -> medoid node_id
        # Reverse adjacency for tombstone_delete
        self.reverse_adj: List[Set[int]] = []        # reverse_adj[u] = {v : u in Lnow[v]}
        self._delete_count: int = 0                  # trigger _compact_cat_index every 1000

        # Per-category insertion counter for medoid recomputation every 500
        self._cat_insert_count: Dict[str, int] = {}

    @property
    def N(self) -> int:
        return len(self.vectors)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _cosine_sim(self, u: int, v: int) -> float:
        return float(np.dot(self.vectors[u], self.vectors[v]))

    def _cosine_sim_vec(self, q: np.ndarray, v: int) -> float:
        return float(np.dot(q, self.vectors[v]))

    def _same_cat(self, u: int, v: int) -> float:
        return 1.0 if self.cat_sets[u] & self.cat_sets[v] else 0.0

    def _score(self, u: int, v: int) -> float:
        return self.alpha * self._same_cat(u, v) + (1.0 - self.alpha) * self._cosine_sim(u, v)

    # ------------------------------------------------------------------
    # HNT operations
    # ------------------------------------------------------------------

    def _hnt_init(self, u: int, neighbor_ids: List[int], t: int):
        """Initialize Lnow[u] and empty HNT for node u.
        Also registers u in reverse_adj of each neighbor."""
        self.Lnow[u] = [(nid, t) for nid in neighbor_ids]
        self.hnt[u] = []
        for nid in neighbor_ids:
            self.reverse_adj[nid].add(u)

    def _hnt_append(self, u: int, new_neighbor_ids: List[int], eviction_day: int):
        """Update Lnow[u] to reflect new_neighbor_ids; evicted neighbors go to HNT."""
        old_map = {nid: sd for nid, sd in self.Lnow[u]}  # node_id -> start_day
        new_set = set(new_neighbor_ids)
        old_set = set(old_map.keys())

        # Evicted nodes -> HNT (sorted by end_day via bisect.insort)
        evicted = old_set - new_set
        for nid in evicted:
            original_sd = old_map[nid]
            entry = HNTEntry(end_day=eviction_day, start_day=original_sd, node_id=nid)
            bisect.insort(self.hnt[u], entry)
            # Update reverse adjacency
            self.reverse_adj[nid].discard(u)

        # Rebuild Lnow[u]
        new_lnow = []
        for nid in new_neighbor_ids:
            if nid in old_map:
                new_lnow.append((nid, old_map[nid]))  # carry forward original start_day
            else:
                new_lnow.append((nid, eviction_day))   # new neighbor
                self.reverse_adj[nid].add(u)
        self.Lnow[u] = new_lnow

    def _hnt_reconstruct_window(self, u: int, t_start: int, t_end: int, C: str) -> List[int]:
        """Reconstruct valid neighbors of u in time window [t_start, t_end] with category C.

        Category and expiry checks happen inside this function — not outside.

        hnt[u] is sorted by end_day.  Use bisect_right with a sentinel HNTEntry
        to skip all entries with end_day <= t_start in O(log n) without
        allocating a temporary list, then linear scan the remainder.
        """
        result = []

        # From Lnow[u]: current neighbors
        for nid, sd in self.Lnow[u]:
            if sd <= t_end:
                if C in self.cat_sets[nid]:
                    exp = self.expire_days[nid]
                    if exp is None or exp > t_start:
                        result.append(nid)

        # From hnt[u]: historical neighbors
        # hnt[u] is sorted by end_day (via HNTEntry.__lt__).
        # We need: end_day > t_start AND start_day <= t_end
        # Sentinel with end_day=t_start: bisect_right finds first entry > t_start
        hnt_list = self.hnt[u]
        if hnt_list:
            sentinel = HNTEntry(end_day=t_start, start_day=0, node_id=-1)
            lo = bisect.bisect_right(hnt_list, sentinel)
            for i in range(lo, len(hnt_list)):
                entry = hnt_list[i]
                # end_day > t_start is guaranteed by bisect
                # end_day ordering != start_day ordering, so can't break on start_day
                if entry.start_day > t_end:
                    continue
                nid = entry.node_id
                if C in self.cat_sets[nid]:
                    exp = self.expire_days[nid]
                    if exp is None or exp > t_start:
                        result.append(nid)

        return result

    # ------------------------------------------------------------------
    # Greedy beam search (Filtered-Vamana)
    # ------------------------------------------------------------------

    def _greedy_search(
        self,
        q_norm: np.ndarray,
        entry_points: List[int],
        ef: int,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        cat_filter: Optional[str] = None,
        use_hnt: bool = False,
        build_cat_filter: Optional[Set[str]] = None,
    ) -> Tuple[List[int], int]:
        """
        Beam search over the graph.

        Two modes:
          Build-time (use_hnt=False, build_cat_filter=cats):
            Traverse Lnow edges only.
            Skip expired entry points.
            Only expand neighbors nid where cat_sets[nid] & build_cat_filter
            is non-empty — this is what makes it Filtered-Vamana.

          Query-time (use_hnt=True, cat_filter=C):
            Traverse via _hnt_reconstruct_window(u, t_start, t_end, C)
            which already filters by category + time.

        Returns (node_ids sorted by descending cosine similarity, visited_count).
        """
        if not entry_points:
            return [], 0

        visited: Set[int] = set()

        # Result list: keep top-ef by similarity (min-heap by sim -> pop worst)
        results: List[Tuple[float, int]] = []  # (sim, node_id) -- min-heap
        # Candidate queue: nodes to explore, best-first -> max-heap by sim
        candidates: List[Tuple[float, int]] = []  # (-sim, node_id) -- min-heap on -sim

        for ep in entry_points:
            if ep in visited or ep < 0 or ep >= self.N:
                continue
            # Skip expired entry points during build-time graph construction
            if not use_hnt and self.expire_days[ep] is not None:
                continue
            visited.add(ep)
            sim = float(np.dot(q_norm, self.vectors[ep]))
            heapq.heappush(candidates, (-sim, ep))
            heapq.heappush(results, (sim, ep))
            if len(results) > ef:
                heapq.heappop(results)

        while candidates:
            neg_sim, u = heapq.heappop(candidates)
            cur_sim = -neg_sim

            # Early stop: if current candidate is worse than the worst in
            # results and results is full
            if len(results) >= ef and cur_sim < results[0][0]:
                break

            # Get neighbors
            if use_hnt and t_start is not None and t_end is not None and cat_filter is not None:
                # Query-time: HNT reconstruction already filters category+time
                neighbors = self._hnt_reconstruct_window(u, t_start, t_end, cat_filter)
            else:
                # Build-time: traverse Lnow edges with optional category filter
                neighbors = [nid for nid, _ in self.Lnow[u]]

            for nid in neighbors:
                if nid in visited:
                    continue
                # Build-time: skip expired nodes
                if not use_hnt and self.expire_days[nid] is not None:
                    continue
                # Filtered-Vamana: only expand neighbors matching category set
                if build_cat_filter is not None and not (self.cat_sets[nid] & build_cat_filter):
                    continue
                visited.add(nid)
                sim = float(np.dot(q_norm, self.vectors[nid]))

                if len(results) < ef or sim > results[0][0]:
                    heapq.heappush(candidates, (-sim, nid))
                    heapq.heappush(results, (sim, nid))
                    if len(results) > ef:
                        heapq.heappop(results)

        visited_count = len(visited)

        # Return sorted by similarity descending (best first)
        results.sort(key=lambda x: -x[0])
        return [nid for _, nid in results], visited_count

    # ------------------------------------------------------------------
    # Robust prune (DiskANN-style angular pruning)
    # ------------------------------------------------------------------

    def _robust_prune(self, u: int, scored: List[Tuple[float, int]], M: int) -> List[int]:
        """Select up to M diverse neighbors using angular pruning (DiskANN-style).
        Uses PRUNE_ALPHA=1.2 scaling so pruning is less aggressive.

        For each candidate cid in descending score order: skip if any
        already-selected neighbor s satisfies sim(s, cid) > PRUNE_ALPHA * sim(u, cid).
        Fill greedily if under M after pruning.
        """
        selected: List[int] = []
        vec_u = self.vectors[u]
        candidates = [(s, cid) for s, cid in scored if cid != u]

        for s, cid in candidates:
            if len(selected) >= M:
                break
            vec_cid = self.vectors[cid]
            too_close = False
            sim_u_cid = float(np.dot(vec_u, vec_cid))
            for sel in selected:
                sim_sel_cid = float(np.dot(self.vectors[sel], vec_cid))
                if sim_sel_cid > PRUNE_ALPHA * sim_u_cid:
                    too_close = True
                    break
            if not too_close:
                selected.append(cid)

        # If we don't have enough, fill greedily
        if len(selected) < M:
            selected_set = set(selected)
            for s, cid in candidates:
                if cid not in selected_set:
                    selected.append(cid)
                    selected_set.add(cid)
                    if len(selected) >= M:
                        break

        return selected

    # ------------------------------------------------------------------
    # Medoid computation
    # ------------------------------------------------------------------

    def _recompute_medoid(self, cat: str):
        """Recompute medoid for a category: the node closest to the centroid.
        Only considers live (non-expired) nodes.  Samples up to 500 using
        self._rng (seeded default_rng(42))."""
        nodes = [v for v in self.cat_index.get(cat, []) if self.expire_days[v] is None]
        if not nodes:
            return
        # Sample if too many
        if len(nodes) <= 500:
            sample = nodes
        else:
            sample_indices = self._rng.choice(len(nodes), 500, replace=False)
            sample = [nodes[i] for i in sample_indices]
        vecs = np.array([self.vectors[i] for i in sample])
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        sims = vecs @ centroid
        best_idx = sample[int(np.argmax(sims))]
        self.cat_medoid[cat] = best_idx

    # ------------------------------------------------------------------
    # Compact cat_index (remove expired node IDs)
    # ------------------------------------------------------------------

    def _compact_cat_index(self):
        """Remove expired node IDs from cat_index lists."""
        for cat in self.cat_index:
            self.cat_index[cat] = [
                v for v in self.cat_index[cat] if self.expire_days[v] is None
            ]

    # ------------------------------------------------------------------
    # Insert (Filtered-Vamana with ST-connectivity guard)
    # ------------------------------------------------------------------

    def insert(self, vec: np.ndarray, cats: Set[str], t: int):
        """Insert a new node into the index.

        1. Normalize vec to unit float32
        2. Append to all parallel arrays
        3. Update cat_index and cat_medoid
        4. Filtered-Vamana greedy search from category medoids
        5. Extend with 2-hop neighbors; filter expired
        6. Score candidates with alpha-blended similarity
        7. Robust prune → primary (M), remaining → backup (M)
        8. HNT init; bidirectional linking
        9. ST-connectivity guard per category
        10. Recompute medoid every 500 insertions per category
        """
        u = self.N

        # 1. Normalize before storing
        vec = vec.astype(np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # 2. Append to all parallel arrays
        self.vectors.append(vec)
        self.cat_sets.append(cats)
        self.start_days.append(t)
        self.expire_days.append(None)
        self.Lnow.append([])
        self.B.append([])
        self.hnt.append([])
        self.reverse_adj.append(set())

        # 3. Update cat_index
        for c in cats:
            if c not in self.cat_index:
                self.cat_index[c] = []
            self.cat_index[c].append(u)

        # 4. For categories without medoid yet: set cat_medoid[c] = u
        #    (If first node, set medoid and return)
        if u == 0:
            for c in cats:
                self.cat_medoid[c] = u
                self._cat_insert_count[c] = self._cat_insert_count.get(c, 0) + 1
            self._hnt_init(u, [], t)
            return

        for c in cats:
            if c not in self.cat_medoid:
                self.cat_medoid[c] = u

        # 5. Filtered-Vamana greedy search from category medoids
        #    build_cat_filter=cats ensures we only expand neighbors sharing
        #    at least one category with u → Filtered-Vamana guarantee.
        candidates_set: Set[int] = set()
        for c in cats:
            ep = self.cat_medoid[c]
            found, _ = self._greedy_search(
                vec, [ep], self.ef_construction,
                use_hnt=False, build_cat_filter=cats,
            )
            candidates_set.update(found)

        # FIX: fallback collects all category medoids instead of only node 0
        if not candidates_set:
            eps = [0]
            for c in cats:
                if c in self.cat_medoid and self.cat_medoid[c] != 0:
                    eps.append(self.cat_medoid[c])
            found, _ = self._greedy_search(
                vec, eps, self.ef_construction,
                use_hnt=False, build_cat_filter=cats,
            )
            candidates_set.update(found)

        # 6. Extend with 2-hop neighbors; filter expired
        two_hop: Set[int] = set()
        for cid in candidates_set:
            for nid, _ in self.Lnow[cid]:
                two_hop.add(nid)
        candidates_set.update(two_hop)
        candidates_set.discard(u)
        candidates_set = {c for c in candidates_set if self.expire_days[c] is None}

        if not candidates_set:
            self._hnt_init(u, [], t)
            # Still do medoid bookkeeping
            for c in cats:
                self._cat_insert_count[c] = self._cat_insert_count.get(c, 0) + 1
                if self._cat_insert_count[c] % 500 == 0:
                    self._recompute_medoid(c)
            return

        # 7. Score candidates: alpha * same_cat(u, cid) + (1-alpha) * cosine(u, cid)
        scored = []
        for cid in candidates_set:
            s = self.alpha * (1.0 if cats & self.cat_sets[cid] else 0.0) + \
                (1.0 - self.alpha) * float(np.dot(vec, self.vectors[cid]))
            scored.append((s, cid))
        scored.sort(key=lambda x: -x[0])

        # 8. Robust prune → primary (M), remaining → backup (M)
        primary = self._robust_prune(u, scored, self.M)
        primary_set = set(primary)
        remaining = [cid for _, cid in scored if cid not in primary_set]
        backup = remaining[:self.M]

        # 9. HNT init
        self._hnt_init(u, primary, t)
        self.B[u] = backup

        # 10. Bidirectional linking: for each v in primary, re-score v's pool
        #     including u, robust_prune, if changed _hnt_append
        for v in primary:
            current_nids = [nid for nid, _ in self.Lnow[v]]
            current_nids_set = set(current_nids)
            candidate_pool_v = list(current_nids_set) + [u]
            scored_v = [(self._score(v, cid), cid) for cid in candidate_pool_v]
            scored_v.sort(key=lambda x: -x[0])

            new_primary_v = self._robust_prune(v, scored_v, self.M)

            if set(new_primary_v) != current_nids_set:
                self._hnt_append(v, new_primary_v, eviction_day=t)
                new_primary_v_set = set(new_primary_v)
                remaining_v = [cid for _, cid in scored_v
                               if cid not in new_primary_v_set and cid != v]
                self.B[v] = remaining_v[:self.M]

        # 11. ST-connectivity guard (after bidirectional linking)
        #     For each c in cats: ensure medoid can reach u through
        #     category-consistent path by potentially force-adding u
        #     to medoid's neighbor list.
        for c in cats:
            med = self.cat_medoid[c]
            if med == u:
                continue
            med_lnow_ids = [nid for nid, _ in self.Lnow[med]]
            if not med_lnow_ids:
                # Medoid has no neighbors — force-add u
                new_med_pool = [u]
                scored_med = [(self._score(med, cid), cid) for cid in new_med_pool]
                scored_med.sort(key=lambda x: -x[0])
                new_primary_med = self._robust_prune(med, scored_med, self.M)
                if set(new_primary_med) != set(med_lnow_ids):
                    self._hnt_append(med, new_primary_med, t)
                continue

            cos_med_u = self._cosine_sim(med, u)
            min_cos = min(self._cosine_sim(med, nid) for nid in med_lnow_ids)

            if len(med_lnow_ids) < self.M or cos_med_u > min_cos:
                if u not in set(med_lnow_ids):
                    new_med_pool = med_lnow_ids + [u]
                    scored_med = [(self._score(med, cid), cid) for cid in new_med_pool]
                    scored_med.sort(key=lambda x: -x[0])
                    new_primary_med = self._robust_prune(med, scored_med, self.M)
                    if set(new_primary_med) != set(med_lnow_ids):
                        self._hnt_append(med, new_primary_med, t)

        # 12. Recompute medoid every 500 insertions per category
        for c in cats:
            self._cat_insert_count[c] = self._cat_insert_count.get(c, 0) + 1
            if self._cat_insert_count[c] % 500 == 0:
                self._recompute_medoid(c)

    # ------------------------------------------------------------------
    # Tombstone delete
    # ------------------------------------------------------------------

    def tombstone_delete(self, u: int, t_expire: int):
        """Soft-delete node u at time t_expire.

        1. Set expire_days[u] = t_expire
        2. For each v in reverse_adj[u]:
           a. Remove u from Lnow[v]
           b. Promote backups from B[v] to refill to M slots
           c. If B[v] exhausted AND len < M: re-search with build_cat_filter
           d. _hnt_append(v, new_nids, t_expire)
        3. Clear B[u] and reverse_adj[u]
        4. Every 1000 deletes: compact cat_index
        """
        self.expire_days[u] = t_expire

        # Find all nodes v that have u in their Lnow
        affected = list(self.reverse_adj[u])

        for v in affected:
            current_nids = [nid for nid, _ in self.Lnow[v]]
            if u not in current_nids:
                continue

            # a. Remove u from v's neighbor list
            new_nids = [nid for nid in current_nids if nid != u]

            # b. Promote backups to fill back to M slots
            new_nids_set = set(new_nids)
            for b in list(self.B[v]):
                if len(new_nids) >= self.M:
                    break
                if b != u and self.expire_days[b] is None and b not in new_nids_set:
                    new_nids.append(b)
                    new_nids_set.add(b)
                    self.B[v].remove(b)

            # c. Re-search when backup exhausted and still under M
            if not self.B[v] and len(new_nids) < self.M and self.N > 1:
                # FIX: prefer category medoid over node 0 even when node 0 is alive
                cat_v = next(iter(self.cat_sets[v]), None)
                ep = 0  # default fallback
                if cat_v and cat_v in self.cat_medoid:
                    med_candidate = self.cat_medoid[cat_v]
                    if self.expire_days[med_candidate] is None:
                        ep = med_candidate
                elif self.expire_days[0] is not None:
                    # node 0 expired and no medoid — find any live node
                    for i in range(self.N):
                        if self.expire_days[i] is None:
                            ep = i
                            break
                found, _ = self._greedy_search(
                    self.vectors[v], [ep], self.ef_construction,
                    use_hnt=False, build_cat_filter=self.cat_sets[v],
                )
                replacement = [nid for nid in found
                               if nid != v and nid != u
                               and self.expire_days[nid] is None
                               and nid not in new_nids_set]
                for r in replacement:
                    if len(new_nids) >= self.M:
                        break
                    new_nids.append(r)
                    new_nids_set.add(r)
                # Rebuild B[v] from remaining replacement candidates
                self.B[v] = [nid for nid in replacement
                             if nid not in new_nids_set][:self.M]

            # d. Record eviction in HNT
            self._hnt_append(v, new_nids, eviction_day=t_expire)

        # 3. Clear deleted node's structures
        self.B[u] = []
        self.reverse_adj[u].clear()

        # 4. Every 1000 deletes: compact cat_index
        self._delete_count += 1
        if self._delete_count % 1000 == 0:
            self._compact_cat_index()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, q: np.ndarray, C: str, t_start: int, t_end: int,
              k: int = 10, ef: int = 100) -> Tuple[np.ndarray, int]:
        """Query the index for k nearest neighbors matching category C in [t_start, t_end].

        Returns (result_ids, visited_count) where result_ids is a numpy array of
        node IDs sorted by cosine similarity descending and visited_count is the
        number of nodes touched during beam search.

        1. Normalize q to unit float32, ravel to 1D
        2. Build valid_set: nodes in cat_index[C] where start_days[v] <= t_end
           AND not expired before t_start
        3. If len(valid_set) <= k: brute force, return sorted by cosine sim
        4. Seeds: cat_medoid[C] + top-3 from valid_set by dot product (deterministic)
        5. _greedy_search with use_hnt=True
        6. Filter results to valid_set; sort by cosine sim descending; return top-k
        7. NO brute-force supplement after beam search
        """
        # 1. Normalize query vector
        q = q.astype(np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # 2. Compute valid set: start_days[v] in [t_start, t_end] and not expired
        valid_set = []
        for v in self.cat_index.get(C, []):
            if self.start_days[v] >= t_start and self.start_days[v] <= t_end:
                exp = self.expire_days[v]
                if exp is None or exp > t_start:
                    valid_set.append(v)

        if not valid_set:
            return np.array([], dtype=np.int64), 0

        valid_set_s = set(valid_set)
        # After you build valid_set
        mid_t = (t_start + t_end) // 2

        # 3. Seeds: category medoid if live, else first valid node
        seeds: List[int] = []
        if C in self.cat_medoid:
            med = self.cat_medoid[C]
            if self.expire_days[med] is None:
                seeds.append(med)
        if valid_set:
            # restrict to a small prefix or random subset to keep cost bounded
            sample = valid_set if len(valid_set) <= 128 else valid_set[:128]
            closest = sorted(sample, key=lambda v: abs(self.start_days[v] - mid_t))
            seed_set = set(seeds)
            for v in closest[:2]:
                if v not in seed_set:
                    seeds.append(v)
                    seed_set.add(v)

        # Fallback
        if not seeds:
            seeds.append(valid_set[0])

        # 4. Primary beam search with HNT (temporal + category filtering)
        results, visited_count = self._greedy_search(
            q, seeds, ef,
            t_start=t_start, t_end=t_end,
            cat_filter=C, use_hnt=True,
        )
        filtered = [nid for nid in results if nid in valid_set_s]

        # 5. ACORN-style bounded fallback: 2-hop expansion if < k results
        if len(filtered) < k:
            frontier = filtered if filtered else list(seeds)
            seen: Set[int] = set(results)  # nodes already touched
            max_extra = 300                # hard cap on extra candidates
            extra_candidates: List[int] = []

            for _ in range(2):  # up to 2 hops
                if not frontier or len(extra_candidates) >= max_extra:
                    break
                next_frontier: List[int] = []
                for u in frontier:
                    nbrs = self._hnt_reconstruct_window(u, t_start, t_end, C)
                    for v in nbrs:
                        if v in seen:
                            continue
                        seen.add(v)
                        visited_count += 1
                        next_frontier.append(v)
                        if v in valid_set_s:
                            extra_candidates.append(v)
                            if len(extra_candidates) >= max_extra:
                                break
                    if len(extra_candidates) >= max_extra:
                        break
                frontier = next_frontier

            # merge extra candidates not already in filtered
            extra_unique = [v for v in extra_candidates if v not in filtered]
            filtered.extend(extra_unique)

        if not filtered:
            return np.array([], dtype=np.int64), visited_count

        # 6. Rank by cosine similarity within filtered set and return top-k
        scored = [(float(np.dot(q, self.vectors[v])), v) for v in filtered]
        scored.sort(key=lambda x: -x[0])
        top_k = [v for _, v in scored[:k]]

        # Remap internal IDs → original dataset IDs, if present
        if hasattr(self, "_id_map"):
            top_k = [self._id_map[nid] for nid in top_k]

        return np.array(top_k, dtype=np.int64), visited_count

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, cat_sets_list,
              start_days_list):
        """Build index by inserting all nodes in chronological order.

        Sort by start_day ascending, call insert for each in order.
        After all inserts: recompute medoid for all categories.
        Print progress every 10,000 nodes.
        No _compact_cat_index at build time (no deletions have occurred).
        """
        start_days_list = [int(d) for d in start_days_list]  # normalize numpy → int
        n = len(vectors)
        assert n == len(cat_sets_list) == len(start_days_list)

        # Sort by start_day ascending
        order = sorted(range(n), key=lambda i: start_days_list[i])
        self._id_map = list(order)  # internal_id → original_id

        for count, idx in enumerate(order):
            cats = set(cat_sets_list[idx])
            self.insert(vectors[idx], cats, start_days_list[idx])
            if (count + 1) % 10000 == 0:
                print(f"  Inserted {count + 1}/{n} nodes")

        # Final medoid recomputation for all categories
        for c in self.cat_index:
            self._recompute_medoid(c)

        print(f"  Build complete: {self.N} nodes, {len(self.cat_index)} categories")
