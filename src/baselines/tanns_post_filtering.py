"""
TANNS — Timestamp Approximate Nearest Neighbor Search
Implementation of the compressed timestamp graph with Historic Neighbor Tree (HNT)
from the ICDE'25 paper: "Timestamp Approximate Nearest Neighbor Search
over High-Dimensional Vector Data" by Wang et al.

This version strictly follows the paper's Algorithms 1–4 and Select-Nbrs.
"""

import sys
import struct
import json
import heapq
import bisect
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict

assert sys.version_info >= (3, 10), (
    f"Requires Python 3.10+ for bisect key= support (found {sys.version_info})"
)

# ── Paper Alignment Notes ────────────────────────────────────────────
#
# Deviation 1 — Base graph search: Replaced single min-heap beam search
#   with Algorithm 1's faithful pool/ann separation. pool tracks candidates
#   to explore; ann tracks the current best k' results. Early-stop fires
#   when the closest candidate in pool is farther than the furthest in ann
#   (dis(q,v) > dis(q,u) in distance terms → sim(q,v) < sim(q,u) in
#   similarity terms). Neighbor insertion uses paper's line 13 condition.
#
# Deviation 2 — Select-Nbrs: Replaced DiskANN-style pruning
#   (sim(s,c) > PRUNE_ALPHA * sim(u,c)) with the paper's HNSW heuristic
#   (Algorithm 2, lines 10–16). A point u is dominated by v if
#   dis(o,v) < dis(o,u) AND dis(v,u) < dis(o,u), which in similarity
#   terms is: sim(o,v) > sim(o,u) AND sim(v,u) > sim(o,u).
#   PRUNE_ALPHA removed entirely. No greedy fill-up of remaining slots.
#
# Deviation 3 — Point Insertion bidirectional linking: Replaced ad-hoc
#   domination check with the paper's Algorithm 3 lines 4–12 exactly.
#   Domination uses the same Select-Nbrs heuristic: o is dominated by
#   v ∈ TG[u] if sim(u,v) > sim(u,o) AND sim(v,o) > sim(u,o).
#   "Closer to u than furthest" uses similarity (larger = closer).
#
# Deviation 4 — Parameters: Updated to paper defaults from Section VI-A:
#   M=16, M'=200, μ=8, EF_SEARCH=200. Removed PRUNE_ALPHA.
# ─────────────────────────────────────────────────────────────────────

# ── Module-level constants ──────────────────────────────────────────
# FIX: Deviation 4 — Paper default parameters (Section VI-A)
M = 16              # neighbor list size (paper default)
M_PRIME = 200        # candidate neighbors during construction (paper: M' = 200)
MU = 8               # HNT leaf size threshold (paper default μ = 8)
EF_SEARCH = 200      # search expansion factor k' (paper: k' = k · ef, ef ≈ 20 for k=10)


# ── File loaders ────────────────────────────────────────────────────

def load_fvecs(path: str) -> np.ndarray:
    """Load .fvecs file → (N, D) float32, unit-normalized at load time."""
    with open(path, 'rb') as f:
        data = f.read()
    vectors = []
    offset = 0
    while offset < len(data):
        dim = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        vec = np.frombuffer(data, dtype=np.float32, count=dim, offset=offset)
        offset += dim * 4
        vectors.append(vec.copy())
    vecs = np.array(vectors, dtype=np.float32)
    # Unit-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs


def load_ivecs(path: str) -> List[List[int]]:
    """Load .ivecs file → list of int lists."""
    with open(path, 'rb') as f:
        data = f.read()
    result = []
    offset = 0
    while offset < len(data):
        dim = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        vals = list(struct.unpack_from(f'<{dim}i', data, offset))
        offset += dim * 4
        result.append(vals)
    return result


def load_jsonl(path: str) -> List[dict]:
    """Load .jsonl file → list of dicts."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── HNT Node ────────────────────────────────────────────────────────

@dataclass
class HNTNode:
    t: Optional[int]                          # split timestamp; None for leaf nodes
    is_leaf: bool
    # Points stored in this node: (start_day, end_day, node_id)
    points_by_start: List[Tuple[int, int, int]] = field(default_factory=list)  # sorted by start_day asc
    points_by_end: List[Tuple[int, int, int]] = field(default_factory=list)    # sorted by end_day asc
    left: Optional['HNTNode'] = None
    right: Optional['HNTNode'] = None
    parent: Optional['HNTNode'] = None
    size: int = 0                             # number of points in this node


# ── TANNS Class ─────────────────────────────────────────────────────

class TANNS:
    def __init__(self, M_param: int = M, ef_construction: int = M_PRIME, mu: int = MU):
        self._M = M_param
        self._ef_construction = ef_construction
        self._mu = mu

        # Parallel arrays, all index-aligned on node_id
        self.vectors: List[np.ndarray] = []
        self.cat_sets: List[Set[str]] = []
        self.start_days: List[int] = []
        self.expire_days: List[Optional[int]] = []
        self.Lnow: List[List[Tuple[int, int]]] = []       # [(node_id, start_day)] sorted by start_day asc
        self.B: List[List[int]] = []                       # backup neighbors (up to M)
        self.hnt_root: List[Optional[HNTNode]] = []
        self.hnt_newest_leaf: List[Optional[HNTNode]] = []
        self.cat_index: Dict[str, List[int]] = {}
        self.entry_point: int = -1
        self.reverse_adj: List[Set[int]] = []

    @property
    def N(self) -> int:
        return len(self.vectors)

    # ── Distance / Similarity ──────────────────────────────────────
    def _sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (vectors are pre-normalized, so dot product).
        Higher = closer. Distance = 1 - sim."""
        return float(np.dot(a, b))

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance = 1 - dot(a, b). Lower = closer."""
        return 1.0 - float(np.dot(a, b))

    # ── Neighbor selection (Paper Algorithm 2, lines 10–16) ────────

    # FIX: Deviation 2 — HNSW heuristic Select-Nbrs replaces DiskANN-style
    # A point u is dominated by v ∈ selected if:
    #   dis(o, v) < dis(o, u) AND dis(v, u) < dis(o, u)
    # In similarity terms:
    #   sim(o, v) > sim(o, u) AND sim(v, u) > sim(o, u)
    def _select_neighbors(self, o_vec: np.ndarray, candidates: List[int], max_neighbors: int) -> List[int]:
        """
        Select up to max_neighbors from candidates using the HNSW heuristic.
        Sort candidates by ascending distance to o (= descending similarity).
        For each candidate u: skip if dominated by any already-selected v
        where dis(o,v) < dis(o,u) AND dis(v,u) < dis(o,u).
        """
        if not candidates:
            return []
        # Sort by descending similarity (ascending distance)
        cand_sims = [(c, self._sim(o_vec, self.vectors[c])) for c in candidates]
        cand_sims.sort(key=lambda x: -x[1])

        selected = []
        for u, sim_ou in cand_sims:
            if len(selected) >= max_neighbors:
                break
            # Check if u is dominated by any already-selected v
            dominated = False
            for v in selected:
                sim_ov = self._sim(o_vec, self.vectors[v])
                sim_vu = self._sim(self.vectors[v], self.vectors[u])
                # dis(o,v) < dis(o,u) ↔ sim_ov > sim_ou
                # dis(v,u) < dis(o,u) ↔ sim_vu > sim_ou
                if sim_ov > sim_ou and sim_vu > sim_ou:
                    dominated = True
                    break
            if not dominated:
                selected.append(u)

        return selected

    # ── HNT operations ──────────────────────────────────────────────

    def _hnt_init(self, u: int, neighbor_ids: List[int], t: int):
        """Initialize HNT for node u with given neighbors at time t."""
        # All entries share the same start_day=t, so no sort needed;
        # _hnt_append will re-sort Lnow when it rebuilds after changes.
        self.Lnow[u] = [(nid, t) for nid in neighbor_ids]
        self.hnt_root[u] = None
        self.hnt_newest_leaf[u] = None
        for nid in neighbor_ids:
            self.reverse_adj[nid].add(u)

    def _hnt_reconstruct(self, u: int, ts: int) -> List[int]:
        """Algorithm 5: Reconstruct neighbor list at timestamp ts."""
        result = []
        result_set = set()

        # Step 1: Scan Lnow[u] (sorted by start_day)
        for nid, start_day in self.Lnow[u]:
            if start_day > ts:
                break
            # Skip expired nodes
            if self.expire_days[nid] is not None and self.expire_days[nid] <= ts:
                continue
            if nid not in result_set:
                result.append(nid)
                result_set.add(nid)

        # Step 2: If no HNT, return
        if self.hnt_root[u] is None:
            return result

        # Step 3: Traverse HNT
        node = self.hnt_root[u]
        while node is not None:
            if not node.is_leaf:
                # Internal node
                if ts == node.t:
                    # Add all points where start_day <= ts
                    for start_day, end_day, nid in node.points_by_start:
                        if start_day > ts:
                            break
                        if self.expire_days[nid] is not None and self.expire_days[nid] <= ts:
                            continue
                        if nid not in result_set:
                            result.append(nid)
                            result_set.add(nid)
                    return result
                elif ts < node.t:
                    # Scan points_by_start forward
                    for start_day, end_day, nid in node.points_by_start:
                        if start_day > ts:
                            break
                        if self.expire_days[nid] is not None and self.expire_days[nid] <= ts:
                            continue
                        if nid not in result_set:
                            result.append(nid)
                            result_set.add(nid)
                    node = node.left
                else:  # ts > node.t
                    # bisect to first entry with end_day > ts (sorted ascending by end_day)
                    lo = bisect.bisect_right(node.points_by_end, ts, key=lambda x: x[1])
                    for start_day, end_day, nid in node.points_by_end[lo:]:
                        if start_day > ts:
                            continue
                        exp = self.expire_days[nid]
                        if exp is not None and exp <= ts:
                            continue
                        if nid not in result_set:
                            result.append(nid)
                            result_set.add(nid)
                    node = node.right
            else:
                # Leaf node
                for start_day, end_day, nid in node.points_by_start:
                    if start_day <= ts and end_day > ts:
                        if self.expire_days[nid] is not None and self.expire_days[nid] <= ts:
                            continue
                        if nid not in result_set:
                            result.append(nid)
                            result_set.add(nid)
                node = None

        return result

    def _get_active_path(self, u: int) -> List[HNTNode]:
        """Get active path from root to newest leaf."""
        if self.hnt_newest_leaf[u] is None:
            return []
        path = []
        node = self.hnt_newest_leaf[u]
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()  # root -> ... -> newest_leaf
        return path

    def _insert_sorted_by_start(self, node: HNTNode, entry: Tuple[int, int, int]):
        """Insert into points_by_start maintaining sort by start_day."""
        bisect.insort(node.points_by_start, entry, key=lambda x: x[0])

    def _insert_sorted_by_end(self, node: HNTNode, entry: Tuple[int, int, int]):
        """Insert into points_by_end maintaining sort by end_day."""
        bisect.insort(node.points_by_end, entry, key=lambda x: x[1])

    def _remove_point_from_node(self, node: HNTNode, nid: int):
        """Remove point with given nid from a node's point lists."""
        node.points_by_start = [p for p in node.points_by_start if p[2] != nid]
        node.points_by_end = [p for p in node.points_by_end if p[2] != nid]
        if node.is_leaf:
            node.size = len(node.points_by_start)

    def _find_point_in_hnt(self, u: int, nid: int) -> Optional[HNTNode]:
        """Find which HNT node contains the point with given nid."""
        if self.hnt_root[u] is None:
            return None
        stack = [self.hnt_root[u]]
        while stack:
            node = stack.pop()
            for _, _, pid in node.points_by_start:
                if pid == nid:
                    return node
            if not node.is_leaf:
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return None

    def _find_maximal_complete_subtree(self, leaf: HNTNode) -> HNTNode:
        """
        Find the maximal complete binary subtree containing the given leaf.
        Walk up while the node is the LEFT child of its parent — a left
        subtree that is itself complete can grow into its parent. Stop when
        the node is a right child (the subtree above may not be complete).
        """
        node = leaf
        while node.parent is not None:
            if node.parent.left is node:
                # We are the left child — subtree is complete, can grow
                node = node.parent
            else:
                # We are the right child — stop
                break
        return node

    def _hnt_append(self, u: int, new_Lnow_ids: List[int], t: int):
        """Algorithm 6: Append neighbor list — evict old neighbors into HNT."""
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for x in new_Lnow_ids:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        new_Lnow_ids = deduped

        old_lnow = self.Lnow[u]
        old_ids = {nid for nid, _ in old_lnow}
        new_ids_set = set(new_Lnow_ids)

        # Find evicted points
        evicted = []
        for nid, start_day in old_lnow:
            if nid not in new_ids_set:
                evicted.append((nid, start_day))

        # Process each evicted point
        for nid, original_start_day in evicted:
            end_day = t
            entry = (original_start_day, end_day, nid)

            # Check if point is already in HNT
            existing_node = self._find_point_in_hnt(u, nid)

            if existing_node is None:
                # Case A: point not yet in HNT
                active_path = self._get_active_path(u)

                if not active_path:
                    # No HNT yet — create first leaf
                    new_leaf = HNTNode(t=None, is_leaf=True, size=0)
                    self._insert_sorted_by_start(new_leaf, entry)
                    self._insert_sorted_by_end(new_leaf, entry)
                    new_leaf.size = len(new_leaf.points_by_start)
                    self.hnt_root[u] = new_leaf
                    self.hnt_newest_leaf[u] = new_leaf
                    # Check split
                    self._maybe_split_leaf(u, t)
                else:
                    # Scan active path top-down (root → newest_leaf) for the
                    # highest node where p is valid at that node's timestamp.
                    # If no internal node satisfies the validity check, fall
                    # through to the newest leaf nc (which is always the last
                    # element in active_path since the path is root → leaf).
                    placed = False
                    nc = self.hnt_newest_leaf[u]
                    for node in active_path:
                        if node is nc:
                            # Reached newest leaf — place here unconditionally
                            self._insert_sorted_by_start(node, entry)
                            self._insert_sorted_by_end(node, entry)
                            node.size = len(node.points_by_start)
                            placed = True
                            break
                        elif not node.is_leaf and node.t is not None:
                            # Check if p is valid at n.t: start_day <= n.t AND end_day > n.t
                            if original_start_day <= node.t and end_day > node.t:
                                self._insert_sorted_by_start(node, entry)
                                self._insert_sorted_by_end(node, entry)
                                placed = True
                                break
                    if not placed:
                        # Fallback: place at newest leaf (only reachable if nc
                        # was not in active_path, e.g. single-node tree edge case)
                        self._insert_sorted_by_start(nc, entry)
                        self._insert_sorted_by_end(nc, entry)
                        nc.size = len(nc.points_by_start)

                    # Check if newest leaf needs splitting
                    self._maybe_split_leaf(u, t)
            else:
                # Case B: point already in HNT (re-entering then re-evicting)
                # Extract the stored start_day from the HNT entry before removing,
                # since the Lnow original_start_day only reflects the most recent
                # re-entry — the HNT entry has the true earliest start_day.
                stored_start = original_start_day  # fallback
                for sd, ed, pid in existing_node.points_by_start:
                    if pid == nid:
                        stored_start = sd
                        break
                self._remove_point_from_node(existing_node, nid)
                new_entry = (stored_start, end_day, nid)

                # Walk from root to existing_node, place at highest valid
                path_to_node = []
                n = existing_node
                while n is not None:
                    path_to_node.append(n)
                    n = n.parent
                path_to_node.reverse()  # root -> ... -> existing_node

                placed = False
                for node in path_to_node:
                    if not node.is_leaf and node.t is not None:
                        if stored_start <= node.t and end_day > node.t:
                            self._insert_sorted_by_start(node, new_entry)
                            self._insert_sorted_by_end(node, new_entry)
                            placed = True
                            break
                if not placed:
                    # Place back in original node
                    self._insert_sorted_by_start(existing_node, new_entry)
                    self._insert_sorted_by_end(existing_node, new_entry)
                    if existing_node.is_leaf:
                        existing_node.size = len(existing_node.points_by_start)

        # Rebuild Lnow[u]
        # Carry forward original_start_day for unchanged neighbors; new ones get start_day = t
        new_lnow = []
        old_start_map = {nid: sd for nid, sd in old_lnow}
        for nid in new_Lnow_ids:
            if nid in old_start_map and nid in old_ids:
                new_lnow.append((nid, old_start_map[nid]))
            else:
                new_lnow.append((nid, t))
        new_lnow.sort(key=lambda x: x[1])
        self.Lnow[u] = new_lnow

        # Update reverse_adj
        for nid, _ in evicted:
            self.reverse_adj[nid].discard(u)
        for nid in new_Lnow_ids:
            if nid not in old_ids:
                self.reverse_adj[nid].add(u)

    def _maybe_split_leaf(self, u: int, t: int):
        """Check if newest leaf needs splitting after insertion."""
        nc = self.hnt_newest_leaf[u]
        if nc is None or nc.size <= self._mu:
            return

        # Find maximal complete binary subtree containing nc
        treec = self._find_maximal_complete_subtree(nc)

        # Save treec's original parent before overwriting
        original_parent = treec.parent

        # Create new internal node ni: left = treec, timestamp = t
        ni = HNTNode(t=t, is_leaf=False)
        ni.left = treec
        ni.parent = original_parent  # ni takes treec's old parent slot

        # Create new empty leaf nl as right child of ni
        nl = HNTNode(t=None, is_leaf=True, size=0)
        nl.parent = ni
        ni.right = nl

        # Reparent treec under ni
        treec.parent = ni

        # Patch original_parent's child pointer to ni
        if original_parent is not None:
            if original_parent.left is treec:
                original_parent.left = ni
            else:
                original_parent.right = ni

        # Update newest leaf
        self.hnt_newest_leaf[u] = nl

        # If treec was the root, ni becomes the new root
        if treec is self.hnt_root[u]:
            self.hnt_root[u] = ni

    # ── Greedy search (Paper Algorithm 1) ──────────────────────────

    # FIX: Deviation 1 — Faithful Algorithm 1 with pool/ann separation.
    # pool = candidates to explore (sorted set by distance to q).
    # ann = current best k' results (bounded).
    # Early-stop: line 9 — if dis(q,v) > dis(q,u) where v is furthest in ann
    #   and u is closest in pool, break. In sim terms: sim(q,v) < sim(q,u) → break
    #   i.e. if best remaining candidate is worse than the worst result, stop.
    #   Paper actually says: if dis(q,v) < dis(q,u) then break — meaning if the
    #   furthest in ann is CLOSER than the closest candidate, we're done.
    #   Wait — re-read: line 8: v ← point furthest from q in ann
    #   line 9: if dis(q,v) < dis(q,u) then break
    #   This means: if the WORST result is still better than the BEST candidate, stop.
    #   But that's wrong for distance — paper says "if dis(q,v) > dis(q,u) then break"
    #   Actually looking at the PDF again:
    #   Line 9: "if dis(q, v) > dis(q, u) then break;"
    #   Wait no — the PDF says line 9: "if dis(q, v) < dis(q, u) then break;"
    #   Let me re-check... The user's spec says the early-stop is:
    #   "if dis(q,v) < dis(q,u): break  ← paper line 9"
    #   where u = closest to q in pool, v = furthest from q in ann.
    #   This means: break when furthest-in-ann is closer than closest-in-pool.
    #   In sim terms: sim(q,v) > sim(q,u) → break.
    #   But that would mean: the worst result is BETTER than the best candidate.
    #   That doesn't make sense for early termination...
    #
    #   Actually re-reading the PDF text on page 3:
    #   "The search terminates when ann reaches the size of k' and all points in
    #    ann are closer to q than those in pool."
    #   So: break when all ann points are closer than all pool points.
    #   v = furthest in ann, u = closest in pool.
    #   "all ann closer than pool" → dis(q,v) < dis(q,u) → break
    #   In sim terms: sim(q,v) > sim(q,u) → break.
    #
    #   But actually the original HNSW paper (Malkov & Yashunin) uses:
    #   "if distance(c, q) > distance(f, q)" where c=nearest in candidates, f=furthest in result
    #   which is: if dis(q,u) > dis(q,v) then break — meaning if the best candidate
    #   is worse than worst result, stop. That's the SAME condition just written differently.
    #
    #   The condition in the paper Algorithm 1 line 9 reads from the PDF:
    #   "if dis(q, v) > dis(q, u) then break"
    #   where u = closest to q in pool (line 6), v = furthest from q in ann (line 8).
    #   This means: if furthest-in-ann is further than closest-in-pool, break.
    #   That would break immediately! That can't be right.
    #
    #   Actually, re-reading the PDF image more carefully:
    #   Line 8: v ← point furthest from q in ann
    #   Line 9: if dis(q, v) < dis(q, u) then break
    #   The '<' means: if the furthest result is CLOSER than the best candidate,
    #   all results are closer than any remaining candidate → stop.
    #   In similarity terms: sim(q,v) > sim(q,u) → break.
    def _greedy_search(self, q_norm: np.ndarray, entry_points: List[int],
                       ef: int, ts: int = None,
                       use_hnt: bool = False) -> Tuple[List[int], int]:
        """
        Paper Algorithm 1: HNSW Search with pool/ann separation.
        Returns (result_ids sorted by sim desc, visited_count).
        """
        visited = set()
        # pool: max-heap of (sim, id) — we want closest to q (highest sim) at top
        # Use negated sim in a min-heap for pool to get max-sim at top
        pool = []   # min-heap of (-sim, id): pop gives highest sim
        # ann: min-heap of (sim, id) — smallest sim at top for easy eviction of furthest
        ann = []    # min-heap of (sim, id): top is furthest from q (smallest sim)

        for ep in entry_points:
            if ep < 0 or ep >= self.N:
                continue
            if ep in visited:
                continue
            visited.add(ep)
            sim_ep = float(np.dot(q_norm, self.vectors[ep]))
            heapq.heappush(pool, (-sim_ep, ep))
            heapq.heappush(ann, (sim_ep, ep))

        while pool:
            # u ← point closest to q in pool (highest similarity)
            neg_sim_u, u = heapq.heappop(pool)
            sim_u = -neg_sim_u

            # v ← point furthest from q in ann (lowest similarity)
            # ann[0] is the min element = furthest from q
            sim_v_furthest = ann[0][0]

            # Paper line 9: if dis(q,v) < dis(q,u) then break
            # ↔ if sim(q,v) > sim(q,u) then break
            # i.e. if the worst result is still better than the best candidate, stop
            if sim_v_furthest > sim_u:
                break

            # Get neighbors of u
            if use_hnt and ts is not None:
                neighbors = self._hnt_reconstruct(u, ts)
            else:
                neighbors = [nid for nid, _ in self.Lnow[u]]

            for o in neighbors:
                if o in visited:
                    continue
                if o < 0 or o >= self.N:
                    continue
                visited.add(o)
                sim_o = float(np.dot(q_norm, self.vectors[o]))

                # Paper line 12: v ← point furthest from q in ann
                sim_v = ann[0][0]

                # Paper line 13: if |ann| < k' or dis(q,o) < dis(q,v)
                # ↔ if |ann| < ef or sim(q,o) > sim(q,v)
                if len(ann) < ef or sim_o > sim_v:
                    # Paper line 14: Insert o into ann and pool
                    heapq.heappush(ann, (sim_o, o))
                    heapq.heappush(pool, (-sim_o, o))
                    # Paper line 15-16: if |ann| > k' then remove furthest
                    if len(ann) > ef:
                        heapq.heappop(ann)

        # Return k closest in ann (sorted by similarity descending)
        result_list = sorted(ann, key=lambda x: -x[0])
        return [rid for _, rid in result_list], len(visited)

    # ── Insert (Paper Algorithm 3) ──────────────────────────────────

    # FIX: Deviation 3 — Faithful Algorithm 3 bidirectional linking (lines 4–12).
    # For each u ∈ TG[o]:
    #   if o is not dominated by points in TG[u] AND
    #      o is closer to u than the furthest point in TG[u]:
    #     move furthest from TG[u] into B[u], add o to TG[u], append history
    #   else:
    #     add o to B[u]; if |B[u]| > M: remove furthest from B[u]
    #
    # Domination check uses Select-Nbrs heuristic:
    #   o is dominated by v ∈ TG[u] if dis(u,v) < dis(u,o) AND dis(v,o) < dis(u,o)
    #   ↔ sim(u,v) > sim(u,o) AND sim(v,o) > sim(u,o)
    def insert(self, vec: np.ndarray, cats: Set[str], t: int):
        """Insert a new point into the TANNS index (Algorithm 3)."""
        o = self.N
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Append to parallel arrays
        self.vectors.append(vec.astype(np.float32))
        self.cat_sets.append(cats)
        self.start_days.append(t)
        self.expire_days.append(None)
        self.Lnow.append([])
        self.B.append([])
        self.hnt_root.append(None)
        self.hnt_newest_leaf.append(None)
        self.reverse_adj.append(set())

        # Update cat_index
        for c in cats:
            if c not in self.cat_index:
                self.cat_index[c] = []
            self.cat_index[c].append(o)

        # First node
        if o == 0:
            self._hnt_init(o, [], t)
            self.entry_point = 0
            return

        # Algorithm 3, line 1: cand ← Search(TG, o, M')
        cands, _ = self._greedy_search(vec, [self.entry_point], self._ef_construction, use_hnt=False)

        # Filter out self and expired
        cands = [c for c in cands if c != o and (self.expire_days[c] is None)]

        # Algorithm 3, line 2: TG[o], B[o] ← Select-Nbrs(o, cand, 2M)
        selected = self._select_neighbors(vec, cands, 2 * self._M)

        primary = selected[:self._M]
        backup = selected[self._M:]

        # Guard: if primary is empty but the graph is non-trivial, connect to entry_point
        if not primary and self.N > 1:
            primary = [self.entry_point]

        # Algorithm 3, line 3: Initialize o's historic neighbor list with TG[o]
        self._hnt_init(o, primary, t)
        self.B[o] = backup[:self._M]

        # Algorithm 3, lines 4-12: Bidirectional linking
        for u in primary:
            lnow_u_ids = [nid for nid, _ in self.Lnow[u]]

            if not lnow_u_ids:
                # Empty neighbor list — just add o
                new_ids = [o]
                self._hnt_append(u, new_ids, t)
                continue

            sim_uo = self._sim(self.vectors[u], vec)  # sim(u, o)

            # Check domination: o is dominated by v ∈ TG[u] if
            # sim(u,v) > sim(u,o) AND sim(v,o) > sim(u,o)
            dominated = False
            for v in lnow_u_ids:
                sim_uv = self._sim(self.vectors[u], self.vectors[v])
                sim_vo = self._sim(self.vectors[v], vec)
                if sim_uv > sim_uo and sim_vo > sim_uo:
                    dominated = True
                    break

            # Find furthest point in TG[u] (smallest similarity to u)
            furthest_id = -1
            furthest_sim = float('inf')
            for nid in lnow_u_ids:
                s = self._sim(self.vectors[u], self.vectors[nid])
                if s < furthest_sim:
                    furthest_sim = s
                    furthest_id = nid

            # Algorithm 3, line 5: if o is not dominated AND closer to u than furthest
            # "closer to u than furthest" ↔ sim(u,o) > sim(u, furthest)
            if not dominated and sim_uo > furthest_sim:
                # Line 6: Move furthest from TG[u] into B[u]
                self.B[u].append(furthest_id)
                if len(self.B[u]) > self._M:
                    # Trim B[u] — remove furthest from u
                    b_sims = [(bid, self._sim(self.vectors[u], self.vectors[bid])) for bid in self.B[u]]
                    b_sims.sort(key=lambda x: -x[1])
                    self.B[u] = [bid for bid, _ in b_sims[:self._M]]

                # Line 7: Add o to TG[u]
                # Line 8: Append TG[u] to historic neighbor list of u
                new_ids = [nid for nid in lnow_u_ids if nid != furthest_id] + [o]
                self._hnt_append(u, new_ids, t)
            else:
                # Line 10: Add o to B[u]
                self.B[u].append(o)
                # Line 11-12: if |B[u]| > M: remove furthest from B[u]
                if len(self.B[u]) > self._M:
                    b_sims = [(bid, self._sim(self.vectors[u], self.vectors[bid])) for bid in self.B[u]]
                    b_sims.sort(key=lambda x: -x[1])
                    self.B[u] = [bid for bid, _ in b_sims[:self._M]]

        # Periodic entry point recomputation
        if self.N > 1 and self.N % 500 == 0:
            self._recompute_entry_point()

    # ── Tombstone delete (Algorithm 4) ──────────────────────────────

    def tombstone_delete(self, u: int, t_expire: int):
        """Mark node u as expired at time t_expire."""
        self.expire_days[u] = t_expire

        # Process reverse adjacency
        rev = list(self.reverse_adj[u])
        for v in rev:
            if v == u:
                continue  # skip self-loops (defensive)
            if u in self.B[v]:
                self.B[v].remove(u)
            else:
                lnow_v_ids = [nid for nid, _ in self.Lnow[v]]
                if u in lnow_v_ids:
                    # Remove u from Lnow[v]
                    remaining = [nid for nid in lnow_v_ids if nid != u]

                    if self.B[v]:
                        candidates = remaining + self.B[v]
                        # Filter out expired
                        candidates = [c for c in candidates if self.expire_days[c] is None]
                        selected = self._select_neighbors(self.vectors[v], candidates, 2 * self._M)
                        new_primary = selected[:self._M]
                        self.B[v] = selected[self._M:]
                    else:
                        cands, _ = self._greedy_search(self.vectors[v], [self.entry_point],
                                                       self._ef_construction, use_hnt=False)
                        cands = [c for c in cands if c != v and self.expire_days[c] is None]
                        selected = self._select_neighbors(self.vectors[v], cands, 2 * self._M)
                        new_primary = selected[:self._M]
                        self.B[v] = selected[self._M:]

                    # Guard: if all candidates expired, keep the non-expired
                    # remaining neighbors rather than disconnecting v
                    if not new_primary and remaining:
                        new_primary = [c for c in remaining if self.expire_days[c] is None][:self._M]

                    self._hnt_append(v, new_primary, t_expire)

        # Clear B[u] and reverse_adj[u]; keep HNT and Lnow for historical queries
        self.B[u] = []
        self.reverse_adj[u].clear()

        # Periodically compact cat_index after heavy deletes
        self._delete_count = getattr(self, '_delete_count', 0) + 1
        if self._delete_count % 1000 == 0:
            self._compact_cat_index()

    # ── Query ───────────────────────────────────────────────────────

    def query(self, q: np.ndarray, C: str, ts: int, k: int = 10, ef: int = 100) -> Tuple[np.ndarray, int]:
        """
        TANNS query: find k nearest neighbors in category C valid at timestamp ts.
        Returns (top_k_ids as np.ndarray, visited_count).
        """
        # Normalize query
        norm = np.linalg.norm(q)
        if norm > 0:
            q_norm = q / norm
        else:
            q_norm = q
        q_norm = q_norm.ravel()  # handle (1,D) input → (D,)

        # Build valid set
        valid_set = []
        for v in self.cat_index.get(C, []):
            if self.start_days[v] <= ts and (self.expire_days[v] is None or self.expire_days[v] > ts):
                valid_set.append(v)

        if len(valid_set) == 0:
            return np.array([], dtype=np.int64), 0

        if len(valid_set) <= k or (self.N > 0 and len(valid_set) / self.N < 0.01):
            # Brute force
            sims = [(v, float(np.dot(q_norm, self.vectors[v]))) for v in valid_set]
            sims.sort(key=lambda x: -x[1])
            top_k = [v for v, _ in sims[:k]]
            return np.array(top_k, dtype=np.int64), 0

        # Seeds: entry_point (if alive) + top-3 nearest from valid_set
        valid_vecs = np.array([self.vectors[v] for v in valid_set], dtype=np.float32)
        dots = valid_vecs @ q_norm
        top3_idx = np.argsort(-dots)[:3]
        seeds = []
        if self.expire_days[self.entry_point] is None:
            seeds.append(self.entry_point)
        seeds += [valid_set[i] for i in top3_idx]
        if not seeds:
            seeds = [valid_set[0]]
        seeds = list(dict.fromkeys(seeds))  # deduplicate preserving order

        results, visited = self._greedy_search(q_norm, seeds, ef, ts=ts, use_hnt=True)

        # Post-filter
        valid_set_s = set(valid_set)
        filtered = [nid for nid in results if C in self.cat_sets[nid] and nid in valid_set_s]

        # Sort by cosine similarity descending
        filtered_sims = [(nid, float(np.dot(q_norm, self.vectors[nid]))) for nid in filtered]
        filtered_sims.sort(key=lambda x: -x[1])
        top_k = [nid for nid, _ in filtered_sims[:k]]

        return np.array(top_k, dtype=np.int64), visited

    # ── Build ───────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, cat_sets_list: List[Set[str]], start_days_list: List[int]):
        """Build TANNS index from sorted data."""
        # Sort by start_day ascending
        order = sorted(range(len(start_days_list)), key=lambda i: start_days_list[i])

        total = len(order)
        for idx, i in enumerate(order):
            self.insert(vectors[i], cat_sets_list[i], start_days_list[i])
            if (idx + 1) % 10000 == 0:
                print(f"  Inserted {idx + 1}/{total} nodes")

        # Recompute entry point after full build
        self._recompute_entry_point()
        # No compact needed here — no deletions have occurred
        print(f"  Build complete: {self.N} nodes indexed")

    # ── Entry point recomputation ───────────────────────────────────

    def _compact_cat_index(self):
        """Remove expired nodes from cat_index to prevent unbounded growth."""
        for c in self.cat_index:
            self.cat_index[c] = [v for v in self.cat_index[c]
                                 if self.expire_days[v] is None]

    def _recompute_entry_point(self):
        """Recompute entry point: sample 500 live nodes, pick closest to centroid."""
        live = [i for i in range(self.N) if self.expire_days[i] is None]
        if not live:
            return

        sample_size = min(500, len(live))
        rng = np.random.default_rng()  # unseeded — avoids deterministic clustering bias
        sample = rng.choice(live, size=sample_size, replace=False)

        # Compute centroid of sample
        vecs = np.array([self.vectors[i] for i in sample], dtype=np.float32)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        # Find closest to centroid
        dots = vecs @ centroid
        best_idx = np.argmax(dots)
        self.entry_point = int(sample[best_idx])
