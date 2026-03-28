#!/usr/bin/env python3
"""
recall_qps_sweep.py — Recall@10 vs QPS sweep for TACO-GANN baselines.

Mirrors the ACORN paper benchmark style (Figs 7, 9, 10, 11):
  - For each method, sweeps a range of ef values.
  - At each ef, runs all queries and records (mean Recall@10, QPS).
  - Each (recall, QPS) pair becomes one point on the trade-off curve.
  - Results are segmented by selectivity bin so you get one subplot per bin.

Output files (written to --results-dir):
  _recall_qps_curve.json   — full curve data per method per bin per ef
  figures/recall_qps_*.png — one PNG per selectivity bin (+ one aggregate)

Prerequisites:
  results/_state.pkl must contain: gt, Ms, N, NQ, (optional) qs

Usage:
  python benchmarks/recall_qps_sweep.py
  python benchmarks/recall_qps_sweep.py \\
      --data-dir data --results-dir results \\
      --ef-sweep 10 20 40 80 150 300 500 \\
      --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hnswlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_fvecs, load_metadata, generate_queries
from src.baselines.tanns_post_filtering import TANNS
from src.taco_gann import TACOGANN

logger = logging.getLogger(__name__)

# ── Selectivity bins (same as compute_selectivity.py) ────────────────────────
BIN_EDGES  = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
BIN_LABELS = ["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%"]

# ── Plot style ────────────────────────────────────────────────────────────────
METHOD_STYLE: dict[str, dict] = {
    "PostFilter":  {"color": "#e69f00", "ls": "-",  "lw": 2, "label": "HNSW Postfilter"},
    "TANNS+Post":  {"color": "#56b4e9", "ls": "--", "lw": 2, "label": "TANNS+Post"},
    "TACO-GANN":   {"color": "#009e73", "ls": "-",  "lw": 2, "label": "TACO-GANN (ours)"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def recall_at_k_single(retrieved: list[int], gt_arr, k: int) -> float | None:
    """Recall@k for a single query. Returns None when GT is empty."""
    if len(gt_arr) == 0:
        return None
    gt_topk = set(int(x) for x in gt_arr[:k])
    ret_topk = set(int(x) for x in retrieved[:k])
    denom = min(k, len(gt_topk))
    return len(ret_topk & gt_topk) / denom if denom > 0 else 1.0


def run_queries_at_ef(
    hnsw: hnswlib.Index,
    tanns: TANNS,
    tacogann: TACOGANN,
    queries: list[dict],
    gt: list,
    Ms: list,
    N: int,
    NQ: int,
    ef: int,
    k: int,
) -> dict[str, dict]:
    """
    Run all queries for each method at a single ef value.

    Returns per-method dict with keys:
        recalls    : list[float | None]  — per-query recall@k
        latencies  : list[float]         — per-query wall-clock seconds
    """
    results: dict[str, dict] = {
        m: {"recalls": [None] * NQ, "latencies": [0.0] * NQ}
        for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]
    }

    # HNSW PostFilter: retrieve ef*10 candidates then dual post-filter
    k_search = min(ef * 10, N)
    hnsw.set_ef(max(k_search, ef))

    for qi in range(NQ):
        if len(gt[qi]) == 0:
            continue
        q   = queries[qi]
        qv  = q["query_vector"]
        cat = q["target_category"]
        ts, te = q["t_start"], q["t_end"]
        mask = Ms[qi]

        # ── PostFilter-HNSW ───────────────────────────────────────────────
        t0 = time.perf_counter()
        labels, _ = hnsw.knn_query(qv.reshape(1, -1), k=k_search)
        lat = time.perf_counter() - t0
        filtered = [int(x) for x in labels[0] if mask[int(x)]][:k]
        results["PostFilter"]["recalls"][qi]   = recall_at_k_single(filtered, gt[qi], k)
        results["PostFilter"]["latencies"][qi] = lat

        # ── TANNS + post-filter ───────────────────────────────────────────
        t0 = time.perf_counter()
        ids_t, _ = tanns.query(qv, cat, te, k=k, ef=ef)
        lat = time.perf_counter() - t0
        ids_t_f = [int(x) for x in ids_t if mask[int(x)]][:k]
        results["TANNS+Post"]["recalls"][qi]   = recall_at_k_single(ids_t_f, gt[qi], k)
        results["TANNS+Post"]["latencies"][qi] = lat

        # ── TACO-GANN ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        ids_c, _ = tacogann.query(qv, cat, ts, te, k=k, ef=ef)
        lat = time.perf_counter() - t0
        ids_c_f = [int(x) for x in ids_c if mask[int(x)]][:k]
        results["TACO-GANN"]["recalls"][qi]   = recall_at_k_single(ids_c_f, gt[qi], k)
        results["TACO-GANN"]["latencies"][qi] = lat

    return results


def aggregate_curve_point(
    results: dict[str, dict],
    query_indices: np.ndarray,
    k: int,
) -> dict[str, dict[str, float]]:
    """
    Aggregate mean recall and QPS for a given subset of query indices.

    Returns: {method: {"recall": float, "qps": float}}
    """
    out: dict[str, dict] = {}
    for m, data in results.items():
        recalls = [
            data["recalls"][qi]
            for qi in query_indices
            if data["recalls"][qi] is not None
        ]
        lats = [
            data["latencies"][qi]
            for qi in query_indices
            if data["recalls"][qi] is not None  # only valid queries
        ]
        if not recalls:
            out[m] = {"recall": 0.0, "qps": 0.0}
        else:
            mean_recall = float(np.mean(recalls))
            total_lat   = float(np.sum(lats))
            qps = len(recalls) / total_lat if total_lat > 0 else 0.0
            out[m] = {"recall": round(mean_recall, 5), "qps": round(qps, 2)}
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_recall_qps(
    curve_data: dict,   # bin_label -> method -> list of {ef, recall, qps}
    figures_dir: str,
    k: int,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # ── Per-bin subplots ──────────────────────────────────────────────────
    for bin_label, method_curves in curve_data.items():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for m, points in method_curves.items():
            xs = [p["qps"]    for p in points if p["qps"] > 0]
            ys = [p["recall"] for p in points if p["qps"] > 0]
            if not xs:
                continue
            style = METHOD_STYLE.get(m, {})
            ax.semilogx(
                xs, ys,
                color=style.get("color", "gray"),
                ls=style.get("ls", "-"),
                lw=style.get("lw", 1.5),
                label=style.get("label", m),
                marker="o", markersize=4,
            )
        ax.set_xlim(left=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("QPS (log scale)", fontsize=9)
        ax.set_ylabel(f"Recall@{k}", fontsize=9)
        ax.set_title(f"Selectivity {bin_label}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.3)
        safe_label = bin_label.replace("<", "lt").replace("%", "pct").replace("-", "_")
        fname = os.path.join(figures_dir, f"recall_qps_{safe_label}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", fname)

    # ── All-bins grid figure (mirrors ACORN Fig. 9 layout) ────────────────
    bins_with_data = [b for b in BIN_LABELS if b in curve_data]
    n = len(bins_with_data)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, bin_label in enumerate(bins_with_data):
        ax = axes_flat[idx]
        method_curves = curve_data[bin_label]
        for m, points in method_curves.items():
            xs = [p["qps"]    for p in points if p["qps"] > 0]
            ys = [p["recall"] for p in points if p["qps"] > 0]
            if not xs:
                continue
            style = METHOD_STYLE.get(m, {})
            ax.semilogx(
                xs, ys,
                color=style.get("color", "gray"),
                ls=style.get("ls", "-"),
                lw=style.get("lw", 1.5),
                label=style.get("label", m),
                marker="o", markersize=4,
            )
        ax.set_xlim(left=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("QPS (log scale)", fontsize=8)
        ax.set_ylabel(f"Recall@{k}", fontsize=8)
        ax.set_title(f"Sel. {bin_label}", fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused subplot panels
    for idx in range(len(bins_with_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Recall@{k} vs QPS — ArXiv (TACO-GANN)", fontsize=11, y=1.01)
    fig.tight_layout()
    grid_path = os.path.join(figures_dir, "recall_qps_all_bins.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grid figure: %s", grid_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Sweep ef values to produce Recall@k vs QPS trade-off curves"
    )
    parser.add_argument("--data-dir",    default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument(
        "--ef-sweep",
        type=int, nargs="+",
        default=[10, 20, 40, 80, 150, 300, 500],
        help="ef values to sweep (ascending for clean curve)",
    )
    parser.add_argument("--k", type=int, default=10, help="k for Recall@k (default 10)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    figures_dir = os.path.join(args.results_dir, "..", "figures")

    # ── Load state ─────────────────────────────────────────────────────────
    state_path = os.path.join(args.results_dir, "_state.pkl")
    logger.info("Loading state from %s", state_path)
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    gt = state["gt"]
    Ms = state["Ms"]
    N  = state["N"]
    NQ = state["NQ"]

    # ── Load vectors and metadata ──────────────────────────────────────────
    for vec_name in ["database_vectors_small.fvecs", "database_vectors.fvecs"]:
        vec_path = os.path.join(args.data_dir, vec_name)
        if os.path.exists(vec_path):
            break
    for attr_name in ["database_attributes_small.jsonl", "database_attributes.jsonl"]:
        attr_path = os.path.join(args.data_dir, attr_name)
        if os.path.exists(attr_path):
            break

    logger.info("Loading vectors from %s", vec_path)
    V = load_fvecs(vec_path)
    cats, udays = load_metadata(attr_path)

    queries = state.get("qs") or generate_queries(V, cats, udays, n_queries=NQ, seed=42)
    assert V.shape[0] == N
    udays_list = [int(d) for d in udays]

    # ── Selectivity bins for each query ───────────────────────────────────
    selectivity_pct = np.array([int(m.sum()) for m in Ms]) / N * 100.0
    query_bins = np.digitize(selectivity_pct, BIN_EDGES) - 1  # 0-indexed

    # ── Build indices (once) ───────────────────────────────────────────────
    logger.info("Building HNSW PostFilter index...")
    hnsw = hnswlib.Index(space="cosine", dim=V.shape[1])
    hnsw.init_index(max_elements=N, M=32, ef_construction=200)
    hnsw.add_items(V, ids=np.arange(N))

    logger.info("Building TANNS+Post index...")
    tanns = TANNS()
    tanns.build(V, cats, udays_list)

    logger.info("Building TACO-GANN index...")
    tacogann = TACOGANN()
    tacogann.build(V, cats, udays_list)

    # ── ef sweep ──────────────────────────────────────────────────────────
    # curve_data[bin_label][method] = list of {ef, recall, qps}
    curve_data: dict[str, dict[str, list]] = {
        bl: {m: [] for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]}
        for bl in BIN_LABELS
    }
    # also aggregate across all bins
    curve_data["ALL"] = {m: [] for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]}

    ef_list = sorted(set(args.ef_sweep))
    logger.info("Starting ef sweep: %s", ef_list)

    for ef in ef_list:
        logger.info("── ef = %d ──", ef)
        results = run_queries_at_ef(
            hnsw, tanns, tacogann,
            queries, gt, Ms, N, NQ, ef, args.k,
        )

        # Aggregate over ALL queries
        all_indices = np.where([len(gt[qi]) > 0 for qi in range(NQ)])[0]
        agg_all = aggregate_curve_point(results, all_indices, args.k)
        for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]:
            curve_data["ALL"][m].append({"ef": ef, **agg_all[m]})

        # Aggregate per selectivity bin
        for bin_idx, bin_label in enumerate(BIN_LABELS):
            q_idx = np.where(
                (query_bins == bin_idx) &
                np.array([len(gt[qi]) > 0 for qi in range(NQ)])
            )[0]
            if len(q_idx) == 0:
                for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]:
                    curve_data[bin_label][m].append({"ef": ef, "recall": 0.0, "qps": 0.0})
                continue
            agg = aggregate_curve_point(results, q_idx, args.k)
            for m in ["PostFilter", "TANNS+Post", "TACO-GANN"]:
                curve_data[bin_label][m].append({"ef": ef, **agg[m]})

        logger.info(
            "  ef=%d | TACO-GANN recall=%.3f qps=%.1f | PostFilter recall=%.3f qps=%.1f",
            ef,
            agg_all["TACO-GANN"]["recall"], agg_all["TACO-GANN"]["qps"],
            agg_all["PostFilter"]["recall"], agg_all["PostFilter"]["qps"],
        )

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "_recall_qps_curve.json")
    with open(out_path, "w") as f:
        json.dump({"k": args.k, "ef_values": ef_list, "curves": curve_data}, f, indent=2)
    logger.info("Saved curve data to %s", out_path)

    # ── Print summary table ────────────────────────────────────────────────
    methods = ["PostFilter", "TANNS+Post", "TACO-GANN"]
    header = f"{'ef':>6}  " + "  ".join(f"{m+' R@'+str(args.k):>18} {'QPS':>8}" for m in methods)
    logger.info("\n%s", header)
    for i, ef in enumerate(ef_list):
        row = f"{ef:>6}  "
        for m in methods:
            pt = curve_data["ALL"][m][i]
            row += f"  {pt['recall']:>18.4f} {pt['qps']:>8.1f}"
        logger.info(row)

    # ── Generate figures ───────────────────────────────────────────────────
    logger.info("Generating figures...")
    plot_recall_qps(
        {bl: curve_data[bl] for bl in BIN_LABELS if bl in curve_data},
        figures_dir=figures_dir,
        k=args.k,
    )
    logger.info("Done. Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
