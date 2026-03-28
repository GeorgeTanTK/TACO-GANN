#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for TACO-GANN paper.

Methods compared (3 baselines only — no ACORN, no FDiskANN):
  - PostFilter-HNSW : HNSW + dual post-filter (category AND time)
  - TANNS+Post      : TANNS timestamp graph + category post-filter
  - TACO-GANN         : Full system (category-aware graph + HNT)

Style: matplotlib, 3.5-inch width (two-column paper),
PDF-compatible fonts (Type 42), 300 DPI.

Generates Figures 1, 2, 3, 5, 6 as both PNG and PDF.
Figure 4 (ablation) removed — no ablation variants in this study.

Usage:
    python benchmarks/generate_figures.py
    python benchmarks/generate_figures.py --results-dir results/ --output-dir figures/
"""

import argparse
import csv
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)
# reduce noise from fontTools during PDF saving
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

# ─── Global style ────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "legend.fontsize":    8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "axes.axisbelow":     True,
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.5,
    "lines.markersize":   6,
})

# ─── Color palette (3 methods, colorblind-safe) ───────────────────────

COLORS = {
    "PostFilter": "#CC4444",  # red
    "TANNS+Post": "#EE7733",  # orange
    "TACO-GANN":    "#222222",  # black (ours)
}

MARKERS = {
    "PostFilter": "v",
    "TANNS+Post": "^",
    "TACO-GANN":    "*",
}

DISPLAY_NAMES = {
    "PostFilter": "Post-filter (HNSW)",
    "TANNS+Post": "TANNS+Post",
    "TACO-GANN":    "TACO-GANN (ours)",
}

# CSV method name → internal key
METHOD_MAP = {
    "PostFilter-HNSW":  "PostFilter",
    "TANNS+PostFilter": "TANNS+Post",
    "TANNS+Post":       "TANNS+Post",
    "TACO-GANN":          "TACO-GANN",
}

# Plot order: baselines first, TACO-GANN last (on top)
PLOT_ORDER = ["PostFilter", "TANNS+Post", "TACO-GANN"]


# ─── Helpers ─────────────────────────────────────────────────────────

def load_csv(results_dir):
    """Load baseline_results.csv → {internal_key: [row_dicts]}."""
    data = {}
    csv_path = os.path.join(results_dir, "baseline_results.csv")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["method"]
            key = METHOD_MAP.get(raw, raw)
            if key not in PLOT_ORDER:
                continue  # skip any stale entries
            if key not in data:
                data[key] = []
            data[key].append({
                "ef":      row.get("expansion_factor", row.get("ef_search", "0")),
                "r10":     float(row["recall@10"]),
                "r100":    float(row.get("recall@100", 0.0)),
                "qps":     float(row["QPS"]),
                "latency": float(row["latency_ms"]),
            })
    return data


def save_fig(fig, name, outdir):
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  Saved {name}.png + .pdf")


def _plot_series(ax, data, y_key):
    """Plot all methods on ax using data[key][y_key] vs QPS."""
    for key in PLOT_ORDER:
        points = data.get(key, [])
        if not points:
            continue
        qps_vals = [p["qps"]    for p in points]
        y_vals   = [p[y_key]    for p in points]
        lw      = 2.5 if key == "TACO-GANN" else 1.4
        ms      = 9   if key == "TACO-GANN" else 5
        zorder  = 10  if key == "TACO-GANN" else 3
        ax.plot(
            qps_vals, y_vals,
            marker=MARKERS[key], color=COLORS[key],
            label=DISPLAY_NAMES[key],
            linewidth=lw, markersize=ms, zorder=zorder,
            markeredgewidth=0.5, markeredgecolor="#333",
        )


# # ═════════════════════════════════════════════════════════════════════
# # Figure 1 — System Architecture Diagram
# # ═════════════════════════════════════════════════════════════════════

# def fig1_architecture(outdir):
#     logger.info("Figure 1: Architecture diagram...")
#     fig, ax = plt.subplots(figsize=(3.5, 3.2))
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 7.5)
#     ax.axis("off")

#     box_kw = dict(boxstyle="round,pad=0.3", linewidth=1.2)

#     # Input
#     ax.add_patch(FancyBboxPatch((0.3, 6.2), 9.2, 0.9, **box_kw,
#                                 facecolor="#E8E8E8", edgecolor="#333"))
#     ax.text(5.0, 6.65, "Query: (q, C, [t_start, t_end], k)",
#             ha="center", va="center", fontsize=9, fontweight="bold")

#     # Two pillars: Category-Aware Graph | Temporal HNT
#     pillar_info = [
#         ("#CCE5FF", "Category-Aware Graph\n(Filtered-Vamana)",
#          "Per-cat medoid entry points\nST-connectivity per label\nAlpha-blended scoring"),
#         ("#D4EDDA", "Temporal HNT\n(per-node history)",
#          "Flat sorted HNTEntry list\nbisect_right reconstruction\nTombstone-safe history"),
#     ]
#     x_positions = [0.5, 5.2]
#     widths = [4.3, 4.3]
#     for (x, w), (color, title, desc) in zip(
#         zip(x_positions, widths), pillar_info
#     ):
#         ax.add_patch(FancyBboxPatch((x, 2.4), w, 3.4, **box_kw,
#                                     facecolor=color, edgecolor="#555"))
#         ax.text(x + w/2, 5.1, title, ha="center", va="center",
#                 fontsize=7.5, fontweight="bold", linespacing=1.3)
#         ax.text(x + w/2, 3.6, desc, ha="center", va="center",
#                 fontsize=6.2, color="#444", linespacing=1.3)
#         # Arrow from input
#         ax.annotate("", xy=(x + w/2, 5.8), xytext=(x + w/2, 6.2),
#                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

#     # Merge label
#     ax.text(5.0, 1.95, "Filtered Beam Search",
#             ha="center", va="center", fontsize=7.5, fontweight="bold",
#             fontstyle="italic", color="#333",
#             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
#                       edgecolor="#999", linewidth=0.5))

#     # Arrows from pillars to merge
#     for x, w in zip(x_positions, widths):
#         ax.annotate("", xy=(5.0, 2.1), xytext=(x + w/2, 2.4),
#                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))

#     # Output
#     ax.add_patch(FancyBboxPatch((2.0, 0.3), 5.8, 0.9, **box_kw,
#                                 facecolor="#E8E8E8", edgecolor="#333"))
#     ax.text(5.0, 0.75, "Output: Top-k filtered nearest neighbors",
#             ha="center", va="center", fontsize=9, fontweight="bold")
#     ax.annotate("", xy=(5.0, 1.3), xytext=(5.0, 1.9),
#                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

#     save_fig(fig, "fig1_architecture", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 2 — Recall@10 vs QPS
# ═════════════════════════════════════════════════════════════════════

def fig2_recall10_vs_qps(results_dir, outdir):
    logger.info("Figure 2: Recall@10 vs QPS...")
    data = load_csv(results_dir)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _plot_series(ax, data, "r10")
    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=1,
              handletextpad=0.4)
    save_fig(fig, "fig2_recall10_vs_qps", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 3 — Recall@100 vs QPS
# ═════════════════════════════════════════════════════════════════════

def fig3_recall100_vs_qps(results_dir, outdir):
    logger.info("Figure 3: Recall@100 vs QPS...")
    data = load_csv(results_dir)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _plot_series(ax, data, "r100")
    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec)")
    ax.set_ylabel("Recall@100")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=1,
              handletextpad=0.4)
    save_fig(fig, "fig3_recall100_vs_qps", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 4 — Recall@10 vs Visited Nodes (efficiency curve)
# Replaces the ablation bar chart.
# Reads per-query data from _baselines_simple.json produced by
# run_baselines.py. X-axis = mean visited_count, Y = Recall@10.
# ═════════════════════════════════════════════════════════════════════

def fig4_visited_vs_recall(results_dir, outdir):
    logger.info("Figure 4: Recall@10 vs Visited Nodes...")

    path = os.path.join(results_dir, "_baselines_simple.json")
    if not os.path.exists(path):
        logger.warning(f"  {path} not found — skipping Figure 4")
        return

    with open(path) as f:
        baseline_data = json.load(f)

    summary = baseline_data.get("summary", {})
    per_query = baseline_data.get("per_query", {})

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for key in PLOT_ORDER:
        if key not in summary:
            continue
        pq = per_query.get(key, [])
        if not pq:
            continue

        # Bucket by selectivity to get visited vs recall curves
        # Group into coarse bins and compute mean visited / mean recall
        valid = [(p["visited"], p["recall@10"]) for p in pq
                 if p["visited"] is not None and p["recall@10"] is not None]
        if not valid:
            continue

        visited_arr = np.array([v for v, _ in valid])
        recall_arr  = np.array([r for _, r in valid])

        # Sort by visited count (ascending) for a clean curve
        order = np.argsort(visited_arr)
        visited_sorted = visited_arr[order]
        recall_sorted  = recall_arr[order]

        lw     = 2.5 if key == "TACO-GANN" else 1.4
        ms     = 9   if key == "TACO-GANN" else 5
        zorder = 10  if key == "TACO-GANN" else 3

        ax.scatter(visited_sorted, recall_sorted,
                   color=COLORS[key], label=DISPLAY_NAMES[key],
                   s=ms**2, alpha=0.5, zorder=zorder)

    ax.set_xlabel("Visited nodes (beam search)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, handletextpad=0.4)
    save_fig(fig, "fig4_visited_vs_recall", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 5 — Selectivity Analysis
# ═════════════════════════════════════════════════════════════════════

def fig5_selectivity(results_dir, outdir):
    logger.info("Figure 5: Selectivity analysis...")

    sel_path = os.path.join(results_dir, "_selectivity_recall.json")
    if not os.path.exists(sel_path):
        logger.warning(f"  {sel_path} not found — skipping Figure 5")
        return

    with open(sel_path) as f:
        raw = json.load(f)

    # Support both old format (top-level method keys) and new
    # format ({bins, methods, results: {method: {bin: {mean_recall, count}}}})
    if "results" in raw:
        sel_data = raw["results"]
    else:
        sel_data = raw

    bin_labels  = ["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%"]
    bin_centers = [0.25, 0.75, 1.5, 2.5, 4.0, 7.5, 15.0]

    # Map stored method names to internal keys
    sel_keyed = {}
    for raw_key, bins in sel_data.items():
        internal = METHOD_MAP.get(raw_key, raw_key)
        if internal in PLOT_ORDER:
            sel_keyed[internal] = bins

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for key in PLOT_ORDER:
        if key not in sel_keyed:
            continue
        xs, ys = [], []
        for bl, bc in zip(bin_labels, bin_centers):
            entry = sel_keyed[key].get(bl, {})
            if entry and entry.get("mean_recall") is not None:
                xs.append(bc)
                ys.append(entry["mean_recall"])
        if not xs:
            continue

        lw     = 2.5 if key == "TACO-GANN" else 1.4
        ms     = 9   if key == "TACO-GANN" else 5
        zorder = 10  if key == "TACO-GANN" else 3

        ax.plot(xs, ys,
                marker=MARKERS[key], color=COLORS[key],
                label=DISPLAY_NAMES[key],
                linewidth=lw, markersize=ms, zorder=zorder,
                markeredgewidth=0.5, markeredgecolor="#333")

    ax.set_xlabel("Filter selectivity (% of dataset)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.4, 1.05)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.25%", "0.5%", "1%", "2%", "5%", "10%", "20%"],
                       fontsize=7.5)

    # Shade low-selectivity region (< 1%) — where TACO-GANN's advantage is largest
    ax.axvspan(0.1, 1.0, alpha=0.07, color="#FF0000")
    ax.text(0.42, 0.44, "low\nselectivity", fontsize=6, color="#CC0000",
            ha="center", va="bottom", fontstyle="italic")

    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    save_fig(fig, "fig5_selectivity", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 6 — Construction Time and Memory
# ═════════════════════════════════════════════════════════════════════

def fig6_construction(results_dir, outdir):
    logger.info("Figure 6: Construction cost...")

    costs_path = os.path.join(results_dir, "_construction_costs.json")
    if not os.path.exists(costs_path):
        logger.warning(f"  {costs_path} not found — skipping Figure 6")
        return

    with open(costs_path) as f:
        costs = json.load(f)

    # Only include methods that are present in costs AND in PLOT_ORDER
    # Display labels for bars (shorter than DISPLAY_NAMES)
    BAR_DISPLAY = {
        "PostFilter": "PostFilter\n(HNSW)",
        "TANNS+Post": "TANNS\n+Post",
        "TACO-GANN":    "TACO-GANN\n(ours)",
    }

    methods_present = [m for m in PLOT_ORDER if m in costs]
    if not methods_present:
        logger.warning("  No matching methods in _construction_costs.json — skipping Figure 6")
        return

    display      = [BAR_DISPLAY[m] for m in methods_present]
    build_times  = [costs[m]["build_time_s"] for m in methods_present]
    memories     = [costs[m]["peak_mem_mb"]  for m in methods_present]
    bar_colors   = [COLORS[m] for m in methods_present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5))
    x = np.arange(len(methods_present))
    w = 0.5

    # Build time
    bars1 = ax1.bar(x, build_times, color=bar_colors,
                    edgecolor="#444", linewidth=0.5, width=w)
    ax1.set_ylabel("Build time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display, fontsize=7, linespacing=1.0)
    for bar, val in zip(bars1, build_times):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(build_times)*0.01,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=5.5)

    # Peak memory
    bars2 = ax2.bar(x, memories, color=bar_colors,
                    edgecolor="#444", linewidth=0.5, width=w)
    ax2.set_ylabel("Peak memory (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display, fontsize=7, linespacing=1.0)
    for bar, val in zip(bars2, memories):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(memories)*0.01,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "fig6_construction", outdir)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate publication figures for TACO-GANN"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(REPO_ROOT, "results"),
        help="Directory containing benchmark result files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # fig1_architecture(args.output_dir)
    fig2_recall10_vs_qps(args.results_dir, args.output_dir)
    fig3_recall100_vs_qps(args.results_dir, args.output_dir)
    fig4_visited_vs_recall(args.results_dir, args.output_dir)
    fig5_selectivity(args.results_dir, args.output_dir)
    fig6_construction(args.results_dir, args.output_dir)

    logger.info(f"\nAll figures saved to {args.output_dir}/")
