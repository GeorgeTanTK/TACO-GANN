#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for TANNS-C paper.

Style: matplotlib, font size 12, 3.5-inch width (two-column paper),
PDF-compatible fonts (Type 1 via ps.fonttype=42), 300 DPI.

Generates Figures 1–6 as both PNG and PDF.

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# ─── Global style ────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,      # TrueType (Type 42) — PDF compatible
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})

# ─── Color palette (publication-friendly, colorblind-safe) ───────────

COLORS = {
    "PostFilter":      "#CC4444",  # red
    "PreFilter":       "#228833",  # green
    "ACORN-1":         "#4477AA",  # blue
    "TANNS+Post":      "#EE7733",  # orange
    "TANNS+Pre":       "#AA3377",  # magenta
    "FDiskANN+Post":   "#66CCEE",  # cyan
    "FDiskANN+Pre":    "#BBBBBB",  # gray
    "TANNS-C":         "#222222",  # black (ours — prominent)
}

MARKERS = {
    "PostFilter":      "v",   # triangle down
    "PreFilter":       "D",   # diamond
    "ACORN-1":         "s",   # square
    "TANNS+Post":      "^",   # triangle up
    "TANNS+Pre":       "p",   # pentagon
    "FDiskANN+Post":   "<",   # triangle left
    "FDiskANN+Pre":    ">",   # triangle right
    "TANNS-C":         "*",   # star
}

# Map CSV method names to short keys
METHOD_MAP = {
    "PostFilter-HNSW":      "PostFilter",
    "PreFilter-BruteForce": "PreFilter",
    "ACORN-1":              "ACORN-1",
    "TANNS+PostFilter":     "TANNS+Post",
    "TANNS+PreFilter":      "TANNS+Pre",
    "FDiskANN+PostFilter":  "FDiskANN+Post",
    "FDiskANN+PreFilter":   "FDiskANN+Pre",
    "TANNS-C":              "TANNS-C",
}

# Display names for legend
DISPLAY_NAMES = {
    "PostFilter":      "Post-filter",
    "PreFilter":       "Pre-filter",
    "ACORN-1":         "ACORN-1",
    "TANNS+Post":      "TANNS+Post",
    "TANNS+Pre":       "TANNS+Pre",
    "FDiskANN+Post":   "FDiskANN+Post",
    "FDiskANN+Pre":    "FDiskANN+Pre",
    "TANNS-C":         "TANNS-C (ours)",
}


# ─── Load data ───────────────────────────────────────────────────────

def load_csv(results_dir):
    """Load baseline_results.csv from the results directory."""
    data = {}
    csv_path = os.path.join(results_dir, "baseline_results.csv")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in data:
                data[method] = []
            data[method].append({
                "ef": row["expansion_factor"],
                "r10": float(row["recall@10"]),
                "r100": float(row["recall@100"]),
                "qps": float(row["QPS"]),
                "latency": float(row["latency_ms"]),
            })
    return data


def save_fig(fig, name, outdir):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  Saved {name}.png + .pdf")


# ═════════════════════════════════════════════════════════════════════
# Figure 1 — System Architecture Diagram
# ═════════════════════════════════════════════════════════════════════

def fig1_architecture(outdir):
    """Generate Figure 1: Architecture diagram."""
    logger.info("Figure 1: Architecture diagram...")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.3", linewidth=1.2)

    # Input box
    inp = FancyBboxPatch((0.3, 6.0), 9.2, 1.0, **box_kw,
                         facecolor="#E8E8E8", edgecolor="#333")
    ax.add_patch(inp)
    ax.text(5.0, 6.5, "Query: (q, C, [t_start, t_end], k)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Three pillars
    pillar_colors = ["#CCE5FF", "#D4EDDA", "#FFF3CD"]
    pillar_labels = [
        "Pillar 1\nCategory-Aware\nGraph",
        "Pillar 2\nTemporal\nSnapshots",
        "Pillar 3\nACORN-Style\nFallback",
    ]
    pillar_descs = [
        "Per-cat medoid\nentry points,\ncategory-aware\nneighbor selection",
        "Yearly snapshot\nadjacency lists,\ntemporal pruning",
        "Two-hop expansion\nfor thin pools,\npre-filter fallback\nfor low selectivity",
    ]

    x_positions = [0.4, 3.5, 6.6]
    for i, (x, color, label, desc) in enumerate(
        zip(x_positions, pillar_colors, pillar_labels, pillar_descs)
    ):
        box = FancyBboxPatch((x, 2.2), 2.7, 3.3, **box_kw,
                             facecolor=color, edgecolor="#555")
        ax.add_patch(box)
        ax.text(x + 1.35, 4.8, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold")
        ax.text(x + 1.35, 3.3, desc, ha="center", va="center",
                fontsize=5.8, color="#444", linespacing=1.2)

        # Arrow from input to pillar
        ax.annotate("", xy=(x + 1.35, 5.5), xytext=(x + 1.35, 6.0),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    # Output box
    out = FancyBboxPatch((2.0, 0.3), 5.8, 1.0, **box_kw,
                         facecolor="#E8E8E8", edgecolor="#333")
    ax.add_patch(out)
    ax.text(5.0, 0.8, "Output: Top-k filtered nearest neighbors",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows from pillars to output
    for x in x_positions:
        ax.annotate("", xy=(5.0, 1.3), xytext=(x + 1.35, 2.2),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))

    # Merge label
    ax.text(5.0, 1.8, "Filtered Beam Search", ha="center", va="center",
            fontsize=7, fontweight="bold", fontstyle="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#999", linewidth=0.5))

    save_fig(fig, "fig1_architecture", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 2 — Recall@10 vs QPS
# ═════════════════════════════════════════════════════════════════════

def fig2_recall10_vs_qps(results_dir, outdir):
    """Generate Figure 2: Recall@10 vs QPS."""
    logger.info("Figure 2: Recall@10 vs QPS...")
    csv_data = load_csv(results_dir)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Plot order: baselines first, TANNS-C last (on top)
    plot_order = ["PostFilter", "ACORN-1", "TANNS+Post", "TANNS+Pre",
                  "FDiskANN+Post", "FDiskANN+Pre", "PreFilter", "TANNS-C"]

    for key in plot_order:
        csv_name = [k for k, v in METHOD_MAP.items() if v == key]
        if not csv_name:
            continue
        points = csv_data.get(csv_name[0], [])
        if not points:
            continue

        qps_vals = [p["qps"] for p in points]
        r10_vals = [p["r10"] for p in points]

        lw = 2.5 if key == "TANNS-C" else 1.2
        ms = 8 if key == "TANNS-C" else 5
        zorder = 10 if key == "TANNS-C" else 3

        ax.plot(qps_vals, r10_vals,
                marker=MARKERS[key], color=COLORS[key],
                label=DISPLAY_NAMES[key], linewidth=lw, markersize=ms,
                zorder=zorder, markeredgewidth=0.5, markeredgecolor="#333")

    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2,
              columnspacing=0.5, handletextpad=0.3)

    save_fig(fig, "fig2_recall10_vs_qps", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 3 — Recall@100 vs QPS
# ═════════════════════════════════════════════════════════════════════

def fig3_recall100_vs_qps(results_dir, outdir):
    """Generate Figure 3: Recall@100 vs QPS."""
    logger.info("Figure 3: Recall@100 vs QPS...")
    csv_data = load_csv(results_dir)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    plot_order = ["PostFilter", "ACORN-1", "TANNS+Post", "TANNS+Pre",
                  "FDiskANN+Post", "FDiskANN+Pre", "PreFilter", "TANNS-C"]

    for key in plot_order:
        csv_name = [k for k, v in METHOD_MAP.items() if v == key]
        if not csv_name:
            continue
        points = csv_data.get(csv_name[0], [])
        if not points:
            continue

        qps_vals = [p["qps"] for p in points]
        r100_vals = [p["r100"] for p in points]

        lw = 2.5 if key == "TANNS-C" else 1.2
        ms = 8 if key == "TANNS-C" else 5
        zorder = 10 if key == "TANNS-C" else 3

        ax.plot(qps_vals, r100_vals,
                marker=MARKERS[key], color=COLORS[key],
                label=DISPLAY_NAMES[key], linewidth=lw, markersize=ms,
                zorder=zorder, markeredgewidth=0.5, markeredgecolor="#333")

    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec)")
    ax.set_ylabel("Recall@100")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2,
              columnspacing=0.5, handletextpad=0.3)

    save_fig(fig, "fig3_recall100_vs_qps", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 4 — Ablation Study (bar chart)
# ═════════════════════════════════════════════════════════════════════

def fig4_ablation(results_dir, outdir):
    """Generate Figure 4: Ablation bar chart."""
    logger.info("Figure 4: Ablation bar chart...")
    csv_data = load_csv(results_dir)

    # Extract ablation data at ef=200
    ablation = {}
    for method_name, points in csv_data.items():
        if "TANNS-C" in method_name:
            for p in points:
                if p["ef"] == "200":
                    ablation[method_name] = p

    # Labels and values
    labels = ["P1 only", "P2 only", "P1+P2", "Full"]
    keys = ["TANNS-C (P1 only)", "TANNS-C (P2 only)", "TANNS-C (P1+P2)", "TANNS-C"]
    r10_vals = [ablation[k]["r10"] for k in keys]
    qps_vals = [ablation[k]["qps"] for k in keys]

    bar_colors = ["#4477AA", "#228833", "#AA3377", "#222222"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5))

    # Recall@10
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, r10_vals, color=bar_colors, edgecolor="#444", linewidth=0.5, width=0.6)
    ax1.set_ylabel("Recall@10")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax1.set_ylim(0.985, 1.002)
    ax1.axhline(y=r10_vals[-1], color="#222", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add value labels (offset overlapping ones)
    for i, (bar, val) in enumerate(zip(bars1, r10_vals)):
        ha = "center"
        x_off = 0
        # If two adjacent bars are very close in height, shift labels
        if i == 1:  # P2 only — shift left
            ha = "right"
            x_off = -0.02
        elif i == 2:  # P1+P2 — shift right
            ha = "left"
            x_off = 0.02
        ax1.text(bar.get_x() + bar.get_width()/2 + x_off,
                 bar.get_height() + 0.0005,
                 f"{val:.4f}", ha=ha, va="bottom", fontsize=5.5)

    # QPS
    bars2 = ax2.bar(x, qps_vals, color=bar_colors, edgecolor="#444", linewidth=0.5, width=0.6)
    ax2.set_ylabel("QPS")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")

    for bar, val in zip(bars2, qps_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "fig4_ablation", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 5 — Selectivity Analysis
# ═════════════════════════════════════════════════════════════════════

def fig5_selectivity(results_dir, outdir):
    """Generate Figure 5: Selectivity analysis."""
    logger.info("Figure 5: Selectivity analysis...")

    sel_path = os.path.join(results_dir, "_selectivity_recall.json")
    with open(sel_path) as f:
        sel_data = json.load(f)

    bin_labels = ["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%"]
    bin_centers = [0.25, 0.75, 1.5, 2.5, 4.0, 7.5, 15.0]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    methods_to_plot = ["PostFilter", "ACORN-1", "FDiskANN+Post", "TANNS+Post", "TANNS-C"]

    for method in methods_to_plot:
        if method not in sel_data:
            continue
        xs, ys = [], []
        for bl, bc in zip(bin_labels, bin_centers):
            if bl in sel_data[method]:
                xs.append(bc)
                ys.append(sel_data[method][bl]["mean_recall"])

        lw = 2.5 if method == "TANNS-C" else 1.2
        ms = 7 if method == "TANNS-C" else 5
        zorder = 10 if method == "TANNS-C" else 3

        ax.plot(xs, ys, marker=MARKERS.get(method, "o"), color=COLORS[method],
                label=DISPLAY_NAMES[method], linewidth=lw, markersize=ms,
                zorder=zorder, markeredgewidth=0.5, markeredgecolor="#333")

    ax.set_xlabel("Filter selectivity (% of dataset)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.5, 1.05)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.25%", "0.5%", "1%", "2%", "5%", "10%", "20%"])
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)

    # Shade low-selectivity region
    ax.axvspan(0.1, 1.0, alpha=0.08, color="#FF0000")
    ax.text(0.45, 0.53, "low\nselectivity", fontsize=6, color="#CC0000",
            ha="center", va="bottom", fontstyle="italic")

    save_fig(fig, "fig5_selectivity", outdir)


# ═════════════════════════════════════════════════════════════════════
# Figure 6 — Construction Time and Memory
# ═════════════════════════════════════════════════════════════════════

def fig6_construction(results_dir, outdir):
    """Generate Figure 6: Construction cost."""
    logger.info("Figure 6: Construction cost...")

    costs_path = os.path.join(results_dir, "_construction_costs.json")
    with open(costs_path) as f:
        costs = json.load(f)

    methods = ["PostFilter", "ACORN-1", "TANNS+Post", "FDiskANN+Post", "TANNS-C"]
    display = ["PostFilt", "ACORN-1", "TANNS", "FDiskANN", "TANNS-C"]
    build_times = [costs[m]["build_time_s"] for m in methods]
    memories = [costs[m]["peak_mem_mb"] for m in methods]

    bar_colors = [COLORS[m] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5))

    x = np.arange(len(methods))
    w = 0.55

    # Build time
    bars1 = ax1.bar(x, build_times, color=bar_colors, edgecolor="#444",
                    linewidth=0.5, width=w)
    ax1.set_ylabel("Build time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display, fontsize=6, rotation=30, ha="right")
    for bar, val in zip(bars1, build_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)

    # Memory
    bars2 = ax2.bar(x, memories, color=bar_colors, edgecolor="#444",
                    linewidth=0.5, width=w)
    ax2.set_ylabel("Peak memory (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display, fontsize=6, rotation=30, ha="right")
    for bar, val in zip(bars2, memories):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "fig6_construction", outdir)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate publication figures for TANNS-C")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"),
                        help="Directory containing benchmark result files")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fig1_architecture(args.output_dir)
    fig2_recall10_vs_qps(args.results_dir, args.output_dir)
    fig3_recall100_vs_qps(args.results_dir, args.output_dir)
    fig4_ablation(args.results_dir, args.output_dir)
    fig5_selectivity(args.results_dir, args.output_dir)
    fig6_construction(args.results_dir, args.output_dir)
    logger.info(f"\nAll figures saved to {args.output_dir}/")
