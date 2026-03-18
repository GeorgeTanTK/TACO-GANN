# TANNS-C: Temporal- and Category-Aware Approximate Nearest Neighbour Search

TANNS-C is a filtered approximate nearest neighbour (ANN) index that natively supports **temporal range** and **categorical** predicates.  It extends the HNSW graph with three structural pillars:

1. **Category-Partitioned Sub-graphs (P1)** — one HNSW sub-graph per category, so filtered search never touches irrelevant vectors.
2. **Temporal Edge Weighting (P2)** — edge weights encode recency, letting the greedy search favour temporally relevant neighbours.
3. **Composite Scoring (P3)** — query-time scoring blends vector similarity with a temporal-decay factor, returning results that are both semantically close and temporally fresh.

```
┌───────────────────────────────────────────┐
│               TANNS-C Index               │
│                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Cat = ML │ │ Cat = CV │ │ Cat = NLP│  │
│  │  (HNSW)  │ │  (HNSW)  │ │  (HNSW)  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │  temporal   │  temporal   │        │
│       │  edge wts   │  edge wts   │        │
│       ▼             ▼             ▼        │
│  ┌─────────────────────────────────────┐  │
│  │   Composite Score = α·sim + β·decay │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Benchmark Results](#benchmark-results)
- [Ablation Study](#ablation-study)
- [Running Individual Steps](#running-individual-steps)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/tanns-c.git
cd tanns-c
pip install -r requirements.txt

# 2. Run the full pipeline (download → ground truth → baselines → figures)
bash run_all.sh

# 3. Or skip the download if data/ is already populated
bash run_all.sh --skip-download
```

Results are written to `results/` and publication figures to `figures/`.

---

## Project Structure

```
tanns-c/
├── run_all.sh                  # End-to-end pipeline script
├── download_data.py            # Fetch arxiv-for-FANNS from HuggingFace
├── requirements.txt            # Pinned dependencies
├── src/
│   ├── data_loader.py          # Dataset loading, query generation
│   ├── tanns_c.py              # TANNS-C index (all 3 pillars)
│   └── baselines/
│       ├── postfilter.py       # HNSW + post-filter
│       ├── prefilter.py        # Brute-force pre-filter
│       ├── acorn1.py           # ACORN-1 (expanded graph)
│       ├── tanns.py            # TANNS (2 variants)
│       └── filtered_diskann.py # Filtered-DiskANN (2 variants)
├── benchmarks/
│   ├── compute_ground_truth.py # Exact k-NN ground truth
│   ├── evaluate_all.py         # Run all methods, log recall/QPS
│   ├── run_ablation.py         # TANNS-C pillar ablation
│   ├── compute_selectivity.py  # Per-selectivity-bin recall
│   ├── measure_construction.py # Index build time/memory
│   └── generate_figures.py     # 6 publication-quality figures
├── results/                    # CSV / JSON outputs
├── figures/                    # Generated PNG + PDF figures
├── docs/
│   └── literature_matrix.md    # 22-paper survey of filtered ANN
└── data/                       # Dataset (populated by download)
```

---

## Dataset

**arxiv-for-FANNS** (small split): 1,000 Specter-v2 embeddings (4,096-d) of arXiv papers spanning 2007–2024 across 40 CS categories.  Each vector carries a category label and an integer timestamp (epoch day).

The benchmark generates 1,000 queries, each with a random category filter and a temporal window drawn uniformly between 1 and 17 years.  Mean selectivity is ~3.05 %.

To download separately:

```bash
python download_data.py --output-dir data
```

---

## Benchmark Results

All methods evaluated on the same 1,000-query workload with Recall@10 and queries per second (QPS).

| Method | Best Recall@10 | QPS (at best R@10) | Build Time (s) |
|---|---:|---:|---:|
| PreFilter (brute-force) | 1.0000 | 9,685 | — |
| TANNS + PostFilter (ef=100) | 1.0000 | 614 | 0.13 |
| TANNS + PreFilter | 1.0000 | 2,579 | 0.13 |
| FDiskANN + PostFilter (ef=500) | 1.0000 | 1,462 | 0.41 |
| FDiskANN + PreFilter | 1.0000 | 2,003 | 0.41 |
| **TANNS-C** (ef=200) | **0.9949** | **265** | **2.09** |
| ACORN-1 (ef=1000) | 0.9929 | 120 | 1.06 |
| PostFilter-HNSW (ef=50) | 0.8144 | 1,282 | 0.60 |

> **Note:** At N=1,000 the dataset is small enough for brute-force and simple baselines to dominate on both recall and QPS. TANNS-C's structural advantages (partitioned sub-graphs, temporal weighting) are expected to show clearer gains at larger scale (N ≥ 100 K).

---

## Ablation Study

Contribution of each TANNS-C pillar (ef=200):

| Variant | Recall@10 | QPS |
|---|---:|---:|
| P1 only (category partitions) | 0.9989 | 138 |
| P2 only (temporal weighting) | 0.9926 | 250 |
| P1 + P2 | 0.9926 | 256 |
| **P1 + P2 + P3 (full)** | **0.9949** | **265** |

P1 alone gives the largest recall boost; P3 (composite scoring) improves QPS by letting the search converge faster.

---

## Running Individual Steps

Each benchmark script supports `--data-dir` and `--output` flags:

```bash
# Ground truth
python -m benchmarks.compute_ground_truth --data-dir data

# Evaluate baselines
python -m benchmarks.evaluate_all --data-dir data --output results/baseline_results.csv

# Ablation
python -m benchmarks.run_ablation --data-dir data --output results/baseline_results.csv

# Selectivity analysis
python -m benchmarks.compute_selectivity --data-dir data --output results/selectivity_recall.json

# Construction costs
python -m benchmarks.measure_construction --data-dir data --output results/construction_costs.json

# Generate figures (requires results)
python -m benchmarks.generate_figures \
  --results results/baseline_results.csv \
  --selectivity results/selectivity_recall.json \
  --construction results/construction_costs.json \
  --output-dir figures/
```

---

## Limitations

1. **Small-scale evaluation only.** All benchmarks use the 1,000-vector small split.  At this scale, brute-force pre-filter is both faster and exact.  Results should be validated on the full 4.27 M-vector dataset.
2. **Synthetic queries.** Category and temporal windows are drawn uniformly at random; real workloads may have skewed distributions.
3. **Single-threaded.** QPS numbers reflect single-threaded Python execution; production deployments would use C++/Rust with batch parallelism.
4. **No disk-tier evaluation.** All indices are in-memory.  Filtered-DiskANN's on-disk design is not exercised.
5. **Fixed hyper-parameters.** Temporal decay rate λ and composite weight α were not tuned via grid search on validation queries.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{tanns-c-2026,
  title   = {{TANNS-C}: Temporal- and Category-Aware Approximate Nearest Neighbour Search},
  author  = {Tan, T.K.},
  year    = {2026},
  note    = {MPhil thesis, HKUST}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
