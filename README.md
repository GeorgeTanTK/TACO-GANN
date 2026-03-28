# TACO-GANN: Temporal And Category Optimised Graph Approximate Nearest Neighbour Search

TACO-GANN is a single‑graph approximate nearest neighbour (ANN) index that natively supports **time‑range filtering** and **category filtering** on large vector datasets. It answers queries of the form “k‑nearest neighbors at time *t* with category *C*” without resorting to brute‑force post‑filtering.

TACO-GANN is designed for users who need:

- Time-aware ANN search with **temporal filters** (sliding or fixed time windows)
- Category-aware ANN search with **first-class category filters**
- Combined **time and category filtering** over high-dimensional vectors (e.g., ArXiv embeddings with subject areas and timestamps)

It combines:

- A **category-aware Filtered‑Vamana graph** (per‑label ST‑connectivity, medoid entry points, alpha‑blended neighbour scoring), and  
- **Per‑node Historic Neighbour Tables (HNTs)** that reconstruct neighbours valid in a query time window.

The target query is:

> “Top‑k nearest neighbours of vector **q**, with category **C**, valid in time window **[t_start, t_end]**.”

The reference dataset is **SPCL/arxiv‑for‑fanns‑medium** (100K ArXiv embeddings with subject categories and submission days)[cite:114].
text

---

## Table of Contents

- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Implemented Methods for Time + Category Filtering](#Implemented-Methods-for-Time-and-Category-Filtering)
- [Benchmark Pipeline](#benchmark-pipeline)
- [Figures](#figures)
- [Limitations and TODOs](#limitations-and-todos)
- [Citation](#citation)
- [License](#license)

---

## Use Cases

- Time-filtered semantic search over documents (e.g., “papers similar to this, in cs.CL, from 2020–2022”)
- Temporal recommendation with category constraints (e.g., “similar items in category X purchased in the last 7 days”)
- Any vector database workload that needs **temporal + categorical filters** without materialising per-filter indices

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/georgetktan/taco-gann.git
cd taco-gann
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download the SPCL/arxiv-for-fanns dataset into data/ Option (small/medium)/ default small
python download_data.py --out-dir data 

# 3. Run the full benchmark pipeline (ground truth → baselines → figs) anything after split optional
bash run_all.sh --split small
```

Results are written to `results/` and figures to `figures/`.

---

## Project Structure

```text
TACO-GANN/
├── run_all.sh                  # End-to-end pipeline script
├── download_data.py            # Fetch SPCL/arxiv-for-fanns-medium from HuggingFace
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # Dataset loading, metadata, query generation
│   ├── taco_gann.py              # TACO-GANN index (Filtered-Vamana + HNT)
│   └── baselines/
│       ├── postfilter.py       # HNSW + dual post-filter (category AND time)
|       ├── prefilter.py        # HNSW + dual pre-filter (category AND time)
│       └── tanns_post_filtering.py
│                               # TANNS implementation (ICDE'25 style)
├── benchmarks/
│   ├── compute_ground_truth.py # Exact k-NN ground truth for (C, [t_s, t_e]) queries
│   ├── compute_selectivity.py  # Per-selectivity-bin recall for each method
│   ├── measure_construction.py # Index build time / peak memory
│   ├── run_baselines.py        # Run HNSW PostFilter, TANNS+Post, TACO-GANN
│   └── generate_figures.py     # Figures (PNG + PDF)
├── results/                    # CSV / JSON outputs (created by scripts)
├── figures/                    # Generated figures
├── docs/
│   └── TACO-GANN-Design-Summary-and-Novelty-Brief.md
│                               # Design + novelty write-up
└── data/                       # Dataset files (populated by download)
```

---

## Dataset

This repo benchmarks on the **SPCL/arxiv‑for‑fanns‑medium** dataset[cite:114]:

- **100K** ArXiv papers with **4096‑dim** SPECTER‑style embeddings  
- Per‑paper metadata:
  - `categories`: subject area labels (e.g., `cs.AI`, `cs.CL`)[cite:114]
  - `submission_day`: integer day index (temporal order)[cite:114]
  - additional attributes (not all used yet)

TANNS‑C uses:

- `database_vectors.fvecs` — database vectors (100K × 4096)  
- `database_attributes.jsonl` — category sets and submission days per vector  
- `query_vectors.fvecs` plus query attribute JSONL to define (C, [t_start, t_end]) workloads[cite:114][cite:115]

To fetch the dataset:

```bash
python download_data.py --output-dir data
```

---

## Implemented Methods for Time and Category Filtering

The current benchmark compares **three** methods:

| Internal name   | Description |
| ---             | --- |
| `PostFilter-HNSW` | Vanilla HNSW (no metadata). Search over all vectors, then apply **time filter** and **category filter** as a post-filter mask. |
| `TANNS+Post`      | Timestamp graph + HNT (ICDE'25 TANNS) with **temporal k-NN** support, but **category filtering** is applied only as a post-filter at query time. |
| `TACO-GANN`       | This work: single Filtered‑Vamana graph with per‑label ST‑connectivity and per‑node HNT, enabling **joint temporal and category-aware ANN search** at query time. |

Older baselines (ACORN, Filtered‑DiskANN, ablative pillars P1/P2/P3) are **not** wired into this repo yet; the code focuses on a clean comparison between:

- naïve filtered HNSW  
- temporal‑only TANNS (category as post‑filter)  
- fully temporal‑ and category‑aware TANNS‑C

---

## Benchmark Pipeline

The recommended workflow mirrors the paper:

1. **Ground truth for conjunctive queries**

   ```bash
   python benchmarks/compute_ground_truth.py \
       --data-dir data \
       --results-dir results
   ```

   This computes exact top‑k under `(category C AND submission_day ∈ [t_start, t_end])` for each query and stores:

   - `gt`: ground-truth ids per query  
   - `Ms`: boolean masks for valid set membership per query  
   - `N`, `NQ`: corpus size and number of queries  

2. **Baseline runs (HNSW PostFilter, TANNS+Post, TANNS‑C)**

   ```bash
   python benchmarks/run_baselines.py \
       --data-dir data \
       --results-dir results
   ```

   This:

   - builds all three indices  
   - runs the full query set  
   - writes per‑method, per‑query stats and an overall summary to:
     - `results/baseline_results.csv`  (aggregated curves: recall vs QPS etc.)
     - `results/_baselines_simple.json` (per‑query recall, visited_count, selectivity)

3. **Selectivity analysis**

   ```bash
   python benchmarks/compute_selectivity.py \
       --data-dir data \
       --results-dir results
   ```

   This bins queries by **selectivity** \(|valid\_set| / N\) and computes Recall@10 per method within each bin, writing:

   - `results/_selectivity_recall.json`

4. **Construction cost**

   ```bash
   python benchmarks/measure_construction.py \
       --data-dir data \
       --results-dir results
   ```

   This measures index build time and peak RSS memory for each method and writes:

   - `results/_construction_costs.json`

5. **Figures**

   ```bash
   python benchmarks/generate_figures.py \
       --results-dir results \
       --output-dir figures
   ```

   This generates all paper figures as `PNG` + `PDF` under `figures/`.

---

## Figures

`generate_figures.py` currently produces:

- **Figure 1** – Architecture diagram:  
  category-aware Filtered‑Vamana layer + HNT temporal layer, with query `(q, C, [t_start, t_end], k)` and beam search over the temporal graph.

- **Figure 2** – Recall@10 vs QPS:  
  log‑scale QPS (x‑axis) against Recall@10 (y‑axis) for `PostFilter-HNSW`, `TANNS+Post`, `TANNS‑C`.

- **Figure 3** – Recall@100 vs QPS:  
  same as Fig. 2 but Recall@100.

- **Figure 4** – Recall@10 vs visited nodes:  
  efficiency curve using per‑query `visited_count` from `run_baselines.py`. Shows how many nodes each method touches to reach a given recall.

- **Figure 5** – Selectivity analysis:  
  Recall@10 as a function of query selectivity (% of corpus satisfying C × [t_start, t_end]). Highlights the **low‑selectivity regime** (< 1%) where TANNS‑C is designed to shine.

- **Figure 6** – Construction time and peak memory:  
  per‑method build cost on the SPCL/arxiv‑for‑fanns‑medium dataset.

---

## Limitations and TODOs

1. **Limited method coverage.**  
   Only three methods are implemented end‑to‑end: `PostFilter-HNSW`, `TANNS+Post`, `TANNS‑C`. Planned (but not yet wired) baselines include ACORN and Filtered‑DiskANN.

2. **Single dataset / medium scale.**  
   Experiments currently use the **100K** SPCL/arxiv‑for‑fanns‑medium split[cite:114]. Extending to the large split (or additional filtered‑ANN benchmarks) would better showcase scalability and structural gains.

3. **Python implementation.**  
   All indices are implemented in Python / NumPy. Performance is suitable for research but not production; a C++/Rust implementation would be significantly faster.

4. **Static benchmark.**  
   While TANNS‑C supports dynamic insert and tombstone delete in principle, the current scripts only evaluate static build + query workloads.

5. **No hyper‑parameter sweep.**  
   Default parameters (e.g., `M`, `ef_construction`, alpha in the blended score) follow the design document, not a tuned grid search.

---

## Citation

If you use this code or ideas in scientific work, please cite:

```bibtex
@misc{tanns-c-2026,
  title   = {{taco-gan}: Temporal And Category Optimised Graph Approximate Nearest Neighbour Search Search},
  author  = {Tan, T.K.},
  year    = {2026},
  note    = {HKUST}
}
```

---

## License

This project is released under the [MIT License](LICENSE).