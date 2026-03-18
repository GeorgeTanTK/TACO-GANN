# Literature Matrix: Filtered / Temporal / Metadata-Aware Approximate Nearest Neighbor Search (2023–2026)

**Compiled:** 2026-03-05  
**Context:** Related work survey for TANNS-C (Temporal- and Category-Aware ANN Index)  
**Scope:** All papers on filtered, temporal, or metadata-aware ANNS published 2023–2026, plus key precursors

---

## Part I: Core Literature Matrix

### A. Temporal-Aware ANN Systems

| Field | **TANNS** | **TiGER** | **FreshDiskANN** |
|---|---|---|---|
| **Title** | Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data | A Versioned Unified Graph Index for Dynamic Timestamp-Aware Nearest Neighbor Search | FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search |
| **Authors** | Y. Wang, Z. He, Y. Tong, Z. Zhou, Y. Zhong | J.W. Chung, W. Zhao | A. Singh, S.J. Subramanya, R. Krishnaswamy, H.V. Simhadri |
| **Venue, Year** | ICDE 2025 | ICLR 2026 (withdrawn Nov 2025) | arXiv 2021 / NeurIPS'23 streaming baseline |
| **Core Method** | Timestamp Graph on HNSW: single unified proximity graph with per-node Historic Neighbor Tree (HNT) compressing temporal neighbor versions to O(MN) space. Backup-neighbor mechanism maintains connectivity after expirations. | Versioned proximity graph with validity-annotated edges [t_insert, t_expire], predecessor links for global reachability, and sparse edge database for O(1) range-minimum lookups over contiguous timestamp queries. | FreshVamana: streaming Vamana graph supporting real-time inserts/deletes. StreamingMerge merges in-memory updates into SSD-resident long-term index with write cost proportional to change set. |
| **Filter Type** | Range/timestamp (single query timestamp; validity interval per vector) | Range/timestamp (contiguous or disjoint timestamp sets) | None (pure streaming ANN; no attribute filtering) |
| **Graph Base** | HNSW (extended) | Custom proximity graph (HNSW-inspired) | Vamana (DiskANN) |
| **Key Datasets** | SIFT-1M, GIST-1M, DEEP-1M, GloVe-1M | SIFT-1M, GloVe-100-1M | SIFT-1B (~900M), MS Turing-30M, Wiki-35M |
| **Best Recall@10** | >0.99 on all 4 datasets (SIFT, GIST, DEEP, GloVe) | N/A (reports Recall@100 only) | ~0.95 5-recall@5 (streaming) |
| **Best Recall@100** | Not reported (k=10 only) | ~0.90+ (from QPS-recall curves) | Not primary metric |
| **QPS at 0.9 recall** | ~50K QPS at 0.95 recall (SIFT/DEEP); 7K (GIST); 4.5K (GloVe). 4.4×–138.1× speedup over baselines | ~10K–100K (SIFT, from figure plots; ~5× over baselines) | ~1,000 search/sec on 900M pts (SSD); concurrent 1.8K inserts + 1.8K deletes/sec |
| **Temporal Evolution** | **Yes** (core contribution: dynamic insert/expire) | **Yes** (dynamic insertions with predecessor links) | **Yes** (streaming inserts + deletes) |
| **Category/Label Filters** | **No** | **No** | **No** |
| **Open-Source Code** | Not released | Not available (withdrawn) | [github.com/microsoft/DiskANN](https://github.com/microsoft/DiskANN) |

---

### B. Category/Label-Filtered ANN Systems

| Field | **Filtered-DiskANN** | **NHQ** | **ACORN** | **SIEVE** | **RWalks** |
|---|---|---|---|---|---|
| **Title** | Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters | An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint | ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data | SIEVE: Effective Filtered Vector Search with Collection of Indexes | RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search |
| **Authors** | S. Gollapudi, N. Karia, V. Sivashankar, R. Krishnaswamy et al. (Microsoft) | M. Wang, L. Lv, X. Xu, Y. Wang, Q. Yue, J. Ni | L. Patel, P. Kraft, C. Guestrin, M. Zaharia | Z. Li, S. Huang, W. Ding, Y. Park, J. Chen | A. Ait Aomar, K. Echihabi, M. Arnaboldi, I. Alagiannis, D. Hilloulin, M. Cherkaoui |
| **Venue, Year** | WWW 2023 | NeurIPS 2023 | SIGMOD 2024 | VLDB 2025 | SIGMOD 2025 |
| **Core Method** | FilteredVamana / StitchedVamana: category-aware graph construction with filter-aware RobustPrune; per-label entry points; FilteredGreedySearch routes only through label-matching nodes. | Composite proximity graph using "fusion distance" jointly encoding vector distance + attribute match; NHQ pruning preserves attribute diversity; two NPG variants (NPG_kgraph, NPG_nsw). | Predicate-agnostic hybrid search on HNSW. ACORN-γ expands neighbor lists by factor γ=1/s_min at build time; predicate subgraph traversal at query time restricts to matching nodes. ACORN-1 does expansion at search time. | Workload-aware collection of many specialized HNSW indexes. 3D analytical model selects fastest index per query for target recall. Amortizes build cost across predicate patterns. | Index-agnostic method: attribute-aware random walks diffuse attribute information through graph, guiding traversal toward predicate-satisfying regions without modifying index construction. |
| **Filter Type** | Category/label (single equality predicate) | Multi-attribute categorical (equality, compound) | Predicate-agnostic (equality, range, regex, compound) | Predicate-agnostic (equality, compound, multi-label) | Predicate-agnostic (unique + composite filters) |
| **Graph Base** | Vamana | NSW / KGraph (+ DiskANN extension) | HNSW (FAISS extension) | HNSW (collection) | Agnostic (tested on HNSW) |
| **Key Datasets** | Turing-2.6M, Prep-1M, DANN-3.3M, SIFT-1M, GIST-1M | 10 datasets: SIFT-1M, GIST-1M, GloVe-1.2M, Crawl-2M, Paper-2M, BIGANN-100M, etc. | SIFT-1M, Paper-2M, TripClick-1M, LAION-1M/25M | YFCC-10M, Paper, UQV, GIST, SIFT, MSONG | 4 real-world datasets up to 100M vectors |
| **Best Recall@10** | >0.90 at thousands of QPS (SSD) | >0.99 on SIFT, GIST, GloVe, etc. | ~0.95+ across datasets (from curves) | >0.99 on low-selectivity sets | Not tabulated (2× faster than ACORN at same recall) |
| **Best Recall@100** | Not reported | 0.80–1.00 range | Not reported | Not reported | Not reported |
| **QPS at 0.9 recall** | StitchedVamana: 6–7.5× over IVF baseline (Prep/DANN); thousands QPS on SSD | 10³–10⁴ on SIFT-1M; 315× faster than AF-DiskANN on BIGANN-100M | 2–1,000× over baselines depending on selectivity (LAION-25M: >1,000×) | Up to 8.06× over ACORN (YFCC); 10.61× over CAPS (Paper) | 2× faster than ACORN; 76× faster build; 13× faster unfiltered |
| **Temporal Evolution** | **No** | **No** | **No** | **No** | **No** |
| **Category/Label Filters** | **Yes** (core contribution) | **Yes** (core contribution) | **Yes** (predicate-agnostic) | **Yes** (workload-aware) | **Yes** (attribute diffusion) |
| **Open-Source Code** | [github.com/microsoft/DiskANN](https://github.com/microsoft/DiskANN) | [github.com/YujianFu97/NHQ](https://github.com/YujianFu97/NHQ) | [github.com/stanford-futuredata/ACORN](https://github.com/stanford-futuredata/ACORN) | Not released | Not released |

---

### C. Range-Filtered ANN Systems

| Field | **SeRF** | **Dynamic SeRF (DSG)** | **iRangeGraph** | **DIGRA** | **UNIFY** |
|---|---|---|---|---|---|
| **Title** | SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search | Dynamic Range-Filtering Approximate Nearest Neighbor Search | iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search | DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter | UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search |
| **Authors** | C. Zuo, M. Qiao, W. Zhou, F. Li, D. Deng | Z. Peng, M. Qiao, W. Zhou, F. Li, D. Deng | Y. Xu, J. Gao, Y. Gou, C. Long, C.S. Jensen | M. Jiang, Z. Yang, F. Zhang, G. Hou, J. Shi, W. Zhou, F. Li, S. Wang | A. Liang, P. Zhang, B. Yao, Z. Chen, Y. Song, G. Cheng |
| **Venue, Year** | SIGMOD 2024 | VLDB 2025 | SIGMOD 2025 | SIGMOD 2025 | VLDB 2025 |
| **Core Method** | Segment graph losslessly compresses O(n) prefix-range HNSW indexes into single structure with same size as one HNSW; 2D segment graph for general ranges at O(n log n). | Dynamic segment graph with rectangle-labeled edges for streaming insert-then-query workloads. Insertions add O(log \|D\|) edges in expectation. | Segment tree of RNG-based HNSW elemental graphs; on-the-fly reconstruction of range-specific graph from O(log n) elemental graphs at query time. | Dynamic B-tree-like multi-way tree with NSW indices at each node; lazy weight-based split/merge yields O(log n) amortized update overhead—5 orders of magnitude cheaper than rebuild. | Segmented Inclusive Graph (SIG): segments by attribute values so any contiguous range's proximity graph is a subgraph. Hierarchical SIG (HSIG) adds HNSW-like levels. |
| **Filter Type** | Numeric range (single ordered attribute) | Numeric range (dynamic/streaming) | Numeric range (single + multi-attribute via probabilistic extension) | Numeric range (single attribute) | Numeric range (single attribute, all widths) |
| **Graph Base** | HNSW | HNSW | HNSW (segment tree) | NSW (multi-way tree) | HNSW-derived (SIG/HSIG) |
| **Key Datasets** | SIFT, GIST, GloVe, Text2Image (all 1M) | SIFT, GIST, GloVe, Text2Image | WIT-1M, TripClick-1M, Redcaps-1M, YouTube-RGB/Audio-1M | SIFT-1M, Redcaps-1M, GIST-1M, WIT-1M | SIFT-1M, GIST-1M, DEEP-1M/100M, Text2Image |
| **Best Recall@10** | >0.95 (SIFT, varied range widths) | ~0.90+ (competitive with static) | >0.90 on all datasets across selectivities | >0.90, stable post-updates | State-of-the-art across all range widths |
| **Best Recall@100** | Not reported | Not reported | Not reported | Not reported | Not reported |
| **QPS at 0.9 recall** | Up to ~10× over naive methods at narrow ranges | Competitive with or better than static baselines | 2–5× over best baselines; ~2K–10K QPS on 1M datasets | ~2K–4K (SIFT); ~400–800 (Redcaps); stable under 10% inserts | Best or competitive at all range widths |
| **Temporal Evolution** | **No** (static) | **Yes** (streaming inserts) | **No** (static) | **Yes** (inserts + deletes, O(log n)) | **No** (static) |
| **Category/Label Filters** | **No** | **No** | **No** | **No** | **No** |
| **Open-Source Code** | [github.com/Chaoji-zuo/SeRF](https://github.com/Chaoji-zuo/SeRF) | Not released | [github.com/YuexuanXu7/iRangeGraph](https://github.com/YuexuanXu7/iRangeGraph) | [github.com/CUHK-DBGroup/DIGRA](https://github.com/CUHK-DBGroup/DIGRA) | [github.com/Liang-Anqi/UNIFY](https://github.com/Liang-Anqi/UNIFY) |

---

### D. Additional Range, Compound, and System Papers

| Field | **KHI** | **WinFilter (β-WST)** | **ESG** | **RangePQ** | **WoW** |
|---|---|---|---|---|---|
| **Title** | Efficient Approximate Nearest Neighbor Search under Multi-Attribute Range Filter | Approximate Nearest Neighbor Search with Window Filters | ESG: Elastic Graphs for Range-Filtering Approximate k-Nearest Neighbor Search | Efficient Dynamic Indexing for Range Filtered Approximate Nearest Neighbor Search | WoW: A Window-to-Window Incremental Index for Range-Filtering ANN Search |
| **Authors** | Y. Yu, D. Cheng, Y. Zhang, L. Qin, W. Zhang, X. Lin | J. Engels, B. Landrum, S. Yu, L. Dhulipala, J. Shun | M. Yang, W. Li, Z. Shen, C. Xiao, W. Wang | F. Zhang, M. Jiang, G. Hou, J. Shi, H. Fan, W. Zhou, F. Li, S. Wang | Z. Wang, J. Zhang, W. Hu |
| **Venue, Year** | arXiv Feb 2026 | arXiv Feb 2024 | arXiv Apr 2025 | SIGMOD 2025 | arXiv Aug 2025 |
| **Core Method** | Key-value Hybrid Index: skew-aware k-d-tree partitioning of multi-attribute space + HNSW per tree node. First to tackle multi-attribute RFANNS. | Modular tree-based framework wrapping any c-approximate NN index for window (range) search. Formal approximation guarantees. Up to 75× speedup. | Elastic relaxation: allows controlled out-of-range inclusion during search, reducing required subranges to at most 2 (vs. O(log N) in SeRF). | PQ-based range-filtered ANN with O(n log K) space; hybrid two-layer structure reduces to O(n). Supports dynamic updates natively. | Hierarchical window graphs with incremental insertion; optimizes relevant window search based on range selectivity. |
| **Filter Type** | Multi-attribute range (compound range predicates) | Numeric range/window (single ordered attribute) | Numeric range (single attribute) | Numeric range (single attribute) | Numeric range (single attribute) |
| **Graph Base** | HNSW (per tree node) | Agnostic (wraps any ANN index) | Proximity graph (HNSW-based) | PQ-based (not graph) | Window-graph (proximity-based) |
| **Key Datasets** | 4 real-world datasets (typical ANN benchmarks) | SIFT + YFCC with real timestamps, adversarial embeddings | SIFT, GIST, DEEP | SIFT, GIST, DEEP, Text2Image | Standard RFANNS benchmarks |
| **Best Recall@10** | Not explicitly stated (QPS improvement focus) | Same recall, 75× faster | High accuracy maintained | Competitive with graph methods | Matches best static index |
| **QPS at 0.9 recall** | 2.46× avg; up to 16.22× over single-attr baselines | Up to 75× over prior methods | 1.5–6× over state-of-the-art | Competitive; native update support | 4× faster than best incremental index |
| **Temporal Evolution** | **No** | **No** (static; handles timestamp ranges as attribute) | **No** | **Yes** (inserts/deletes) | **Yes** (incremental insertion) |
| **Category/Label Filters** | **No** (range only) | **No** | **No** | **No** | **No** |
| **Open-Source Code** | Not released | [github.com/parlayann/window_filters](https://github.com/parlayann/window_filters) | Not released | Not released | Not released |

---

### E. Systems, Benchmarks, and Surveys

| Field | **Curator** | **HQANN** | **NaviX** | **PASE** | **FANNS Benchmark (ETH)** |
|---|---|---|---|---|---|
| **Title** | Curator: Efficient Indexing for Multi-Tenant Vector Databases | HQANN: Efficient and Robust Similarity Search for Hybrid Queries with Structured and Unstructured Constraints | NaviX: A Native Vector Index Design for Graph DBMSs With Robust Filtered Search | PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension | Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors |
| **Authors** | Y. Jin, Y. Wu, W. Hu, B.M. Maggs, X. Zhang, D. Zhuo | W. Wu, J. He, Y. Qiao, G. Fu, L. Liu, J. Yu | G. Sehgal, S. Salihoğlu | W. Yang, T. Li, G. Fang, H. Wei | P. Iff, P. Bruegger, M. Chrapek, M. Besta, T. Hoefler |
| **Venue, Year** | arXiv 2024 | CIKM 2022 | VLDB 2025 | SIGMOD 2020 (Industrial) | arXiv Jul 2025 (ETH Zurich) |
| **Core Method** | Global Clustering Tree (hierarchical k-means) shared across tenants; per-tenant Tenant Clustering Trees encoded via Bloom filters. Matches per-tenant indexing speed at shared-index memory. | Attribute navigation structure alongside proximity graph; fuses vector similarity with structured constraint matching during unified traversal. | Adaptive-local heuristic for filtered kNN in graph DBMS (KùzuDB): dynamically picks expansion strategy based on per-node estimated selectivity. | PostgreSQL extension supporting HNSW + IVFFlat with composite SQL predicate queries over high-dimensional vectors. | Comprehensive benchmark: 11 FANNS methods on arxiv-for-fanns dataset (2.7M arXiv abstracts, 11 real attributes). No single winner; method choice depends on filter type and selectivity. |
| **Filter Type** | Tenant-identity/label (binary access control per tenant) | Compound/multi-attribute (equality, structured constraints) | Predicate-agnostic (equality, compound via pre-filtering) | Compound SQL (equality, range, multi-attribute) | All (benchmark covers categorical, range, compound) |
| **Graph Base** | IVF (hierarchical k-means) | HNSW (or any proximity graph) | HNSW | HNSW + IVFFlat | Multiple (benchmark) |
| **Key Datasets** | YFCC100M-1M (192d, 1K tenants), arXiv-2M (384d, 100 tenants) | GloVe-1.2M | YFCC-1M, Wikipedia embeddings | Alibaba production, SIFT | arxiv-for-fanns: 2.7M abstracts, 4096d, 11 attributes |
| **Best Recall@10** | On par with per-tenant indexing | 0.99 on GloVe-1.2M (~50µs latency) | Robust at medium-to-low selectivity | Production-level (Alibaba) | Varies by method; no universal winner |
| **QPS at 0.9 recall** | 37.2× over metadata filtering | ~10× over prior hybrid ANNS | Outperforms blind/one-hop heuristics | Not tabulated | Benchmark results per method |
| **Temporal Evolution** | **Partial** (insert/delete/access-revoke; GCT fixed) | **No** | **No** | **Yes** (PostgreSQL native updates) | **No** |
| **Category/Label Filters** | **Yes** (tenant ID) | **Yes** (multi-attribute) | **Yes** (equality + compound) | **Yes** (SQL predicates) | **Yes** (benchmark) |
| **Open-Source Code** | Not released | Not released | [github.com/kuzudb/kuzu](https://github.com/kuzudb/kuzu) | [github.com/B-tree-cloud/pase](https://github.com/B-tree-cloud/pase) | Dataset: [HuggingFace SPCL/arxiv-for-fanns](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium) |

---

## Part II: Consolidated Capability Matrix

The table below cross-references all 22 papers/systems on the two dimensions most relevant to TANNS-C: **temporal evolution support** and **category/label filter support**.

| Paper | Year | Venue | Temporal Evolution | Category/Label Filter | Filter Type | Graph Base |
|---|---|---|---|---|---|---|
| **TANNS** | 2025 | ICDE | ✅ Yes | ❌ No | Timestamp | HNSW |
| **TiGER** | 2025 | ICLR (withdrawn) | ✅ Yes | ❌ No | Timestamp set | Custom |
| **FreshDiskANN** | 2021 | arXiv | ✅ Yes | ❌ No | None | Vamana |
| **Dynamic SeRF (DSG)** | 2025 | VLDB | ✅ Yes | ❌ No | Numeric range | HNSW |
| **DIGRA** | 2025 | SIGMOD | ✅ Yes | ❌ No | Numeric range | NSW |
| **RangePQ** | 2025 | SIGMOD | ✅ Yes | ❌ No | Numeric range | PQ |
| **WoW** | 2025 | arXiv | ✅ Yes | ❌ No | Numeric range | Window graph |
| **Curator** | 2024 | arXiv | ⚠️ Partial | ✅ Yes (tenant) | Tenant label | IVF |
| **PASE** | 2020 | SIGMOD | ✅ Yes (RDBMS) | ✅ Yes (SQL) | Compound SQL | HNSW+IVF |
| **Filtered-DiskANN** | 2023 | WWW | ❌ No | ✅ Yes | Category/label | Vamana |
| **NHQ** | 2023 | NeurIPS | ❌ No | ✅ Yes | Multi-attr categorical | NSW/KGraph |
| **ACORN** | 2024 | SIGMOD | ❌ No | ✅ Yes | Predicate-agnostic | HNSW |
| **SIEVE** | 2025 | VLDB | ❌ No | ✅ Yes | Predicate-agnostic | HNSW |
| **RWalks** | 2025 | SIGMOD | ❌ No | ✅ Yes | Predicate-agnostic | Agnostic |
| **HQANN** | 2022 | CIKM | ❌ No | ✅ Yes | Compound | HNSW |
| **NaviX** | 2025 | VLDB | ❌ No | ✅ Yes | Predicate-agnostic | HNSW |
| **SeRF** | 2024 | SIGMOD | ❌ No | ❌ No | Numeric range | HNSW |
| **iRangeGraph** | 2025 | SIGMOD | ❌ No | ❌ No | Numeric range | HNSW |
| **UNIFY** | 2025 | VLDB | ❌ No | ❌ No | Numeric range | HNSW |
| **KHI** | 2026 | arXiv | ❌ No | ❌ No | Multi-attr range | HNSW |
| **WinFilter (β-WST)** | 2024 | arXiv | ❌ No | ❌ No | Numeric range | Agnostic |
| **ESG** | 2025 | arXiv | ❌ No | ❌ No | Numeric range | Prox. graph |

**Legend:** ✅ = native support, ❌ = not supported, ⚠️ = partial/limited

---

## Part III: Gap Analysis — Motivation for TANNS-C

### Gap 1: No Existing System Jointly Addresses Temporal Evolution and Category-Aware Filtering

The literature on filtered approximate nearest neighbor search has advanced rapidly along two largely independent axes. On the temporal axis, TANNS (Wang et al., ICDE 2025) introduced the Timestamp Graph and Historic Neighbor Tree for efficient single-timestamp ANN queries, while TiGER (Chung & Zhao, 2025), Dynamic SeRF (Peng et al., VLDB 2025), DIGRA (Jiang et al., SIGMOD 2025), and FreshDiskANN (Singh et al., 2021) each address dynamic range-filtered or streaming ANN workloads — but none of these systems support category or label predicates. On the category-filtering axis, Filtered-DiskANN (Gollapudi et al., WWW 2023) embeds per-label graph construction with FilteredVamana, NHQ (Wang et al., NeurIPS 2023) fuses attribute matching into composite proximity graphs, and ACORN (Patel et al., SIGMOD 2024) enables predicate-agnostic traversal over expanded HNSW — yet all three are static indexes with no mechanism for handling temporal validity or evolving neighbor relationships. Even the most recent entries — SIEVE (Li et al., VLDB 2025), RWalks (Ait Aomar et al., SIGMOD 2025), NaviX (Sehgal & Salihoğlu, VLDB 2025), and KHI (Yu et al., 2026) — each advance either range filtering or label-based filtering in isolation, without jointly modeling temporal neighbor evolution. The only systems that touch both dimensions are PASE (Yang et al., SIGMOD 2020), which relies on PostgreSQL's generic transactional layer rather than temporal-aware indexing, and Curator (Jin et al., 2024), which handles only binary tenant-access labels without temporal neighbor versioning. In short, **no existing ANN index natively combines (i) category-aware graph construction, (ii) temporal neighbor versioning, and (iii) adaptive fallback for sparse filter combinations** into a single, unified structure.

### Gap 2: The Joint Temporal + Category Query Is a Real and Growing Workload

This gap is not merely theoretical. Real-world vector retrieval workloads increasingly require conjunctive temporal-categorical predicates — for example, "retrieve the k nearest documents in category cs.AI that were active before January 2025" or "find the most similar product embeddings in category Electronics inserted in the last 30 days." The FANNS benchmark (Iff et al., ETH Zurich 2025) demonstrates that filter selectivity, attribute type, and predicate compound structure all materially impact recall-QPS tradeoffs, yet its 11-attribute arXiv dataset is evaluated only with static indexes. TANNS-C addresses this gap by proposing a three-pillar architecture: (1) a Filtered-DiskANN-style category-aware graph with per-label entry points ensuring navigability within each category subgraph; (2) coarse temporal neighbor snapshots (a simplified Timestamp Graph) that version each node's neighbor list across time without the full O(M²N) overhead of the original Timestamp Graph; and (3) an ACORN-style two-hop fallback for sparse filter combinations where the intersection of a category and a time window yields too few candidates for reliable graph traversal. This combination is strictly more expressive than any single existing method, and the independent maturity of each component (validated in TANNS, Filtered-DiskANN, and ACORN respectively) provides strong evidence that their composition is both feasible and likely to outperform naive alternatives (pre-filtering, post-filtering, or sequential application of temporal and category filters).

---

## Part IV: Source URLs

| Paper | Primary Source |
|---|---|
| TANNS | [ICDE 2025 PDF](https://hufudb.com/static/paper/2025/ICDE25-wang.pdf) |
| TiGER | [OpenReview (withdrawn)](https://openreview.net/forum?id=nadglckd3z) |
| FreshDiskANN | [arXiv:2105.09613](https://arxiv.org/abs/2105.09613) |
| Filtered-DiskANN | [ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3543507.3583552) / [PDF](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf) |
| NHQ | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html) |
| ACORN | [arXiv:2403.04871](https://arxiv.org/abs/2403.04871) / [GitHub](https://github.com/stanford-futuredata/ACORN) |
| SIEVE | [VLDB 2025](https://dl.acm.org/doi/10.14778/3749646.3749725) |
| RWalks | [SIGMOD 2025](https://dl.acm.org/doi/10.1145/3725349) |
| SeRF | [SIGMOD 2024](https://dl.acm.org/doi/10.1145/3639324) |
| Dynamic SeRF (DSG) | [VLDB 2025](https://dl.acm.org/doi/10.14778/3748191.3748193) |
| iRangeGraph | [arXiv:2409.02571](https://arxiv.org/abs/2409.02571) / [GitHub](https://github.com/YuexuanXu7/iRangeGraph) |
| DIGRA | [SIGMOD 2025](https://dl.acm.org/doi/10.1145/3725399) / [GitHub](https://github.com/CUHK-DBGroup/DIGRA) |
| UNIFY | [VLDB 2025](https://dl.acm.org/doi/10.14778/3717755.3717770) |
| KHI | [Semantic Scholar](https://www.semanticscholar.org/paper/2a87d5497ccba98e591cb6da0a0a299fcadd4fa0) |
| WinFilter (β-WST) | [arXiv:2402.00943](https://arxiv.org/abs/2402.00943) / [GitHub](https://github.com/parlayann/window_filters) |
| ESG | [arXiv:2504.04018](https://arxiv.org/abs/2504.04018) |
| RangePQ | [SIGMOD 2025](https://dl.acm.org/doi/10.1145/3725401) |
| WoW | [arXiv:2508.18617](https://arxiv.org/abs/2508.18617) |
| Curator | [arXiv:2401.07119](https://arxiv.org/abs/2401.07119) |
| HQANN | [CIKM 2022](https://dl.acm.org/doi/10.1145/3511808.3557610) |
| NaviX | [VLDB 2025](https://www.vldb.org/pvldb/vol18/p4438-sehgal.pdf) |
| PASE | [SIGMOD 2020](https://dl.acm.org/doi/10.1145/3318464.3386131) |
| FANNS Benchmark | [arXiv:2507.21989](https://arxiv.org/abs/2507.21989) / [Dataset](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium) |
