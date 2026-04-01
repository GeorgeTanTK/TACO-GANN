# Literature Matrix: Temporal, Filtered, and Metadata-Aware Approximate Nearest Neighbor Search (2020–2026)

**Compiled:** 2026‑03‑27  
**Context:** Related work survey for TANNS‑C (Temporal‑ and Category‑Aware ANN Index)  
**Scope:** Temporal, filtered, and metadata‑aware ANN methods that are relevant to conjunctive queries of the form  
> “Top‑k nearest neighbours of q within category C that are valid in time window [t_start, t_end].”

Where possible, tables focus on:

- **Temporal evolution** (dynamic inserts/updates, time‑stamped validity, historic neighbours)
- **Filter support** (labels, attributes, ranges, predicates)
- **Index structure** (graph type, range structures, hybrid systems)

---

## Part I: Core Literature Tables

### A. Temporal-Aware ANN Systems

These systems explicitly model time or support streaming inserts/expirations, but do **not** make category/label filters a first‑class part of the index.

| Field | **TANNS** | **TiGER** | **FreshDiskANN** |
|---|---|---|---|
| **Title** | Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data | A Versioned Unified Graph Index for Dynamic Timestamp-Aware Nearest Neighbor Search | FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search |
| **Authors** | Y. Wang, Z. He, Y. Tong, Z. Zhou, Y. Zhong | J.W. Chung, W. Zhao | A. Singh, S.J. Subramanya, R. Krishnaswamy, H.V. Simhadri |
| **Venue, Year** | ICDE 2025 | ICLR 2026 (withdrawn Nov 2025) | arXiv 2021 (used in later streaming work) |
| **Core Method** | **Timestamp Graph**: single unified proximity graph, plus per‑node **Historic Neighbor Tree (HNT)** that compresses neighbour histories so all timestamps share one graph with \(O(M^2 N)\) space[cite:74]. Fast insert/expire operations maintain temporal consistency. | Versioned timestamp‑aware proximity graph. Each edge stores a validity interval \([t_\text{insert}, t_\text{expire}]\); predecessor links maintain global reachability across versions. | **FreshVamana**: streaming Vamana graph supporting continuous inserts and deletes; **StreamingMerge** folds in-memory updates into an SSD‑resident long‑term graph with write cost proportional to the change set. |
| **Filter Type** | Timestamp (single query time; each vector has a validity interval) | Timestamp sets / ranges | None (pure similarity search) |
| **Graph Base** | HNSW‑style graph with temporal extensions[cite:74] | Custom versioned proximity graph | Vamana / DiskANN family |
| **Key Datasets** | SIFT‑1M, GIST‑1M, DEEP‑1M, GloVe‑1M | SIFT‑1M, GloVe‑100‑1M | SIFT‑1B (~900M), MS Turing‑30M, Wiki‑35M |
| **Temporal Evolution** | **Yes** (insert + expire, historic neighbours) | **Yes** (versioned graph) | **Yes** (streaming inserts + deletes) |
| **Category / Label Filters** | **No** | **No** | **No** |
| **Open-Source Code** | Not released at time of writing | Not released (paper withdrawn) | [Microsoft DiskANN GitHub][cite:136] (FreshDiskANN concepts later influence streaming DiskANN variants) |

---

### B. Category / Label-Filtered ANN Systems

These systems make attribute or label filters first‑class, mainly via graph construction and traversal, but treat time either as a static attribute or ignore it.

| Field | **Filtered-DiskANN** | **NHQ** | **ACORN** | **SIEVE** | **RWalks** |
|---|---|---|---|---|---|
| **Title** | Graph Algorithms for Approximate Nearest Neighbor Search with Filters | An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint | ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data | SIEVE: Effective Filtered Vector Search with Collection of Indexes | RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search |
| **Authors** | S. Gollapudi, N. Karia, V. Sivashankar, R. Krishnaswamy et al. | M. Wang, L. Lv, X. Xu, Y. Wang, Q. Yue, J. Ni | L. Patel, P. Kraft, C. Guestrin, M. Zaharia | Z. Li, S. Huang, W. Ding, Y. Park, J. Chen | A. Ait Aomar, K. Echihabi, M. Arnaboldi, I. Alagiannis, D. Hilloulin, M. Cherkaoui |
| **Venue, Year** | WWW 2023 | NeurIPS 2023 | SIGMOD 2024 | VLDB 2025 | SIGMOD 2025 |
| **Core Method** | **FilteredVamana / StitchedVamana**: graph construction is label‑aware via filter‑constrained **RobustPrune** and **FilteredGreedySearch**; per‑label entry points and ST‑connectivity guarantees for each label[cite:136]. | Composite “fusion distance” that jointly encodes vector similarity and attribute proximity; builds an attribute‑aware proximity graph with diversity‑preserving pruning. | **Predicate‑agnostic hybrid search**: HNSW index plus a filter‑aware traversal. ACORN‑γ expands neighbor lists by factor γ during construction; ACORN‑1 expands at query time. Traversal navigates **predicate subgraphs** of HNSW without duplicating indices. | **Workload‑aware multi‑index**: builds many specialized HNSW indices tuned to different filters/predicates; a 3D analytical cost model chooses the best index per query. | Index‑agnostic: uses attribute‑biased random walks over existing graphs to “diffuse” attribute information, biasing traversal towards nodes satisfying the predicate without changing the underlying index. |
| **Filter Type** | Single label equality (AND of multiple labels via Cartesian expansion) | Multi‑attribute categorical (equality, compound) | Predicate‑agnostic (equality, ranges, regex, compound predicates) | Predicate‑agnostic (equality and compound, multi‑label) | Predicate‑agnostic attribute filters |
| **Graph Base** | Vamana / DiskANN | NSW / KGraph (+ DiskANN adaptations) | HNSW (FAISS‑based) | HNSW (collection) | Agnostic (evaluated on HNSW) |
| **Temporal Evolution** | **No** (static index) | **No** | **No** (time is just another attribute) | **No** | **No** |
| **Category / Label Filters** | **Yes** (core contribution) | **Yes** (core contribution) | **Yes** (generic predicates) | **Yes** | **Yes** |
| **Open-Source Code** | [DiskANN GitHub][cite:136] (Filtered‑DiskANN code integrated) | [NHQ GitHub][cite:136] | [ACORN GitHub][cite:2] | Not public at time of writing | Not public at time of writing |

---

### C. Range-Filtered ANN Systems

These methods build range‑sensitive indexes for numeric attributes (e.g., price, timestamp). They are an important reference for temporal windows, but generally **do not** model temporal neighbour evolution in the TANNS sense.

| Field | **SeRF** | **Dynamic SeRF (DSG)** | **iRangeGraph** | **DIGRA** | **UNIFY** |
|---|---|---|---|---|---|
| **Title** | SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search | Dynamic Range-Filtering Approximate Nearest Neighbor Search | iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search | DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter | UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search |
| **Authors** | C. Zuo, M. Qiao, W. Zhou, F. Li, D. Deng | Z. Peng, M. Qiao, W. Zhou, F. Li, D. Deng | Y. Xu, J. Gao, Y. Gou, C. Long, C.S. Jensen | M. Jiang, Z. Yang, F. Zhang, G. Hou, J. Shi, W. Zhou, F. Li, S. Wang | A. Liang, P. Zhang, B. Yao, Z. Chen, Y. Song, G. Cheng |
| **Venue, Year** | SIGMOD 2024 | VLDB 2025 | SIGMOD 2025 | SIGMOD 2025 | VLDB 2025 |
| **Core Method** | **Segment Graph**: compresses many per‑range HNSW indexes into one structure; supports arbitrary numeric ranges with near‑optimal graph reuse. | Dynamic segment graph with rectangle‑labeled edges and logarithmic expected update cost for streaming insert‑then‑query workloads. | Segment tree of RNG/HNSW “elemental graphs”; range queries combine a small number of elemental graphs at query time. | Dynamic multi‑way tree where each node stores an NSW index; supports inserts and deletes with \(O(\log n)\) amortized cost while preserving range‑filtered ANN quality. | **Segmented Inclusive Graph (SIG)**: segments by attribute value so any contiguous range’s graph is a subgraph; hierarchical SIG (HSIG) adds multiple levels akin to HNSW. |
| **Filter Type** | Numeric range (single attribute) | Numeric range (streaming) | Numeric range (single and multi‑attribute variants) | Numeric range (single attribute) | Numeric range (single attribute) |
| **Graph Base** | HNSW | HNSW | HNSW / RNG | NSW | HNSW‑like |
| **Temporal Evolution** | Static | **Yes** (streaming inserts) | Static | **Yes** (updates) | Static |
| **Category / Label Filters** | **No** | **No** | **No** | **No** | **No** |

---

### D. Systems, Hybrid Queries, and Benchmarks

These works support hybrid or filtered search in broader system contexts (DBMSs, multi‑tenant systems, general FANNS benchmarks).

| Field | **Curator** | **HQANN** | **NaviX** | **PASE** | **FANNS Benchmark (ETH)** |
|---|---|---|---|---|---|
| **Title** | Curator: Efficient Indexing for Multi-Tenant Vector Databases | HQANN: Efficient and Robust Similarity Search for Hybrid Queries with Structured and Unstructured Constraints | NaviX: A Native Vector Index Design for Graph DBMSs With Robust Filtered Search | PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension | Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors |
| **Authors** | Y. Jin, Y. Wu, W. Hu, B.M. Maggs, X. Zhang, D. Zhuo | W. Wu, J. He, Y. Qiao, G. Fu, L. Liu, J. Yu | G. Sehgal, S. Salihoğlu | W. Yang, T. Li, G. Fang, H. Wei | P. Iff, P. Bruegger, M. Chrapek, M. Besta, T. Hoefler |
| **Venue, Year** | arXiv 2024 | CIKM 2022 | VLDB 2025 | SIGMOD 2020 (Industrial) | arXiv 2025 (ETH Zurich) |
| **Core Method** | Global clustering tree shared across tenants; per‑tenant clustering encoded via Bloom filters. Optimizes **multi‑tenant label filtering**, not temporal neighbour evolution. | Hybrid query index combining structured constraints with ANN; navigates both attribute and vector space during a unified search. | Native vector index inside Kùzu DBMS; adaptively chooses traversal strategies for filtered k‑NN queries based on estimated selectivity and graph structure. | PostgreSQL extension with HNSW/IVFFlat, supporting full SQL predicates over vectors and structured columns. | Introduces **arxiv‑for‑fanns** datasets (small/medium/large) with transformer embeddings and 11 attributes, and benchmarks a wide range of FANNS methods[cite:114][cite:133][cite:134]. No method jointly models temporal neighbour evolution and category. |
| **Filter Type** | Tenant labels, access control | Compound structured attributes | Predicate‑agnostic filters in a graph DBMS | SQL predicates (equality, ranges, joins) | All FANNS filter types (exact‑match, range, set‑membership) |
| **Temporal Evolution** | Partial (tenants can change, but neighbour evolution not explicitly modeled) | No | No | Uses Postgres update mechanisms | No (static datasets) |
| **Category / Label Filters** | Yes (tenant/label) | Yes | Yes | Yes | Yes (as workload, not as method) |

---

## Part II: Consolidated Capability Matrix

This matrix cross‑references representative systems on:

- **Temporal evolution support**: does the index explicitly track how neighbour relations change over time (via inserts, deletes, time‑stamped validity, or historic structures like HNT)?  
- **First‑class filter support**: does the index structure (not just post‑filtering) understand labels/attributes/ranges and use them in construction or traversal?

| Paper / System | Year | Venue | Temporal Evolution | Category / Label Filter | Filter Type | Graph Base |
|---|---|---|---|---|---|---|
| TANNS | 2025 | ICDE | ✅ Yes | ❌ No | Timestamp | HNSW + HNT[cite:74] |
| TiGER | 2026 | ICLR (withdrawn) | ✅ Yes | ❌ No | Timestamp sets | Custom versioned graph |
| FreshDiskANN | 2021 | arXiv | ✅ Yes | ❌ No | None | Vamana (DiskANN family) |
| Dynamic SeRF (DSG) | 2025 | VLDB | ✅ Yes | ❌ No | Numeric range | HNSW |
| DIGRA | 2025 | SIGMOD | ✅ Yes | ❌ No | Numeric range | NSW |
| RangePQ | 2025 | SIGMOD | ✅ Yes | ❌ No | Numeric range | PQ‑based |
| WoW | 2025 | arXiv | ✅ Yes | ❌ No | Numeric range | Window graph |
| Curator | 2024 | arXiv | ⚠️ Partial | ✅ Yes (tenant) | Tenant labels | IVF‑style |
| PASE | 2020 | SIGMOD | ✅ Yes (via DBMS) | ✅ Yes (SQL predicates) | General SQL | HNSW + IVF |
| Filtered-DiskANN | 2023 | WWW | ❌ No | ✅ Yes | Single label equality | Vamana / DiskANN[cite:136] |
| NHQ | 2023 | NeurIPS | ❌ No | ✅ Yes | Multi‑attribute categorical | NSW / KGraph |
| ACORN | 2024 | SIGMOD | ❌ No | ✅ Yes | Predicate‑agnostic | HNSW[cite:2] |
| SIEVE | 2025 | VLDB | ❌ No | ✅ Yes | Predicate‑agnostic | HNSW |
| RWalks | 2025 | SIGMOD | ❌ No | ✅ Yes | Predicate‑agnostic | Agnostic (on HNSW) |
| HQANN | 2022 | CIKM | ❌ No | ✅ Yes | Compound hybrid queries | HNSW |
| NaviX | 2025 | VLDB | ❌ No | ✅ Yes | Predicate‑agnostic | HNSW |
| SeRF | 2024 | SIGMOD | ❌ No | ❌ No | Numeric range | HNSW |
| iRangeGraph | 2025 | SIGMOD | ❌ No | ❌ No | Numeric range | HNSW / RNG |
| UNIFY | 2025 | VLDB | ❌ No | ❌ No | Numeric range | HNSW‑like |
| KHI | 2026 | arXiv | ❌ No | ❌ No | Multi‑attribute range | HNSW |
| WinFilter (β‑WST) | 2024 | arXiv | ❌ No | ❌ No | Numeric range | Index‑agnostic |
| ESG | 2025 | arXiv | ❌ No | ❌ No | Numeric range | Proximity graph |
| FANNS Benchmark | 2025 | arXiv | ❌ No | ✅ Yes (as workload) | EM, R, EMIS filters | Multiple (benchmark only)[cite:114][cite:134] |

**Legend:** ✅ = native/explicit support, ❌ = no support, ⚠️ = partial / indirect support.

---

## Part III: Gap Analysis and Motivation for TANNS‑C

### Gap 1: Temporal Evolution vs. Filtered Search Have Evolved Separately

Over the last five years, the ANN literature has progressed rapidly along **two mostly independent dimensions**:

1. **Temporal evolution of neighbour relations.**  
   TANNS proposes the **Timestamp Graph** and **Historic Neighbor Tree**, compressing neighbour histories so that a single graph supports timestamp‑aware queries with efficient insertions and expirations[cite:74]. FreshDiskANN and Dynamic SeRF focus on streaming inserts and dynamic range filters, but either ignore attributes altogether or treat numeric ranges as a separate partitioning structure. DIGRA and RangePQ similarly target range‑filtered dynamic workloads, but again treat time as just another numeric attribute, without explicit historic neighbour reconstruction.

2. **First‑class filtered / hybrid queries.**  
   In parallel, methods like **Filtered‑DiskANN** build label‑aware Vamana graphs with filter‑constrained pruning and per‑label entry points[cite:136]. NHQ, HQANN, and NaviX integrate attribute constraints directly into graph construction and traversal for hybrid structured‑plus‑vector search. ACORN, SIEVE, and RWalks push **predicate‑agnostic** filtered search over HNSW, demonstrating that generic graph‑level techniques can deliver strong recall/QPS trade‑offs across many filter types. However, all of these works assume a **static** notion of the graph: neighbour relations do not explicitly evolve with time, and “time” (if present) is simply another predicate.

No existing index simultaneously:

- tracks **how neighbours change over time** (as TANNS does via HNT), and  
- treats **category/label predicates as first‑class routing constraints** in the graph (as Filtered‑DiskANN/ACORN/SIEVE do for static filters).

### Gap 2: Conjunctive Temporal + Category Queries Are Real, but Poorly Supported

FANNS workloads increasingly involve **conjunctive temporal‑categorical filters**, e.g.:

- “Top‑k nearest papers in subject area `cs.AI` *before* January 2025,”  
- “Most similar products in category `Electronics` that were active last month,”  
- “Neighbour suggestions restricted to the current regulatory regime (temporal) within jurisdiction `EU` (categorical).”

The ETH **FANNS benchmark** and the associated **arxiv‑for‑fanns** datasets were explicitly designed to stress such filtered workloads, with transformer embeddings of ArXiv abstracts plus 11 real‑world attributes[cite:114][cite:133][cite:134]. However:

- Existing **filtered** methods on this dataset (e.g., Filtered‑DiskANN, ACORN, SIEVE, RWalks) either ignore time or treat it purely as another filter dimension.  
- Existing **temporal** methods (TANNS, FreshDiskANN, Dynamic SeRF) either ignore attributes or support only generic range predicates, without category‑aware graph construction or medoid routing per label.

For conjunctive queries of the form **(q, C, [t_start, t_end])**, this means:

- Time is typically handled as a **post‑filter** (HNSW+post) or **pre‑filter** (scan then ANN), which destroys the structural guarantees of either the temporal graph or the filtered graph.  
- No prior index is designed so that the **graph itself approximates how neighbours looked “around time t” inside category C**, while keeping traversal constrained to that label’s subgraph.

### Gap 3: Unified, Single-Graph Treatment of Time and Category

Several systems conceptually get close but stop short of a unified, temporal‑and‑category‑aware graph:

- **ACORN** builds a single dense HNSW graph and defines predicate subgraphs at query time via expanded neighbors‑of‑neighbors traversal, but does not track how those predicate subgraphs evolve across timestamps; time is just another attribute.  
- **Filtered‑DiskANN** ensures per‑label ST‑connectivity within a Vamana graph, but has no notion of historic neighbour lists or validity intervals.  
- **TANNS** provides exactly that historic temporal view, but it works over an HNSW‑like structure that is **label‑agnostic**.

There is therefore a clear, unfilled niche for:

> A **single** graph index that (i) guarantees navigability within each label/category, (ii) reconstructs neighbours as they existed near a query time window, and (iii) supports practical recall/QPS trade‑offs on realistic filtered workloads like arxiv‑for‑fanns‑medium.

TANNS‑C is designed to inhabit precisely this niche.

---

## Part IV: Where TANNS‑C Fits

TANNS‑C, as implemented in this repository, combines ideas from both axes:

- A **Filtered‑Vamana–style category‑aware graph** (per‑label entry points, ST‑connectivity invariants, and alpha‑blended neighbour scoring) inspired by Filtered‑DiskANN[cite:136]; and  
- **Per‑node historic neighbour trees (HNT‑style)** that track how each node’s neighbours change across time, enabling reconstruction of neighbours valid for \([t_start, t_end]\) at query time, in the spirit of TANNS[cite:74].

On top of this graph, TANNS‑C executes beam search that:

1. Starts from category‑specific medoids.  
2. Traverses only neighbours that are both:
   - valid in the query time window, and  
   - within the target category (or its label set).  
3. Can fall back to broader graph walks (ACORN‑style) when the category×time intersection is sparse (planned future extension).

In the capability matrix above, TANNS‑C would be the **first entry** with:

- ✅ Temporal evolution support (HNT‑style neighbour history), and  
- ✅ Native category / label filtering (Filtered‑Vamana‑style graph),  

within a **single navigable graph** for conjunctive temporal+category queries. No existing publication occupies that exact combination today.

---

## Part V: Selected Source Links

The following links are useful entry points to the primary literature and datasets referenced above.

| Paper / Dataset | Primary Source |
|---|---|
| TANNS (Timestamp Graph / HNT) | ICDE 2025 PDF[cite:74] |
| Filtered-DiskANN | ACM DL / PDF[cite:136] |
| ACORN | arXiv & GitHub[cite:2] |
| SeRF | SIGMOD 2024 (DL) |
| Dynamic SeRF (DSG) | VLDB 2025 (DL) |
| iRangeGraph | arXiv & GitHub |
| DIGRA | SIGMOD 2025 & GitHub |
| UNIFY | VLDB 2025 |
| FreshDiskANN / DiskANN | arXiv 2105.09613 & Microsoft GitHub[cite:136] |
| HQANN | CIKM 2022 (ACM DL) |
| NaviX | VLDB 2025 (PVLDB) |
| Curator | arXiv 2401.07119 |
| PASE | SIGMOD 2020 (ACM DL) |
| FANNS Benchmark paper | ETH SPCL publication page[cite:134] |
| SPCL/arxiv-for-fanns-medium dataset | HuggingFace dataset page[cite:114] |

This matrix is intended as a **living document**. As new temporal or filter‑aware ANN methods appear, they can be slotted into the tables and capability matrix above by filling in their temporal and filter characteristics.
