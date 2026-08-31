# SEAM — Stateful E-commerce Agent for Matching

SEAM is an offline-first conversational shopping agent for the TechJam Conversational E-Commerce Search Challenge. It turns a short, evolving shopper conversation into ranked products from a 50,000-item Amazon catalog—without requiring an API key, model download, or network access during evaluation.

## Results

Public development-set results on 200 released sessions:

| Agent | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Released BM25 baseline | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |
| **SEAM** | **1.000** | **0.972000** | **2.070** | **0.893** | **0.970200** |

SEAM reaches a 100% Hit Rate@10 across Buying, Browsing, Intent Override, and Boundary scenarios. These are public development measurements, not private leaderboard estimates.

## Why SEAM

Traditional catalog search treats every query as a static keyword lookup. SEAM instead maintains an explicit shopping state: it separates hard requirements from soft preferences, asks useful follow-up questions, and rewrites the active intent whenever the customer changes their mind.

```text
Customer message
      │
      ▼
Session state ──► Intent router ──► Multi-route retrieval ──► Evidence-aware ranking
      │                  │                    │                          │
      │                  ├─ Buying             ├─ exact constraints       ├─ hard-match tiers
      │                  └─ Browsing           ├─ phrase / keyword        ├─ category & sequence
      ▼                                       └─ category routes          └─ adaptive output breadth
Adaptive clarification ◄───────────────────────────────────────────────────────────────┘
```

## Architecture

### 1. Intent routing and hybrid retrieval

- **Buying mode** protects explicit requirements with exact constraint intersections and strict hard-constraint ranking tiers.
- **Browsing mode** keeps discovery broad with keyword, phrase, and category retrieval routes.
- **Offline wording-variant mode** expands a small one-way synonym map only for out-of-vocabulary wording and repairs only high-confidence typos. It shortlists catalog terms with boundary-aware character trigrams, verifies them with bounded token edit similarity, and rejects ambiguous corrections.
- **Simulator-likelihood reranking** reconstructs each candidate's catalog-derived intent slots and rewards exact disclosed phrases, expected hard/soft placement, field provenance, and detail order while penalizing contradictions and missing generated values.
- **Reciprocal Rank Fusion** merges routes without assuming their raw scores are comparable.
- A SQLite FTS5 index and in-memory exact-constraint maps keep search local, fast, and reproducible.

### 2. Multi-turn dialogue strategy

`SessionState` converts messages into weighted evidence and supports append, replace, and exclusion operations. It handles both gradual preference accumulation and abrupt intent overrides such as “actually, ignore that.”

The question planner compares live candidates across material, colour, size, style, use case, budget, brand, category, and feature facets. It asks for the facet with the strongest expected value instead of using a fixed question script.

### 3. Dynamic context programming

SEAM continuously rebuilds the active query from the current dialogue state:

- new constraints accumulate when compatible;
- replacements remove superseded values for that attribute;
- exclusions remove conflicting positive evidence;
- profile tags provide a bounded, non-conflicting tie-break.

This keeps retrieval aligned with the customer’s latest intent rather than treating the first message as permanent.

### 4. Evidence-aware ranking and efficient conversion

The final ranking combines exact constraint coverage, category specificity, simulator-likelihood, cohesive detail sequences, lexical relevance, budget fit, and bounded profile and popularity signals. The simulator tier sits below exact hard-constraint matching and above popularity. An adaptive recommendation policy uses rank margin, entropy, constraint coverage, question value, and remaining turns to decide whether to return one confident match, a shortlist, or broader recall.

## Design choices

SEAM intentionally uses a deterministic offline path as its default:

- The public simulator exposes catalog-like metadata phrases, making exact constraint matching more reliable than an unconstrained semantic guess.
- The default path has zero token usage, no credentials, and works when network access is unavailable.
- An optional in-memory embedding experiment exists, but it is disabled by default because the evaluated lexical/constraint route is more reliable on the released development set.

This is a deliberate feasibility choice, not a dependency limitation: semantic reranking can be added behind the retrieval interface when varied natural language or an approved local model is available.

## Quick start

### Prerequisites

- Python 3.10+
- The frozen catalog at `data/catalog.jsonl`

Download the catalog archive from the participant release, then decompress it:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the catalog with the supplied SHA256 checksum before evaluation.

### Reproduce the public result

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.build_catalog_index
python -m evaluator.local_evaluator --output results.json
```

The evaluator writes aggregate and per-session metrics to `results.json`. Do not modify the evaluator or public labels when reporting results.

`scripts.build_catalog_index` is an optional preprocessing step. It creates a read-only SQLite artifact with FTS5 and exact-constraint maps; if that artifact is unavailable, the agent safely rebuilds equivalent indices in memory.

## Demo frontend

The SEAM frontend uses the same agent contract, session state, question planning, retrieval, and ranking logic evaluated above.

```bash
python -m backend.api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The local backend keeps a warm agent instance, resets browser sessions through the published contract, and enriches returned ASINs using the same frozen catalog.

## Optional semantic route

The repository includes an experimental, opt-in in-memory vector reranker. It is not used for the reported result and requires a local embedding artifact.

```bash
python -m scripts.generate_catalog_embeddings
```

```python
from starter.agent import Agent
from starter.config import AgentConfig

agent = Agent(config=AgentConfig(enable_vector_reranker=True))
```

If the artifact, credentials, or network are unavailable, the agent continues using its offline retrieval pipeline. Do not commit API keys or generated model artifacts.

## Evaluation metrics

```text
TechnicalScore = 0.50 × Hit Rate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

- **Hit Rate@10**: the target appears in the first ten valid recommendations.
- **MRR**: reciprocal rank of the target at its first successful turn.
- **MTTC**: mean turn at which the target first appears.

Only exact `parent_asin` equality is scored.

## Limitations

- The released simulator often reveals exact catalog metadata; real customer language with typos, synonyms, or vague descriptions would benefit from a semantic reranking route.
- Some catalog siblings share the same disclosed metadata, making the exact purchased parent ASIN intrinsically ambiguous.
- The catalog is static, text-only, and limited to Clothing, Shoes & Jewelry.

## Contributions

| Name | Contribution |
|---|---|
| Naren | _Add contribution_ |
| Anson | _Led the development and optimisation of SEAM’s offline conversational product-search engine, implementing constraint-aware retrieval, adaptive ranking, confidence-driven recommendations, dynamic clarification planning, a preprocessed catalogue index and built extensive evaluation and testing infrastructure._ |
| Joseph | _Contributed to Runtime Adaptation by leveraging accumulated dialog history to perform Personalized Context Distillation, continuously updating short-term session states and long-term user profiles._ |
| Harry | _Add contribution_ |

## Repository map

```text
starter/                 Agent, dialogue state, ranking, retrieval, and indexing
evaluator/               Deterministic public-set evaluator
scripts/                 Index build and development calibration utilities
backend/                 Local API for the demo frontend
frontend/                SEAM demo experience
data/                    Frozen catalog and released public sessions
docs/                    API contract, rules, calibration, and reproducibility notes
```

## References and data attribution

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for dataset terms and attribution, and [docs/submission_rules.md](docs/submission_rules.md) for submission rules.
