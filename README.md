# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

The Agent implementation is in `starter/`. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Implemented Agent

This repository now includes a stateful, offline-first conversational search
agent. It uses only the Python standard library and requires no API key, model
download, or network access during evaluation.

The pipeline has seven stages:

1. `starter.dialogue.SessionState` turns each message into weighted preference
   evidence, tracks requested attributes, ignores explicit no-preference
   answers, and applies replace, append, or exclude operations to each
   attribute's active value set.
2. `starter.constraint_index.ConstraintIndex` derives exact feature, detail,
   material, colour, and category inverted maps from the catalog at startup.
   Intersections from these maps form an uncapped candidate route and an exact
   ranking tier; no evaluator labels or target identifiers are used.
3. `starter.retrieval.CatalogSearch` generates additional candidates through accumulated
   keyword, exact-phrase, and category routes using a weighted SQLite FTS5
   index.
4. Reciprocal-rank fusion combines the lexical routes without assuming their raw
   scores are calibrated.
5. A deterministic reranker scores constraint coverage, exact metadata
   phrases, budget proximity, a small aggregate-profile match, and a
   log-scaled product-popularity prior. In Buying mode, an independently
   calibrated rating-history alignment provides a bounded tie-break below the
   hard-constraint and category tiers. Across both routing modes, a cohesive
   sequence tier prefers products whose distinct disclosed details occur
   together in catalog order; it remains below hard-constraint and category
   tiers and does not use evaluator labels. Product text is normalized once on
   first retrieval and retained in a bounded 5,000-entry LRU feature cache;
   query evidence is compiled once per turn and reused for every candidate.
6. An adaptive question planner measures how much the live candidates differ
   across material, colour, size, style, use case, price, brand, category, and
   features. It returns the selected attribute and message together with raw
   information gain, estimated answerability, and their expected value. There
   is no fixed question order or per-attribute question-text dictionary.
7. An immutable recommendation policy chooses output breadth from the live
   Top-1/Top-2 margin, normalized candidate entropy, exact hard-constraint
   coverage, clarification expected value, and the remaining turn budget. It
   stays narrow for a decisive winner or a high-value next question, exposes a
   five-item shortlist for a low-value question, and uses full breadth when a
   low-value question coincides with an ambiguous ranking. The runtime policy
   never reads evaluator scenario labels or target identifiers.

The current public-set results are:

| Agent | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Released BM25 baseline | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |
| Stateful agent with constraint index, profile alignment, and cohesive-detail ranking | **1.000** | **0.963173** | **2.120** | **0.8880** | **0.966552** |

Scenario Hit Rate@10 is `1.0` for Buying, Intent Override, Browsing, and
Boundary. These are public development-set measurements, not estimates of the
private leaderboard score. The method does not memorize public target
identifiers; public labels are used only by the evaluator and optional
diagnostic/calibration scripts.

### Reproduce the Results

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.build_catalog_index
python -m evaluator.local_evaluator
```

`scripts.build_catalog_index` preprocesses the frozen catalog into
`data/catalog_index.sqlite3`, including both FTS5 and the six exact-constraint
maps. Runtime verifies the catalog SHA-256 and schema version before opening
the artifact read-only. If it is absent, corrupt, or stale, the agent safely
rebuilds the same indices in memory, so preprocessing changes startup cost but
not ranking behavior.

The breadth policy is calibrated with a scenario-stratified 160/40
development/validation split. The script records each ranking trajectory once,
compares the fixed `(1,1,3)`, `(1,2,3)`, `(1,3,5)`, `(1,1,5)`, and `(1,3,10)`
schedules plus full breadth, searches 1,944 adaptive margin/entropy policy
configurations, and compares 14 focused question-value cutoff/width variants.
Only the development fold selects the runtime parameters; the validation fold
is reported untouched in
`docs/recommendation_breadth_calibration.json`.

Reproduce the calibration with:

```bash
python -m scripts.calibrate_recommendation_breadth
```

### Optional Semantic Vector Route

The evaluated default is deterministic and offline: it does not construct a
vector index or call an embedding API. The retrieval pipeline can optionally
add exact in-memory cosine search over normalized OpenAI catalog embeddings.
Generate the local artifact once with:

```bash
python -m scripts.generate_catalog_embeddings
```

Then opt into the experiment explicitly:

```python
from starter.agent import Agent
from starter.config import AgentConfig

agent = Agent(config=AgentConfig(enable_vector_reranker=True))
```

The command uses `text-embedding-3-small` with 256 dimensions, resumes from a
completed batch after interruption, and writes an ignored approximately 49 MiB
`data/catalog_embeddings.npy` file plus checked metadata. At runtime, active
intent is filtered and rendered as one structured category/features/use-case
query and embedded once. Raw cosine similarity can then make a capped adjustment
among close lexical candidates only when calibrated similarity and margin gates
pass and the candidate matches the requested category. Exact hard-constraint
matches and lexical leads larger than the semantic cap are protected.
Category-only or still-exploring queries skip the vector route.
`OPENAI_API_KEY` may be supplied through
the environment or an ignored `.env` file; the existing `OPENAI_APIKEY` alias
is also accepted for compatibility. If credentials, network access, or
the validated artifact are unavailable, the agent continues with its existing
offline retrieval routes.

On hosts where Python 3.13 rejects an older enterprise CA solely because its
Basic Constraints extension is not marked critical, set
`OPENAI_SYSTEM_CA_COMPAT=1`. This retains certificate-chain and hostname
verification while disabling only Python's X.509 strict compatibility flag.

The checked calibration artifact is `docs/vector_gate_calibration.json`. It can
be regenerated without running the evaluator:

```bash
python -m scripts.calibrate_vector_gates
```

For a turn-by-turn inspection of one labelled development session:

```bash
python -m scripts.analyze_session public_0053
```

The diagnostic script is development-only and is not imported by the Agent.

### Cost, Latency, and Limitations

- Model/API cost and reported token usage are zero. The evaluated path is
  deterministic and standard-library-only.
- On the development machine, preprocessing the 50,000-product catalog took
  31.0 seconds and produced a 200.9 MiB ignored SQLite artifact. Opening the
  verified artifact took approximately 0.46 seconds, compared with 15.8
  seconds for the in-memory fallback. Generate it during untimed setup; if
  setup time is included and only one Agent is constructed, the fallback is
  faster overall. A ten-turn benchmark averaged approximately 172 ms per
  response with adaptive question analysis. On a
  deterministic 20-session slice, a 5,000-product feature cache reduced
  evaluation time from 12.654 seconds with effectively no reuse to 9.220
  seconds, a 27.1% reduction with identical metrics. These measurements are
  hardware-dependent.
- The released simulator reveals constraints copied from catalog metadata, so
  exact phrases are especially informative. More varied real customer language
  would benefit from an optional local semantic-retrieval route.
- Very broad categories paired only with generic attributes can remain
  intrinsically ambiguous, and private performance may be lower than the
  development score.
- The popularity feature is log-scaled and subordinate to textual constraints,
  but it can still favor established products when several candidates are
  otherwise indistinguishable.

### Design References

- [SQLite FTS5](https://www.sqlite.org/fts5.html) documents phrase queries,
  column weights, and the sign/order semantics of its BM25 implementation.
- Cormack, Clarke, and Buettcher's
  [Reciprocal Rank Fusion paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
  motivates robust rank-based fusion of heterogeneous retrieval routes.
- The [Sentence Transformers retrieve-and-rerank
  documentation](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  supports the two-stage candidate-generation/reranking architecture. A dense
  model is deliberately not required here so the official path stays fully
  offline and reproducible.
- Aliannejadi et al.'s [clarifying-question retrieval
  framework](https://arxiv.org/abs/1907.06554) motivates treating question
  selection as part of retrieval rather than as free-form chat generation.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
