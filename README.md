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

The pipeline has five stages:

1. `starter.dialogue.SessionState` turns each message into weighted positive
   evidence, tracks requested attributes, ignores explicit no-preference
   answers, and removes superseded opening preferences on intent override.
2. `starter.retrieval.CatalogSearch` generates candidates through accumulated
   keyword, exact-phrase, and category routes using a weighted SQLite FTS5
   index.
3. Reciprocal-rank fusion combines the routes without assuming their raw
   scores are calibrated.
4. A deterministic reranker scores constraint coverage, exact metadata
   phrases, budget proximity, a small aggregate-profile match, and a
   log-scaled product-popularity prior.
5. An adaptive question planner measures how much the live candidates differ
   across material, colour, size, style, use case, price, brand, category, and
   features. It selects the facet with the greatest estimated information gain
   and generates the question from observed candidate values. There is no
   fixed question order or per-attribute question-text dictionary.

The current public-set results are:

| Agent | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Released BM25 baseline | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |
| Stateful adaptive multi-route agent | **0.995** | **0.721635** | **1.845** | **0.9155** | **0.897091** |

Scenario Hit Rate@10 is `0.9875` for Buying and `1.0` for Browsing, Intent
Override, and Boundary. These are development-set measurements, not estimates
of the private leaderboard score. The method does not memorize public target
identifiers; public labels are used only by the evaluator and optional
diagnostic script.

For comparison, the earlier fixed clarification sequence scored `0.899689`.
The adaptive policy gives up `0.002598` TechnicalScore while preserving the
same `0.995` Hit Rate, in exchange for questions that react to the current
candidate pool rather than assuming one predetermined conversation path.

### Reproduce the Results

```bash
python -m unittest discover -s tests -v
python -m evaluator.local_evaluator
```

For a turn-by-turn inspection of one labelled development session:

```bash
python -m scripts.analyze_session public_0053
```

The diagnostic script is development-only and is not imported by the Agent.

### Cost, Latency, and Limitations

- Model/API cost and reported token usage are zero. The evaluated path is
  deterministic and standard-library-only.
- On the development machine with Python 3.13.5, building the 50,000-product
  in-memory index took approximately 1.81 seconds. A small ten-turn benchmark
  averaged approximately 163 ms per response with adaptive question analysis.
  The complete 200-session public evaluator, including its own catalog loading
  and Agent startup, took about 62 seconds. These measurements are
  hardware-dependent.
- The released simulator reveals constraints copied from catalog metadata, so
  exact phrases are especially informative. More varied real customer language
  would benefit from an optional local semantic-retrieval route.
- Very broad categories paired only with generic attributes can remain
  intrinsically ambiguous. The public run misses one such session, and private
  performance may be lower than the development score.
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
