"""Calibrate dense-retrieval confidence gates without running the evaluator."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from starter.vector_index import create_openai_client, load_openai_api_key


BROAD_CATEGORIES = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|fabric)\b", re.I
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)


def _clean(value: object, limit: int = 140) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" -;,.\t\n")[:limit].rstrip()


def _category(values: object) -> str:
    cleaned: list[str] = []
    for value in values if isinstance(values, list) else []:
        for part in str(value).split(","):
            part = _clean(part)
            if part and part.casefold() not in BROAD_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _requirements(product: dict) -> list[str]:
    searchable = " ".join(
        str(value)
        for field in ("title", "features", "details", "description")
        for value in (
            product.get(field, {}).values()
            if isinstance(product.get(field), dict)
            else product.get(field, [])
            if isinstance(product.get(field), list)
            else [product.get(field)]
        )
        if value
    )
    candidates: list[str] = []
    if material := MATERIAL_RE.search(searchable):
        candidates.append(material.group(1).lower())
    if color := COLOR_RE.search(searchable):
        candidates.append(f"color: {color.group(1).lower()}")
    candidates.extend(
        _clean(value)
        for value in product.get("features") or []
        if _clean(value)
    )
    return list(dict.fromkeys(candidates))[:2]


def _query(product: dict) -> str | None:
    requirements = _requirements(product)
    if not requirements:
        return None
    return (
        f"Product category: {_category(product.get('categories'))}\n"
        f"Required features: {'; '.join(requirements)}"
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    return {
        name: round(float(np.quantile(values, quantile)), 6)
        for name, quantile in (("p05", 0.05), ("p10", 0.10), ("p50", 0.50),
                               ("p90", 0.90), ("p95", 0.95))
    }


def calibrate(args: argparse.Namespace) -> dict:
    if not load_openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is unavailable")

    with Path(args.catalog).open(encoding="utf-8") as handle:
        products = [json.loads(line) for line in handle if line.strip()]
    vectors = np.load(args.vectors, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    if len(products) != len(vectors):
        raise RuntimeError("catalog and embedding row counts differ")

    eligible = [(index, query) for index, product in enumerate(products)
                if (query := _query(product))]
    sample_count = min(max(8, int(args.sample_size)), len(eligible))
    selected = [eligible[index] for index in np.linspace(
        0, len(eligible) - 1, sample_count, dtype=np.int64
    )]
    queries = [query for _, query in selected]

    client = create_openai_client()
    response = client.embeddings.create(
        input=queries,
        model=str(metadata["model"]),
        dimensions=int(metadata["dimensions"]),
        encoding_format="float",
    )
    query_vectors = np.asarray(
        [item.embedding for item in sorted(response.data, key=lambda item: int(item.index))],
        dtype=np.float32,
    )
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    target_scores: list[float] = []
    alternative_scores: list[float] = []
    winning_target_scores: list[float] = []
    winning_margins: list[float] = []
    for (target_index, _), query_vector in zip(selected, query_vectors):
        scores = np.asarray(vectors @ query_vector).copy()
        target_score = float(scores[target_index])
        scores[target_index] = -np.inf
        alternative_score = float(np.max(scores))
        target_scores.append(target_score)
        alternative_scores.append(alternative_score)
        if target_score > alternative_score:
            winning_target_scores.append(target_score)
            winning_margins.append(target_score - alternative_score)

    if len(winning_target_scores) < 8:
        raise RuntimeError("too few high-confidence calibration queries")

    similarity_threshold = float(np.quantile(winning_target_scores, 0.10))
    margin_threshold = float(np.quantile(winning_margins, 0.10))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "sample": "deterministic evenly spaced catalog products with descriptive fields",
            "similarity_gate": "10th percentile of target cosine when target beats every alternative",
            "margin_gate": "10th percentile of target-minus-best-alternative margin for those wins",
            "evaluator_run": False,
        },
        "sample_count": sample_count,
        "target_wins": len(winning_target_scores),
        "distributions": {
            "target_cosine": _quantiles(target_scores),
            "best_alternative_cosine": _quantiles(alternative_scores),
            "winning_target_cosine": _quantiles(winning_target_scores),
            "winning_margin": _quantiles(winning_margins),
        },
        "selected_thresholds": {
            "minimum_cosine_similarity": round(similarity_threshold, 6),
            "minimum_top_margin": round(margin_threshold, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--vectors", default="data/catalog_embeddings.npy")
    parser.add_argument("--metadata", default="data/catalog_embeddings.meta.json")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--output", default="docs/vector_gate_calibration.json")
    args = parser.parse_args()
    result = calibrate(args)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} from {result['sample_count']} diagnostic queries")


if __name__ == "__main__":
    main()
