from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from openai import OpenAI

from starter.vector_index import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    catalog_sha256,
    create_openai_client,
    load_openai_api_key,
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(_text(item) for item in value)
    return str(value)


def product_embedding_text(product: dict) -> str:
    fields = (
        ("Title", product.get("title")),
        ("Categories", product.get("categories")),
        ("Features", product.get("features")),
        ("Details", product.get("details")),
        ("Store", product.get("store")),
        ("Description", product.get("description")),
    )
    return "\n".join(f"{name}: {_text(value)}" for name, value in fields if _text(value))


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _catalog_info(path: Path) -> tuple[int, str]:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count, catalog_sha256(path)


def _embed_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimensions: int,
    attempts: int = 6,
) -> object:
    for attempt in range(attempts):
        try:
            return client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
                encoding_format="float",
            )
        except Exception:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError("unreachable")


def generate(args: argparse.Namespace) -> None:
    catalog_path = Path(args.catalog)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    partial_path = output_path.with_name(output_path.name + ".partial.npy")
    progress_path = output_path.with_name(output_path.name + ".progress.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count, checksum = _catalog_info(catalog_path)
    identity = {
        "model": args.model,
        "dimensions": args.dimensions,
        "row_count": row_count,
        "catalog_sha256": checksum,
        "normalized": True,
    }

    if output_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()):
            print(f"Embedding artifact is already complete: {output_path}")
            return
        raise RuntimeError("existing embedding artifact does not match requested configuration")

    completed_rows = 0
    prompt_tokens = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not all(progress.get(key) == value for key, value in identity.items()):
            raise RuntimeError("partial embedding artifact does not match requested configuration")
        completed_rows = int(progress.get("completed_rows", 0))
        prompt_tokens = int(progress.get("prompt_tokens", 0))
        vectors = np.load(partial_path, mmap_mode="r+", allow_pickle=False)
        if vectors.shape != (row_count, args.dimensions) or vectors.dtype != np.float32:
            raise RuntimeError("partial embedding matrix has an invalid shape or dtype")
        print(f"Resuming at catalog row {completed_rows:,}")
    else:
        vectors = open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, args.dimensions),
        )
        _atomic_json(
            progress_path,
            {**identity, "completed_rows": 0, "prompt_tokens": 0},
        )

    if not load_openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    client = create_openai_client()

    batch: list[str] = []
    batch_start = completed_rows
    with catalog_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < completed_rows:
                continue
            batch.append(product_embedding_text(json.loads(line)))
            if len(batch) < args.batch_size and index + 1 < row_count:
                continue

            response = _embed_with_retry(client, batch, args.model, args.dimensions)
            items = sorted(response.data, key=lambda item: int(item.index))
            if len(items) != len(batch):
                raise RuntimeError("embedding response length mismatch")
            matrix = np.asarray([item.embedding for item in items], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
                raise RuntimeError("embedding response contains an invalid vector")
            matrix /= norms
            end = batch_start + len(batch)
            vectors[batch_start:end] = matrix
            vectors.flush()
            usage = getattr(response, "usage", None)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completed_rows = end
            _atomic_json(
                progress_path,
                {
                    **identity,
                    "completed_rows": completed_rows,
                    "prompt_tokens": prompt_tokens,
                },
            )
            print(f"Embedded {completed_rows:,}/{row_count:,} rows", flush=True)
            batch.clear()
            batch_start = completed_rows

    if completed_rows != row_count:
        raise RuntimeError(f"expected {row_count} vectors, generated {completed_rows}")
    del vectors
    os.replace(partial_path, output_path)
    _atomic_json(
        metadata_path,
        {
            **identity,
            "prompt_tokens": prompt_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    progress_path.unlink(missing_ok=True)
    print(f"Wrote {output_path} and {metadata_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normalized catalog embeddings")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/catalog_embeddings.npy")
    parser.add_argument("--metadata", default="data/catalog_embeddings.meta.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.dimensions <= 0 or args.batch_size <= 0:
        parser.error("dimensions and batch-size must be positive")
    return args


if __name__ == "__main__":
    generate(parse_args(sys.argv[1:]))
