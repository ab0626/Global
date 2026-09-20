"""Embed document titles with a multilingual sentence encoder and cache the vectors.

Reads ``documents.parquet`` from ``attention.atomic`` and writes, under ``--output``:

* ``title_embeddings.npy``   float16 L2-normalised matrix, one row per embedded doc.
* ``title_embedding_ids.npy`` int64 ``document_id`` for each row.
* ``embed.json``             model, dimension, counts, runtime.

Only documents with a non-null title are embedded (no title -> no row), so the
legacy metadata channels stay the only signal for title-less documents. Identical
titles (wire copies, syndication) are encoded once and the vector is shared.
Encoding runs in shards written to ``--output/shards/``; an interrupted run resumes
from the shards already on disk. Vectors are cached per document window;
re-running with the same model is a no-op unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TITLE_TOKENS = 64
SHARD_SIZE = 20_000


def embed_titles(
    titles: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    shard_dir: Path | None = None,
) -> np.ndarray:
    """L2-normalised vectors for ``titles`` (one row each). With ``shard_dir``, every
    ``SHARD_SIZE`` titles are saved as ``shard_{i:05d}.npy`` and shards already present
    are reused, so a killed run continues where it stopped."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_TITLE_TOKENS

    def encode(chunk: list[str]) -> np.ndarray:
        if not chunk:
            dim = model.get_embedding_dimension() or 0
            return np.zeros((0, dim), dtype=np.float32)
        vectors = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    if shard_dir is None:
        return encode(titles)
    return encode_sharded(encode, titles, shard_dir)


def encode_sharded(
    encode: Callable[[list[str]], np.ndarray],
    titles: list[str],
    shard_dir: Path,
    shard_size: int = SHARD_SIZE,
) -> np.ndarray:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[np.ndarray] = []
    total = (len(titles) + shard_size - 1) // shard_size
    for index, start in enumerate(range(0, len(titles), shard_size)):
        path = shard_dir / f"shard_{index:05d}.npy"
        if path.is_file():
            shard = np.load(path).astype(np.float32)
        else:
            print(f"shard {index + 1}/{total}", flush=True)
            shard = encode(titles[start : start + shard_size])
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, shard.astype(np.float16))
            tmp.replace(path)
        shards.append(shard)
    return np.concatenate(shards) if shards else np.zeros((0, 0), dtype=np.float32)


def distinct_titles(titled: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Sorted distinct titles and, per row of ``titled``, the index of its title."""
    unique = titled.select("title").unique().sort("title")
    row_of = titled.join(
        unique.with_row_index("row"), on="title", how="left", maintain_order="left"
    )["row"].to_numpy()
    return unique, row_of


def load_embeddings(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    vectors = directory / "title_embeddings.npy"
    ids = directory / "title_embedding_ids.npy"
    if not vectors.is_file() or not ids.is_file():
        return None
    return np.load(vectors).astype(np.float32), np.load(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="attention.atomic output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "embed.json"
    if meta_path.is_file() and not args.force:
        previous = json.loads(meta_path.read_text())
        if previous.get("model") == args.model and load_embeddings(args.output) is not None:
            print(f"Cached embeddings for {args.model} found; use --force to redo.")
            return

    started = perf_counter()
    documents = pl.read_parquet(
        args.features / "documents.parquet", columns=["document_id", "title"]
    )
    titled = documents.drop_nulls("title").filter(pl.col("title").str.len_chars() > 0)
    unique, row_of = distinct_titles(titled)
    print(
        f"Embedding {unique.height} distinct titles for {titled.height} of "
        f"{documents.height} documents with {args.model}..."
    )
    unique_vectors = (
        embed_titles(unique["title"].to_list(), args.model, args.batch_size, args.output / "shards")
        if unique.height
        else np.zeros((0, 0), dtype=np.float32)
    )
    vectors = unique_vectors[row_of] if unique.height else unique_vectors
    np.save(args.output / "title_embeddings.npy", vectors.astype(np.float16))
    np.save(
        args.output / "title_embedding_ids.npy", titled["document_id"].to_numpy().astype(np.int64)
    )
    meta = {
        "model": args.model,
        "dimension": int(vectors.shape[1]) if vectors.size else 0,
        "documents": documents.height,
        "embedded": titled.height,
        "distinct_titles": unique.height,
        "title_coverage": titled.height / max(documents.height, 1),
        "runtime_seconds": perf_counter() - started,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
