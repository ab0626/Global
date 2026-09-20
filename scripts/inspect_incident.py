"""Inspect the modularity partition: sample non-keyword docs from the top quake
clusters to estimate true purity beyond the keyword proxy."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import igraph as ig
import numpy as np
import polars as pl
from leiden_sweep import QUAKE


def main() -> None:
    clusters, features = Path(sys.argv[1]), Path(sys.argv[2])
    resolution = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    docs = pl.read_parquet(features / "documents.parquet").sort("document_id")
    pairs = pl.read_parquet(clusters / "pair_features.parquet")
    mins = {"title": 0.5, "event": 0.05, "url": 0.2, "entity": 0.15}
    evidence = sum((pl.col(f"{c}_score") >= v).cast(pl.Int64) for c, v in mins.items())
    gate = (evidence >= 2) | pl.col("single_channel_strong")
    edges = pairs.filter(gate & (pl.col("combined") >= 0.3)).select("left", "right", "combined")
    graph = ig.Graph(n=docs.height, edges=edges.select("left", "right").iter_rows())
    ig.set_random_number_generator(random.Random(2026))
    member = np.asarray(
        graph.community_leiden(
            weights=edges["combined"].to_list(),
            objective_function="modularity",
            resolution=resolution,
            n_iterations=-1,
        ).membership
    )
    member[np.asarray(graph.degree()) == 0] = -1
    frame = docs.with_columns(
        pl.Series("incident", member),
        pl.col("title").str.contains(QUAKE).fill_null(False).alias("q"),
    )
    per = (
        frame.filter(pl.col("incident") >= 0)
        .group_by("incident")
        .agg(
            pl.len().alias("size"),
            pl.col("q").sum().alias("hits"),
            pl.col("first_seen").min().alias("start"),
            pl.col("first_seen").max().alias("end"),
            pl.col("publisher_country").n_unique().alias("countries"),
            pl.col("language").n_unique().alias("languages"),
        )
        .sort("hits", descending=True)
    )
    print(per.head(6))
    for inc in per["incident"].head(3):
        print(f"--- incident {inc}: 20 random NON-keyword titles")
        sample = (
            frame.filter((pl.col("incident") == inc) & ~pl.col("q"))
            .select("publisher_country", "language", "title")
            .sample(20, seed=3)
        )
        for row in sample.iter_rows():
            print("   ", row)
    print("--- largest 8 incidents overall")
    big = per.sort("size", descending=True).head(8)
    for inc, size in zip(big["incident"], big["size"], strict=True):
        titles = frame.filter(pl.col("incident") == inc)["title"].drop_nulls().sample(5, seed=1)
        print(inc, size, [t[:70] for t in titles])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
