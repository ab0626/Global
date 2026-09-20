"""Re-run only gating + Leiden from a saved pair_features.parquet to tune the
evidence gate and the Leiden objective without recomputing candidates.

usage: uv run python scripts/leiden_sweep.py data/clusters/20230206 data/features/20230206
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import igraph as ig
import numpy as np
import polars as pl

QUAKE = (
    r"(?i)earthquake|quake|erdbeben|s[ée]isme|terremoto|sismo|deprem"
    r"|地震|землетрясен|zemětřesení|tremblement"
)


def evaluate(membership: np.ndarray, docs: pl.DataFrame, label: str) -> None:
    n = len(membership)
    m = pl.Series("incident", membership)
    frame = docs.with_columns(m)
    assigned = frame.filter(pl.col("incident") >= 0)
    sizes = assigned.group_by("incident").len().sort("len", descending=True)
    quake = frame.with_columns(pl.col("title").str.contains(QUAKE).fill_null(False).alias("q"))
    total_q = quake["q"].sum()
    per = (
        quake.filter(pl.col("incident") >= 0)
        .group_by("incident")
        .agg(pl.len().alias("size"), pl.col("q").sum().alias("hits"))
        .sort("hits", descending=True)
    )
    top = per.row(0, named=True)
    print(
        f"[{label}] incidents={sizes.height} unassigned={n - assigned.height} "
        f"largest={sizes['len'][0]} ge50={int((sizes['len'] >= 50).sum())} "
        f"quake-cluster size={top['size']} hits={top['hits']} "
        f"purity={top['hits'] / top['size']:.2f} recall={top['hits'] / max(total_q, 1):.2f}"
    )
    print("  top quake incidents:", per.head(5).to_dicts())
    ex = (
        quake.filter(pl.col("incident") == top["incident"])
        .select("publisher_country", "language", "title")
        .sample(min(12, top["size"]), seed=1)
    )
    for row in ex.iter_rows():
        print("   ", row)


def main() -> None:
    clusters, features = Path(sys.argv[1]), Path(sys.argv[2])
    docs = pl.read_parquet(features / "documents.parquet").sort("document_id")
    count = docs.height
    pairs = pl.read_parquet(clusters / "pair_features.parquet")
    mins = {"title": 0.5, "event": 0.05, "url": 0.2, "entity": 0.15}
    evidence = sum((pl.col(f"{c}_score") >= v).cast(pl.Int64) for c, v in mins.items())
    pairs = pairs.with_columns(evidence.alias("evidence2"))
    gate = (pl.col("evidence2") >= 2) | pl.col("single_channel_strong")
    edges = pairs.filter(gate & (pl.col("combined") >= 0.3)).select("left", "right", "combined")
    print(f"edges: old={pairs.filter(pl.col('gated') >= 0.3).height:,} new={edges.height:,}")
    graph = ig.Graph(n=count, edges=edges.select("left", "right").iter_rows())
    isolated = np.asarray(graph.degree()) == 0
    weights = edges["combined"].to_list()
    for objective, res in [("modularity", 5.0), ("modularity", 20.0), ("CPM", 0.02), ("CPM", 0.05)]:
        ig.set_random_number_generator(random.Random(2026))
        member = np.asarray(
            graph.community_leiden(
                weights=weights, objective_function=objective, resolution=res, n_iterations=-1
            ).membership,
            dtype=np.int64,
        )
        member[isolated] = -1
        evaluate(member, docs, f"{objective} r={res}")


if __name__ == "__main__":
    main()
