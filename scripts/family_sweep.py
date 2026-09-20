"""Two-level experiment on saved pair_features: CPM incidents on the document
graph, then families = Leiden-CPM over incident centroids (all-pairs cosine)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import igraph as ig
import numpy as np
import polars as pl
from leiden_sweep import QUAKE


def cpm(graph: ig.Graph, weights: list[float], resolution: float, seed: int = 2026) -> np.ndarray:
    ig.set_random_number_generator(random.Random(seed))
    return np.asarray(
        graph.community_leiden(
            weights=weights, objective_function="CPM", resolution=resolution, n_iterations=-1
        ).membership,
        dtype=np.int64,
    )


def report(frame: pl.DataFrame, column: str, label: str) -> None:
    total_q = frame["q"].sum()
    per = (
        frame.filter(pl.col(column) >= 0)
        .group_by(column)
        .agg(
            pl.len().alias("size"),
            pl.col("q").sum().alias("hits"),
            pl.col("publisher_country").n_unique().alias("countries"),
            pl.col("language").n_unique().alias("languages"),
        )
        .sort("hits", descending=True)
    )
    sizes = per.sort("size", descending=True)
    top = per.row(0, named=True)
    print(
        f"[{label}] groups={per.height} largest={sizes['size'][0]} "
        f"ge50={int((per['size'] >= 50).sum())} "
        f"quake: size={top['size']} hits={top['hits']} purity={top['hits'] / top['size']:.2f} "
        f"recall={top['hits'] / total_q:.2f} countries={top['countries']} langs={top['languages']}"
    )
    print("   next quake groups:", per.head(6).select("size", "hits").to_dicts()[1:])
    print("   largest groups:", sizes.head(5).select("size", "hits").to_dicts())
    rest = frame.filter((pl.col(column) == top[column]) & ~pl.col("q")).select(
        "publisher_country", "language", "title"
    )
    sample = rest.sample(min(15, rest.height), seed=5)
    for row in sample.iter_rows():
        print("     ", row)


def main() -> None:
    clusters, features, embeddings = (Path(p) for p in sys.argv[1:4])
    docs = pl.read_parquet(features / "documents.parquet").sort("document_id")
    count = docs.height
    pairs = pl.read_parquet(clusters / "pair_features.parquet")
    mins = {"title": 0.5, "event": 0.05, "url": 0.2, "entity": 0.15}
    evidence = sum((pl.col(f"{c}_score") >= v).cast(pl.Int64) for c, v in mins.items())
    gate = (evidence >= 2) | pl.col("single_channel_strong")
    edges = pairs.filter(gate & (pl.col("combined") >= 0.3)).select("left", "right", "combined")
    graph = ig.Graph(n=count, edges=edges.select("left", "right").iter_rows())
    isolated = np.asarray(graph.degree()) == 0
    incident = cpm(graph, edges["combined"].to_list(), 0.05)
    incident[isolated] = -1
    # compact ids
    _, incident = np.unique(incident, return_inverse=True)
    incident = incident - 1 if isolated.any() else incident
    n_inc = int(incident.max()) + 1
    print(f"incidents={n_inc}")

    vectors = np.load(embeddings / "title_embeddings.npy")
    ids = np.load(embeddings / "title_embedding_ids.npy")
    row_of = np.full(count, -1, dtype=np.int64)
    row_of[ids] = np.arange(len(ids))
    assigned = np.flatnonzero((incident >= 0) & (row_of >= 0))
    centroids = np.zeros((n_inc, vectors.shape[1]), dtype=np.float32)
    np.add.at(centroids, incident[assigned], vectors[row_of[assigned]])
    norms = np.linalg.norm(centroids, axis=1)
    has = norms > 0
    centroids[has] /= norms[has][:, None]

    src_all, dst_all, w_all = [], [], []
    for start in range(0, n_inc, 1024):
        block = centroids[start : start + 1024] @ centroids.T
        s, d = np.nonzero(block >= 0.5)
        keep = (s + start) < d
        src_all.append(s[keep] + start)
        dst_all.append(d[keep])
        w_all.append(block[s[keep], d[keep]])
    src, dst, w = np.concatenate(src_all), np.concatenate(dst_all), np.concatenate(w_all)
    print(f"incident-graph edges (cos>=0.5): {len(src):,}")
    igraph_inc = ig.Graph(n=n_inc, edges=list(zip(src.tolist(), dst.tolist(), strict=True)))
    frame = docs.with_columns(
        pl.Series("incident", incident),
        pl.col("title").str.contains(QUAKE).fill_null(False).alias("q"),
    )
    report(frame, "incident", "incidents CPM 0.05")
    for gamma in [0.6]:
        fam_of_inc = cpm(igraph_inc, w.tolist(), gamma)
        family = np.where(incident >= 0, fam_of_inc[np.maximum(incident, 0)], -1)
        fam = frame.with_columns(pl.Series("family", family))
        report(fam, "family", f"families CPM {gamma}")
        big = (
            fam.filter(pl.col("family") >= 0).group_by("family").len().sort("len", descending=True)
        )
        for fid, size in zip(big["family"].head(6), big["len"].head(6), strict=True):
            sub = fam.filter(pl.col("family") == fid)
            print(
                f"== family {fid} size={size} null_titles={sub['title'].null_count()} "
                f"incidents={sub['incident'].n_unique()}"
            )
            for row in (
                sub.select("publisher_country", "language", "title").sample(12, seed=2).iter_rows()
            ):
                print("      ", row)


if __name__ == "__main__":
    main()
