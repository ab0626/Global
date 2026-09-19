"""Blocked document clustering for one day: event-ID, URL-token and GKG-entity channels.

Nodes are wire-deduplicated documents (one per ``content_fingerprint`` from
``attention.atomic``). Three sparse TF-IDF channels are built, candidates are
retrieved per channel with top-k neighbours (never a global O(N^2) comparison),
pairs are scored with the v2 corroboration gate (>= 2 channels agree, or one
channel is very strong) and the sparse graph is partitioned with Leiden.

Outputs ``document_clusters.parquet`` (document_id, node, cluster), ``pairs.parquet``
and ``audit.json``. Cluster ids are arbitrary; ``cluster == -1`` marks documents
that received no edge above threshold (unassigned), which are kept explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlsplit

import igraph as ig
import numpy as np
import polars as pl
from scipy import sparse

from clustering_experiment import ArticleSettings, channel_candidates, tfidf_channel, url_words
from clustering_v2 import V2Settings, clean_token, host_filter


@dataclass(frozen=True)
class ClusterSettings:
    threshold: float = 0.3
    resolution: float = 1.0
    seed: int = 2026
    max_entity_share: float = 0.1
    channels: str = "event,url,entity"


def slug_path(url: str) -> str:
    try:
        return unquote(urlsplit(url).path).lower()
    except ValueError:
        return ""


def node_table(documents: pl.DataFrame) -> pl.DataFrame:
    """One node per content fingerprint; the earliest document is the representative."""
    return (
        documents.sort("first_seen", "document_id")
        .group_by("content_fingerprint", maintain_order=True)
        .agg(
            pl.col("document_id").first().alias("representative"),
            pl.col("document_id").alias("document_ids"),
        )
        .with_row_index("node")
    )


def feature_matrix(rows: pl.DataFrame, count: int) -> sparse.csr_matrix:
    pairs = rows.select("node", "feature").unique()
    vocabulary = {f: i for i, f in enumerate(pairs["feature"].unique(maintain_order=True))}
    columns = np.fromiter(
        (vocabulary[f] for f in pairs["feature"]), dtype=np.int64, count=pairs.height
    )
    return sparse.csr_matrix(
        (np.ones(pairs.height), (pairs["node"].to_numpy(), columns)),
        shape=(count, len(vocabulary)),
    )


def event_rows(links: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    return (
        links.join(membership, on="document_id")
        .select("node", pl.col("GlobalEventID").cast(pl.String).alias("feature"))
        .unique()
    )


def url_rows(
    documents: pl.DataFrame, membership: pl.DataFrame, settings: V2Settings
) -> pl.DataFrame:
    frame = documents.join(membership, on="document_id").select(
        "node", "domain", "canonical_url", pl.col("first_seen").dt.date().alias("day")
    )
    rows = [
        (node, domain, day, word)
        for node, domain, url, day in frame.iter_rows()
        for word in url_words(slug_path(url))
        if clean_token(word)
    ]
    table = pl.DataFrame(
        rows,
        schema={"node": pl.UInt32, "domain": pl.String, "day": pl.Date, "feature": pl.String},
        orient="row",
    ).unique()
    table = table.rename({"node": "vertex"})
    table, _ = host_filter(table, settings)
    return table.rename({"vertex": "node"}).select("node", "feature")


def entity_rows(
    documents: pl.DataFrame, membership: pl.DataFrame, max_share: float
) -> pl.DataFrame:
    docs = documents.join(membership, on="document_id").filter(pl.col("has_gkg"))
    parts = [
        docs.select("node", "persons")
        .explode("persons")
        .select("node", (pl.lit("p:") + pl.col("persons")).alias("feature")),
        docs.select("node", "organizations")
        .explode("organizations")
        .select("node", (pl.lit("o:") + pl.col("organizations")).alias("feature")),
        docs.select("node", "locations")
        .explode("locations")
        .select(
            "node",
            (pl.lit("l:") + pl.col("locations").struct.field("name").str.to_lowercase()).alias(
                "feature"
            ),
        ),
    ]
    rows = pl.concat(parts).drop_nulls().unique()
    frequent = (
        rows.group_by("feature").len().filter(pl.col("len") > max_share * docs["node"].n_unique())
    )
    return rows.join(frequent.select("feature"), on="feature", how="anti")


def score(
    channels: dict[str, sparse.csr_matrix], count: int, article: ArticleSettings, v2: V2Settings
) -> tuple[pl.DataFrame, dict]:
    retrieved = {name: channel_candidates(m, article) for name, m in channels.items()}
    keys = np.unique(np.concatenate(list(retrieved.values())))
    left, right = keys // count, keys % count
    frame = pl.DataFrame({"left": left, "right": right})
    available = np.zeros(len(keys), dtype=np.int64)
    evidence = np.zeros(len(keys), dtype=np.int64)
    total = np.zeros(len(keys), dtype=np.float64)
    best = np.zeros(len(keys), dtype=np.float64)
    for name, matrix in channels.items():
        has = np.asarray(matrix.getnnz(axis=1) > 0).ravel()
        scores = np.zeros(len(keys), dtype=np.float64)
        batch = article.scoring_batch_size
        for start in range(0, len(keys), batch):
            stop = start + batch
            scores[start:stop] = np.asarray(
                matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
            ).ravel()
        scores = np.clip(scores, 0, 1)
        frame = frame.with_columns(pl.Series(f"{name}_score", scores))
        available += (has[left] & has[right]).astype(np.int64)
        evidence += (scores > 0).astype(np.int64)
        total += scores
        best = np.maximum(best, scores)
    combined = total / np.maximum(available, 1)
    gated = np.where(
        (evidence >= v2.corroboration_min_channels) | (best >= v2.single_channel_floor),
        combined,
        0.0,
    )
    frame = frame.with_columns(
        pl.Series("available_channels", available),
        pl.Series("evidence_channels", evidence),
        pl.Series("combined", combined),
        pl.Series("gated", gated),
    )
    return frame, {
        "directed_neighbors": {k: int(len(v)) for k, v in retrieved.items()},
        "candidate_pairs": frame.height,
        "pairs_single_channel": int((evidence == 1).sum()),
    }


def leiden(pairs: pl.DataFrame, count: int, settings: ClusterSettings) -> np.ndarray:
    selected = pairs.filter(pl.col("gated") >= settings.threshold)
    graph = ig.Graph(n=count, edges=selected.select("left", "right").iter_rows())
    ig.set_random_number_generator(random.Random(settings.seed))
    membership = np.asarray(
        graph.community_leiden(
            weights=selected["gated"].to_list(),
            objective_function="modularity",
            resolution=settings.resolution,
            n_iterations=-1,
        ).membership,
        dtype=np.int64,
    )
    isolated = np.asarray(graph.degree()) == 0
    membership[isolated] = -1
    return membership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="attention.atomic output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=ClusterSettings.threshold)
    parser.add_argument("--resolution", type=float, default=ClusterSettings.resolution)
    parser.add_argument("--channels", default=ClusterSettings.channels)
    parser.add_argument("--neighbors", type=int, default=ArticleSettings.neighbors)
    args = parser.parse_args()
    settings = ClusterSettings(
        threshold=args.threshold, resolution=args.resolution, channels=args.channels
    )
    article = ArticleSettings(neighbors=args.neighbors, max_feature_articles=10**9)
    v2 = V2Settings()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    documents = pl.read_parquet(args.features / "documents.parquet")
    links = pl.read_parquet(args.features / "document_events.parquet")
    nodes = node_table(documents)
    membership = (
        nodes.select("node", "document_ids")
        .explode("document_ids")
        .rename({"document_ids": "document_id"})
    )
    count = nodes.height
    builders = {
        "event": lambda: event_rows(links, membership),
        "url": lambda: url_rows(documents, membership, v2),
        "entity": lambda: entity_rows(documents, membership, settings.max_entity_share),
    }
    channels: dict[str, sparse.csr_matrix] = {}
    audit: dict = {
        "command": sys.argv,
        "settings": asdict(settings),
        "nodes": count,
        "channels": {},
    }
    for name in settings.channels.split(","):
        print(f"Building channel: {name}...", flush=True)
        rows = builders[name]()
        channels[name], audit["channels"][name] = tfidf_channel(
            feature_matrix(rows, count), article
        )
    print("Retrieving and scoring candidates...", flush=True)
    pairs, pair_audit = score(channels, count, article, v2)
    audit["pairs"] = pair_audit
    print("Leiden...", flush=True)
    clusters = leiden(pairs, count, settings)
    assignment = nodes.with_columns(pl.Series("cluster", clusters)).select(
        "node", "cluster", "document_ids"
    )
    out = assignment.explode("document_ids").rename({"document_ids": "document_id"})
    sizes = (
        out.filter(pl.col("cluster") >= 0).group_by("cluster").len().sort("len", descending=True)
    )
    audit.update(
        {
            "clusters": sizes.height,
            "unassigned_nodes": int((clusters < 0).sum()),
            "unassigned_documents": out.filter(pl.col("cluster") < 0).height,
            "largest_cluster_documents": sizes.select(pl.col("len").max().fill_null(0)).item(),
            "clusters_ge_10_documents": sizes.filter(pl.col("len") >= 10).height,
            "clusters_ge_50_documents": sizes.filter(pl.col("len") >= 50).height,
            "runtime_seconds": perf_counter() - started,
        }
    )
    out.write_parquet(args.output / "document_clusters.parquet")
    pairs.write_parquet(args.output / "pairs.parquet")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "command"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
