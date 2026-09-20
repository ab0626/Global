"""Blocked document clustering for one window: title, event-ID, URL-token and entity channels.

Every canonical document is its own graph node (wire copies are *not* collapsed
before clustering; the earlier fingerprint collapse caused false merges). Candidate
pairs come from per-channel top-k retrieval only (title kNN over the cached
multilingual embeddings via FAISS; sparse TF-IDF kNN for event IDs, URL slug tokens
and GKG entities), never from a global O(N^2) comparison. Pairs further apart than
``candidate_max_hours`` are discarded. Each surviving pair is scored on every
channel, fused with fixed channel weights and kept only when the evidence gate
passes (>= 2 channels agree, or one channel is above its own floor). Leiden
partitions the gated graph into incidents; incidents are linked into story
families; every document gets an ``assignment_score`` (share of its edge weight
inside its incident) and optional secondary memberships.

Outputs under ``--output`` (the lineage the API and tests rely on):

* ``candidate_pairs.parquet``      left, right, channels that proposed the pair.
* ``pair_features.parquet``        per-channel scores, fused score, gate decision.
* ``graph_edges.parquet``          gated pairs above threshold (the Leiden input).
* ``incident_memberships.parquet`` document_id, incident_id, family_id,
  assignment_score, is_primary (``incident_id == -1`` marks unassigned docs).
* ``incidents.parquet``            per-incident size, family, wire groups, span.
* ``run.json``                     settings, seed, model, counts, runtime.
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

from attention.embed import load_embeddings
from clustering_experiment import ArticleSettings, channel_candidates, tfidf_channel, url_words
from clustering_v2 import V2Settings, clean_token, host_filter

CHANNEL_WEIGHTS = {"title": 2.0, "event": 1.5, "url": 1.0, "entity": 0.5}
CHANNEL_FLOORS = {"title": 0.65, "event": 0.8, "url": 0.8, "entity": 0.9}
TITLE_CANDIDATE_MIN = 0.35


@dataclass(frozen=True)
class ClusterSettings:
    threshold: float = 0.3
    resolution: float = 1.0
    seed: int = 2026
    max_entity_share: float = 0.1
    channels: str = "title,event,url,entity"
    neighbors: int = 30
    candidate_max_hours: float = 48.0
    corroboration_min_channels: int = 2
    family_title_threshold: float = 0.55
    family_entity_threshold: float = 0.2
    secondary_min_score: float = 0.25


def slug_path(url: str) -> str:
    try:
        return unquote(urlsplit(url).path).lower()
    except ValueError:
        return ""


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


def event_rows(links: pl.DataFrame) -> pl.DataFrame:
    return links.select(
        pl.col("document_id").alias("node"),
        pl.col("GlobalEventID").cast(pl.String).alias("feature"),
    ).unique()


def url_rows(documents: pl.DataFrame, settings: V2Settings) -> pl.DataFrame:
    frame = documents.select(
        pl.col("document_id").alias("node"),
        "domain",
        "canonical_url",
        pl.col("first_seen").dt.date().alias("day"),
    )
    rows = [
        (node, domain, day, word)
        for node, domain, url, day in frame.iter_rows()
        for word in url_words(slug_path(url))
        if clean_token(word)
    ]
    table = pl.DataFrame(
        rows,
        schema={"node": pl.Int64, "domain": pl.String, "day": pl.Date, "feature": pl.String},
        orient="row",
    ).unique()
    table = table.rename({"node": "vertex"})
    table, _ = host_filter(table, settings)
    return table.rename({"vertex": "node"}).select("node", "feature")


def entity_rows(documents: pl.DataFrame, max_share: float) -> pl.DataFrame:
    """Persons and organizations only: GDELT locations bias clusters toward geography."""
    docs = documents.filter(pl.col("has_gkg")).select(
        pl.col("document_id").alias("node"), "persons", "organizations"
    )
    parts = [
        docs.select("node", "persons")
        .explode("persons")
        .select("node", (pl.lit("p:") + pl.col("persons")).alias("feature")),
        docs.select("node", "organizations")
        .explode("organizations")
        .select("node", (pl.lit("o:") + pl.col("organizations")).alias("feature")),
    ]
    rows = pl.concat(parts).drop_nulls().unique()
    frequent = (
        rows.group_by("feature").len().filter(pl.col("len") > max_share * docs["node"].n_unique())
    )
    return rows.join(frequent.select("feature"), on="feature", how="anti")


class TitleChannel:
    """Dense multilingual title vectors with FAISS HNSW retrieval."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, count: int) -> None:
        self.dimension = vectors.shape[1] if vectors.size else 0
        self.row_of = np.full(count, -1, dtype=np.int64)
        self.row_of[ids] = np.arange(len(ids))
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.has = self.row_of >= 0

    def candidates(self, neighbors: int) -> np.ndarray:
        import faiss

        count = len(self.row_of)
        rows = len(self.vectors)
        if rows < 2 or self.dimension == 0:
            return np.zeros(0, dtype=np.int64)
        if rows <= 20_000:
            index = faiss.IndexFlatIP(self.dimension)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 96
        index.add(self.vectors)
        k = min(neighbors + 1, rows)
        scores, found = index.search(self.vectors, k)
        ids = np.flatnonzero(self.has)
        source = np.repeat(ids, k)
        target_rows = found.ravel()
        valid = (target_rows >= 0) & (scores.ravel() >= TITLE_CANDIDATE_MIN)
        source, target_rows = source[valid], target_rows[valid]
        target = ids[target_rows]
        valid = source != target
        source, target = source[valid], target[valid]
        return np.unique(np.minimum(source, target) * count + np.maximum(source, target))

    def scores(self, left: np.ndarray, right: np.ndarray, batch: int) -> np.ndarray:
        out = np.zeros(len(left), dtype=np.float64)
        lrow, rrow = self.row_of[left], self.row_of[right]
        ok = (lrow >= 0) & (rrow >= 0)
        where = np.flatnonzero(ok)
        for start in range(0, len(where), batch):
            idx = where[start : start + batch]
            out[idx] = np.einsum("ij,ij->i", self.vectors[lrow[idx]], self.vectors[rrow[idx]])
        return np.clip(out, 0.0, 1.0)


def score_pairs(
    keys: np.ndarray,
    proposed_by: dict[str, np.ndarray],
    count: int,
    sparse_channels: dict[str, sparse.csr_matrix],
    title: TitleChannel | None,
    first_seen: np.ndarray,
    settings: ClusterSettings,
    batch: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    left, right = keys // count, keys % count
    hours = np.abs(first_seen[left] - first_seen[right]) / 3600.0
    frame = pl.DataFrame({"left": left, "right": right, "delta_hours": hours})
    for name, proposed in proposed_by.items():
        frame = frame.with_columns(pl.Series(f"from_{name}", np.isin(keys, proposed)))
    candidates = frame
    in_window = hours <= settings.candidate_max_hours
    left, right = left[in_window], right[in_window]
    frame = frame.filter(pl.Series(in_window))

    weight_sum = np.zeros(len(left), dtype=np.float64)
    weighted = np.zeros(len(left), dtype=np.float64)
    evidence = np.zeros(len(left), dtype=np.int64)
    strong = np.zeros(len(left), dtype=bool)

    def absorb(name: str, scores: np.ndarray, has: np.ndarray) -> None:
        nonlocal weight_sum, weighted, evidence, strong
        available = has[left] & has[right]
        weight = CHANNEL_WEIGHTS[name]
        weight_sum += available * weight
        weighted += scores * weight
        evidence += (scores > 0).astype(np.int64)
        strong |= scores >= CHANNEL_FLOORS[name]

    if title is not None:
        title_scores = title.scores(left, right, batch)
        absorb("title", title_scores, title.has)
        frame = frame.with_columns(pl.Series("title_score", title_scores))
    for name, matrix in sparse_channels.items():
        has = np.asarray(matrix.getnnz(axis=1) > 0).ravel()
        scores = np.zeros(len(left), dtype=np.float64)
        for start in range(0, len(left), batch):
            stop = start + batch
            scores[start:stop] = np.asarray(
                matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
            ).ravel()
        scores = np.clip(scores, 0, 1)
        frame = frame.with_columns(pl.Series(f"{name}_score", scores))
        absorb(name, scores, has)
    combined = np.divide(weighted, weight_sum, out=np.zeros_like(weighted), where=weight_sum > 0)
    gate = (evidence >= settings.corroboration_min_channels) | strong
    gated = np.where(gate, combined, 0.0)
    frame = frame.with_columns(
        pl.Series("evidence_channels", evidence),
        pl.Series("single_channel_strong", strong),
        pl.Series("combined", combined),
        pl.Series("gated", gated),
    )
    return candidates, frame


def leiden(edges: pl.DataFrame, count: int, settings: ClusterSettings) -> np.ndarray:
    graph = ig.Graph(n=count, edges=edges.select("left", "right").iter_rows())
    ig.set_random_number_generator(random.Random(settings.seed))
    membership = np.asarray(
        graph.community_leiden(
            weights=edges["gated"].to_list(),
            objective_function="modularity",
            resolution=settings.resolution,
            n_iterations=-1,
        ).membership,
        dtype=np.int64,
    )
    isolated = np.asarray(graph.degree()) == 0
    membership[isolated] = -1
    return membership


def memberships(
    edges: pl.DataFrame, incident: np.ndarray, settings: ClusterSettings
) -> pl.DataFrame:
    """Primary membership from Leiden plus edge-weight shares into every incident a
    document touches; secondaries are shares >= ``secondary_min_score``."""
    count = len(incident)
    both = pl.concat(
        [
            edges.select(
                pl.col("left").alias("document_id"), pl.col("right").alias("other"), "gated"
            ),
            edges.select(
                pl.col("right").alias("document_id"), pl.col("left").alias("other"), "gated"
            ),
        ]
    )
    both = both.with_columns(pl.Series("other_incident", incident[both["other"].to_numpy()]))
    totals = both.group_by("document_id").agg(pl.col("gated").sum().alias("total"))
    shares = (
        both.filter(pl.col("other_incident") >= 0)
        .group_by("document_id", "other_incident")
        .agg(pl.col("gated").sum().alias("weight"))
        .join(totals, on="document_id")
        .with_columns((pl.col("weight") / pl.col("total")).alias("assignment_score"))
        .rename({"other_incident": "incident_id"})
    )
    primary = pl.DataFrame(
        {
            "document_id": np.arange(count, dtype=np.int64),
            "incident_id": incident,
        }
    )
    primary_rows = (
        primary.join(shares, on=["document_id", "incident_id"], how="left")
        .with_columns(
            pl.when(pl.col("incident_id") < 0)
            .then(0.0)
            .otherwise(pl.col("assignment_score").fill_null(1.0))
            .alias("assignment_score"),
            pl.lit(True).alias("is_primary"),
        )
        .select("document_id", "incident_id", "assignment_score", "is_primary")
    )
    secondary_rows = (
        shares.join(primary, on="document_id", suffix="_primary")
        .filter(
            (pl.col("incident_id") != pl.col("incident_id_primary"))
            & (pl.col("assignment_score") >= settings.secondary_min_score)
        )
        .select("document_id", "incident_id", "assignment_score", pl.lit(False).alias("is_primary"))
    )
    return pl.concat([primary_rows, secondary_rows]).sort(
        "document_id", "is_primary", descending=[False, True]
    )


def link_families(
    incident: np.ndarray,
    documents: pl.DataFrame,
    title: TitleChannel | None,
    settings: ClusterSettings,
) -> np.ndarray:
    """Group incidents into story families: connected components over incidents
    whose title centroids are close (when both have titles) or whose top entities
    overlap strongly (legacy fallback). Returns family id per incident id."""
    n_incidents = int(incident.max()) + 1 if incident.size and incident.max() >= 0 else 0
    if n_incidents == 0:
        return np.zeros(0, dtype=np.int64)
    edges: set[tuple[int, int]] = set()
    if title is not None and title.dimension:
        rows = title.row_of
        assigned = np.flatnonzero((incident >= 0) & (rows >= 0))
        centroids = np.zeros((n_incidents, title.dimension), dtype=np.float32)
        np.add.at(centroids, incident[assigned], title.vectors[rows[assigned]])
        norms = np.linalg.norm(centroids, axis=1)
        has_centroid = norms > 0
        centroids[has_centroid] /= norms[has_centroid][:, None]
        for start in range(0, n_incidents, 2048):
            block = centroids[start : start + 2048] @ centroids.T
            src, dst = np.nonzero(block >= settings.family_title_threshold)
            for a, b in zip(src + start, dst, strict=True):
                if a < b and has_centroid[a] and has_centroid[b]:
                    edges.add((int(a), int(b)))
    entities = (
        documents.filter(pl.col("has_gkg"))
        .select(
            pl.Series(
                "incident_id",
                incident[documents.filter(pl.col("has_gkg"))["document_id"].to_numpy()],
            ),
            pl.concat_list(
                pl.col("persons").fill_null([]), pl.col("organizations").fill_null([])
            ).alias("entity"),
        )
        .filter(pl.col("incident_id") >= 0)
        .explode("entity")
        .drop_nulls("entity")
        .group_by("incident_id", "entity")
        .len()
        .sort("len", descending=True)
        .group_by("incident_id", maintain_order=True)
        .agg(pl.col("entity").head(10))
    )
    top: dict[int, set[str]] = {
        int(i): set(e) for i, e in zip(entities["incident_id"], entities["entity"], strict=True)
    }
    by_entity: dict[str, list[int]] = {}
    for inc, ents in top.items():
        for ent in ents:
            by_entity.setdefault(ent, []).append(inc)
    for members in by_entity.values():
        if len(members) > 50:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                lo, hi = min(a, b), max(a, b)
                if (lo, hi) in edges:
                    continue
                inter = len(top[lo] & top[hi])
                union = len(top[lo] | top[hi])
                if union and inter / union >= settings.family_entity_threshold:
                    edges.add((lo, hi))
    graph = ig.Graph(n=n_incidents, edges=sorted(edges))
    return np.asarray(graph.connected_components().membership, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="attention.atomic output")
    parser.add_argument("--embeddings", type=Path, help="attention.embed output (title channel)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=ClusterSettings.threshold)
    parser.add_argument("--resolution", type=float, default=ClusterSettings.resolution)
    parser.add_argument("--channels", default=ClusterSettings.channels)
    parser.add_argument("--neighbors", type=int, default=ClusterSettings.neighbors)
    parser.add_argument(
        "--candidate-max-hours", type=float, default=ClusterSettings.candidate_max_hours
    )
    parser.add_argument("--seed", type=int, default=ClusterSettings.seed)
    args = parser.parse_args()
    settings = ClusterSettings(
        threshold=args.threshold,
        resolution=args.resolution,
        channels=args.channels,
        neighbors=args.neighbors,
        candidate_max_hours=args.candidate_max_hours,
        seed=args.seed,
    )
    run(args.features, args.embeddings, args.output, settings)


def run(features: Path, embeddings: Path | None, output: Path, settings: ClusterSettings) -> dict:
    article = ArticleSettings(neighbors=settings.neighbors, max_feature_articles=10**9)
    v2 = V2Settings()
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    documents = pl.read_parquet(features / "documents.parquet").sort("document_id")
    links = pl.read_parquet(features / "document_events.parquet")
    count = documents.height
    if documents["document_id"].to_list() != list(range(count)):
        raise ValueError("documents.parquet must have contiguous document_id 0..N-1")
    first_seen = documents["first_seen"].dt.epoch("s").to_numpy().astype(np.float64)
    wanted = [c for c in settings.channels.split(",") if c]

    title: TitleChannel | None = None
    embed_meta: dict = {}
    if "title" in wanted and embeddings is not None:
        loaded = load_embeddings(embeddings)
        if loaded is not None and loaded[0].size:
            title = TitleChannel(loaded[0], loaded[1], count)
            embed_meta = json.loads((embeddings / "embed.json").read_text())
    resolution_model = "title_multilingual_v1" if title is not None else "legacy_metadata_v1"

    builders = {
        "event": lambda: event_rows(links),
        "url": lambda: url_rows(documents, v2),
        "entity": lambda: entity_rows(documents, settings.max_entity_share),
    }
    audit: dict = {
        "command": sys.argv,
        "settings": asdict(settings),
        "resolution_model": resolution_model,
        "embedding": embed_meta,
        "documents": count,
        "channels": {},
    }
    proposed: dict[str, np.ndarray] = {}
    sparse_channels: dict[str, sparse.csr_matrix] = {}
    if title is not None:
        print("Title kNN...", flush=True)
        proposed["title"] = title.candidates(settings.neighbors)
        audit["channels"]["title"] = {
            "documents_with_title": int(title.has.sum()),
            "directed_pairs": int(len(proposed["title"])),
        }
    for name in wanted:
        if name not in builders:
            continue
        print(f"Building channel: {name}...", flush=True)
        rows = builders[name]()
        if rows.is_empty():
            audit["channels"][name] = {"features_retained": 0, "directed_pairs": 0}
            continue
        matrix, channel_audit = tfidf_channel(feature_matrix(rows, count), article)
        sparse_channels[name] = matrix
        proposed[name] = channel_candidates(matrix, article)
        channel_audit["directed_pairs"] = int(len(proposed[name]))
        audit["channels"][name] = channel_audit
    keys = (
        np.unique(np.concatenate(list(proposed.values())))
        if proposed
        else np.zeros(0, dtype=np.int64)
    )
    print(f"Scoring {len(keys):,} candidate pairs...", flush=True)
    candidates, pairs = score_pairs(
        keys,
        proposed,
        count,
        sparse_channels,
        title,
        first_seen,
        settings,
        article.scoring_batch_size,
    )
    edges = pairs.filter(pl.col("gated") >= settings.threshold).select(
        "left", "right", "gated", "combined", "evidence_channels"
    )
    print(f"Leiden over {edges.height:,} edges...", flush=True)
    incident = (
        leiden(edges, count, settings) if edges.height else np.full(count, -1, dtype=np.int64)
    )
    family_of_incident = link_families(incident, documents, title, settings)
    family = np.where(incident >= 0, family_of_incident[np.maximum(incident, 0)], -1)
    member = (
        memberships(edges, incident, settings)
        .join(
            pl.DataFrame(
                {"incident_id": np.arange(len(family_of_incident)), "family_id": family_of_incident}
            ),
            on="incident_id",
            how="left",
        )
        .with_columns(pl.col("family_id").fill_null(-1))
    )
    member = member.select(
        "document_id", "incident_id", "family_id", "assignment_score", "is_primary"
    )
    incidents = (
        documents.with_columns(pl.Series("incident_id", incident), pl.Series("family_id", family))
        .filter(pl.col("incident_id") >= 0)
        .group_by("incident_id")
        .agg(
            pl.col("family_id").first(),
            pl.len().alias("documents"),
            pl.col("domain").n_unique().alias("unique_domains"),
            pl.col("wire_group").n_unique().alias("effective_reports"),
            pl.col("publisher_country").drop_nulls().n_unique().alias("publisher_countries"),
            pl.col("language").drop_nulls().n_unique().alias("languages"),
            pl.col("first_seen").min().alias("start_time"),
            pl.col("last_seen").max().alias("end_time"),
        )
        .sort("documents", descending=True)
    )
    audit.update(
        {
            "candidate_pairs": candidates.height,
            "pairs_in_window": pairs.height,
            "pairs_gated": int((pairs["gated"] > 0).sum()),
            "edges": edges.height,
            "incidents": incidents.height,
            "families": int(len(np.unique(family_of_incident))),
            "unassigned_documents": int((incident < 0).sum()),
            "unassigned_rate": float((incident < 0).mean()) if count else 0.0,
            "largest_incident_documents": incidents.select(
                pl.col("documents").max().fill_null(0)
            ).item(),
            "incidents_ge_10_documents": incidents.filter(pl.col("documents") >= 10).height,
            "incidents_ge_50_documents": incidents.filter(pl.col("documents") >= 50).height,
            "secondary_memberships": member.filter(~pl.col("is_primary")).height,
            "runtime_seconds": perf_counter() - started,
        }
    )
    candidates.write_parquet(output / "candidate_pairs.parquet")
    pairs.write_parquet(output / "pair_features.parquet")
    edges.write_parquet(output / "graph_edges.parquet")
    member.write_parquet(output / "incident_memberships.parquet")
    incidents.write_parquet(output / "incidents.parquet")
    (output / "run.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "command"}, indent=2, default=str))
    return audit


if __name__ == "__main__":
    main()
