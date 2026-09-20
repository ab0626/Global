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
CHANNEL_FLOORS = {"title": 0.55, "event": 0.8, "url": 0.8, "entity": 0.9}
TITLE_CANDIDATE_MIN = 0.35
# a channel only counts as corroborating evidence above these scores; any
# positive title cosine is not evidence (nearly every pair has one)
CHANNEL_EVIDENCE_MIN = {
    "title": 0.4,
    "event": 0.05,
    "url": 0.2,
    "entity": 0.15,
}
# title cosine is calibrated against the language pair's random-pair background
BACKGROUND_QUANTILE = 0.9
# prior for language pairs too small to estimate: median same-language p90 after
# centering on the Feb 2023 window (eng 0.13 ... kor 0.38)
BACKGROUND_PRIOR = 0.17
BACKGROUND_SAMPLES = 4000
BACKGROUND_MIN_DOCS = 500
# a title repeated this often by one domain is site boilerplate ("ТАСС", "Stock Market")
BOILERPLATE_TITLE_REPEATS = 10


@dataclass(frozen=True)
class ClusterSettings:
    threshold: float = 0.3
    resolution: float = 0.05
    family_resolution: float = 0.5
    seed: int = 2026
    max_entity_share: float = 0.1
    max_feature_articles: int = 2000
    max_pair_contributions: int = 2_000_000_000
    channels: str = "title,event,url,entity"
    neighbors: int = 30
    candidate_max_hours: float = 48.0
    corroboration_min_channels: int = 2
    # a lone strong event/url/entity channel cannot carry a pair whose titles both
    # exist and clearly disagree (site-wide GDELT events, URL boilerplate)
    single_channel_title_veto: float = 0.15
    # ...and both documents need this many features in that channel: a one-token URL
    # path or a single site-wide GlobalEventID gives cosine 1.0 without meaning it
    single_channel_min_features: int = 2
    # a pair where either document has no usable title (missing, or site boilerplate
    # excluded from the title channel) needs this many evidence channels: such nodes
    # are the bridges that glue unrelated incidents ("Primeira Edição", untitled feeds)
    titleless_min_channels: int = 2
    # a lone title match between documents in different languages must clear this
    # (background-calibrated) score: the multilingual encoder rates topical kin
    # ("train derails in Ohio" ~ "Indian Railways hygiene") ~0.55-0.6 across languages
    cross_language_title_floor: float = 0.65
    family_title_threshold: float = 0.4
    family_entity_threshold: float = 0.2
    family_strong_title: float = 0.7
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


def boilerplate_titles(documents: pl.DataFrame) -> np.ndarray:
    """document_ids whose (domain, title) repeats >= BOILERPLATE_TITLE_REPEATS times."""
    return (
        documents.filter(pl.col("title").is_not_null())
        .filter(pl.len().over("domain", "title") >= BOILERPLATE_TITLE_REPEATS)["document_id"]
        .to_numpy()
    )


class TitleChannel:
    """Dense multilingual title vectors with FAISS HNSW retrieval."""

    def __init__(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        count: int,
        languages: np.ndarray | None = None,
        seed: int = 2026,
        domains: np.ndarray | None = None,
    ) -> None:
        self.dimension = vectors.shape[1] if vectors.size else 0
        self.row_of = np.full(count, -1, dtype=np.int64)
        self.row_of[ids] = np.arange(len(ids))
        self.vectors = np.array(vectors, dtype=np.float32, order="C")
        self.has = self.row_of >= 0
        self.language_codes: list[str] = ["other"]
        self.code_of_row = np.zeros(len(ids), dtype=np.int64)
        self.background = np.full((1, 1), BACKGROUND_PRIOR, dtype=np.float64)
        if languages is not None and len(ids):
            if domains is not None:
                self.center(np.asarray(domains)[ids].astype(str))
            self.calibrate(np.asarray(languages)[ids].astype(str), seed)

    def center(self, groups: np.ndarray) -> None:
        """Subtract each group's mean direction (groups with >= BACKGROUND_MIN_DOCS rows)
        and renormalise (Mu & Viswanath 2018, "all-but-the-top"). Used per publisher
        domain to cancel site boilerplate in titles ("... | Fakti.bg - News") and per
        language to cancel the script's common component, which otherwise makes
        unrelated same-site or same-language titles -- and their centroids -- alike."""
        values, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        for g in np.flatnonzero(counts >= BACKGROUND_MIN_DOCS):
            rows = np.flatnonzero(inverse == g)
            self.vectors[rows] -= self.vectors[rows].mean(axis=0, keepdims=True)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = np.ascontiguousarray(
            np.divide(self.vectors, norms, out=np.zeros_like(self.vectors), where=norms > 0)
        )

    def calibrate(self, languages: np.ndarray, seed: int) -> None:
        """Random-pair cosine background per language pair. The multilingual encoder
        is far less discriminative for scripts it tokenises poorly (Korean, Bengali,
        Kannada random pairs sit at 0.45-0.6 cosine vs 0.03 for English), so a fixed
        cosine threshold would glue unrelated same-language titles together."""
        values, counts = np.unique(languages, return_counts=True)
        kept = [str(v) for v, c in zip(values, counts, strict=True) if c >= BACKGROUND_MIN_DOCS]
        self.language_codes = ["other", *kept]
        code = {lang: i + 1 for i, lang in enumerate(kept)}
        self.code_of_row = np.fromiter(
            (code.get(str(lang), 0) for lang in languages), dtype=np.int64, count=len(languages)
        )
        rng = np.random.default_rng(seed)
        n = len(self.language_codes)
        rows_of = [np.flatnonzero(self.code_of_row == i) for i in range(n)]
        self.center(languages)
        background = np.full((n, n), BACKGROUND_PRIOR, dtype=np.float64)
        for a in range(n):
            if len(rows_of[a]) < BACKGROUND_MIN_DOCS:
                continue
            for b in range(a, n):
                if len(rows_of[b]) < BACKGROUND_MIN_DOCS:
                    continue
                left = rng.choice(rows_of[a], BACKGROUND_SAMPLES)
                right = rng.choice(rows_of[b], BACKGROUND_SAMPLES)
                same = left == right
                cos = np.einsum("ij,ij->i", self.vectors[left], self.vectors[right])[~same]
                q = float(np.quantile(cos, BACKGROUND_QUANTILE)) if cos.size else 0.0
                background[a, b] = background[b, a] = min(max(q, 0.0), 0.99)
        self.background = background

    def background_table(self) -> pl.DataFrame:
        rows = [
            (self.language_codes[a], self.language_codes[b], float(self.background[a, b]))
            for a in range(len(self.language_codes))
            for b in range(a, len(self.language_codes))
        ]
        return pl.DataFrame(
            rows,
            schema={
                "language_left": pl.String,
                "language_right": pl.String,
                "quantile": pl.Float64,
            },
            orient="row",
        )

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

    def cosines(self, left: np.ndarray, right: np.ndarray, batch: int) -> np.ndarray:
        out = np.zeros(len(left), dtype=np.float64)
        lrow, rrow = self.row_of[left], self.row_of[right]
        ok = (lrow >= 0) & (rrow >= 0)
        where = np.flatnonzero(ok)
        for start in range(0, len(where), batch):
            idx = where[start : start + batch]
            out[idx] = np.einsum("ij,ij->i", self.vectors[lrow[idx]], self.vectors[rrow[idx]])
        return np.clip(out, 0.0, 1.0)

    def scores(self, left: np.ndarray, right: np.ndarray, batch: int) -> np.ndarray:
        """Cosine rescaled so the language pair's background quantile maps to 0 and
        identical titles to 1: ``(cos - q) / (1 - q)`` clipped to [0, 1]."""
        cos = self.cosines(left, right, batch)
        lrow, rrow = self.row_of[left], self.row_of[right]
        ok = (lrow >= 0) & (rrow >= 0)
        q = np.zeros(len(left), dtype=np.float64)
        q[ok] = self.background[self.code_of_row[lrow[ok]], self.code_of_row[rrow[ok]]]
        return np.clip((cos - q) / (1.0 - q), 0.0, 1.0)


def apply_gate(frame: pl.DataFrame, settings: ClusterSettings) -> pl.DataFrame:
    """(Re)derive ``evidence_channels``, ``single_channel_strong`` and ``gated`` from the
    per-channel scores in a pair_features frame, so gate settings can be swept without
    rescoring. A pair passes when >= ``corroboration_min_channels`` channels show
    evidence, or one channel is above its floor -- but a lone event/url/entity channel
    cannot carry a pair whose titles both exist and disagree, nor one where either
    document has fewer than ``single_channel_min_features`` features in that channel,
    nor one where either document lacks a usable title unless
    ``titleless_min_channels`` channels agree. A lone title carries a cross-language
    pair (``same_language`` false) only above ``cross_language_title_floor``."""
    n = frame.height
    evidence = np.zeros(n, dtype=np.int64)
    strong = np.zeros(n, dtype=bool)
    title_agrees = np.ones(n, dtype=bool)
    both = np.ones(n, dtype=bool)
    if "title_score" in frame.columns:
        scores = frame["title_score"].to_numpy()
        both = frame["title_both"].to_numpy()
        title_agrees = ~both | (scores >= settings.single_channel_title_veto)
        evidence += (scores >= CHANNEL_EVIDENCE_MIN["title"]).astype(np.int64)
        floor = np.full(n, CHANNEL_FLOORS["title"])
        if "same_language" in frame.columns:
            cross = ~frame["same_language"].to_numpy()
            floor[cross] = max(CHANNEL_FLOORS["title"], settings.cross_language_title_floor)
        strong |= scores >= floor
    for name in ("event", "url", "entity"):
        if f"{name}_score" not in frame.columns:
            continue
        scores = frame[f"{name}_score"].to_numpy()
        rich = frame[f"{name}_min_features"].to_numpy() >= settings.single_channel_min_features
        evidence += (scores >= CHANNEL_EVIDENCE_MIN[name]).astype(np.int64)
        strong |= (scores >= CHANNEL_FLOORS[name]) & rich & title_agrees
    gate = (evidence >= settings.corroboration_min_channels) | strong
    gate &= both | (evidence >= settings.titleless_min_channels)
    combined = frame["combined"].to_numpy()
    return frame.with_columns(
        pl.Series("evidence_channels", evidence),
        pl.Series("single_channel_strong", strong),
        pl.Series("gated", np.where(gate, combined, 0.0)),
    )


def score_pairs(
    keys: np.ndarray,
    proposed_by: dict[str, np.ndarray],
    count: int,
    sparse_channels: dict[str, sparse.csr_matrix],
    title: TitleChannel | None,
    first_seen: np.ndarray,
    settings: ClusterSettings,
    batch: int,
    languages: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    left, right = keys // count, keys % count
    hours = np.abs(first_seen[left] - first_seen[right]) / 3600.0
    frame = pl.DataFrame({"left": left, "right": right, "delta_hours": hours})
    if languages is not None:
        frame = frame.with_columns(pl.Series("same_language", languages[left] == languages[right]))
    for name, proposed in proposed_by.items():
        frame = frame.with_columns(pl.Series(f"from_{name}", np.isin(keys, proposed)))
    candidates = frame
    in_window = hours <= settings.candidate_max_hours
    left, right = left[in_window], right[in_window]
    frame = frame.filter(pl.Series(in_window))

    weight_sum = np.zeros(len(left), dtype=np.float64)
    weighted = np.zeros(len(left), dtype=np.float64)

    def absorb(name: str, scores: np.ndarray, has: np.ndarray) -> None:
        nonlocal weight_sum, weighted
        available = has[left] & has[right]
        weight = CHANNEL_WEIGHTS[name]
        weight_sum += available * weight
        weighted += scores * weight

    if title is not None:
        title_scores = title.scores(left, right, batch)
        absorb("title", title_scores, title.has)
        frame = frame.with_columns(
            pl.Series("title_cosine", title.cosines(left, right, batch)),
            pl.Series("title_score", title_scores),
            pl.Series("title_both", title.has[left] & title.has[right]),
        )
    for name, matrix in sparse_channels.items():
        nnz = np.asarray(matrix.getnnz(axis=1)).ravel()
        scores = np.zeros(len(left), dtype=np.float64)
        for start in range(0, len(left), batch):
            stop = start + batch
            scores[start:stop] = np.asarray(
                matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
            ).ravel()
        scores = np.clip(scores, 0, 1)
        absorb(name, scores, nnz > 0)
        frame = frame.with_columns(
            pl.Series(f"{name}_score", scores),
            # feature count of the sparser document: a pair sharing one feature each
            # scores cosine 1.0, so the gate needs to know how thin that evidence is
            pl.Series(f"{name}_min_features", np.minimum(nnz[left], nnz[right]).astype(np.int32)),
        )
    combined = np.divide(weighted, weight_sum, out=np.zeros_like(weighted), where=weight_sum > 0)
    frame = apply_gate(frame.with_columns(pl.Series("combined", combined)), settings)
    return candidates, frame


def cpm_leiden(graph: ig.Graph, weights: list[float], resolution: float, seed: int) -> np.ndarray:
    """Leiden with the Constant Potts Model: a community is kept only while its mean
    intra-pair weight stays above ``resolution``. Unlike modularity, this has no
    resolution limit, so cluster granularity does not drift with the number of
    documents in the window (Traag et al. 2019)."""
    if graph.ecount() == 0:
        # igraph's "iterate until stable" never terminates on an edgeless graph
        return np.arange(graph.vcount(), dtype=np.int64)
    ig.set_random_number_generator(random.Random(seed))
    return np.asarray(
        graph.community_leiden(
            weights=weights,
            objective_function="CPM",
            resolution=resolution,
            n_iterations=-1,
        ).membership,
        dtype=np.int64,
    )


def leiden(edges: pl.DataFrame, count: int, settings: ClusterSettings) -> np.ndarray:
    graph = ig.Graph(n=count, edges=edges.select("left", "right").iter_rows())
    membership = cpm_leiden(graph, edges["gated"].to_list(), settings.resolution, settings.seed)
    isolated = np.asarray(graph.degree()) == 0
    membership[isolated] = -1
    # compact ids so incident_id is 0..k-1 over non-empty incidents only
    kept = membership >= 0
    _, compact = np.unique(membership[kept], return_inverse=True)
    membership[kept] = compact
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


def incident_top_entities(
    incident: np.ndarray, documents: pl.DataFrame, top_k: int = 10
) -> dict[int, set[str]]:
    """Most frequent persons/organizations per incident (locations excluded)."""
    gkg = documents.filter(pl.col("has_gkg"))
    entities = (
        gkg.select(
            pl.Series("incident_id", incident[gkg["document_id"].to_numpy()]),
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
        .agg(pl.col("entity").head(top_k))
    )
    return {
        int(i): set(e) for i, e in zip(entities["incident_id"], entities["entity"], strict=True)
    }


def incident_corroboration(
    incident: np.ndarray,
    documents: pl.DataFrame,
    links: pl.DataFrame | None,
    n_incidents: int,
) -> tuple[dict[int, set[str]], dict[int, set[int]], np.ndarray, np.ndarray]:
    """Per incident: top entities, GlobalEventIDs, and [start, end] of first_seen (s)."""
    top = incident_top_entities(incident, documents)
    events: dict[int, set[int]] = {}
    if links is not None and links.height:
        linked = (
            pl.DataFrame({"document_id": np.arange(len(incident)), "incident": incident})
            .filter(pl.col("incident") >= 0)
            .join(links.select("document_id", "GlobalEventID"), on="document_id")
            .group_by("incident")
            .agg(pl.col("GlobalEventID").unique())
        )
        events = {
            int(i): set(e) for i, e in zip(linked["incident"], linked["GlobalEventID"], strict=True)
        }
    seen = documents["first_seen"].dt.epoch("s").to_numpy().astype(np.float64)
    assigned = incident >= 0
    start = np.full(n_incidents, np.inf)
    end = np.full(n_incidents, -np.inf)
    np.minimum.at(start, incident[assigned], seen[assigned])
    np.maximum.at(end, incident[assigned], seen[assigned])
    return top, events, start, end


def link_families(
    incident: np.ndarray,
    documents: pl.DataFrame,
    title: TitleChannel | None,
    settings: ClusterSettings,
    links: pl.DataFrame | None = None,
) -> np.ndarray:
    """Group incidents into story families. Title-capable windows: an incident graph
    weighted by calibrated centroid cosine, keeping a pair only when (a) the score is
    >= ``family_title_threshold``, (b) the incidents' first_seen ranges lie within
    ``candidate_max_hours`` of each other, and (c) they share a top entity or a
    GlobalEventID unless the score is >= ``family_strong_title``; partitioned with
    CPM-Leiden at ``family_resolution``. Legacy windows: top-entity Jaccard edges
    instead. Returns family id per incident id."""
    n_incidents = int(incident.max()) + 1 if incident.size and incident.max() >= 0 else 0
    if n_incidents == 0:
        return np.zeros(0, dtype=np.int64)
    edges: dict[tuple[int, int], float] = {}
    if title is not None and title.dimension:
        top, events, t0, t1 = incident_corroboration(incident, documents, links, n_incidents)
        max_gap = settings.candidate_max_hours * 3600.0
        rows = title.row_of
        assigned = np.flatnonzero((incident >= 0) & (rows >= 0))
        centroids = np.zeros((n_incidents, title.dimension), dtype=np.float32)
        np.add.at(centroids, incident[assigned], title.vectors[rows[assigned]])
        norms = np.linalg.norm(centroids, axis=1)
        has_centroid = norms > 0
        centroids[has_centroid] /= norms[has_centroid][:, None]
        # dominant language per incident; centroid cosine is calibrated against the
        # same random-pair background as document pairs
        code_counts = np.zeros((n_incidents, len(title.language_codes)), dtype=np.int64)
        np.add.at(code_counts, (incident[assigned], title.code_of_row[rows[assigned]]), 1)
        dominant = code_counts.argmax(axis=1)
        for start in range(0, n_incidents, 1024):
            block = centroids[start : start + 1024] @ centroids.T
            q = title.background[dominant[start : start + 1024]][:, dominant]
            block = np.clip((block - q) / (1.0 - q), 0.0, 1.0)
            src, dst = np.nonzero(block >= settings.family_title_threshold)
            src_abs = src + start
            keep = (src_abs < dst) & has_centroid[src_abs] & has_centroid[dst]
            gap = np.maximum(t0[src_abs] - t1[dst], t0[dst] - t1[src_abs])
            keep &= gap <= max_gap
            for a, b, w in zip(
                src_abs[keep].tolist(),
                dst[keep].tolist(),
                block[src[keep], dst[keep]].tolist(),
                strict=True,
            ):
                corroborated = bool(top.get(a, set()) & top.get(b, set())) or bool(
                    events.get(a, set()) & events.get(b, set())
                )
                if corroborated or w >= settings.family_strong_title:
                    edges[(a, b)] = float(w)
        graph = ig.Graph(n=n_incidents, edges=list(edges))
        return cpm_leiden(graph, list(edges.values()), settings.family_resolution, settings.seed)
    top = incident_top_entities(incident, documents)
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
                    edges[(lo, hi)] = inter / union
    graph = ig.Graph(n=n_incidents, edges=list(edges))
    return cpm_leiden(graph, list(edges.values()), settings.family_entity_threshold, settings.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="attention.atomic output")
    parser.add_argument("--embeddings", type=Path, help="attention.embed output (title channel)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=ClusterSettings.threshold)
    parser.add_argument("--resolution", type=float, default=ClusterSettings.resolution)
    parser.add_argument(
        "--family-resolution", type=float, default=ClusterSettings.family_resolution
    )
    parser.add_argument("--channels", default=ClusterSettings.channels)
    parser.add_argument("--neighbors", type=int, default=ClusterSettings.neighbors)
    parser.add_argument(
        "--candidate-max-hours", type=float, default=ClusterSettings.candidate_max_hours
    )
    parser.add_argument(
        "--max-feature-articles", type=int, default=ClusterSettings.max_feature_articles
    )
    parser.add_argument(
        "--max-pair-contributions", type=int, default=ClusterSettings.max_pair_contributions
    )
    parser.add_argument("--seed", type=int, default=ClusterSettings.seed)
    parser.add_argument(
        "--corroboration-min-channels",
        type=int,
        default=ClusterSettings.corroboration_min_channels,
    )
    parser.add_argument(
        "--single-channel-title-veto",
        type=float,
        default=ClusterSettings.single_channel_title_veto,
    )
    parser.add_argument(
        "--single-channel-min-features",
        type=int,
        default=ClusterSettings.single_channel_min_features,
    )
    parser.add_argument(
        "--family-title-threshold", type=float, default=ClusterSettings.family_title_threshold
    )
    parser.add_argument(
        "--family-entity-threshold", type=float, default=ClusterSettings.family_entity_threshold
    )
    parser.add_argument(
        "--family-strong-title", type=float, default=ClusterSettings.family_strong_title
    )
    parser.add_argument(
        "--reuse-pairs",
        type=Path,
        help="previous cluster output: skip retrieval/scoring, re-gate its pair_features",
    )
    args = parser.parse_args()
    settings = ClusterSettings(
        threshold=args.threshold,
        resolution=args.resolution,
        family_resolution=args.family_resolution,
        channels=args.channels,
        neighbors=args.neighbors,
        candidate_max_hours=args.candidate_max_hours,
        max_feature_articles=args.max_feature_articles,
        max_pair_contributions=args.max_pair_contributions,
        seed=args.seed,
        corroboration_min_channels=args.corroboration_min_channels,
        single_channel_title_veto=args.single_channel_title_veto,
        single_channel_min_features=args.single_channel_min_features,
        family_title_threshold=args.family_title_threshold,
        family_entity_threshold=args.family_entity_threshold,
        family_strong_title=args.family_strong_title,
    )
    run(args.features, args.embeddings, args.output, settings, reuse_pairs=args.reuse_pairs)


def document_languages(documents: pl.DataFrame) -> np.ndarray:
    return documents["language"].fill_null("und").to_numpy()


def with_same_language(pairs: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
    """Backfill ``same_language`` on pair_features written before the column existed."""
    if "same_language" in pairs.columns:
        return pairs
    languages = document_languages(documents)
    left, right = pairs["left"].to_numpy(), pairs["right"].to_numpy()
    return pairs.with_columns(pl.Series("same_language", languages[left] == languages[right]))


def load_inputs(
    features: Path, embeddings: Path | None, settings: ClusterSettings
) -> tuple[pl.DataFrame, pl.DataFrame, TitleChannel | None, dict]:
    documents = pl.read_parquet(features / "documents.parquet").sort("document_id")
    links = pl.read_parquet(features / "document_events.parquet")
    count = documents.height
    if documents["document_id"].to_list() != list(range(count)):
        raise ValueError("documents.parquet must have contiguous document_id 0..N-1")
    title: TitleChannel | None = None
    embed_meta: dict = {}
    if "title" in settings.channels.split(",") and embeddings is not None:
        loaded = load_embeddings(embeddings)
        if loaded is not None and loaded[0].size:
            vectors, ids = loaded
            boilerplate = boilerplate_titles(documents)
            keep = ~np.isin(ids, boilerplate)
            languages = documents["language"].fill_null("und").to_numpy()
            domains = documents["domain"].fill_null("").to_numpy()
            title = TitleChannel(vectors[keep], ids[keep], count, languages, settings.seed, domains)
            embed_meta = json.loads((embeddings / "embed.json").read_text())
            embed_meta["boilerplate_titles_excluded"] = int(len(boilerplate))
    return documents, links, title, embed_meta


def run(
    features: Path,
    embeddings: Path | None,
    output: Path,
    settings: ClusterSettings,
    reuse_pairs: Path | None = None,
) -> dict:
    article = ArticleSettings(
        neighbors=settings.neighbors,
        max_feature_articles=settings.max_feature_articles,
        max_pair_contributions=settings.max_pair_contributions,
    )
    v2 = V2Settings()
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    documents, links, title, embed_meta = load_inputs(features, embeddings, settings)
    count = documents.height
    first_seen = documents["first_seen"].dt.epoch("s").to_numpy().astype(np.float64)
    wanted = [c for c in settings.channels.split(",") if c]
    if title is not None:
        title.background_table().write_parquet(output / "title_background.parquet")
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
    if reuse_pairs is not None:
        previous = json.loads((reuse_pairs / "run.json").read_text())
        audit["channels"] = previous["channels"]
        audit["reused_pairs_from"] = str(reuse_pairs)
        candidates = pl.read_parquet(reuse_pairs / "candidate_pairs.parquet")
        pairs = apply_gate(
            with_same_language(pl.read_parquet(reuse_pairs / "pair_features.parquet"), documents),
            settings,
        )
        return partition(
            candidates, pairs, documents, links, title, settings, output, audit, started
        )
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
        document_languages(documents),
    )
    return partition(candidates, pairs, documents, links, title, settings, output, audit, started)


def partition(
    candidates: pl.DataFrame,
    pairs: pl.DataFrame,
    documents: pl.DataFrame,
    links: pl.DataFrame,
    title: TitleChannel | None,
    settings: ClusterSettings,
    output: Path,
    audit: dict,
    started: float,
    write_pairs: bool = True,
) -> dict:
    """Gate -> Leiden incidents -> story families -> memberships, from scored pairs."""
    count = documents.height
    edges = pairs.filter(pl.col("gated") >= settings.threshold).select(
        "left", "right", "gated", "combined", "evidence_channels"
    )
    print(f"Leiden over {edges.height:,} edges...", flush=True)
    incident = (
        leiden(edges, count, settings) if edges.height else np.full(count, -1, dtype=np.int64)
    )
    family_of_incident = link_families(incident, documents, title, settings, links)
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
            "families": int(len(np.unique(family[family >= 0]))),
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
    if write_pairs:
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
