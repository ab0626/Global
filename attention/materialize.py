"""Materialise the queryable event/attention store from one clustering run.

Tables written to ``--output``:

* ``macro_events.parquet``            one row per retained incident (macro-event),
  with its ``incident_id``/``family_id``, representative title, label, geography,
  provisional event types, raw/unique/effective measures, coherence and the
  ``resolution_model`` that produced it.
* ``event_families.parquet``          story families: linked incidents.
* ``macro_event_documents.parquet``   every document (assigned or not) with its
  memberships, ``assignment_score``, publisher country (+confidence), observed time.
* ``country_event_attention.parquet`` macro-event x publisher-country x hour.
* ``country_event_summary.parquet``   macro-event x publisher-country totals, raw and
  effective shares, attention ratio, onset ingredients and lag versus world onset.
* ``country_baseline.parquet``        per-country window denominators.
* ``sources.parquet``                 domain -> publisher country, copied from clean.
* ``meta.json``                       window, denominators, filters, run provenance.

Measures: ``raw_documents`` = canonical URLs; ``unique_domains`` = distinct
publisher domains; ``effective_reports`` = distinct wire groups (syndicated copies
count once). Shares divide by the same measure over the whole window for that
country; ``attention_ratio`` = effective_share(country) / effective_share(world).
Onset per country = the later of the hour the 3rd distinct outlet appears and the
hour cumulative documents reach 10%; countries with < 3 outlets are suppressed
(null onset). Documents whose publisher country is unresolved or below
``--min-country-confidence`` are excluded from country tables and counted in meta.

Only incidents with at least ``--min-documents`` documents, ``--min-effective-reports``
wire groups and coherence >= ``--min-confidence`` become macro-events; everything
else stays in ``macro_event_documents`` under ``macro_event_id = -1``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from attention.embed import load_embeddings
from attention.preprocess import load_domain_lookup
from clustering_experiment import url_words
from clustering_v2 import clean_token

ONSET_OUTLETS = 3
ONSET_QUANTILE = 0.1
TYPE_MIN_SHARE = 0.3
CAMEO_MIN_SHARE = 0.5
TOP_ENTITIES = 3

THEME_TYPES: list[tuple[str, str]] = [
    ("natural_disaster", r"^NATURAL_DISASTER"),
    ("fire", r"^DISASTER_FIRE|WILDFIRE"),
    ("accident", r"^MANMADE_DISASTER"),
    ("armed_conflict", r"^ARMEDCONFLICT|^WB_2432_FRAGILITY_CONFLICT"),
    ("terrorism", r"^TERROR|^SUICIDE_ATTACK|^EXTREMISM"),
    ("protest", r"^PROTEST|^GENERAL_STRIKE"),
    ("election", r"^ELECTION"),
    ("legal_judicial", r"^TRIAL|^ARREST|^SOC_JUDICIAL|^LEGALIZE"),
    ("crime", r"^CRIME|^KIDNAP|^DRUG_TRADE|^SECURITY_SERVICES"),
    ("death", r"^TAX_FNCACT_VICTIM|^CRISISLEX_T03_DEAD"),
    ("health", r"^HEALTH_|^MEDICAL|^EPIDEMIC|^PANDEMIC|^TAX_DISEASE"),
    ("economy_business", r"^ECON_|^BUS_"),
    ("environment_climate", r"^ENV_|^CLIMATE"),
    ("politics_government", r"^LEGISLATION|^EPU_POLICY|^USPEC_POLICY|^GENERAL_GOVERNMENT"),
    ("religion_culture", r"^RELIGION|^TAX_RELIGION"),
    ("sports", r"^SPORT|^TAX_SPORTS"),
]
MAX_TYPES = 3
CAMEO_ROOT_TYPES = {
    "14": "protest",
    "17": "coercion_repression",
    "18": "armed_conflict",
    "19": "armed_conflict",
    "20": "armed_conflict",
}
LABEL_STOP = {
    "news",
    "article",
    "story",
    "world",
    "local",
    "national",
    "politics",
    "latest",
    "says",
    "new",
    "html",
    "id",
    "content",
    "nation",
    "us",
    "uk",
}


def coherence(members: pl.DataFrame) -> pl.DataFrame:
    """Share of a cluster's documents mentioning any of its top-k people/organizations."""
    entities = (
        members.select(
            "cluster",
            "document_id",
            pl.concat_list(
                pl.col("persons").fill_null([]), pl.col("organizations").fill_null([])
            ).alias("entity"),
        )
        .explode("entity")
        .drop_nulls("entity")
        .unique()
    )
    top = (
        entities.group_by("cluster", "entity")
        .len()
        .sort("len", descending=True)
        .group_by("cluster", maintain_order=True)
        .head(TOP_ENTITIES)
        .select("cluster", "entity")
    )
    covered = entities.join(top, on=["cluster", "entity"]).select("cluster", "document_id").unique()
    sizes = members.group_by("cluster").agg(pl.len().alias("n"))
    return (
        covered.group_by("cluster")
        .len()
        .join(sizes, on="cluster", how="right")
        .with_columns((pl.col("len").fill_null(0) / pl.col("n")).alias("entity_coherence"))
        .select("cluster", "entity_coherence")
    )


def title_coherence(members: pl.DataFrame, vectors: np.ndarray, ids: np.ndarray) -> pl.DataFrame:
    """Mean cosine of member titles to their cluster centroid, plus the medoid title."""
    row_of = {int(i): k for k, i in enumerate(ids)}
    titled = members.filter(pl.col("title").is_not_null()).select("cluster", "document_id", "title")
    rows = np.fromiter((row_of.get(int(d), -1) for d in titled["document_id"]), dtype=np.int64)
    keep = rows >= 0
    titled = titled.filter(pl.Series(keep))
    rows = rows[keep]
    if titled.is_empty():
        return pl.DataFrame(
            schema={"cluster": pl.Int64, "title_coherence": pl.Float64, "title": pl.String}
        )
    clusters = titled["cluster"].to_numpy()
    order = np.argsort(clusters, kind="stable")
    clusters, rows = clusters[order], rows[order]
    titles = titled["title"].to_numpy()[order]
    bounds = np.flatnonzero(np.diff(clusters)) + 1
    out_cluster, out_coh, out_title = [], [], []
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(clusters)], strict=True):
        block = vectors[rows[start:stop]]
        centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        sims = block @ (centroid / norm) if norm > 0 else np.zeros(len(block))
        out_cluster.append(int(clusters[start]))
        out_coh.append(float(sims.mean()))
        out_title.append(str(titles[start + int(np.argmax(sims))]))
    return pl.DataFrame(
        {"cluster": out_cluster, "title_coherence": out_coh, "title": out_title},
        schema={"cluster": pl.Int64, "title_coherence": pl.Float64, "title": pl.String},
    )


def onset(assigned: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Observed attention onset per group: the later of (a) the hour in which the
    ONSET_OUTLETS-th distinct outlet appears and (b) the hour in which cumulative
    documents reach ONSET_QUANTILE of the group's total. Null when (a) is never reached."""
    hourly = assigned.with_columns(pl.col("observed_time").dt.truncate("1h").alias("time_bucket"))
    kth = (
        hourly.sort("time_bucket")
        .unique([*keys, "source_domain"], keep="first", maintain_order=True)
        .with_columns(pl.cum_count("source_domain").over(keys).alias("k"))
        .filter(pl.col("k") == ONSET_OUTLETS)
        .select(*keys, pl.col("time_bucket").alias("third_source_seen"))
    )
    p10 = hourly.group_by(keys).agg(
        pl.col("time_bucket")
        .cast(pl.Int64)
        .quantile(ONSET_QUANTILE, interpolation="higher")
        .cast(pl.Datetime("us", "UTC"))
        .alias("p10_seen")
    )
    return kth.join(p10, on=keys).with_columns(
        pl.max_horizontal("third_source_seen", "p10_seen").alias("onset")
    )


def label_for(urls: list[str]) -> str:
    counts: dict[str, int] = {}
    for url in urls:
        path = re.sub(r"^https?://[^/]+", "", url)
        for word in url_words(path):
            if clean_token(word) and word not in LABEL_STOP and not word.isdecimal():
                counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return " ".join(word for word, _ in ranked)


def event_types(members: pl.DataFrame, atomic_links: pl.DataFrame) -> pl.DataFrame:
    sizes = members.group_by("cluster").agg(pl.len().alias("n"))
    themed = (
        members.select("cluster", "document_id", "themes")
        .explode("themes")
        .drop_nulls("themes")
        .unique()
    )
    frames = []
    for name, pattern in THEME_TYPES:
        hit = (
            themed.filter(pl.col("themes").str.contains(pattern))
            .select("cluster", "document_id")
            .unique()
            .group_by("cluster")
            .len()
            .join(sizes, on="cluster")
            .with_columns((pl.col("len") / pl.col("n")).alias("share"), pl.lit(name).alias("type"))
            .filter(pl.col("share") >= TYPE_MIN_SHARE)
            .select("cluster", "type", "share")
        )
        frames.append(hit)
    cameo = (
        atomic_links.with_columns(
            pl.col("EventRootCode").replace_strict(CAMEO_ROOT_TYPES, default=None).alias("type")
        )
        .drop_nulls("type")
        .select("cluster", "document_id", "type")
        .unique()
        .group_by("cluster", "type")
        .len()
        .join(sizes, on="cluster")
        .with_columns((pl.col("len") / pl.col("n")).alias("share"))
        .filter(pl.col("share") >= CAMEO_MIN_SHARE)
        .select("cluster", "type", "share")
    )
    frames.append(cameo)
    return (
        pl.concat(frames)
        .group_by("cluster", "type")
        .agg(pl.col("share").max())
        .sort("share", descending=True)
        .group_by("cluster", maintain_order=True)
        .agg(
            pl.col("type").head(MAX_TYPES).alias("event_types"),
            pl.col("share").head(MAX_TYPES).alias("event_type_shares"),
        )
    )


def top_list(column: str, k: int = 5) -> pl.Expr:
    return (
        pl.col(column).explode().drop_nulls().value_counts(sort=True).head(k).struct.field(column)
    )


def country_centroids(gkg: pl.DataFrame) -> pl.DataFrame:
    """Representative (lat, lon) per FIPS country code, taken as the median of the
    coordinates GDELT attaches to country-level (geo_type 1) location mentions;
    used only to place publisher-country markers."""
    return (
        gkg.select(pl.col("locations").explode())
        .unnest("locations")
        .filter((pl.col("geo_type") == 1) & pl.col("country_code").is_not_null())
        .group_by(pl.col("country_code").alias("publisher_country"))
        .agg(pl.col("lat").median(), pl.col("lon").median())
    )


def country_names(lookup_path: Path | None) -> pl.DataFrame:
    """FIPS code -> most common display name in the GDELT domain list (empty when
    no list is given; the API then falls back to the code)."""
    if lookup_path is None:
        return pl.DataFrame(schema={"publisher_country": pl.String, "country_name": pl.String})
    pairs = list(load_domain_lookup(lookup_path).values())
    return (
        pl.DataFrame(
            {"publisher_country": [c for c, _ in pairs], "country_name": [n for _, n in pairs]}
        )
        .group_by("publisher_country")
        .agg(pl.col("country_name").mode().first())
    )


def country_tables(
    docs: pl.DataFrame, baseline: pl.DataFrame, world_docs: int, world_reports: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hourly attention and per-country summary for assigned, country-resolved docs."""
    keys = ["macro_event_id", "publisher_country"]
    hourly = (
        docs.with_columns(pl.col("observed_time").dt.truncate("1h").alias("time_bucket"))
        .group_by(*keys, "time_bucket")
        .agg(
            pl.len().alias("raw_documents"),
            pl.col("source_domain").n_unique().alias("unique_domains"),
            pl.col("wire_group").n_unique().alias("effective_reports"),
        )
        .sort(*keys, "time_bucket")
        .with_columns(
            pl.col("raw_documents").cum_sum().over(keys).alias("cumulative_documents"),
        )
        .join(baseline, on="publisher_country")
        .with_columns(
            (pl.col("raw_documents") / pl.col("country_documents")).alias("raw_share"),
        )
    )
    outlets = onset(docs, keys)
    hourly = (
        hourly.join(outlets.select(*keys, "onset"), on=keys, how="left")
        .with_columns(
            (pl.col("time_bucket") == pl.col("onset")).fill_null(False).alias("onset_flag")
        )
        .drop(
            "onset",
            "country_documents",
            "country_domains",
            "country_effective_reports",
            "lat",
            "lon",
            "country_name",
        )
    )
    world_onset = onset(docs, ["macro_event_id"]).select(
        "macro_event_id", pl.col("onset").alias("world_onset")
    )
    event_totals = docs.group_by("macro_event_id").agg(
        pl.len().alias("event_documents"),
        pl.col("wire_group").n_unique().alias("event_effective_reports"),
    )
    summary = (
        docs.group_by(keys)
        .agg(
            pl.len().alias("raw_documents"),
            pl.col("source_domain").n_unique().alias("unique_domains"),
            pl.col("wire_group").n_unique().alias("effective_reports"),
            pl.col("observed_time").min().alias("first_seen"),
            pl.col("observed_time").max().alias("last_seen"),
        )
        .join(baseline, on="publisher_country")
        .join(event_totals, on="macro_event_id")
        .join(outlets, on=keys, how="left")
        .join(world_onset, on="macro_event_id", how="left")
        .with_columns(
            (pl.col("raw_documents") / pl.col("country_documents")).alias("raw_share"),
            (pl.col("effective_reports") / pl.col("country_effective_reports")).alias(
                "effective_share"
            ),
            (pl.col("event_documents") / world_docs).alias("world_raw_share"),
            (pl.col("event_effective_reports") / world_reports).alias("world_effective_share"),
        )
        .with_columns(
            (pl.col("effective_share") / pl.col("world_effective_share")).alias("attention_ratio"),
            ((pl.col("onset") - pl.col("world_onset")).dt.total_minutes() / 60).alias("lag_hours"),
            (pl.col("unique_domains") < ONSET_OUTLETS).alias("suppressed"),
        )
        .sort("macro_event_id", "raw_documents", descending=[False, True])
    )
    return hourly, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-documents", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-effective-reports", type=int, default=5)
    parser.add_argument("--min-country-confidence", type=float, default=0.5)
    parser.add_argument(
        "--domain-lookup", type=Path, help="GDELT domain list; supplies country display names"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    documents = pl.read_parquet(args.features / "documents.parquet")
    links = pl.read_parquet(args.features / "document_events.parquet")
    atomic = pl.read_parquet(args.features / "atomic_events.parquet")
    membership = pl.read_parquet(args.clusters / "incident_memberships.parquet")
    sources = pl.read_parquet(args.clean / "sources.parquet")
    run = json.loads((args.clusters / "run.json").read_text())
    resolution_model = run["resolution_model"]
    embeddings = load_embeddings(args.embeddings) if args.embeddings else None

    primary = membership.filter(pl.col("is_primary")).select(
        "document_id", pl.col("incident_id").alias("cluster"), "family_id", "assignment_score"
    )
    members = documents.join(primary, on="document_id")
    assigned_members = members.filter(pl.col("cluster") >= 0)
    atomic_links = (
        links.join(primary.select("document_id", "cluster"), on="document_id")
        .join(
            atomic.select(
                "GlobalEventID",
                "EventRootCode",
                "ActionGeo_CountryCode",
                "ActionGeo_Lat",
                "ActionGeo_Long",
                "Actor1Name",
                "Actor2Name",
            ),
            on="GlobalEventID",
            how="left",
        )
        .filter(pl.col("cluster") >= 0)
    )
    entity_coh = coherence(assigned_members)
    if embeddings is not None and embeddings[0].size:
        title_coh = title_coherence(assigned_members, *embeddings)
    else:
        title_coh = pl.DataFrame(
            schema={"cluster": pl.Int64, "title_coherence": pl.Float64, "title": pl.String}
        )
    types = event_types(assigned_members, atomic_links)

    geo = (
        atomic_links.drop_nulls("ActionGeo_CountryCode")
        .group_by("cluster", "ActionGeo_CountryCode")
        .agg(
            pl.len().alias("n"), pl.col("ActionGeo_Lat").median(), pl.col("ActionGeo_Long").median()
        )
        .sort("n", descending=True)
        .group_by("cluster", maintain_order=True)
        .agg(
            pl.col("ActionGeo_CountryCode").first().alias("event_country"),
            pl.col("ActionGeo_Lat").first().alias("lat"),
            pl.col("ActionGeo_Long").first().alias("lon"),
        )
    )
    actors = (
        atomic_links.select("cluster", pl.concat_list("Actor1Name", "Actor2Name").alias("actor"))
        .explode("actor")
        .drop_nulls("actor")
        .group_by("cluster", "actor")
        .len()
        .sort("len", descending=True)
        .group_by("cluster", maintain_order=True)
        .agg(pl.col("actor").head(5).alias("actors"))
    )
    events = (
        assigned_members.group_by("cluster")
        .agg(
            pl.col("family_id").first(),
            pl.col("first_seen").min().alias("start_time"),
            pl.col("last_seen").max().alias("end_time"),
            top_list("persons").alias("people"),
            top_list("organizations").alias("organizations"),
            top_list("themes", 8).alias("themes"),
            pl.len().alias("raw_documents"),
            pl.col("domain").n_unique().alias("unique_domains"),
            pl.col("wire_group").n_unique().alias("effective_reports"),
            pl.col("publisher_country").drop_nulls().n_unique().alias("publisher_country_count"),
            pl.col("language").drop_nulls().n_unique().alias("language_count"),
            pl.col("title").drop_nulls().len().alias("titled_documents"),
            pl.col("canonical_url").alias("urls"),
        )
        .join(
            atomic_links.group_by("cluster").agg(
                pl.col("GlobalEventID").n_unique().alias("atomic_event_count")
            ),
            on="cluster",
            how="left",
        )
        .join(entity_coh, on="cluster", how="left")
        .join(title_coh, on="cluster", how="left")
        .join(types, on="cluster", how="left")
        .join(geo, on="cluster", how="left")
        .join(actors, on="cluster", how="left")
        .with_columns(
            pl.col("urls").map_elements(label_for, return_dtype=pl.String).alias("label"),
            pl.col("atomic_event_count").fill_null(0),
            pl.col("event_types").fill_null(pl.lit([], dtype=pl.List(pl.String))),
            pl.col("event_type_shares").fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
            pl.col("actors").fill_null(pl.lit([], dtype=pl.List(pl.String))),
            pl.coalesce("title_coherence", "entity_coherence").alias("cluster_confidence"),
            pl.lit(resolution_model).alias("resolution_model"),
        )
        .drop("urls")
    )
    retained = (
        events.filter(
            (pl.col("raw_documents") >= args.min_documents)
            & (pl.col("effective_reports") >= args.min_effective_reports)
            & (pl.col("cluster_confidence") >= args.min_confidence)
        )
        .sort("effective_reports", "raw_documents", "cluster", descending=[True, True, False])
        .with_row_index("macro_event_id")
        .with_columns(pl.col("macro_event_id").cast(pl.Int64))
    )
    macro_events = retained.select(
        "macro_event_id",
        pl.col("cluster").alias("incident_id"),
        "family_id",
        "title",
        "label",
        "start_time",
        "end_time",
        "event_country",
        "lat",
        "lon",
        "event_types",
        "event_type_shares",
        "actors",
        "people",
        "organizations",
        "themes",
        "atomic_event_count",
        "raw_documents",
        "unique_domains",
        "effective_reports",
        "publisher_country_count",
        "language_count",
        "titled_documents",
        "entity_coherence",
        "title_coherence",
        "cluster_confidence",
        "resolution_model",
    )
    id_map = retained.select("cluster", "macro_event_id")
    families = (
        macro_events.group_by("family_id")
        .agg(
            pl.col("macro_event_id").sort_by("effective_reports", descending=True),
            pl.col("title").sort_by("effective_reports", descending=True).first(),
            pl.col("label").sort_by("effective_reports", descending=True).first(),
            pl.len().alias("incident_count"),
            pl.col("raw_documents").sum(),
            pl.col("effective_reports").sum(),
            pl.col("start_time").min(),
            pl.col("end_time").max(),
        )
        .rename({"macro_event_id": "macro_event_ids"})
        .sort("effective_reports", descending=True)
    )

    primary_event = (
        links.sort("confidence", descending=True)
        .group_by("document_id", maintain_order=True)
        .agg(
            pl.col("GlobalEventID").first().alias("global_event_id"),
            pl.col("confidence").first(),
            pl.col("in_raw_text").first(),
        )
    )
    macro_event_documents = (
        membership.join(id_map, left_on="incident_id", right_on="cluster", how="left")
        .join(documents, on="document_id")
        .join(primary_event, on="document_id", how="left")
        .select(
            pl.col("macro_event_id").fill_null(-1),
            "document_id",
            "incident_id",
            "family_id",
            "assignment_score",
            "is_primary",
            "global_event_id",
            pl.col("canonical_url").alias("url"),
            "title",
            "language",
            pl.col("domain").alias("source_domain"),
            "publisher_country",
            "publisher_country_confidence",
            pl.col("first_seen").alias("observed_time"),
            "confidence",
            "in_raw_text",
            "wire_group",
        )
        .sort("document_id", "is_primary", descending=[False, True])
    )

    resolved = documents.filter(
        pl.col("publisher_country").is_not_null()
        & (pl.col("publisher_country_confidence") >= args.min_country_confidence)
    )
    baseline = (
        resolved.group_by("publisher_country")
        .agg(
            pl.len().alias("country_documents"),
            pl.col("domain").n_unique().alias("country_domains"),
            pl.col("wire_group").n_unique().alias("country_effective_reports"),
        )
        .join(
            country_centroids(pl.read_parquet(args.clean / "gkg.parquet", columns=["locations"])),
            on="publisher_country",
            how="left",
        )
        .join(country_names(args.domain_lookup), on="publisher_country", how="left")
    )
    world_documents = documents.height
    world_reports = documents["wire_group"].n_unique()
    attention_docs = macro_event_documents.filter(
        (pl.col("macro_event_id") >= 0)
        & pl.col("is_primary")
        & pl.col("publisher_country").is_not_null()
        & (pl.col("publisher_country_confidence") >= args.min_country_confidence)
    )
    hourly, summary = country_tables(attention_docs, baseline, world_documents, world_reports)

    macro_events.write_parquet(args.output / "macro_events.parquet")
    families.write_parquet(args.output / "event_families.parquet")
    macro_event_documents.write_parquet(args.output / "macro_event_documents.parquet")
    hourly.write_parquet(args.output / "country_event_attention.parquet")
    summary.write_parquet(args.output / "country_event_summary.parquet")
    sources.write_parquet(args.output / "sources.parquet")
    baseline.write_parquet(args.output / "country_baseline.parquet")
    in_events = macro_event_documents.filter(pl.col("macro_event_id") >= 0)
    meta = {
        "window_start": str(documents["first_seen"].min()),
        "window_end": str(documents["last_seen"].max()),
        "resolution_model": resolution_model,
        "run_seed": run["settings"]["seed"],
        "documents_total": world_documents,
        "effective_reports_total": world_reports,
        "documents_with_publisher_country": documents["publisher_country"].drop_nulls().len(),
        "documents_country_resolved_at_min_confidence": resolved.height,
        "excluded_document_share": 1 - resolved.height / max(world_documents, 1),
        "documents_with_title": documents["title"].drop_nulls().len(),
        "incidents_total": run["incidents"],
        "families_total": run["families"],
        "macro_events": macro_events.height,
        "documents_in_macro_events": in_events.filter(pl.col("is_primary")).height,
        "unassigned_rate": run["unassigned_rate"],
        "filters": {
            "min_documents": args.min_documents,
            "min_confidence": args.min_confidence,
            "min_effective_reports": args.min_effective_reports,
            "min_country_confidence": args.min_country_confidence,
            "onset_outlets": ONSET_OUTLETS,
            "onset_quantile": ONSET_QUANTILE,
        },
        "denominators": {
            "raw_share": "raw_documents(country, event) / raw_documents(country, window)",
            "effective_share": (
                "effective_reports(country, event) / effective_reports(country, window)"
            ),
            "attention_ratio": "effective_share(country) / effective_share(world)",
            "raw_documents": "canonical URLs (GKG + web Mentions), wire copies counted separately",
            "effective_reports": "distinct wire groups (title / GKG-entity fingerprint)",
        },
        "timing": (
            "observed_time = GDELT observation time (UTC, GKG DATEADDED or MentionTimeDate), "
            "not publication time; onset/lag describe observed media attention, not awareness"
        ),
        "publisher_country": "publisher location from sources.parquet, not event geography",
        "event_types": "provisional rule-based labels from GKG themes and CAMEO root codes",
        "cluster_run": {k: v for k, v in run.items() if k != "command"},
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "cluster_run"}, indent=2, default=str))


if __name__ == "__main__":
    main()
