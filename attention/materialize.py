"""Materialise the queryable event/attention store from one clustering run.

Tables written to ``--output``:

* ``macro_events.parquet``            one row per retained cluster (macro-event).
* ``macro_event_documents.parquet``   document membership with provenance.
* ``country_event_attention.parquet`` macro-event x publisher-country x hour.
* ``country_event_summary.parquet``   macro-event x publisher-country totals, share,
  over-attention ratio, k-outlet onset and lag versus global onset.
* ``sources.parquet``                 domain -> publisher country, copied from clean.
* ``meta.json``                       window, denominators, run provenance.

Only clusters with at least ``--min-documents`` documents and an entity-coherence
score of at least ``--min-confidence`` become macro-events; everything else stays
in ``macro_event_documents`` under ``macro_event_id = -1`` so nothing is dropped
silently. Event types are provisional rule-based labels from GKG themes and
CAMEO root codes, not a validated taxonomy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

from clustering_experiment import url_words
from clustering_v2 import clean_token

ONSET_OUTLETS = 3
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
        .with_columns((pl.col("len").fill_null(0) / pl.col("n")).alias("cluster_confidence"))
        .select("cluster", "cluster_confidence")
    )


def onset(assigned: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Observed attention onset per group: the later of (a) the hour in which the
    ONSET_OUTLETS-th distinct outlet appears and (b) the hour in which cumulative
    documents reach 10% of the group's total. Null when (a) is never reached."""
    hourly = assigned.with_columns(pl.col("mention_time").dt.truncate("1h").alias("time_bucket"))
    kth = (
        hourly.sort("time_bucket")
        .unique([*keys, "source_domain"], keep="first", maintain_order=True)
        .with_columns(pl.cum_count("source_domain").over(keys).alias("k"))
        .filter(pl.col("k") == ONSET_OUTLETS)
        .select(*keys, pl.col("time_bucket").alias("onset_outlets_time"))
    )
    p10 = hourly.group_by(keys).agg(
        pl.col("time_bucket")
        .cast(pl.Int64)
        .quantile(0.1, interpolation="higher")
        .cast(pl.Datetime("us", "UTC"))
        .alias("onset_p10_time")
    )
    return kth.join(p10, on=keys).with_columns(
        pl.max_horizontal("onset_outlets_time", "onset_p10_time").alias("onset_time")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-documents", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--min-effective-sources", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    documents = pl.read_parquet(args.features / "documents.parquet")
    links = pl.read_parquet(args.features / "document_events.parquet")
    atomic = pl.read_parquet(args.features / "atomic_events.parquet")
    assignment = pl.read_parquet(args.clusters / "document_clusters.parquet")
    sources = pl.read_parquet(args.clean / "sources.parquet")
    cluster_audit = json.loads((args.clusters / "audit.json").read_text())

    members = documents.join(assignment.select("document_id", "cluster"), on="document_id")
    atomic_links = links.join(assignment.select("document_id", "cluster"), on="document_id").join(
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
    confidence = coherence(members.filter(pl.col("cluster") >= 0))
    types = event_types(
        members.filter(pl.col("cluster") >= 0), atomic_links.filter(pl.col("cluster") >= 0)
    )

    geo = (
        atomic_links.filter(pl.col("cluster") >= 0)
        .drop_nulls("ActionGeo_CountryCode")
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
        atomic_links.filter(pl.col("cluster") >= 0)
        .select("cluster", pl.concat_list("Actor1Name", "Actor2Name").alias("actor"))
        .explode("actor")
        .drop_nulls("actor")
        .group_by("cluster", "actor")
        .len()
        .sort("len", descending=True)
        .group_by("cluster", maintain_order=True)
        .agg(pl.col("actor").head(5).alias("actors"))
    )

    def top_list(column: str, k: int = 5) -> pl.Expr:
        return (
            pl.col(column)
            .explode()
            .drop_nulls()
            .value_counts(sort=True)
            .head(k)
            .struct.field(column)
        )

    events = (
        members.filter(pl.col("cluster") >= 0)
        .group_by("cluster")
        .agg(
            pl.col("first_seen").min().alias("start_time"),
            pl.col("last_seen").max().alias("end_time"),
            top_list("persons").alias("people"),
            top_list("organizations").alias("organizations"),
            top_list("themes", 8).alias("themes"),
            pl.len().alias("document_count"),
            pl.col("domain").n_unique().alias("source_count"),
            pl.col("content_fingerprint").n_unique().alias("effective_source_count"),
            pl.col("source_country").drop_nulls().n_unique().alias("country_count"),
            pl.col("canonical_url").alias("urls"),
        )
        .join(
            atomic_links.filter(pl.col("cluster") >= 0)
            .group_by("cluster")
            .agg(pl.col("GlobalEventID").n_unique().alias("atomic_event_count")),
            on="cluster",
        )
        .join(confidence, on="cluster")
        .join(types, on="cluster", how="left")
        .join(geo, on="cluster", how="left")
        .join(actors, on="cluster", how="left")
        .with_columns(
            pl.col("urls").map_elements(label_for, return_dtype=pl.String).alias("label"),
            pl.col("event_types").fill_null(pl.lit([], dtype=pl.List(pl.String))),
            pl.col("event_type_shares").fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
            pl.col("actors").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        )
        .drop("urls")
    )
    retained = (
        events.filter(
            (pl.col("document_count") >= args.min_documents)
            & (pl.col("effective_source_count") >= args.min_effective_sources)
            & (pl.col("cluster_confidence") >= args.min_confidence)
        )
        .sort("effective_source_count", "document_count", descending=[True, True])
        .with_row_index("macro_event_id")
    )
    macro_events = retained.select(
        "macro_event_id",
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
        "document_count",
        "source_count",
        "effective_source_count",
        "country_count",
        "cluster_confidence",
        pl.col("cluster").alias("cluster_id"),
    )
    id_map = retained.select("cluster", pl.col("macro_event_id").cast(pl.Int64))

    primary = (
        links.sort("confidence", descending=True)
        .group_by("document_id", maintain_order=True)
        .agg(
            pl.col("GlobalEventID").first().alias("global_event_id"),
            pl.col("confidence").first(),
            pl.col("in_raw_text").first(),
        )
    )
    macro_event_documents = (
        members.join(id_map, on="cluster", how="left")
        .join(primary, on="document_id", how="left")
        .select(
            pl.col("macro_event_id").fill_null(-1),
            "document_id",
            "global_event_id",
            pl.col("canonical_url").alias("url"),
            pl.col("domain").alias("source_domain"),
            "source_country",
            pl.col("first_seen").alias("mention_time"),
            "confidence",
            "in_raw_text",
            pl.col("content_fingerprint").alias("wire_group"),
            pl.lit(None, dtype=pl.Float64).alias("semantic_similarity"),
            pl.col("cluster").alias("cluster_id"),
        )
    )

    baseline = (
        documents.drop_nulls("source_country")
        .group_by("source_country")
        .agg(
            pl.len().alias("country_documents"),
            pl.col("domain").n_unique().alias("country_domains"),
        )
    )
    world_documents = documents.height
    assigned = macro_event_documents.filter(pl.col("macro_event_id") >= 0).drop_nulls(
        "source_country"
    )
    hourly = (
        assigned.with_columns(pl.col("mention_time").dt.truncate("1h").alias("time_bucket"))
        .group_by("macro_event_id", "source_country", "time_bucket")
        .agg(
            pl.len().alias("documents"),
            pl.col("source_domain").n_unique().alias("sources"),
            pl.col("wire_group").n_unique().alias("effective_sources"),
        )
        .sort("macro_event_id", "source_country", "time_bucket")
        .with_columns(
            pl.col("documents")
            .cum_sum()
            .over("macro_event_id", "source_country")
            .alias("cumulative_attention"),
        )
        .join(baseline, on="source_country")
        .with_columns(
            pl.col("documents").alias("raw_attention"),
            (pl.col("documents") / pl.col("country_documents")).alias("normalized_attention"),
        )
    )
    outlets = onset(assigned, ["macro_event_id", "source_country"])
    hourly = (
        hourly.join(outlets, on=["macro_event_id", "source_country"], how="left")
        .with_columns(
            (pl.col("time_bucket") == pl.col("onset_time")).fill_null(False).alias("onset_flag")
        )
        .rename({"source_country": "country"})
        .drop("onset_time", "onset_outlets_time", "onset_p10_time")
    )
    global_onset = onset(assigned, ["macro_event_id"]).select(
        "macro_event_id", pl.col("onset_time").alias("global_onset")
    )
    event_totals = assigned.group_by("macro_event_id").agg(pl.len().alias("event_documents"))
    summary = (
        assigned.group_by("macro_event_id", "source_country")
        .agg(
            pl.len().alias("documents"),
            pl.col("source_domain").n_unique().alias("sources"),
            pl.col("wire_group").n_unique().alias("effective_sources"),
            pl.col("mention_time").min().alias("first_seen"),
            pl.col("mention_time").max().alias("last_seen"),
        )
        .join(baseline, on="source_country")
        .join(event_totals, on="macro_event_id")
        .join(outlets, on=["macro_event_id", "source_country"], how="left")
        .join(global_onset, on="macro_event_id", how="left")
        .with_columns(
            (pl.col("documents") / pl.col("country_documents")).alias("share"),
            (pl.col("event_documents") / world_documents).alias("world_share"),
        )
        .with_columns(
            (pl.col("share") / pl.col("world_share")).alias("attention_ratio"),
            ((pl.col("onset_time") - pl.col("global_onset")).dt.total_minutes() / 60).alias(
                "lag_hours"
            ),
            (pl.col("sources") < ONSET_OUTLETS).alias("suppressed"),
        )
        .rename({"source_country": "country"})
        .sort("macro_event_id", "documents", descending=[False, True])
    )

    macro_events.write_parquet(args.output / "macro_events.parquet")
    macro_event_documents.write_parquet(args.output / "macro_event_documents.parquet")
    hourly.write_parquet(args.output / "country_event_attention.parquet")
    summary.write_parquet(args.output / "country_event_summary.parquet")
    sources.write_parquet(args.output / "sources.parquet")
    baseline.write_parquet(args.output / "country_baseline.parquet")
    meta = {
        "window_start": str(documents["first_seen"].min()),
        "window_end": str(documents["last_seen"].max()),
        "documents_total": world_documents,
        "documents_with_source_country": documents["source_country"].drop_nulls().len(),
        "clusters_total": cluster_audit["clusters"],
        "macro_events": macro_events.height,
        "documents_in_macro_events": int(
            macro_event_documents.filter(pl.col("macro_event_id") >= 0).height
        ),
        "min_documents": args.min_documents,
        "min_confidence": args.min_confidence,
        "min_effective_sources": args.min_effective_sources,
        "onset_outlets": ONSET_OUTLETS,
        "denominators": {
            "share": "documents(country, event) / documents(country, window)",
            "attention_ratio": "share / (documents(event) / documents(window))",
            "documents": "canonical web URLs observed in Mentions; wire copies counted separately",
            "effective_sources": "distinct GKG entity fingerprints (wire-collapsed)",
        },
        "timestamps": "MentionTimeDate = GDELT observation time (UTC), not publication time",
        "event_types": "provisional rule-based labels from GKG themes and CAMEO root codes",
        "cluster_run": cluster_audit,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "cluster_run"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
