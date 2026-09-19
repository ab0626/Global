"""Aggregate cleaned Mentions/Events/GKG into per-GlobalEventID atomic-event features.

Reads the typed Parquet from ``attention.preprocess`` and writes:

* ``documents.parquet``        one row per canonical web URL: domain, publisher
  country, first/last observation, GKG themes/entities (when GKG covered the URL).
* ``document_events.parquet``  URL x GlobalEventID links with confidence and
  raw-text support.
* ``atomic_events.parquet``    per-GlobalEventID aggregates: document, source and
  effective-source counts, confidence statistics, raw-text support rate, entity
  sets, IDF-weighted top themes, CAMEO codes, action geography, time window.

"Effective sources" counts distinct publisher domains with distinct GKG entity
fingerprints, so syndicated wire copies collapse to one; it is a floor on
independence, not proof of it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl

TOP_THEMES = 10


def build_documents(
    mentions: pl.DataFrame, gkg: pl.DataFrame, sources: pl.DataFrame
) -> pl.DataFrame:
    web = mentions.filter(
        (pl.col("MentionType") == 1) & pl.col("MentionIdentifier").str.contains(r"(?i)^https?://")
    )
    docs = (
        web.group_by("canonical_url")
        .agg(
            pl.col("domain").first(),
            pl.col("MentionTimeDate").min().alias("first_seen"),
            pl.col("MentionTimeDate").max().alias("last_seen"),
            pl.col("GlobalEventID").n_unique().alias("event_count"),
            pl.col("translated").any().alias("translated"),
            pl.col("source_language").drop_nulls().first().alias("source_language"),
            pl.col("MentionDocTone").mean().alias("mention_tone"),
        )
        .join(
            gkg.select(
                "canonical_url",
                "themes",
                "persons",
                "organizations",
                "locations",
                "tone",
                "word_count",
                pl.lit(True).alias("has_gkg"),
            ),
            on="canonical_url",
            how="left",
        )
        .join(
            sources.select(
                pl.col("source_domain").alias("domain"),
                pl.col("country").alias("source_country"),
                pl.col("country_confidence").alias("source_country_confidence"),
            ),
            on="domain",
            how="left",
        )
        .with_columns(pl.col("has_gkg").fill_null(False))
        .sort("first_seen", "canonical_url")
        .with_row_index("document_id")
    )
    # Wire fingerprint: identical (persons, organizations, themes) sets from GKG.
    fingerprint = (
        pl.when(pl.col("has_gkg"))
        .then(
            pl.concat_str(
                [
                    pl.col("persons").list.sort().list.join("|"),
                    pl.col("organizations").list.sort().list.join("|"),
                    pl.col("themes").list.sort().list.join("|"),
                ],
                separator="##",
            )
        )
        .otherwise(pl.col("canonical_url"))
        .hash()
        .alias("content_fingerprint")
    )
    return docs.with_columns(fingerprint)


def build_document_events(mentions: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
    web = mentions.filter(pl.col("MentionType") == 1)
    return (
        web.join(documents.select("canonical_url", "document_id"), on="canonical_url")
        .group_by("document_id", "GlobalEventID")
        .agg(
            pl.col("Confidence").max().alias("confidence"),
            pl.col("InRawText").max().alias("in_raw_text"),
            pl.col("MentionTimeDate").min().alias("mention_time"),
            pl.col("EventTimeDate").first().alias("event_time"),
        )
    )


def theme_idf(documents: pl.DataFrame) -> pl.DataFrame:
    with_gkg = documents.filter(pl.col("has_gkg"))
    n_docs = max(with_gkg.height, 1)
    counts = (
        with_gkg.select("document_id", "themes")
        .explode("themes")
        .drop_nulls()
        .group_by("themes")
        .len()
    )
    return counts.with_columns(
        (pl.lit(math.log(n_docs + 1)) - (pl.col("len") + 1).log()).alias("idf")
    ).rename({"themes": "theme", "len": "theme_document_count"})


def build_atomic_events(
    links: pl.DataFrame, documents: pl.DataFrame, events: pl.DataFrame, idf: pl.DataFrame
) -> pl.DataFrame:
    joined = links.join(documents, on="document_id")
    base = joined.group_by("GlobalEventID").agg(
        pl.len().alias("document_count"),
        pl.col("domain").n_unique().alias("source_count"),
        pl.col("content_fingerprint").n_unique().alias("effective_source_count"),
        pl.col("source_country").drop_nulls().n_unique().alias("source_country_count"),
        pl.col("confidence").mean().alias("confidence_mean"),
        pl.col("confidence").median().alias("confidence_median"),
        pl.col("in_raw_text").mean().alias("raw_text_rate"),
        pl.col("has_gkg").mean().alias("gkg_coverage"),
        pl.col("mention_time").min().alias("first_seen"),
        pl.col("mention_time").max().alias("last_seen"),
        pl.col("event_time").first().alias("event_time"),
        pl.col("persons").explode().drop_nulls().value_counts(sort=True).head(10).alias("people"),
        pl.col("organizations")
        .explode()
        .drop_nulls()
        .value_counts(sort=True)
        .head(10)
        .alias("organizations"),
        pl.col("document_id").alias("document_ids"),
    )
    themes = (
        joined.select("GlobalEventID", "themes")
        .explode("themes")
        .drop_nulls()
        .group_by("GlobalEventID", "themes")
        .len()
        .join(idf, left_on="themes", right_on="theme")
        .with_columns((pl.col("len") * pl.col("idf")).alias("weight"))
        .sort("weight", descending=True)
        .group_by("GlobalEventID", maintain_order=True)
        .agg(
            pl.struct(theme=pl.col("themes"), weight=pl.col("weight"))
            .head(TOP_THEMES)
            .alias("top_themes")
        )
    )
    event_cols = events.select(
        "GlobalEventID",
        "Actor1Name",
        "Actor2Name",
        "Actor1CountryCode",
        "Actor2CountryCode",
        "EventCode",
        "EventBaseCode",
        "EventRootCode",
        "QuadClass",
        "GoldsteinScale",
        "AvgTone",
        "ActionGeo_Type",
        "ActionGeo_FullName",
        "ActionGeo_CountryCode",
        "ActionGeo_Lat",
        "ActionGeo_Long",
        "NumMentions",
        "NumSources",
        "NumArticles",
    ).unique("GlobalEventID", keep="first")
    return (
        base.join(themes, on="GlobalEventID", how="left")
        .join(event_cols, on="GlobalEventID", how="left")
        .with_columns(
            pl.col("people").list.eval(pl.element().struct.field("persons")).alias("people"),
            pl.col("organizations")
            .list.eval(pl.element().struct.field("organizations"))
            .alias("organizations"),
            pl.col("top_themes").fill_null(
                pl.lit([], dtype=pl.List(pl.Struct({"theme": pl.String, "weight": pl.Float64})))
            ),
        )
        .sort("document_count", descending=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True, help="attention.preprocess output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mentions = pl.read_parquet(args.clean / "mentions.parquet")
    events = pl.read_parquet(args.clean / "events.parquet")
    gkg = pl.read_parquet(args.clean / "gkg.parquet")
    sources = pl.read_parquet(args.clean / "sources.parquet")

    documents = build_documents(mentions, gkg, sources)
    links = build_document_events(mentions, documents)
    idf = theme_idf(documents)
    atomic = build_atomic_events(links, documents, events, idf)
    documents.write_parquet(args.output / "documents.parquet")
    links.write_parquet(args.output / "document_events.parquet")
    idf.write_parquet(args.output / "theme_idf.parquet")
    atomic.write_parquet(args.output / "atomic_events.parquet")
    summary = {
        "documents": documents.height,
        "documents_with_gkg": int(documents["has_gkg"].sum()),
        "documents_with_source_country": documents["source_country"].drop_nulls().len(),
        "document_event_links": links.height,
        "atomic_events": atomic.height,
        "atomic_events_without_events_row": atomic["EventCode"].null_count(),
        "atomic_events_single_document": atomic.filter(pl.col("document_count") == 1).height,
        "distinct_fingerprints": documents["content_fingerprint"].n_unique(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
