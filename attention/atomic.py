"""Aggregate cleaned Mentions/Events/GKG into per-GlobalEventID atomic-event features.

Reads the typed Parquet from ``attention.preprocess`` and writes:

* ``documents.parquet``        one row per canonical web URL seen in GKG or in a
  web Mention: domain, publisher country (+confidence), first/last observation,
  title, language, GKG themes/entities (when GKG covered the URL), wire group.
* ``document_events.parquet``  URL x GlobalEventID links with confidence and
  raw-text support.
* ``atomic_events.parquet``    per-GlobalEventID aggregates: document, source and
  effective-source counts, confidence statistics, raw-text support rate, entity
  sets, IDF-weighted top themes, CAMEO codes, action geography, time window.

``wire_group`` identifies syndicated copies: documents sharing a normalised title
(when titles exist) or an identical GKG (persons, organizations, themes) fingerprint.
``effective_reports`` counts one report per wire group; it is a floor on
independence, not proof of it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl

TOP_THEMES = 10


def normalised_title() -> pl.Expr:
    return (
        pl.col("title")
        .str.to_lowercase()
        .str.replace_all(r"\s+[-|–—:]\s+[^-|–—:]{1,60}$", "")
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )


def build_documents(
    mentions: pl.DataFrame, gkg: pl.DataFrame, sources: pl.DataFrame
) -> pl.DataFrame:
    web = mentions.filter(
        (pl.col("MentionType") == 1) & pl.col("MentionIdentifier").str.contains(r"(?i)^https?://")
    )
    mention_docs = web.group_by("canonical_url").agg(
        pl.col("domain").first().alias("mention_domain"),
        pl.col("MentionTimeDate").min().alias("mention_first"),
        pl.col("MentionTimeDate").max().alias("mention_last"),
        pl.col("GlobalEventID").n_unique().alias("event_count"),
        pl.col("translated").any().alias("mention_translated"),
        pl.col("source_language").drop_nulls().first().alias("mention_language"),
        pl.col("MentionDocTone").mean().alias("mention_tone"),
    )
    gkg_docs = gkg.filter(pl.col("canonical_url").str.contains(r"(?i)^https?://")).select(
        "canonical_url",
        pl.col("domain").alias("gkg_domain"),
        pl.col("gkg_time"),
        pl.col("page_title").alias("title"),
        pl.col("source_language").alias("gkg_language"),
        pl.col("translated").alias("gkg_translated"),
        "themes",
        "persons",
        "organizations",
        "locations",
        "tone",
        "word_count",
        pl.lit(True).alias("has_gkg"),
    )
    docs = (
        gkg_docs.join(mention_docs, on="canonical_url", how="full", coalesce=True)
        .with_columns(
            pl.coalesce("gkg_domain", "mention_domain").alias("domain"),
            pl.min_horizontal("gkg_time", "mention_first").alias("first_seen"),
            pl.max_horizontal("gkg_time", "mention_last").alias("last_seen"),
            pl.col("has_gkg").fill_null(False),
            (
                pl.col("gkg_translated").fill_null(False)
                | pl.col("mention_translated").fill_null(False)
            ).alias("translated"),
            pl.coalesce("gkg_language", "mention_language").alias("source_language"),
            pl.col("event_count").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("translated"))
            .then(pl.col("source_language"))
            .otherwise(pl.lit("eng"))
            .alias("language")
        )
        .drop(
            "gkg_domain",
            "mention_domain",
            "gkg_time",
            "mention_first",
            "mention_last",
            "gkg_translated",
            "mention_translated",
            "gkg_language",
            "mention_language",
        )
        .join(
            sources.select(
                "domain",
                "publisher_country",
                pl.col("country_confidence").alias("publisher_country_confidence"),
            ),
            on="domain",
            how="left",
        )
        .sort("first_seen", "canonical_url")
        .with_row_index("document_id")
        .with_columns(pl.col("document_id").cast(pl.Int64))
    )
    entity_fingerprint = pl.concat_str(
        [
            pl.col("persons").list.sort().list.join("|"),
            pl.col("organizations").list.sort().list.join("|"),
            pl.col("themes").list.sort().list.join("|"),
        ],
        separator="##",
    )
    has_entities = pl.col("has_gkg") & (
        (pl.col("persons").list.len() + pl.col("organizations").list.len()) > 0
    )
    title_key = normalised_title()
    wire_group = (
        pl.when(pl.col("title").is_not_null() & (title_key.str.len_chars() >= 15))
        .then(pl.lit("t:") + title_key)
        .when(has_entities)
        .then(pl.lit("e:") + entity_fingerprint)
        .otherwise(pl.lit("u:") + pl.col("canonical_url"))
        .hash()
        .alias("wire_group")
    )
    return docs.with_columns(wire_group).with_columns(
        pl.col("wire_group").alias("content_fingerprint")
    )


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
        pl.col("wire_group").n_unique().alias("effective_source_count"),
        pl.col("publisher_country").drop_nulls().n_unique().alias("publisher_country_count"),
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
        "documents_with_title": documents["title"].drop_nulls().len(),
        "documents_with_events": documents.filter(pl.col("event_count") > 0).height,
        "documents_translated": int(documents["translated"].sum()),
        "documents_with_publisher_country": documents["publisher_country"].drop_nulls().len(),
        "document_event_links": links.height,
        "atomic_events": atomic.height,
        "atomic_events_without_events_row": atomic["EventCode"].null_count(),
        "atomic_events_single_document": atomic.filter(pl.col("document_count") == 1).height,
        "distinct_wire_groups": documents["wire_group"].n_unique(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
