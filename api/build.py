"""Turn raw GDELT v2 Mentions/Events files into the tables the API serves.

Nothing here invents content: every served field is either copied from the raw
records or derived from them by a rule recorded in `meta.json`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from api.schema import (
    EVENT_COLUMNS,
    EVENT_ROOT_CODES,
    LANGUAGE_NAMES,
    MENTION_COLUMNS,
    TLD_COUNTRIES,
)

MENTION_INTEGERS = ["GlobalEventID", "MentionType", "Confidence", "MentionDocLen"]
EVENT_INTEGERS = ["GlobalEventID", "QuadClass", "NumMentions", "NumSources", "NumArticles"]
EVENT_FLOATS = [
    "GoldsteinScale",
    "AvgTone",
    "ActionGeo_Lat",
    "ActionGeo_Long",
]
TABLE_FILES = {"mentions": "*.mentions.csv", "events": "*.export.csv"}
STAMP_FORMAT = "%Y%m%d%H%M%S"


def read_raw(directory: Path, table: str) -> tuple[pl.DataFrame, list[str]]:
    """Read every GDELT file of one table in `directory` as strings."""
    columns = MENTION_COLUMNS if table == "mentions" else EVENT_COLUMNS
    paths = sorted(directory.glob(TABLE_FILES[table]))
    if not paths:
        raise ValueError(f"No {table} files found in {directory}")
    frames = [
        pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            quote_char=None,
            encoding="utf8-lossy",
            schema={column: pl.String for column in columns},
        )
        for path in paths
        if path.stat().st_size > 0
    ]
    return pl.concat(frames), [path.name for path in paths]


def parse_stamp(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Datetime, STAMP_FORMAT, strict=False)


def source_language(column: str) -> pl.Expr:
    """`srclc:fra;eng:Moses…` -> a language name; empty translation info means English."""
    code = pl.col(column).str.extract(r"srclc:([a-zA-Z]+)", 1).str.to_lowercase()
    named = pl.coalesce([code.replace_strict(LANGUAGE_NAMES, default=None), code])
    return pl.when(code.is_null()).then(pl.lit("English")).otherwise(named)


def derived_country() -> pl.Expr:
    """Approximate the DOC API's publisher country from the domain's ccTLD."""
    tld = pl.col("domain").str.to_lowercase().str.extract(r"\.([a-z]{2})$", 1)
    return tld.replace_strict(TLD_COUNTRIES, default="").fill_null("")


def derived_title() -> pl.Expr:
    """GDELT's DOC API returns real headlines; the raw archive has none.

    The closest honest substitute is the article slug, so titles are labelled as
    url-derived everywhere they are served.
    """
    slug = (
        pl.col("url")
        .str.replace(r"^https?://[^/]+/?", "")
        .str.split("?")
        .list.first()
        .str.split("#")
        .list.first()
        .str.split("/")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    )
    last = slug.list.last().fill_null("")
    previous = slug.list.slice(-2, 1).list.first().fill_null("")
    tail = (
        pl.when(last.str.contains(r"[A-Za-z]{3}"))
        .then(last)
        .when(previous.str.contains(r"[A-Za-z]{3}"))
        .then(previous)
        .otherwise(pl.lit(""))
    )
    words = (
        tail.str.replace(r"\.(html?|shtml|phtml|php|aspx?|jsp|cms|amp)$", "")
        .str.replace_all(r"[^A-Za-z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    return (
        pl.when(words.str.len_chars() < 3)
        .then(pl.col("domain") + pl.lit(" article"))
        .otherwise(words)
    )


def build_events(directory: Path) -> tuple[pl.DataFrame, list[str]]:
    raw, files = read_raw(directory, "events")
    events = raw.select(
        [
            *[pl.col(column).cast(pl.Int64, strict=False) for column in EVENT_INTEGERS],
            *[pl.col(column).cast(pl.Float64, strict=False) for column in EVENT_FLOATS],
            pl.col("Day"),
            parse_stamp("DATEADDED").alias("dateadded"),
            pl.col("Actor1Name").fill_null(""),
            pl.col("Actor2Name").fill_null(""),
            pl.col("Actor1CountryCode").fill_null(""),
            pl.col("Actor2CountryCode").fill_null(""),
            pl.col("EventCode"),
            pl.col("EventRootCode"),
            pl.col("ActionGeo_FullName").fill_null(""),
            pl.col("ActionGeo_CountryCode").fill_null(""),
            pl.col("SOURCEURL").alias("sourceurl"),
        ]
    )
    events = events.with_columns(
        pl.col("EventRootCode")
        .replace_strict(EVENT_ROOT_CODES, default="Unknown")
        .alias("root_label")
    )
    return events.unique(subset=["GlobalEventID"], keep="first"), files


def build_articles(
    directory: Path, events: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    raw, files = read_raw(directory, "mentions")
    mentions = raw.select(
        [
            *[pl.col(column).cast(pl.Int64, strict=False) for column in MENTION_INTEGERS],
            parse_stamp("MentionTimeDate").alias("seen"),
            pl.col("MentionSourceName").fill_null("").alias("domain"),
            pl.col("MentionIdentifier").alias("url"),
            pl.col("MentionDocTone").cast(pl.Float64, strict=False).alias("tone"),
            source_language("MentionDocTranslationInfo").alias("language"),
        ]
    ).filter(
        (pl.col("MentionType") == 1)
        & pl.col("url").str.starts_with("http")
        & pl.col("seen").is_not_null()
    )

    links = mentions.select(
        pl.col("url"),
        pl.col("GlobalEventID"),
        pl.col("Confidence").alias("confidence"),
        pl.col("seen"),
    ).unique(subset=["url", "GlobalEventID"], keep="first")

    labels = (
        links.join(events, on="GlobalEventID", how="inner")
        .group_by("url")
        .agg(
            pl.col("Actor1Name").filter(pl.col("Actor1Name") != "").unique().alias("actors1"),
            pl.col("Actor2Name").filter(pl.col("Actor2Name") != "").unique().alias("actors2"),
            pl.col("ActionGeo_FullName")
            .filter(pl.col("ActionGeo_FullName") != "")
            .unique()
            .alias("locations"),
            pl.col("ActionGeo_CountryCode")
            .filter(pl.col("ActionGeo_CountryCode") != "")
            .unique()
            .alias("countries"),
            pl.col("root_label").unique().alias("themes"),
            pl.col("QuadClass").drop_nulls().unique().alias("quadclasses"),
            pl.col("AvgTone").mean().alias("event_tone"),
        )
    )

    articles = (
        mentions.group_by("url")
        .agg(
            pl.col("seen").min().alias("seendate"),
            pl.col("domain").first(),
            pl.col("tone").first(),
            pl.col("language").first(),
            pl.col("MentionDocLen").max().alias("doclen"),
            pl.col("Confidence").max().alias("confidence"),
            pl.col("GlobalEventID").unique().alias("eventids"),
        )
        .with_columns(
            pl.col("eventids").list.len().alias("numevents"),
            derived_title().alias("title"),
            derived_country().alias("sourcecountry"),
        )
        .join(labels, on="url", how="left")
    )

    articles = articles.with_columns(
        [
            pl.col(column).fill_null([])
            for column in ["actors1", "actors2", "locations", "countries", "themes", "quadclasses"]
        ]
    ).with_columns(
        pl.concat_str(
            [
                pl.col("title"),
                pl.col("domain"),
                pl.col("actors1").list.join(" "),
                pl.col("actors2").list.join(" "),
                pl.col("locations").list.join(" "),
                pl.col("themes").list.join(" "),
            ],
            separator=" ",
        )
        .str.to_lowercase()
        .alias("searchtext"),
        pl.concat_list([pl.col("actors1"), pl.col("actors2")]).list.unique().alias("actors"),
    )
    return articles.drop(["actors1", "actors2"]), links, files


def build(mentions_dir: Path, events_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, event_files = build_events(events_dir)
    articles, links, mention_files = build_articles(mentions_dir, events)

    # Only events observed in the mention window can be served with article context.
    events = events.join(links.select("GlobalEventID").unique(), on="GlobalEventID", how="semi")

    articles.write_parquet(output_dir / "articles.parquet")
    events.write_parquet(output_dir / "events.parquet")
    links.write_parquet(output_dir / "article_events.parquet")

    span = articles.select(
        pl.col("seendate").min().alias("start"), pl.col("seendate").max().alias("end")
    ).row(0)
    meta = {
        "built_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "s3://gdelt-open-data/v2 (Mentions + Events), TSV, no header",
        "mention_files": len(mention_files),
        "event_files": len(event_files),
        "first_mention_file": mention_files[0],
        "last_mention_file": mention_files[-1],
        "articles": articles.height,
        "events": events.height,
        "article_event_links": links.height,
        "coverage_start": span[0].strftime("%Y%m%d%H%M%S") if span[0] else None,
        "coverage_end": span[1].strftime("%Y%m%d%H%M%S") if span[1] else None,
        "field_provenance": {
            "url": "Mentions.MentionIdentifier (MentionType=1 web documents only)",
            "seendate": "earliest Mentions.MentionTimeDate for the URL, GDELT's observation time",
            "domain": "Mentions.MentionSourceName",
            "tone": "Mentions.MentionDocTone",
            "language": "srclc code in MentionDocTranslationInfo; empty is reported as English",
            "title": "DERIVED from the URL slug; the archive carries no headline text",
            "sourcecountry": "DERIVED from the domain's country-code TLD; blank for generic TLDs",
            "socialimage": "not available in Mentions/Events; always empty",
            "url_mobile": "not available in Mentions/Events; always empty",
            "themes/actors/locations": "CAMEO labels of the events the article mentions",
            "search": "matches the derived title, domain and linked CAMEO labels, not body text",
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
