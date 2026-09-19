"""Convert one day of raw GDELT v2 Mentions, Events and GKG into typed Parquet.

Outputs (under ``--output``):

* ``mentions.parquet``  every Mentions row, typed, with canonical URL and domain.
* ``events.parquet``    every Events row, typed, with parsed geography.
* ``gkg.parquet``       one row per GKG document with de-duplicated themes,
  persons, organizations and locations, document tone and language.
* ``sources.parquet``   source-domain -> publisher country with mapping provenance.
* ``audit.json``        file manifest, hashes, row counts, malformed-row counts.

Timestamps are parsed as UTC. ``MentionTimeDate`` is when GDELT observed the
document, not when it was published; ``EventTimeDate`` is when GDELT first
recorded the event. Rows with a malformed field count are counted and dropped
rather than silently coerced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import polars as pl

from api.schema import EVENT_COLUMNS, MENTION_COLUMNS

GKG_COLUMNS = [
    "GKGRecordID",
    "Date",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]
GKG_KEEP = [
    "GKGRecordID",
    "Date",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Themes",
    "V2Locations",
    "Persons",
    "Organizations",
    "V2Tone",
    "TranslationInfo",
    "Extras",
]
MENTION_INTS = [
    "GlobalEventID",
    "MentionType",
    "SentenceID",
    "Actor1CharOffset",
    "Actor2CharOffset",
    "ActionCharOffset",
    "InRawText",
    "Confidence",
    "MentionDocLen",
]
EVENT_INTS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "IsRootEvent",
    "QuadClass",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "Actor1Geo_Type",
    "Actor2Geo_Type",
    "ActionGeo_Type",
]
EVENT_FLOATS = [
    "FractionDate",
    "GoldsteinScale",
    "AvgTone",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "ActionGeo_Lat",
    "ActionGeo_Long",
]
TRACKING_PARAMS = {"fbclid", "gclid", "ref", "source", "cmp", "ncid", "ito", "mc_cid", "mc_eid"}
GENERIC_TWO_LETTER = {"co", "io", "me", "tv", "fm", "ai", "ly", "to", "cc", "ws", "gg", "am"}


def parse_yyyymmdd(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d")


def expected_files(directory: Path, suffix: str, day: datetime) -> tuple[list[Path], list[str]]:
    stamp = day
    expected = []
    while stamp < day + timedelta(days=1):
        expected.append(f"{stamp:%Y%m%d%H%M%S}.{suffix}")
        stamp += timedelta(minutes=15)
    present = [directory / name for name in expected if (directory / name).is_file()]
    missing = sorted(set(expected) - {p.name for p in present})
    return present, missing


def read_raw(paths: list[Path], columns: list[str], keep: list[str]) -> tuple[pl.DataFrame, dict]:
    frames = []
    manifest = []
    malformed_total = 0
    for path in paths:
        digest = hashlib.sha256()
        rows = 0
        malformed = 0
        good: list[str] = []
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                rows += 1
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if text.count("\t") != len(columns) - 1:
                    malformed += 1
                    continue
                good.append(text)
        malformed_total += malformed
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": rows,
                "malformed_rows": malformed,
                "sha256": digest.hexdigest(),
            }
        )
        if good:
            frame = pl.read_csv(
                ("\n".join(good)).encode("utf-8"),
                separator="\t",
                has_header=False,
                quote_char=None,
                schema={c: pl.String for c in columns},
                encoding="utf8",
            ).select(keep)
            frames.append(frame.with_columns(pl.lit(path.name).alias("source_file")))
    if not frames:
        raise ValueError(f"No rows read for {columns[0]} table")
    data = pl.concat(frames)
    return data, {"files": manifest, "rows": data.height, "malformed_rows": malformed_total}


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (parts.hostname or "").lower().removeprefix("www.")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
    ]
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower() or "http", host, path, urlencode(query), ""))


def url_domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def to_utc(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .str.to_datetime("%Y%m%d%H%M%S", strict=False)
        .dt.replace_time_zone("UTC")
        .alias(column)
    )


def typed_mentions(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.with_columns(
            pl.col(MENTION_INTS).cast(pl.Int64, strict=False),
            pl.col("MentionDocTone").cast(pl.Float64, strict=False),
            to_utc("EventTimeDate"),
            to_utc("MentionTimeDate"),
        )
        .with_columns(
            pl.col("MentionIdentifier")
            .map_elements(canonical_url, return_dtype=pl.String)
            .alias("canonical_url"),
            pl.col("MentionIdentifier")
            .map_elements(url_domain, return_dtype=pl.String)
            .alias("domain"),
            (pl.col("MentionDocTranslationInfo").fill_null("").str.strip_chars() != "").alias(
                "translated"
            ),
            pl.col("MentionDocTranslationInfo")
            .str.extract(r"(?i)srclc:([^; ]+)", 1)
            .alias("source_language"),
        )
        .drop("Extras")
    )


def typed_events(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.with_columns(
        pl.col(EVENT_INTS).cast(pl.Int64, strict=False),
        pl.col(EVENT_FLOATS).cast(pl.Float64, strict=False),
        to_utc("DATEADDED"),
        pl.col("SOURCEURL")
        .map_elements(canonical_url, return_dtype=pl.String)
        .alias("canonical_url"),
    )


def split_list(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).fill_null("") == "")
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col(column).str.split(";"))
        .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        .list.unique(maintain_order=True)
        .alias(column)
    )


def parse_locations(raw: str | None) -> list[dict[str, str | float | int | None]]:
    if not raw:
        return []
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, str | float | int | None]] = []
    for item in raw.split(";"):
        fields = item.split("#")
        if len(fields) < 7:
            continue
        key = (fields[0], fields[1], fields[2], fields[3])
        if key in seen:
            continue
        seen.add(key)
        try:
            lat = float(fields[5]) if fields[5] else None
            lon = float(fields[6]) if fields[6] else None
            geo_type = int(fields[0])
        except ValueError:
            continue
        out.append(
            {
                "geo_type": geo_type,
                "name": fields[1],
                "country_code": fields[2] or None,
                "adm1_code": fields[3] or None,
                "lat": lat,
                "lon": lon,
            }
        )
    return out


LOCATION_DTYPE = pl.List(
    pl.Struct(
        {
            "geo_type": pl.Int64,
            "name": pl.String,
            "country_code": pl.String,
            "adm1_code": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
        }
    )
)


def typed_gkg(raw: pl.DataFrame) -> pl.DataFrame:
    tone = pl.col("V2Tone").str.split(",")
    frame = (
        raw.with_columns(
            to_utc("Date"),
            pl.col("SourceCollectionIdentifier").cast(pl.Int64, strict=False),
            split_list("Themes").alias("themes"),
            split_list("Persons").alias("persons"),
            split_list("Organizations").alias("organizations"),
            pl.col("V2Locations")
            .map_elements(parse_locations, return_dtype=LOCATION_DTYPE)
            .alias("locations"),
            tone.list.get(0, null_on_oob=True).cast(pl.Float64, strict=False).alias("tone"),
            tone.list.get(1, null_on_oob=True).cast(pl.Float64, strict=False).alias("positive"),
            tone.list.get(2, null_on_oob=True).cast(pl.Float64, strict=False).alias("negative"),
            tone.list.get(3, null_on_oob=True).cast(pl.Float64, strict=False).alias("polarity"),
            tone.list.get(6, null_on_oob=True).cast(pl.Int64, strict=False).alias("word_count"),
            pl.col("TranslationInfo")
            .str.extract(r"(?i)srclc:([^; ]+)", 1)
            .alias("source_language"),
            pl.col("Extras").str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>").alias("page_title"),
            pl.col("DocumentIdentifier")
            .map_elements(canonical_url, return_dtype=pl.String)
            .alias("canonical_url"),
            pl.col("DocumentIdentifier")
            .map_elements(url_domain, return_dtype=pl.String)
            .alias("domain"),
        )
        .drop(
            "Themes",
            "Persons",
            "Organizations",
            "V2Locations",
            "V2Tone",
            "TranslationInfo",
            "Extras",
        )
        .rename({"Date": "gkg_time"})
    )
    return frame.sort("gkg_time").unique("canonical_url", keep="first", maintain_order=True)


def load_domain_lookup(path: Path) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[1] == "":
                continue
            lookup[fields[0].lower().removeprefix("www.")] = (fields[1], fields[2])
    return lookup


def resolve_country(
    domain: str, lookup: dict[str, tuple[str, str]]
) -> tuple[str | None, float, str]:
    """Map a domain to a publisher country with (code, confidence, method)."""
    if domain in lookup:
        return lookup[domain][0], 0.9, "gdelt_lookup"
    labels = domain.split(".")
    for i in range(1, len(labels) - 1):
        parent = ".".join(labels[i:])
        if parent in lookup:
            return lookup[parent][0], 0.7, "gdelt_lookup_parent"
    tld = labels[-1] if labels else ""
    if len(tld) == 2 and tld.isalpha() and tld not in GENERIC_TWO_LETTER:
        code = {"uk": "UK"}.get(tld, tld.upper())
        return code, 0.5, "cctld"
    return None, 0.0, "unresolved"


def build_sources(
    mentions: pl.DataFrame, gkg: pl.DataFrame, lookup: dict[str, tuple[str, str]]
) -> pl.DataFrame:
    names = (
        pl.concat(
            [
                mentions.select("domain", pl.col("MentionSourceName").alias("publisher_name")),
                gkg.select("domain", pl.col("SourceCommonName").alias("publisher_name")),
            ]
        )
        .filter(pl.col("domain") != "")
        .group_by("domain")
        .agg(pl.col("publisher_name").drop_nulls().mode().first().alias("publisher_name"))
    )
    resolved = [resolve_country(domain, lookup) for domain in names["domain"].to_list()]
    return names.with_columns(
        pl.Series("country", [r[0] for r in resolved], dtype=pl.String),
        pl.Series("country_confidence", [r[1] for r in resolved], dtype=pl.Float64),
        pl.Series("mapping_method", [r[2] for r in resolved], dtype=pl.String),
    ).rename({"domain": "source_domain"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_yyyymmdd, required=True, metavar="YYYYMMDD")
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--gkg", type=Path, required=True)
    parser.add_argument("--domain-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    audit: dict = {"date": f"{args.date:%Y%m%d}", "command": sys.argv, "tables": {}}
    inputs = {
        "mentions": (args.mentions, "mentions.CSV", MENTION_COLUMNS, MENTION_COLUMNS),
        "events": (args.events, "export.CSV", EVENT_COLUMNS, EVENT_COLUMNS),
        "gkg": (args.gkg, "gkg.csv", GKG_COLUMNS, GKG_KEEP),
    }
    frames: dict[str, pl.DataFrame] = {}
    for table, (directory, suffix, columns, keep) in inputs.items():
        paths, missing = expected_files(directory, suffix, args.date)
        if not paths:
            paths, missing = expected_files(directory, suffix.lower(), args.date)
        if missing and not args.allow_partial:
            raise ValueError(f"{len(missing)} missing {table} files; pass --allow-partial")
        print(f"Reading {len(paths)} {table} files...", flush=True)
        raw, table_audit = read_raw(paths, columns, keep)
        table_audit["missing_files"] = missing
        audit["tables"][table] = table_audit
        frames[table] = raw

    print("Typing mentions...", flush=True)
    mentions = typed_mentions(frames["mentions"])
    print("Typing events...", flush=True)
    events = typed_events(frames["events"])
    print("Typing GKG...", flush=True)
    gkg = typed_gkg(frames["gkg"])
    del frames

    lookup = load_domain_lookup(args.domain_lookup)
    sources = build_sources(mentions, gkg, lookup)
    web = mentions.filter(pl.col("MentionType") == 1)
    joined = (
        web.select("canonical_url")
        .unique()
        .join(gkg.select("canonical_url"), on="canonical_url", how="inner")
    )
    audit["summary"] = {
        "mention_rows": mentions.height,
        "web_mention_rows": web.height,
        "distinct_web_urls": web["canonical_url"].n_unique(),
        "web_urls_with_gkg": joined.height,
        "event_rows": events.height,
        "gkg_documents": gkg.height,
        "gkg_with_page_title": gkg.filter(pl.col("page_title").is_not_null()).height,
        "domains": sources.height,
        "domains_by_mapping_method": sources.group_by("mapping_method")
        .len()
        .sort("len")
        .to_dicts(),
        "mention_time_range": [
            str(mentions["MentionTimeDate"].min()),
            str(mentions["MentionTimeDate"].max()),
        ],
        "null_confidence_rows": mentions["Confidence"].null_count(),
    }
    for name, frame in [
        ("mentions", mentions),
        ("events", events),
        ("gkg", gkg),
        ("sources", sources),
    ]:
        frame.write_parquet(args.output / f"{name}.parquet")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
