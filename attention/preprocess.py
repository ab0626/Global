"""Convert a window of raw GDELT v2 15-minute files into typed Parquet.

Reads the English and translated file families (``gkg``, ``translation.gkg``,
``mentions``, ``translation.mentions``, ``export``, ``translation.export``) from a
directory of ``.zip`` (or unzipped ``.csv``) files as downloaded by
``scripts/fetch_window.py``. Each 15-minute file is processed on its own so memory
stays bounded by one file, not by the window.

Outputs (under ``--output``):

* ``mentions.parquet``  every Mentions row, typed, with canonical URL and domain.
* ``events.parquet``    every Events row, typed, with parsed geography.
* ``gkg.parquet``       one row per GKG document with de-duplicated themes,
  persons, organizations and locations, tone, language and page title (when the
  Extras field carries one; absent before ~2020).
* ``sources.parquet``   domain -> publisher country with confidence and method.
* ``audit.json``        per-file manifest (hash, rows, malformed rows), missing
  files, summary counts.

Timestamps are parsed as UTC. ``MentionTimeDate``/``DATEADDED`` are when GDELT
observed the document, not when it was published; ``EventTimeDate`` is when GDELT
first recorded the event. Rows with a malformed field count are counted and
dropped rather than silently coerced.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import polars as pl

from api.schema import EVENT_COLUMNS, MENTION_COLUMNS

FILE_KINDS: dict[str, tuple[str, ...]] = {
    "mentions": ("mentions.CSV", "translation.mentions.CSV"),
    "events": ("export.CSV", "translation.export.CSV"),
    "gkg": ("gkg.csv", "translation.gkg.csv"),
}

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
GENERIC_TWO_LETTER = {
    "co",
    "io",
    "me",
    "tv",
    "fm",
    "ai",
    "ly",
    "to",
    "cc",
    "ws",
    "gg",
    "am",
    "eu",
    "su",
}
# ccTLD (ISO 3166-1 alpha-2) -> FIPS 10-4, the code system GDELT uses everywhere
# (ISO 'GE' is Georgia but FIPS 'GE' is Germany, so codes must not be mixed).
# Only the codes that differ are listed; identical codes fall through.
ISO2_TO_FIPS = {
    "AD": "AN", "AG": "AC", "AI": "AV", "AQ": "AY", "AS": "AQ", "AT": "AU", "AU": "AS",
    "AW": "AA", "AZ": "AJ", "BA": "BK", "BD": "BG", "BF": "UV", "BG": "BU", "BH": "BA",
    "BI": "BY", "BJ": "BN", "BL": "TB", "BM": "BD", "BN": "BX", "BO": "BL", "BS": "BF",
    "BW": "BC", "BY": "BO", "BZ": "BH", "CC": "CK", "CD": "CG", "CF": "CT", "CG": "CF",
    "CH": "SZ", "CI": "IV", "CK": "CW", "CL": "CI", "CN": "CH", "CR": "CS", "CX": "KT",
    "CZ": "EZ", "DE": "GM", "DK": "DA", "DM": "DO", "DO": "DR", "DZ": "AG", "EE": "EN",
    "ES": "SP", "GA": "GB", "GB": "UK", "GD": "GJ", "GE": "GG", "GF": "FG", "GG": "GK",
    "GM": "GA", "GN": "GV", "GQ": "EK", "GS": "SX", "GU": "GQ", "GW": "PU", "HN": "HO",
    "HT": "HA", "IE": "EI", "IL": "IS", "IQ": "IZ", "IS": "IC", "JP": "JA", "KH": "CB",
    "KI": "KR", "KM": "CN", "KN": "SC", "KP": "KN", "KR": "KS", "KW": "KU", "KY": "CJ",
    "LB": "LE", "LC": "ST", "LI": "LS", "LK": "CE", "LR": "LI", "LS": "LT", "LT": "LH",
    "LV": "LG", "MA": "MO", "MC": "MN", "ME": "MJ", "MG": "MA", "MH": "RM", "MM": "BM",
    "MN": "MG", "MO": "MC", "MP": "CQ", "MQ": "MB", "MS": "MH", "MU": "MP", "MW": "MI",
    "NA": "WA", "NE": "NG", "NG": "NI", "NI": "NU", "NU": "NE", "OM": "MU", "PA": "PM",
    "PF": "FP", "PG": "PP", "PH": "RP", "PM": "SB", "PN": "PC", "PR": "RQ", "PS": "WE",
    "PT": "PO", "PW": "PS", "PY": "PA", "RS": "RI", "RU": "RS", "SB": "BP", "SC": "SE",
    "SD": "SU", "SE": "SW", "SG": "SN", "SK": "LO", "SN": "SG", "SR": "NS", "SS": "OD",
    "ST": "TP", "SV": "ES", "SZ": "WZ", "TC": "TK", "TD": "CD", "TG": "TO", "TJ": "TI",
    "TK": "TL", "TL": "TT", "TM": "TX", "TN": "TS", "TO": "TN", "TR": "TU", "TT": "TD",
    "UA": "UP", "VA": "VT", "VG": "VI", "VI": "VQ", "VN": "VM", "VU": "NH", "YE": "YM",
    "YT": "MF", "ZA": "SF", "ZM": "ZA", "ZW": "ZI",
}  # fmt: skip


def parse_yyyymmdd(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d")


def expected_files(
    directory: Path, suffix: str, start: datetime, end: datetime
) -> tuple[list[Path], list[str]]:
    """Quarter-hour files for ``suffix`` between ``start`` (inclusive) and ``end``
    (exclusive); accepts ``.zip``, exact-case and lower-case ``.csv`` names."""
    stamp = start
    expected = []
    while stamp < end:
        expected.append(f"{stamp:%Y%m%d%H%M%S}.{suffix}")
        stamp += timedelta(minutes=15)
    present = []
    missing = []
    for name in expected:
        candidates = [directory / f"{name}.zip", directory / name, directory / name.lower()]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            missing.append(name)
        else:
            present.append(found)
    return present, missing


def open_raw(path: Path) -> io.BufferedReader | io.BytesIO:
    if path.suffix.lower() != ".zip":
        return path.open("rb")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one member, found {names}")
        return io.BytesIO(archive.read(names[0]))


def read_raw(paths: list[Path], columns: list[str], keep: list[str]) -> tuple[pl.DataFrame, dict]:
    frames = []
    manifest = []
    malformed_total = 0
    for path in paths:
        digest = hashlib.sha256()
        rows = 0
        malformed = 0
        good: list[str] = []
        with open_raw(path) as stream:
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
        data = pl.DataFrame(schema={**{c: pl.String for c in keep}, "source_file": pl.String})
    else:
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


def translated_flag() -> pl.Expr:
    return pl.col("source_file").str.contains("translation").alias("translated")


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
            translated_flag(),
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
        translated_flag(),
    )


def clean_title(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = " ".join(html.unescape(raw).split())
    return text or None


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
            pl.col("Extras")
            .str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")
            .map_elements(clean_title, return_dtype=pl.String, skip_nulls=True)
            .alias("page_title"),
            translated_flag(),
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
    return frame.sort("gkg_time", "translated").unique(
        "canonical_url", keep="first", maintain_order=True
    )


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
        iso = tld.upper()
        return ISO2_TO_FIPS.get(iso, iso), 0.5, "cctld"
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
        pl.Series("publisher_country", [r[0] for r in resolved], dtype=pl.String),
        pl.Series("country_confidence", [r[1] for r in resolved], dtype=pl.Float64),
        pl.Series("mapping_method", [r[2] for r in resolved], dtype=pl.String),
    )


TYPERS = {"mentions": typed_mentions, "events": typed_events, "gkg": typed_gkg}
SCHEMAS = {
    "mentions": (MENTION_COLUMNS, MENTION_COLUMNS),
    "events": (EVENT_COLUMNS, EVENT_COLUMNS),
    "gkg": (GKG_COLUMNS, GKG_KEEP),
}


def load_table(
    table: str, raw_dir: Path, start: datetime, end: datetime, allow_partial: bool
) -> tuple[pl.DataFrame, dict]:
    """Read, type and concatenate every quarter-hour file of ``table`` in the window,
    one file at a time; the English and translated families are both included."""
    columns, keep = SCHEMAS[table]
    typer = TYPERS[table]
    files: list[dict] = []
    missing_all: dict[str, list[str]] = {}
    frames: list[pl.DataFrame] = []
    malformed = 0
    for suffix in FILE_KINDS[table]:
        paths, missing = expected_files(raw_dir, suffix, start, end)
        missing_all[suffix] = missing
        if missing and not allow_partial:
            raise ValueError(f"{len(missing)} missing {suffix} files; pass --allow-partial")
        print(f"Reading {len(paths)} {suffix} files...", flush=True)
        for path in paths:
            raw, part = read_raw([path], columns, keep)
            files.extend(part["files"])
            malformed += part["malformed_rows"]
            if raw.height:
                frames.append(typer(raw))
    if not frames:
        empty = pl.DataFrame(schema={**{c: pl.String for c in keep}, "source_file": pl.String})
        frames.append(typer(empty))
    data = pl.concat(frames)
    if table == "gkg":
        data = data.sort("gkg_time", "translated").unique(
            "canonical_url", keep="first", maintain_order=True
        )
    audit = {
        "files": files,
        "rows": data.height,
        "malformed_rows": malformed,
        "missing_files": missing_all,
        "raw_bytes": sum(int(f["bytes"]) for f in files),
    }
    return data, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="directory of GDELT zips")
    parser.add_argument("--start", type=parse_yyyymmdd, required=True, metavar="YYYYMMDD")
    parser.add_argument(
        "--end", type=parse_yyyymmdd, required=True, metavar="YYYYMMDD", help="inclusive"
    )
    parser.add_argument("--domain-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    end = args.end + timedelta(days=1)

    audit: dict = {
        "window": [f"{args.start:%Y%m%d}", f"{args.end:%Y%m%d}"],
        "command": sys.argv,
        "tables": {},
    }
    tables: dict[str, pl.DataFrame] = {}
    for table in ("mentions", "events", "gkg"):
        tables[table], audit["tables"][table] = load_table(
            table, args.raw, args.start, end, args.allow_partial
        )
    mentions, events, gkg = tables["mentions"], tables["events"], tables["gkg"]

    lookup = load_domain_lookup(args.domain_lookup)
    sources = build_sources(mentions, gkg, lookup)
    web = mentions.filter(pl.col("MentionType") == 1)
    joined = (
        web.select("canonical_url")
        .unique()
        .join(gkg.select("canonical_url"), on="canonical_url", how="inner")
    )
    summary: dict = {
        "mention_rows": mentions.height,
        "web_mention_rows": web.height,
        "distinct_web_urls": web["canonical_url"].n_unique(),
        "web_urls_with_gkg": joined.height,
        "event_rows": events.height,
        "gkg_documents": gkg.height,
        "gkg_translated_documents": int(gkg["translated"].sum()),
        "gkg_with_page_title": gkg.filter(pl.col("page_title").is_not_null()).height,
        "gkg_languages": gkg["source_language"].fill_null("eng").n_unique(),
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
    written = 0
    for name, frame in [
        ("mentions", mentions),
        ("events", events),
        ("gkg", gkg),
        ("sources", sources),
    ]:
        target = args.output / f"{name}.parquet"
        frame.write_parquet(target, compression="zstd")
        written += target.stat().st_size
    raw_bytes = sum(int(t["raw_bytes"]) for t in audit["tables"].values())
    summary["reduction"] = {
        "raw_zip_bytes": raw_bytes,
        "parquet_bytes": written,
        "ratio": round(raw_bytes / max(written, 1), 2),
    }
    audit["summary"] = summary
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
