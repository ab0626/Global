"""Event-centric attention endpoints over the tables written by ``attention.materialize``.

Mounted under ``/search``, ``/events``, ``/families``, ``/documents``,
``/event-types`` and ``/countries``. The store directory comes from
``GDELT_ATTENTION_DATA`` (default ``data/store``); every response carries ``meta``
(window, denominators, timing semantics, resolution model, filters) so charts can
label themselves. Country fields are always ``publisher_country`` (where the outlet
is based) or ``event_country`` (where GDELT geolocated the event); there is no bare
``country``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import cached_property, lru_cache
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from attention.analytics import load_observations

router = APIRouter()

EVENT_SUMMARY_COLUMNS = [
    "macro_event_id",
    "incident_id",
    "family_id",
    "title",
    "label",
    "start_time",
    "end_time",
    "event_country",
    "lat",
    "lon",
    "event_types",
    "atomic_event_count",
    "raw_documents",
    "unique_domains",
    "effective_reports",
    "publisher_country_count",
    "language_count",
    "cluster_confidence",
    "resolution_model",
]
COUNTRY_SUMMARY_COLUMNS = [
    "publisher_country",
    "raw_documents",
    "unique_domains",
    "effective_reports",
    "country_documents",
    "country_effective_reports",
    "raw_share",
    "effective_share",
    "world_raw_share",
    "world_effective_share",
    "attention_ratio",
    "first_seen",
    "third_source_seen",
    "p10_seen",
    "onset",
    "world_onset",
    "lag_hours",
    "suppressed",
]
SPREAD_COLUMNS = [
    "document_id",
    "macro_event_id",
    "incident_id",
    "observed_time",
    "publisher_country",
    "publisher_country_confidence",
    "source_domain",
    "language",
    "title",
    "url",
    "wire_group",
    "assignment_score",
]


def fold(text: str) -> str:
    """Case-fold and strip combining marks so 'Séisme' matches 'seisme'; CJK is untouched."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_expr(column: pl.Expr) -> pl.Expr:
    return column.map_elements(fold, return_dtype=pl.String, skip_nulls=True)


class AttentionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.macro_events = pl.read_parquet(directory / "macro_events.parquet").with_columns(
            fold_expr(
                pl.concat_str(
                    [
                        pl.col("title"),
                        pl.col("label"),
                        pl.col("people").list.join(" "),
                        pl.col("organizations").list.join(" "),
                        pl.col("actors").list.join(" "),
                    ],
                    separator=" ",
                    ignore_nulls=True,
                )
            ).alias("search_text")
        )
        self.families = pl.read_parquet(directory / "event_families.parquet")
        self.documents = pl.read_parquet(directory / "macro_event_documents.parquet")
        self.attention = pl.read_parquet(directory / "country_event_attention.parquet")
        self.summary = pl.read_parquet(directory / "country_event_summary.parquet")
        self.family_attention = pl.read_parquet(directory / "country_family_attention.parquet")
        self.family_summary = pl.read_parquet(directory / "country_family_summary.parquet")
        self.sources = pl.read_parquet(directory / "sources.parquet")
        self.baseline = pl.read_parquet(directory / "country_baseline.parquet")
        evidence_path = directory / "document_evidence.parquet"
        self.evidence = pl.read_parquet(evidence_path) if evidence_path.exists() else None
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.meta = {key: value for key, value in meta.items() if key != "cluster_run"}
        self.title_search = self.documents.filter(
            (pl.col("macro_event_id") >= 0) & pl.col("is_primary") & pl.col("title").is_not_null()
        ).select("macro_event_id", fold_expr(pl.col("title")).alias("search_text"))

    @cached_property
    def observations(self) -> pl.DataFrame:
        """Country response observations (``attention.analytics``), read from the store
        when materialized there, otherwise derived from the summary tables on first use."""
        return load_observations(self.directory)

    @cached_property
    def analytics_countries(self) -> frozenset[str]:
        """Codes usable in analytics queries: every publisher country in the baseline plus
        every event-origin country (which may have no resolved publishers of its own)."""
        return frozenset(self.baseline["publisher_country"].drop_nulls().to_list()) | frozenset(
            self.observations["origin_country"].unique().to_list()
        )

    def country_name(self, code: str) -> str:
        row = self.baseline.filter(pl.col("publisher_country") == code)
        return str(row["country_name"][0]) if row.height and row["country_name"][0] else code

    def require_country(self, code: str) -> str:
        code = code.upper()
        if code not in self.analytics_countries:
            raise HTTPException(404, f"unknown country code {code!r}")
        return code

    def event(self, event_id: int) -> pl.DataFrame:
        frame = self.macro_events.filter(pl.col("macro_event_id") == event_id)
        if frame.is_empty():
            raise HTTPException(404, f"macro-event {event_id} not found")
        return frame

    def family(self, family_id: int) -> pl.DataFrame:
        frame = self.families.filter(pl.col("family_id") == family_id)
        if frame.is_empty():
            raise HTTPException(404, f"family {family_id} not found")
        return frame

    def family_events(self, family_id: int) -> pl.DataFrame:
        return self.macro_events.filter(pl.col("family_id") == family_id).select("macro_event_id")

    def search_families(self, hits: pl.DataFrame) -> pl.DataFrame:
        """Story families of the matching incidents, ranked by matching member titles
        summed over the family, then publisher-country breadth, then effective reports."""
        family_hits = hits.group_by("family_id").agg(pl.col("title_hits").sum())
        return (
            self.families.join(family_hits, on="family_id")
            .sort(
                ["title_hits", "publisher_country_count", "effective_reports"],
                descending=[True, True, True],
            )
            .drop("macro_event_ids")
        )

    def search(self, q: str) -> pl.DataFrame:
        """Macro-events whose summary text or member titles contain every query token.

        Ranked by matching member titles, then publisher-country breadth, then
        effective reports."""
        tokens = [t for t in fold(q).split() if t]
        if not tokens:
            return self.macro_events.head(0).with_columns(pl.lit(0).alias("title_hits"))
        summary_hit = pl.all_horizontal([token_match(t) for t in tokens])
        title_hits = (
            self.title_search.filter(summary_hit)
            .group_by("macro_event_id")
            .agg(pl.len().alias("title_hits"))
        )
        return (
            self.macro_events.join(title_hits, on="macro_event_id", how="left")
            .with_columns(pl.col("title_hits").fill_null(0))
            .filter(summary_hit | (pl.col("title_hits") > 0))
            .sort(
                ["title_hits", "publisher_country_count", "effective_reports"],
                descending=[True, True, True],
            )
        )

    def envelope(self, payload: dict) -> dict:
        return {**payload, "meta": self.meta}


@lru_cache(maxsize=1)
def load(directory: str | None = None) -> AttentionStore:
    path = Path(directory or os.environ.get("GDELT_ATTENTION_DATA", "data/store"))
    return AttentionStore(path)


def store() -> AttentionStore:
    try:
        return load()
    except FileNotFoundError as error:  # pragma: no cover - configuration failure
        raise HTTPException(503, f"attention store not built: {error}") from error


def token_match(token: str) -> pl.Expr:
    """Whole-word match for alphabetic scripts (so 'dame' does not hit 'madame'),
    substring match for scripts without word spacing (CJK)."""
    if re.fullmatch(r"[\w'-]+", token) and re.search(r"[a-z\u0400-\u04ff]", token):
        return pl.col("search_text").str.contains(rf"(^|\W){re.escape(token)}(\W|$)")
    return pl.col("search_text").str.contains(token, literal=True)


def records(frame: pl.DataFrame) -> list[dict]:
    return json.loads(frame.write_json())


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="free text; every token must match"),
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    """Story families first (what a user means by "the event"), then the matching
    incidents; both carry ``title_hits`` so clients can show why they matched."""
    data = store()
    hits = data.search(q)
    families = data.search_families(hits)
    return data.envelope(
        {
            "query": q,
            "total": hits.height,
            "total_families": families.height,
            "families": records(families.head(limit)),
            "events": records(hits.select([*EVENT_SUMMARY_COLUMNS, "title_hits"]).head(limit)),
        }
    )


@router.get("/events")
def list_events(
    q: str | None = Query(None, description="free text over title/label/people/orgs/actors"),
    event_type: str | None = Query(None, alias="type"),
    event_country: str | None = Query(None, description="FIPS country where the event happened"),
    family_id: int | None = Query(None),
    sort: str = Query(
        "effective_reports",
        pattern="^(effective_reports|raw_documents|publisher_countries|start)$",
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    data = store()
    frame = data.search(q) if q else data.macro_events
    if event_type:
        frame = frame.filter(pl.col("event_types").list.contains(event_type))
    if event_country:
        frame = frame.filter(pl.col("event_country") == event_country.upper())
    if family_id is not None:
        frame = frame.filter(pl.col("family_id") == family_id)
    order = {
        "effective_reports": ("effective_reports", True),
        "raw_documents": ("raw_documents", True),
        "publisher_countries": ("publisher_country_count", True),
        "start": ("start_time", False),
    }[sort]
    frame = frame.sort([order[0], "macro_event_id"], descending=[order[1], False])
    return data.envelope(
        {
            "total": frame.height,
            "events": records(frame.select(EVENT_SUMMARY_COLUMNS).slice(offset, limit)),
        }
    )


@router.get("/events/{event_id}")
def event_detail(event_id: int, sample: int = Query(10, ge=0, le=100)) -> dict:
    data = store()
    event = data.event(event_id)
    docs = data.documents.filter((pl.col("macro_event_id") == event_id) & pl.col("is_primary"))
    sample_docs = docs.sort("observed_time", "document_id").unique(
        "wire_group", keep="first", maintain_order=True
    )
    languages = (
        docs.group_by("language").len().sort("len", descending=True).rename({"len": "documents"})
    )
    family = data.families.filter(pl.col("family_id") == event["family_id"][0])
    return data.envelope(
        {
            "event": records(event.drop("search_text"))[0],
            "family": records(family)[0] if family.height else None,
            "languages": records(languages),
            "sample_documents": records(sample_docs.select(SPREAD_COLUMNS).head(sample)),
        }
    )


@router.get("/events/{event_id}/spread")
def event_spread(
    event_id: int,
    include_family: bool = Query(False, description="include sibling incidents of the family"),
    min_country_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Chronological primary documents for the globe: observed time + publisher country.

    Stable ordering (``observed_time``, ``document_id``) so pagination never skips or
    repeats. Documents below ``min_country_confidence`` (or unresolved) are dropped and
    counted in ``excluded_documents``."""
    data = store()
    event = data.event(event_id)
    family_id = int(event["family_id"][0])
    if include_family:
        docs = data.documents.join(data.family_events(family_id), on="macro_event_id")
    else:
        docs = data.documents.filter(pl.col("macro_event_id") == event_id)
    payload = spread_payload(docs, min_country_confidence, limit, offset)
    return data.envelope(
        {
            "macro_event_id": event_id,
            "family_id": family_id,
            "include_family": include_family,
            **payload,
        }
    )


def spread_payload(
    docs: pl.DataFrame, min_country_confidence: float, limit: int, offset: int
) -> dict:
    docs = docs.filter(pl.col("is_primary"))
    kept = docs.filter(
        pl.col("publisher_country").is_not_null()
        & (pl.col("publisher_country_confidence") >= min_country_confidence)
    ).sort("observed_time", "document_id")
    by_country = (
        kept.group_by("publisher_country")
        .agg(
            pl.len().alias("raw_documents"),
            pl.col("wire_group").n_unique().alias("effective_reports"),
            pl.col("observed_time").min().alias("first_seen"),
        )
        .sort("first_seen", "publisher_country")
    )
    return {
        "total": kept.height,
        "excluded_documents": docs.height - kept.height,
        "offset": offset,
        "limit": limit,
        "countries": records(by_country),
        "documents": records(kept.select(SPREAD_COLUMNS).slice(offset, limit)),
    }


@router.get("/events/{event_id}/timeline")
def event_timeline(
    event_id: int,
    publisher_countries: str | None = Query(None, description="comma-separated FIPS codes"),
    value: str = Query(
        "raw_documents", pattern="^(raw_documents|unique_domains|effective_reports|raw_share)$"
    ),
) -> dict:
    data = store()
    data.event(event_id)
    frame = data.attention.filter(pl.col("macro_event_id") == event_id)
    return data.envelope(
        {"macro_event_id": event_id, **timeline_payload(frame, publisher_countries, value)}
    )


def timeline_payload(frame: pl.DataFrame, publisher_countries: str | None, value: str) -> dict:
    if publisher_countries:
        wanted = [c.strip().upper() for c in publisher_countries.split(",") if c.strip()]
        frame = frame.filter(pl.col("publisher_country").is_in(wanted))
    series = (
        frame.sort("time_bucket")
        .group_by("publisher_country", maintain_order=True)
        .agg(
            pl.col("time_bucket"),
            pl.col(value).alias("values"),
            pl.col("cumulative_documents"),
            pl.col("onset_flag"),
            pl.col("raw_documents").sum().alias("total_documents"),
        )
        .sort("total_documents", descending=True)
    )
    world = (
        frame.group_by("time_bucket")
        .agg(
            pl.col("raw_documents").sum(),
            pl.col("unique_domains").sum(),
            pl.col("effective_reports").sum(),
        )
        .sort("time_bucket")
    )
    return {
        "bucket": "1h",
        "value": value,
        "world": records(world),
        "publisher_countries": records(series),
    }


@router.get("/events/{event_id}/countries")
def event_countries(
    event_id: int,
    min_unique_domains: int = Query(1, ge=1),
    include_suppressed: bool = Query(True),
) -> dict:
    data = store()
    data.event(event_id)
    frame = data.summary.filter(pl.col("macro_event_id") == event_id)
    payload = countries_payload(frame, min_unique_domains, include_suppressed)
    return data.envelope({"macro_event_id": event_id, **payload})


def countries_payload(
    frame: pl.DataFrame, min_unique_domains: int, include_suppressed: bool
) -> dict:
    frame = frame.filter(pl.col("unique_domains") >= min_unique_domains)
    if not include_suppressed:
        frame = frame.filter(~pl.col("suppressed"))
    world_onset = (
        records(frame.select("world_onset").head(1))[0]["world_onset"] if frame.height else None
    )
    return {
        "world_onset": world_onset,
        "publisher_countries": records(frame.select(COUNTRY_SUMMARY_COLUMNS)),
    }


@router.get("/families/{family_id}")
def family_detail(family_id: int) -> dict:
    data = store()
    family = data.family(family_id)
    events = data.macro_events.filter(pl.col("family_id") == family_id).sort(
        "effective_reports", descending=True
    )
    return data.envelope(
        {"family": records(family)[0], "events": records(events.select(EVENT_SUMMARY_COLUMNS))}
    )


@router.get("/families/{family_id}/spread")
def family_spread(
    family_id: int,
    min_country_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Chronological primary documents of every incident in the story family."""
    data = store()
    data.family(family_id)
    docs = data.documents.join(data.family_events(family_id), on="macro_event_id")
    payload = spread_payload(docs, min_country_confidence, limit, offset)
    return data.envelope({"family_id": family_id, **payload})


@router.get("/families/{family_id}/timeline")
def family_timeline(
    family_id: int,
    publisher_countries: str | None = Query(None, description="comma-separated FIPS codes"),
    value: str = Query(
        "raw_documents", pattern="^(raw_documents|unique_domains|effective_reports|raw_share)$"
    ),
) -> dict:
    data = store()
    data.family(family_id)
    frame = data.family_attention.filter(pl.col("family_id") == family_id)
    return data.envelope(
        {"family_id": family_id, **timeline_payload(frame, publisher_countries, value)}
    )


@router.get("/families/{family_id}/countries")
def family_countries(
    family_id: int,
    min_unique_domains: int = Query(1, ge=1),
    include_suppressed: bool = Query(True),
) -> dict:
    """Story-level country attention: lag and ratio computed over the whole family,
    so they are comparable across incidents of different sizes."""
    data = store()
    data.family(family_id)
    frame = data.family_summary.filter(pl.col("family_id") == family_id)
    payload = countries_payload(frame, min_unique_domains, include_suppressed)
    return data.envelope({"family_id": family_id, **payload})


@router.get("/event-types")
def list_types() -> dict:
    data = store()
    frame = (
        data.macro_events.select("macro_event_id", "event_types", "raw_documents")
        .explode("event_types")
        .drop_nulls("event_types")
        .group_by("event_types")
        .agg(pl.len().alias("events"), pl.col("raw_documents").sum())
        .sort("events", descending=True)
        .rename({"event_types": "type"})
    )
    return data.envelope({"types": records(frame)})


@router.get("/event-types/{event_type}/countries")
def type_countries(
    event_type: str,
    min_events: int = Query(3, ge=1),
    min_country_documents: int = Query(
        1000, ge=1, description="window-wide documents a country needs before its ratio is ranked"
    ),
) -> dict:
    """Mean attention ratio and median lag per publisher country across all
    macro-events carrying ``event_type``. Countries below the volume floors are
    dropped because a two-document country's share ratio is meaningless."""
    data = store()
    ids = data.macro_events.filter(pl.col("event_types").list.contains(event_type)).select(
        "macro_event_id"
    )
    if ids.is_empty():
        raise HTTPException(404, f"no macro-events typed {event_type!r}")
    frame = (
        data.summary.join(ids, on="macro_event_id")
        .group_by("publisher_country")
        .agg(
            pl.len().alias("events"),
            pl.col("raw_documents").sum(),
            pl.col("effective_reports").sum(),
            pl.col("country_documents").first(),
            pl.col("attention_ratio").mean().alias("mean_attention_ratio"),
            pl.col("lag_hours").median().alias("median_lag_hours"),
            pl.col("lag_hours").drop_nulls().len().alias("events_with_onset"),
        )
        .filter(
            (pl.col("events") >= min_events)
            & (pl.col("country_documents") >= min_country_documents)
        )
        .with_columns(
            (pl.col("raw_documents") / pl.col("country_documents")).alias("share_of_country_output")
        )
        .sort("mean_attention_ratio", descending=True)
    )
    return data.envelope(
        {"type": event_type, "macro_events": ids.height, "publisher_countries": records(frame)}
    )


EVIDENCE_NEIGHBOR_COLUMNS = [
    "document_id",
    "incident_id",
    "macro_event_id",
    "title",
    "language",
    "source_domain",
    "publisher_country",
    "observed_time",
]


@router.get("/documents/{document_id}/evidence")
def document_evidence(document_id: int) -> dict:
    """Why this article sits in its incident: the strongest gated graph edges to
    other documents of the same incident (``supporting``) and the strongest edge to
    any other incident (``competing``), each with per-channel scores. Documents with
    no surviving edge are singletons/unassigned and return empty lists."""
    data = store()
    doc = data.documents.filter((pl.col("document_id") == document_id) & pl.col("is_primary"))
    if doc.is_empty():
        raise HTTPException(404, f"document {document_id} not found")
    if data.evidence is None:
        raise HTTPException(503, "store was built without document_evidence.parquet")
    row = records(doc.select(*SPREAD_COLUMNS, "family_id"))[0]
    edges = data.evidence.filter(pl.col("document_id") == document_id).join(
        data.documents.filter(pl.col("is_primary")).select(EVIDENCE_NEIGHBOR_COLUMNS),
        left_on="neighbor_document_id",
        right_on="document_id",
        how="left",
    )
    edge_records: list[dict] = []
    for edge in edges.sort("same_incident", "rank", descending=[True, False]).iter_rows(named=True):
        neighbor = {
            "document_id": edge["neighbor_document_id"],
            "incident_id": edge["neighbor_incident_id"],
            "macro_event_id": edge["macro_event_id"],
            "title": edge["title"],
            "language": edge["language"],
            "source_domain": edge["source_domain"],
            "publisher_country": edge["publisher_country"],
            "observed_time": edge["observed_time"],
        }
        edge_records.append(
            {
                "neighbor": neighbor,
                "same_incident": edge["same_incident"],
                "rank": edge["rank"],
                "title_score": edge["title_score"],
                "event_score": edge["event_score"],
                "url_score": edge["url_score"],
                "entity_score": edge["entity_score"],
                "delta_hours": edge["delta_hours"],
                "evidence_channels": edge["evidence_channels"],
                "gated": edge["gated"],
                "same_publisher_country": edge["publisher_country"] == row["publisher_country"],
            }
        )
    supporting = [e for e in edge_records if e["same_incident"]]
    competing = [e for e in edge_records if not e["same_incident"]]
    incident = data.macro_events.filter(pl.col("macro_event_id") == row["macro_event_id"])
    family = data.families.filter(pl.col("family_id") == row["family_id"])
    support = edges.filter(pl.col("same_incident"))
    checks = {
        "title_similarity": support["title_score"].max(),
        "shared_gdelt_event": bool((support["event_score"] > 0).any()),
        "shared_url_tokens": bool((support["url_score"] > 0).any()),
        "shared_entities": bool((support["entity_score"] > 0).any()),
        "hours_to_nearest_support": support["delta_hours"].min(),
        "other_publisher_country": bool(
            (support["publisher_country"] != row["publisher_country"]).any()
        ),
    }
    return data.envelope(
        {
            "document": row,
            "incident": records(incident.drop("search_text"))[0] if incident.height else None,
            "family": records(family.drop("macro_event_ids"))[0] if family.height else None,
            "assignment_score": row["assignment_score"],
            "checks": checks,
            "supporting": supporting,
            "competing": competing,
            "note": "scores are the clustering run's pair channels (title = calibrated "
            "multilingual cosine, event = IDF-weighted GlobalEventID overlap, url/entity = "
            "rare-token overlap); gated is the fused edge weight Leiden saw",
        }
    )


@router.get("/countries")
def list_countries() -> dict:
    data = store()
    frame = data.baseline.sort("country_documents", descending=True)
    return data.envelope({"publisher_countries": records(frame)})
