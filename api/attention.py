"""Event-centric attention endpoints over the tables written by ``attention.materialize``.

Mounted under ``/events`` and ``/event-types``. The store directory comes from
``GDELT_ATTENTION_DATA`` (default ``data/store``); every response carries the
denominators and caveats from ``meta.json`` so charts can label themselves.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

EVENT_SUMMARY_COLUMNS = [
    "macro_event_id",
    "label",
    "start_time",
    "end_time",
    "event_country",
    "lat",
    "lon",
    "event_types",
    "atomic_event_count",
    "document_count",
    "source_count",
    "effective_source_count",
    "country_count",
    "cluster_confidence",
]


class AttentionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.macro_events = pl.read_parquet(directory / "macro_events.parquet").with_columns(
            pl.concat_str(
                [
                    pl.col("label"),
                    pl.col("people").list.join(" "),
                    pl.col("organizations").list.join(" "),
                ],
                separator=" ",
                ignore_nulls=True,
            )
            .str.to_lowercase()
            .alias("search_text")
        )
        self.documents = pl.read_parquet(directory / "macro_event_documents.parquet")
        self.attention = pl.read_parquet(directory / "country_event_attention.parquet")
        self.summary = pl.read_parquet(directory / "country_event_summary.parquet")
        self.sources = pl.read_parquet(directory / "sources.parquet")
        self.baseline = pl.read_parquet(directory / "country_baseline.parquet")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.meta = {key: value for key, value in meta.items() if key != "cluster_run"}

    def event(self, event_id: int) -> pl.DataFrame:
        frame = self.macro_events.filter(pl.col("macro_event_id") == event_id)
        if frame.is_empty():
            raise HTTPException(404, f"macro-event {event_id} not found")
        return frame

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


def records(frame: pl.DataFrame) -> list[dict]:
    return json.loads(frame.write_json())


@router.get("/events")
def list_events(
    q: str | None = Query(None, description="case-insensitive match on label/people/orgs"),
    event_type: str | None = Query(None, alias="type"),
    country: str | None = Query(None, description="FIPS event/action country"),
    sort: str = Query(
        "effective_sources", pattern="^(effective_sources|documents|countries|start)$"
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    data = store()
    frame = data.macro_events
    if q:
        needle = q.lower()
        frame = frame.filter(pl.col("search_text").str.contains(needle, literal=True))
    if event_type:
        frame = frame.filter(pl.col("event_types").list.contains(event_type))
    if country:
        frame = frame.filter(pl.col("event_country") == country.upper())
    order = {
        "effective_sources": ("effective_source_count", True),
        "documents": ("document_count", True),
        "countries": ("country_count", True),
        "start": ("start_time", False),
    }[sort]
    frame = frame.sort(order[0], descending=order[1])
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
    docs = data.documents.filter(pl.col("macro_event_id") == event_id)
    sample_docs = docs.sort("mention_time").unique("wire_group", keep="first", maintain_order=True)
    return data.envelope(
        {
            "event": records(event)[0],
            "sample_documents": records(
                sample_docs.select(
                    "document_id", "url", "source_domain", "source_country", "mention_time"
                ).head(sample)
            ),
        }
    )


@router.get("/events/{event_id}/timeline")
def event_timeline(
    event_id: int,
    countries: str | None = Query(None, description="comma-separated FIPS publisher countries"),
    normalized: bool = Query(False),
) -> dict:
    data = store()
    data.event(event_id)
    frame = data.attention.filter(pl.col("macro_event_id") == event_id)
    if countries:
        wanted = [code.strip().upper() for code in countries.split(",") if code.strip()]
        frame = frame.filter(pl.col("country").is_in(wanted))
    value = "normalized_attention" if normalized else "documents"
    series = (
        frame.sort("time_bucket")
        .group_by("country", maintain_order=True)
        .agg(
            pl.col("time_bucket"),
            pl.col(value).alias("values"),
            pl.col("cumulative_attention"),
            pl.col("onset_flag"),
            pl.col("documents").sum().alias("total_documents"),
        )
        .sort("total_documents", descending=True)
    )
    global_series = (
        frame.group_by("time_bucket")
        .agg(pl.col("documents").sum(), pl.col("sources").sum(), pl.col("effective_sources").sum())
        .sort("time_bucket")
    )
    return data.envelope(
        {
            "macro_event_id": event_id,
            "bucket": "1h",
            "value": value,
            "global": records(global_series),
            "countries": records(series),
        }
    )


@router.get("/events/{event_id}/countries")
def event_countries(
    event_id: int,
    min_sources: int = Query(1, ge=1),
    include_suppressed: bool = Query(True),
) -> dict:
    data = store()
    data.event(event_id)
    frame = data.summary.filter(
        (pl.col("macro_event_id") == event_id) & (pl.col("sources") >= min_sources)
    )
    if not include_suppressed:
        frame = frame.filter(~pl.col("suppressed"))
    return data.envelope(
        {
            "macro_event_id": event_id,
            "global_onset": records(frame.select("global_onset").head(1))[0]["global_onset"]
            if frame.height
            else None,
            "countries": records(
                frame.select(
                    "country",
                    "documents",
                    "sources",
                    "effective_sources",
                    "country_documents",
                    "share",
                    "world_share",
                    "attention_ratio",
                    "first_seen",
                    "onset_outlets_time",
                    "onset_p10_time",
                    "onset_time",
                    "lag_hours",
                    "suppressed",
                )
            ),
        }
    )


@router.get("/event-types")
def list_types() -> dict:
    data = store()
    frame = (
        data.macro_events.select("macro_event_id", "event_types", "document_count")
        .explode("event_types")
        .drop_nulls("event_types")
        .group_by("event_types")
        .agg(pl.len().alias("events"), pl.col("document_count").sum().alias("documents"))
        .sort("events", descending=True)
        .rename({"event_types": "type"})
    )
    return data.envelope({"types": records(frame)})


@router.get("/event-types/{event_type}/countries")
def type_countries(event_type: str, min_events: int = Query(1, ge=1)) -> dict:
    """Aggregate over-attention and median lag per publisher country across all
    macro-events carrying ``event_type``; ratios are document-weighted."""
    data = store()
    ids = data.macro_events.filter(pl.col("event_types").list.contains(event_type)).select(
        "macro_event_id"
    )
    if ids.is_empty():
        raise HTTPException(404, f"no macro-events typed {event_type!r}")
    frame = (
        data.summary.join(ids, on="macro_event_id")
        .group_by("country")
        .agg(
            pl.len().alias("events"),
            pl.col("documents").sum(),
            pl.col("sources").sum(),
            pl.col("country_documents").first(),
            pl.col("attention_ratio").mean().alias("mean_attention_ratio"),
            pl.col("lag_hours").median().alias("median_lag_hours"),
            pl.col("lag_hours").drop_nulls().len().alias("events_with_onset"),
        )
        .filter(pl.col("events") >= min_events)
        .with_columns(
            (pl.col("documents") / pl.col("country_documents")).alias("share_of_country_output")
        )
        .sort("mean_attention_ratio", descending=True)
    )
    return data.envelope(
        {"type": event_type, "macro_events": ids.height, "countries": records(frame)}
    )


@router.get("/countries")
def list_countries() -> dict:
    data = store()
    frame = data.baseline.rename({"source_country": "country"}).sort(
        "country_documents", descending=True
    )
    return data.envelope({"countries": records(frame)})
