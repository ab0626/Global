"""Country response analytics: how quickly publisher countries respond to events.

Mounted under ``/analytics``. Every aggregate is a ``summary`` block from
``attention.analytics.summarize`` (coverage with a Wilson interval, latency with a
seeded bootstrap interval, explicit support status) and every number is drillable
to the ``event × origin × destination`` observations that produced it via
``/analytics/events``. Countries are FIPS publisher-country codes as used across
the attention API; ``origin`` is event geography, ``destination`` is where the
covering outlets are based.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query

from api.attention import AttentionStore, store
from attention.analytics import (
    CAVEATS,
    LEVELS,
    REFERENCES,
    Filters,
    Support,
    apply_filters,
    domestic,
    event_records,
    foreign,
    pair_matrix,
    summarize,
    summarize_by,
    summarize_by_magnitude,
    summarize_by_type,
)

router = APIRouter()

MAX_MATRIX_SIDE = 25
LEVEL_PATTERN = "^(" + "|".join(LEVELS) + ")$"
REFERENCE_PATTERN = "^(" + "|".join(REFERENCES) + ")$"


@dataclass(frozen=True)
class Request:
    """Filters + support parsed from the query string, shared by every route."""

    filters: Filters
    support: Support


def codes(values: list[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or []:
        out.extend(v.strip().upper() for v in value.split(",") if v.strip())
    return tuple(dict.fromkeys(out))


Countries = Annotated[
    list[str] | None, Query(description="repeatable or comma-separated FIPS codes")
]


def request_params(
    level: Annotated[
        str,
        Query(
            pattern=LEVEL_PATTERN,
            description="one observation per story family (default) or per incident",
        ),
    ] = "family",
    event_type: Annotated[
        list[str] | None, Query(description="repeatable; any listed type matches")
    ] = None,
    min_event_effective_reports: Annotated[
        int, Query(ge=0, description="event magnitude floor")
    ] = 0,
    start: Annotated[
        datetime | None, Query(description="events observed from this UTC time")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="events observed before this UTC time")
    ] = None,
    resolution_model: str | None = None,
    reference: Annotated[str, Query(pattern=REFERENCE_PATTERN)] = "origin_preferred",
    min_covered_events: Annotated[int, Query(ge=1)] = 5,
    min_effective_reports: Annotated[int, Query(ge=0)] = 15,
) -> Request:
    return Request(
        filters=Filters(
            level=level,
            event_types=tuple(t for t in (event_type or []) if t),
            min_event_effective_reports=min_event_effective_reports,
            start=start,
            end=end,
            resolution_model=resolution_model,
            reference=reference,
        ),
        support=Support(
            min_covered_events=min_covered_events, min_effective_reports=min_effective_reports
        ),
    )


Params = Annotated[Request, Depends(request_params)]


def records(frame: pl.DataFrame) -> list[dict]:
    return json.loads(frame.write_json())


def selected(data: AttentionStore, req: Request) -> pl.DataFrame:
    try:
        return apply_filters(data.observations, req.filters)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


def named(data: AttentionStore, rows: list[dict], *keys: str) -> list[dict]:
    for row in rows:
        for key in keys:
            if row.get(key):
                row[f"{key}_name"] = data.country_name(row[key])
    return rows


def envelope(data: AttentionStore, req: Request, payload: dict) -> dict:
    return data.envelope(
        {
            **payload,
            "filters": req.filters.as_dict(),
            "support": req.support.as_dict(),
            "caveats": CAVEATS,
        }
    )


def pair_frame(frame: pl.DataFrame, origin: str, destination: str) -> pl.DataFrame:
    group = frame.filter(
        (pl.col("origin_country") == origin) & (pl.col("destination_country") == destination)
    )
    return domestic(group) if origin == destination else group


def pair_block(
    data: AttentionStore, frame: pl.DataFrame, origin: str, destination: str, support: Support
) -> dict:
    group = pair_frame(frame, origin, destination)
    return {
        "origin_country": origin,
        "origin_country_name": data.country_name(origin),
        "destination_country": destination,
        "destination_country_name": data.country_name(destination),
        "kind": "domestic" if origin == destination else "foreign",
        "summary": summarize(group, support),
        "by_event_type": summarize_by_type(group, support),
    }


@router.get("/countries")
def country_table(req: Params) -> dict:
    """Every destination country: observed response to foreign events and to its own
    (domestic) events, side by side."""
    data = store()
    frame = selected(data, req)
    rows = []
    for (code,), group in frame.group_by("destination_country", maintain_order=True):
        rows.append(
            {
                "publisher_country": code,
                "publisher_country_name": data.country_name(str(code)),
                "foreign": summarize(foreign(group), req.support),
                "domestic": summarize(domestic(group), req.support),
            }
        )
    rows.sort(
        key=lambda r: (
            -r["foreign"]["covered_events"],
            -r["domestic"]["covered_events"],
            r["publisher_country"],
        )
    )
    return envelope(data, req, {"publisher_countries": rows})


@router.get("/countries/{country}")
def country_overview(country: str, req: Params, top_origins: int = Query(15, ge=1, le=100)) -> dict:
    """Country overview: response to foreign news, observed domestic response, and
    breakdowns by event type, origin country and event magnitude."""
    data = store()
    code = data.require_country(country)
    frame = selected(data, req).filter(pl.col("destination_country") == code)
    abroad, home = foreign(frame), domestic(frame)
    return envelope(
        data,
        req,
        {
            "publisher_country": code,
            "publisher_country_name": data.country_name(code),
            "foreign": summarize(abroad, req.support),
            "domestic": summarize(home, req.support),
            "by_event_type": summarize_by_type(abroad, req.support),
            "by_origin": named(
                data,
                summarize_by(abroad, "origin_country", req.support)[:top_origins],
                "origin_country",
            ),
            "by_magnitude": summarize_by_magnitude(abroad, req.support),
            "domestic_by_event_type": summarize_by_type(home, req.support),
        },
    )


@router.get("/pairs")
def pair(
    req: Params,
    origin: str = Query(..., description="FIPS code of the event-origin country"),
    destination: str = Query(..., description="FIPS code of the publisher country"),
    limit: int = Query(200, ge=0, le=2000),
) -> dict:
    """origin → destination and the reverse relationship, each with its own summary,
    event-type breakdown and contributing events."""
    data = store()
    o, d = data.require_country(origin), data.require_country(destination)
    frame = selected(data, req)
    forward = pair_block(data, frame, o, d, req.support)
    forward["events"] = named(
        data,
        records(event_records(pair_frame(frame, o, d), limit)),
        "origin_country",
        "destination_country",
    )
    payload = {"forward": forward}
    if o != d:
        payload["reverse"] = pair_block(data, frame, d, o, req.support)
    return envelope(data, req, payload)


@router.get("/matrix")
def matrix(req: Params, origin: Countries = None, destination: Countries = None) -> dict:
    """origin × destination cells. With only one side given the other defaults to the
    same countries, so ``?origin=GM&origin=FR`` yields the full pairwise matrix.
    Diagonal cells are observed domestic responses."""
    data = store()
    origins = [data.require_country(c) for c in codes(origin)]
    destinations = [data.require_country(c) for c in codes(destination)]
    if not origins and not destinations:
        raise HTTPException(400, "give at least one origin or destination")
    origins = origins or destinations
    destinations = destinations or origins
    if len(origins) > MAX_MATRIX_SIDE or len(destinations) > MAX_MATRIX_SIDE:
        raise HTTPException(400, f"at most {MAX_MATRIX_SIDE} countries per side")
    frame = selected(data, req)
    cells = pair_matrix(frame, origins, destinations, req.support)
    destination_rows = [
        {
            "publisher_country": d,
            "publisher_country_name": data.country_name(d),
            "foreign": summarize(
                foreign(frame.filter(pl.col("destination_country") == d)), req.support
            ),
            "domestic": summarize(
                domestic(frame.filter(pl.col("destination_country") == d)), req.support
            ),
            "foreign_from_selected_origins": summarize(
                foreign(
                    frame.filter(
                        (pl.col("destination_country") == d)
                        & pl.col("origin_country").is_in([o for o in origins if o != d])
                    )
                ),
                req.support,
            ),
        }
        for d in destinations
    ]
    return envelope(
        data,
        req,
        {
            "origins": [{"code": c, "name": data.country_name(c)} for c in origins],
            "destinations": [{"code": c, "name": data.country_name(c)} for c in destinations],
            "cells": named(data, cells, "origin_country", "destination_country"),
            "destination_rows": destination_rows,
        },
    )


@router.get("/origins/{country}")
def origin_view(country: str, req: Params) -> dict:
    """Events originating in ``country``: how quickly every other publisher country
    responds, plus the world as a whole."""
    data = store()
    code = data.require_country(country)
    frame = selected(data, req).filter(pl.col("origin_country") == code)
    abroad = foreign(frame)
    return envelope(
        data,
        req,
        {
            "origin_country": code,
            "origin_country_name": data.country_name(code),
            "world": summarize(abroad, req.support),
            "domestic": summarize(domestic(frame), req.support),
            "by_destination": named(
                data,
                summarize_by(abroad, "destination_country", req.support),
                "destination_country",
            ),
            "by_event_type": summarize_by_type(abroad, req.support),
        },
    )


@router.get("/destinations/{country}")
def destination_view(country: str, req: Params) -> dict:
    """Publisher country ``country``: response to each origin country's events."""
    data = store()
    code = data.require_country(country)
    frame = selected(data, req).filter(pl.col("destination_country") == code)
    abroad = foreign(frame)
    return envelope(
        data,
        req,
        {
            "destination_country": code,
            "destination_country_name": data.country_name(code),
            "foreign": summarize(abroad, req.support),
            "domestic": summarize(domestic(frame), req.support),
            "by_origin": named(
                data, summarize_by(abroad, "origin_country", req.support), "origin_country"
            ),
        },
    )


@router.get("/events")
def events(
    req: Params,
    origin: Countries = None,
    destination: Countries = None,
    kind: str = Query("all", pattern="^(all|foreign|domestic)$"),
    covered_only: bool = Query(False),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Drilldown: the observations behind any aggregate, covered ones first."""
    data = store()
    origins = [data.require_country(c) for c in codes(origin)]
    destinations = [data.require_country(c) for c in codes(destination)]
    frame = selected(data, req)
    if origins:
        frame = frame.filter(pl.col("origin_country").is_in(origins))
    if destinations:
        frame = frame.filter(pl.col("destination_country").is_in(destinations))
    if kind == "foreign":
        frame = foreign(frame)
    elif kind == "domestic":
        frame = domestic(frame)
    if covered_only:
        frame = frame.filter(pl.col("covered"))
    ordered = event_records(frame, frame.height)
    return envelope(
        data,
        req,
        {
            "total": ordered.height,
            "offset": offset,
            "limit": limit,
            "summary": summarize(frame, req.support),
            "events": named(
                data, records(ordered.slice(offset, limit)), "origin_country", "destination_country"
            ),
        },
    )
