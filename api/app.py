"""A GDELT-shaped read API served from the local v2 Mentions/Events slice.

`/api/v2/doc/doc` and `/api/v2/geo/geo` return the same JSON shapes as the public
GDELT 2.0 APIs so existing clients keep working; `/api/v2/ext/*` exposes the
event-level detail the public APIs do not publish. Fields the raw archive cannot
support (headlines, social images) are documented in `/api/v2/ext/meta`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.analytics import router as analytics_router
from api.attention import router as attention_router
from api.dataset import Dataset, load, parse_datetime, stamp
from api.query import QueryError, filter_frame
from api.schema import EVENT_ROOT_CODES, QUAD_CLASSES

DOC_MODES = ["artlist", "timelinevol", "timelinevolraw", "timelinetone", "tonechart"]
SORT_MODES = ["datedesc", "dateasc", "tonedesc", "toneasc", "hybridrel"]
MAX_RECORDS = 250
TONE_BINS = list(range(-20, 21, 2))

app = FastAPI(
    title="GDELT-shaped API over the open GDELT v2 archive",
    description=__doc__,
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(attention_router, prefix="/api/v2/attention", tags=["attention"])
app.include_router(
    analytics_router, prefix="/api/v2/attention/analytics", tags=["country analytics"]
)


def dataset() -> Dataset:
    try:
        return load()
    except FileNotFoundError as error:  # pragma: no cover - configuration failure
        raise HTTPException(503, f"dataset not built: {error}") from error


def bad_request(message: str) -> HTTPException:
    return HTTPException(400, message)


def selected(
    query: str, startdatetime: str | None, enddatetime: str | None, timespan: str | None
) -> tuple[pl.DataFrame, datetime, datetime]:
    data = dataset()
    try:
        start, end = data.resolve_window(startdatetime, enddatetime, timespan)
        frame = filter_frame(data.window(start, end), query)
    except (QueryError, ValueError) as error:
        raise bad_request(str(error)) from error
    return frame, start, end


def bucket_size(start: datetime, end: datetime) -> timedelta:
    span = end - start
    if span <= timedelta(days=1):
        return timedelta(minutes=15)
    if span <= timedelta(days=14):
        return timedelta(hours=1)
    return timedelta(days=1)


def bucketed(frame: pl.DataFrame, every: timedelta) -> pl.DataFrame:
    return frame.with_columns(pl.col("seendate").dt.truncate(every).alias("bucket"))


def timeline_grid(start: datetime, end: datetime, every: timedelta) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bucket": pl.datetime_range(
                start.replace(second=0, microsecond=0),
                end,
                every,
                eager=True,
            ).dt.truncate(every)
        }
    ).unique(subset=["bucket"], keep="first")


def series(name: str, frame: pl.DataFrame, value_column: str) -> dict:
    return {
        "series": name,
        "data": [
            {"date": stamp(row["bucket"]), "value": row[value_column]}
            for row in frame.sort("bucket").iter_rows(named=True)
        ],
    }


def article_records(frame: pl.DataFrame) -> list[dict]:
    """The eight DOC 2.0 `artlist` fields, plus dataset-only extras."""
    return [
        {
            "url": row["url"],
            "url_mobile": "",
            "title": row["title"],
            "seendate": stamp(row["seendate"]),
            "socialimage": "",
            "domain": row["domain"],
            "language": row["language"],
            "sourcecountry": row["sourcecountry"],
            "tone": round(row["tone"], 4) if row["tone"] is not None else None,
            "numevents": row["numevents"],
            "themes": row["themes"],
            "locations": row["locations"][:5],
            "eventids": row["eventids"][:25],
        }
        for row in frame.iter_rows(named=True)
    ]


def sorted_frame(frame: pl.DataFrame, sort: str) -> pl.DataFrame:
    if sort == "dateasc":
        return frame.sort("seendate")
    if sort == "tonedesc":
        return frame.sort("tone", descending=True, nulls_last=True)
    if sort == "toneasc":
        return frame.sort("tone", nulls_last=True)
    if sort == "hybridrel":
        return frame.sort(["numevents", "seendate"], descending=[True, True])
    return frame.sort("seendate", descending=True)


@app.get("/api/v2/doc/doc")
def doc(
    query: str = Query("", description="GDELT-style query; see /api/v2/ext/meta"),
    mode: str = Query("artlist"),
    format: str = Query("json"),
    maxrecords: int = Query(75, ge=1, le=MAX_RECORDS),
    timespan: str | None = None,
    startdatetime: str | None = None,
    enddatetime: str | None = None,
    sort: str = Query("datedesc"),
) -> JSONResponse:
    mode = mode.lower()
    if mode not in DOC_MODES:
        raise bad_request(f"mode must be one of {', '.join(DOC_MODES)}")
    if format.lower() != "json":
        raise bad_request("only format=json is served")
    if sort.lower() not in SORT_MODES:
        raise bad_request(f"sort must be one of {', '.join(SORT_MODES)}")

    frame, start, end = selected(query, startdatetime, enddatetime, timespan)
    if mode == "artlist":
        records = article_records(sorted_frame(frame, sort.lower()).head(maxrecords))
        return JSONResponse({"articles": records})

    every = bucket_size(start, end)
    grid = timeline_grid(start, end, every)
    if mode in {"timelinevol", "timelinevolraw"}:
        data = dataset()
        matched = bucketed(frame, every).group_by("bucket").agg(pl.len().alias("matched"))
        total = (
            bucketed(data.window(start, end), every).group_by("bucket").agg(pl.len().alias("total"))
        )
        joined = (
            grid.join(matched, on="bucket", how="left")
            .join(total, on="bucket", how="left")
            .with_columns(pl.col("matched").fill_null(0), pl.col("total").fill_null(0))
        )
        if mode == "timelinevol":
            joined = joined.with_columns(
                pl.when(pl.col("total") > 0)
                .then(100 * pl.col("matched") / pl.col("total"))
                .otherwise(0.0)
                .round(6)
                .alias("intensity")
            )
            return JSONResponse({"timeline": [series("Volume Intensity", joined, "intensity")]})
        return JSONResponse(
            {
                "timeline": [
                    series("Article Count", joined, "matched"),
                    series("Total Monitoring Volume", joined, "total"),
                ]
            }
        )

    if mode == "timelinetone":
        tone = (
            bucketed(frame, every)
            .group_by("bucket")
            .agg(pl.col("tone").mean().round(4).alias("tone"))
        )
        joined = grid.join(tone, on="bucket", how="left").with_columns(
            pl.col("tone").fill_null(0.0)
        )
        return JSONResponse({"timeline": [series("Average Tone", joined, "tone")]})

    binned = frame.with_columns(
        pl.col("tone").fill_null(0).clip(TONE_BINS[0], TONE_BINS[-1]).floor().cast(pl.Int64)
        // 2
        * 2
    )
    chart = []
    for value, group in sorted(binned.partition_by("tone", as_dict=True).items()):
        top = sorted_frame(group, "hybridrel").head(10)
        chart.append(
            {
                "bin": value[0],
                "count": group.height,
                "toparts": article_records(top),
            }
        )
    return JSONResponse({"tonechart": chart})


@app.get("/api/v2/geo/geo")
def geo(
    query: str = Query(""),
    mode: str = Query("pointdata"),
    format: str = Query("geojson"),
    timespan: str | None = None,
    startdatetime: str | None = None,
    enddatetime: str | None = None,
    maxpoints: int = Query(500, ge=1, le=5000),
) -> JSONResponse:
    if format.lower() != "geojson":
        raise bad_request("only format=geojson is served")
    if mode.lower() not in {"pointdata", "pointheat"}:
        raise bad_request("mode must be pointdata or pointheat")

    frame, start, end = selected(query, startdatetime, enddatetime, timespan)
    data = dataset()
    located = (
        frame.select("url", "title", "domain", "seendate", "tone", "eventids")
        .explode("eventids")
        .rename({"eventids": "GlobalEventID"})
        .join(
            data.events.select(
                "GlobalEventID",
                "ActionGeo_FullName",
                "ActionGeo_Lat",
                "ActionGeo_Long",
                "root_label",
            ),
            on="GlobalEventID",
            how="inner",
        )
        .filter(pl.col("ActionGeo_Lat").is_not_null() & (pl.col("ActionGeo_FullName") != ""))
    )
    places = (
        located.group_by("ActionGeo_FullName")
        .agg(
            pl.col("ActionGeo_Lat").first().alias("lat"),
            pl.col("ActionGeo_Long").first().alias("lon"),
            pl.col("url").n_unique().alias("count"),
            pl.col("tone").mean().round(3).alias("tone"),
            pl.col("root_label").mode().first().alias("theme"),
            pl.col("url").unique().head(5).alias("urls"),
            pl.col("title").unique().head(5).alias("titles"),
        )
        .sort("count", descending=True)
        .head(maxpoints)
    )
    features = []
    for row in places.iter_rows(named=True):
        links = "".join(
            f'<a href="{url}">{title}</a><br>'
            for url, title in zip(row["urls"], row["titles"], strict=False)
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                "properties": {
                    "name": row["ActionGeo_FullName"],
                    "count": row["count"],
                    "shareimage": "",
                    "html": f"<b>{row['ActionGeo_FullName']}</b><br>{links}",
                    "tone": row["tone"],
                    "theme": row["theme"],
                },
            }
        )
    return JSONResponse(
        {
            "type": "FeatureCollection",
            "query": query,
            "window": {"start": stamp(start), "end": stamp(end)},
            "features": features,
        }
    )


@app.get("/api/v2/ext/events")
def events(
    query: str = Query(""),
    timespan: str | None = None,
    startdatetime: str | None = None,
    enddatetime: str | None = None,
    quadclass: int | None = Query(None, ge=1, le=4),
    rootcode: str | None = None,
    country: str | None = None,
    maxrecords: int = Query(100, ge=1, le=MAX_RECORDS),
) -> JSONResponse:
    frame, start, end = selected(query, startdatetime, enddatetime, timespan)
    data = dataset()
    ids = frame.select(pl.col("eventids").explode().alias("GlobalEventID")).unique()
    selection = data.events.join(ids, on="GlobalEventID", how="semi")
    if quadclass is not None:
        selection = selection.filter(pl.col("QuadClass") == quadclass)
    if rootcode:
        selection = selection.filter(pl.col("EventRootCode") == rootcode.zfill(2))
    if country:
        selection = selection.filter(
            pl.col("ActionGeo_CountryCode").str.to_uppercase() == country.upper()
        )
    selection = selection.sort(["NumArticles", "NumMentions"], descending=True).head(maxrecords)
    return JSONResponse(
        {
            "window": {"start": stamp(start), "end": stamp(end)},
            "count": selection.height,
            "events": [
                {
                    "globaleventid": row["GlobalEventID"],
                    "day": row["Day"],
                    "dateadded": stamp(row["dateadded"]) if row["dateadded"] else None,
                    "actor1": row["Actor1Name"],
                    "actor2": row["Actor2Name"],
                    "actor1country": row["Actor1CountryCode"],
                    "actor2country": row["Actor2CountryCode"],
                    "eventcode": row["EventCode"],
                    "rootcode": row["EventRootCode"],
                    "rootlabel": row["root_label"],
                    "quadclass": row["QuadClass"],
                    "quadlabel": QUAD_CLASSES.get(row["QuadClass"], ""),
                    "goldstein": row["GoldsteinScale"],
                    "avgtone": row["AvgTone"],
                    "nummentions": row["NumMentions"],
                    "numsources": row["NumSources"],
                    "numarticles": row["NumArticles"],
                    "location": row["ActionGeo_FullName"],
                    "countrycode": row["ActionGeo_CountryCode"],
                    "lat": row["ActionGeo_Lat"],
                    "lon": row["ActionGeo_Long"],
                    "sourceurl": row["sourceurl"],
                }
                for row in selection.iter_rows(named=True)
            ],
        }
    )


@app.get("/api/v2/ext/events/{event_id}")
def event_detail(event_id: int, maxrecords: int = Query(50, ge=1, le=MAX_RECORDS)) -> JSONResponse:
    data = dataset()
    row = data.events.filter(pl.col("GlobalEventID") == event_id)
    if row.is_empty():
        raise HTTPException(404, f"event {event_id} is not in this slice")
    urls = data.links.filter(pl.col("GlobalEventID") == event_id).select("url", "confidence")
    articles = data.articles.join(urls, on="url", how="inner").sort("seendate").head(maxrecords)
    record = row.row(0, named=True)
    return JSONResponse(
        {
            "event": {
                "globaleventid": record["GlobalEventID"],
                "day": record["Day"],
                "eventcode": record["EventCode"],
                "rootcode": record["EventRootCode"],
                "rootlabel": record["root_label"],
                "quadclass": record["QuadClass"],
                "quadlabel": QUAD_CLASSES.get(record["QuadClass"], ""),
                "goldstein": record["GoldsteinScale"],
                "avgtone": record["AvgTone"],
                "actor1": record["Actor1Name"],
                "actor2": record["Actor2Name"],
                "location": record["ActionGeo_FullName"],
                "lat": record["ActionGeo_Lat"],
                "lon": record["ActionGeo_Long"],
                "nummentions": record["NumMentions"],
                "numarticles": record["NumArticles"],
                "sourceurl": record["sourceurl"],
            },
            "articles": article_records(articles),
        }
    )


@app.get("/api/v2/ext/facets")
def facets(
    query: str = Query(""),
    timespan: str | None = None,
    startdatetime: str | None = None,
    enddatetime: str | None = None,
    limit: int = Query(15, ge=1, le=100),
) -> JSONResponse:
    frame, start, end = selected(query, startdatetime, enddatetime, timespan)

    def top(column: str) -> list[dict]:
        counted = frame.group_by(column).agg(pl.len().alias("count"))
        return [
            {"value": row[column], "count": row["count"]}
            for row in counted.sort("count", descending=True).head(limit).iter_rows(named=True)
            if row[column]
        ]

    def top_list(column: str) -> list[dict]:
        counted = (
            frame.select(pl.col(column).explode().alias("value"))
            .drop_nulls()
            .filter(pl.col("value") != "")
            .group_by("value")
            .agg(pl.len().alias("count"))
        )
        return counted.sort("count", descending=True).head(limit).to_dicts()

    return JSONResponse(
        {
            "window": {"start": stamp(start), "end": stamp(end)},
            "matched_articles": frame.height,
            "domains": top("domain"),
            "languages": top("language"),
            "sourcecountries": top("sourcecountry"),
            "themes": top_list("themes"),
            "locations": top_list("locations"),
            "actors": top_list("actors"),
        }
    )


@app.get("/api/v2/ext/meta")
def meta() -> JSONResponse:
    data = dataset()
    return JSONResponse(
        {
            **data.meta,
            "coverage": {"start": stamp(data.start), "end": stamp(data.end)},
            "doc_modes": DOC_MODES,
            "sort_modes": SORT_MODES,
            "query_operators": [
                "term",
                '"quoted phrase"',
                "-negated",
                "a OR b",
                "domain:",
                "domainis:",
                "sourcelang:",
                "sourcecountry:",
                "theme:",
                "location:",
                "actor:",
                "quadclass:",
            ],
            "quadclasses": QUAD_CLASSES,
            "rootcodes": EVENT_ROOT_CODES,
        }
    )


@app.get("/api/v2/ext/health")
def health() -> JSONResponse:
    data = dataset()
    return JSONResponse({"status": "ok", "articles": data.articles.height})


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "gdelt-shaped api",
            "docs": "/docs",
            "endpoints": [
                "/api/v2/doc/doc",
                "/api/v2/geo/geo",
                "/api/v2/ext/events",
                "/api/v2/ext/events/{id}",
                "/api/v2/ext/facets",
                "/api/v2/ext/meta",
                "/api/v2/attention/search",
                "/api/v2/attention/events",
                "/api/v2/attention/events/{id}",
                "/api/v2/attention/events/{id}/spread",
                "/api/v2/attention/events/{id}/timeline",
                "/api/v2/attention/events/{id}/countries",
                "/api/v2/attention/families/{id}",
                "/api/v2/attention/event-types",
                "/api/v2/attention/event-types/{type}/countries",
                "/api/v2/attention/countries",
            ],
        }
    )


__all__ = ["app", "parse_datetime"]
