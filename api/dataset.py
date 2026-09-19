"""In-memory access to the built GDELT tables."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

TIMESPAN_UNITS = {"min": "minutes", "h": "hours", "d": "days", "w": "weeks", "m": "days"}
STAMP_FORMAT = "%Y%m%d%H%M%S"


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("-", "").replace(":", "").replace("T", "").rstrip("Z")
    if len(text) == 8:
        text += "000000"
    if len(text) != 14 or not text.isdigit():
        raise ValueError(f"expected YYYYMMDDHHMMSS, got {value!r}")
    return datetime.strptime(text, STAMP_FORMAT)


def parse_timespan(value: str | None) -> timedelta | None:
    """GDELT timespans: 30min, 6h, 3d, 2w, 1m (months are treated as 30 days)."""
    if not value:
        return None
    text = value.strip().lower()
    for suffix in ("min", "h", "d", "w", "m"):
        if text.endswith(suffix) and text[: -len(suffix)].isdigit():
            amount = int(text[: -len(suffix)])
            if suffix == "m":
                amount *= 30
            return timedelta(**{TIMESPAN_UNITS[suffix]: amount})
    raise ValueError(f"unsupported timespan {value!r}")


def stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


class Dataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.articles = pl.read_parquet(directory / "articles.parquet")
        self.events = pl.read_parquet(directory / "events.parquet")
        self.links = pl.read_parquet(directory / "article_events.parquet")
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        span = self.articles.select(
            pl.col("seendate").min().alias("start"), pl.col("seendate").max().alias("end")
        ).row(0)
        self.start, self.end = span

    def window(self, start: datetime | None, end: datetime | None) -> pl.DataFrame:
        frame = self.articles
        if start is not None:
            frame = frame.filter(pl.col("seendate") >= start)
        if end is not None:
            frame = frame.filter(pl.col("seendate") <= end)
        return frame

    def resolve_window(
        self, startdatetime: str | None, enddatetime: str | None, timespan: str | None
    ) -> tuple[datetime, datetime]:
        start = parse_datetime(startdatetime)
        end = parse_datetime(enddatetime)
        delta = parse_timespan(timespan)
        if delta is not None and start is None:
            end = end or self.end
            start = end - delta
        start = start or self.start
        end = end or self.end
        if start > end:
            raise ValueError("startdatetime is after enddatetime")
        return start, end


@lru_cache(maxsize=1)
def load(directory: str | None = None) -> Dataset:
    path = Path(directory or os.environ.get("GDELT_API_DATA", "data/api"))
    return Dataset(path)
