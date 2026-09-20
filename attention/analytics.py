"""Country response analytics over the materialized attention store.

Turns the event-centric tables (``event_families`` / ``macro_events`` and the
per-country ``country_*_summary`` onsets) into one *country response observation*
per ``event × origin_country × destination_country``:

    response_hours(E, O → C) = onset(E, C) - onset(E, O)        (origin-relative)
    response_hours_world(E, C) = onset(E, C) - world_onset(E)  (fallback)
    self_response_hours(E, O)  = onset(E, O) - event_start(E)  (domestic)

``onset`` is the store's robust onset, ``max(third_source_seen, p10_seen)``, and
is never redefined here. The *origin* of an event is the publisher-independent
GDELT event geography: the ``event_country`` carrying most effective reports
across the incidents of a family. Every destination country in
``country_baseline`` gets a row for every eligible event, so countries that never
reach onset stay in the denominator as uncovered (right-censored) observations.

Aggregates report latency (mean / median / quartiles / seeded bootstrap CI of the
median) *separately* from coverage (covered / eligible with a Wilson interval) and
refuse to quote latency below a configurable support floor. All numbers describe
observed media attention in GDELT, not when anyone learned anything.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

SEED = 2026
BOOTSTRAP_SAMPLES = 1000
OBSERVATIONS_FILE = "country_response_observations.parquet"
LEVELS = ("family", "incident")
REFERENCES = ("origin_preferred", "origin_only", "world")
MAGNITUDE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("5-19 effective reports", 5, 20),
    ("20-99 effective reports", 20, 100),
    ("100-499 effective reports", 100, 500),
    ("500+ effective reports", 500, None),
)
CAVEATS = [
    "response = observed media response in GDELT (onset = 3rd outlet & 10th-percentile "
    "document), not when citizens or governments learned about the event",
    "publisher country = where the outlet is based; not its audience and not the event location",
    "GDELT observation time (15-minute batches) is not publication time",
    "origin→destination timing is an observed attention relationship, not causal transmission",
    "origin country = GDELT event geography with most effective reports across the story; events "
    "without geography are excluded",
    "domestic response is measured from the event's first observed document, which may not be the "
    "physical occurrence time; stories already running at the window start are excluded",
    "negative hours mean the destination reached onset before the reference (origin or world)",
]

OBSERVATION_COLUMNS = [
    "level",
    "event_id",
    "family_id",
    "title",
    "event_start",
    "event_end",
    "start_truncated",
    "event_types",
    "event_documents",
    "event_effective_reports",
    "resolution_model",
    "origin_country",
    "origin_share",
    "destination_country",
    "is_domestic",
    "origin_onset",
    "destination_onset",
    "world_onset",
    "response_hours_origin",
    "response_hours_world",
    "self_response_hours",
    "response_reference",
    "response_hours",
    "has_documents",
    "covered",
    "suppressed",
    "censor_hours",
    "raw_documents",
    "unique_domains",
    "effective_reports",
    "attention_ratio",
]


# --------------------------------------------------------------------------- build


def hours(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
    return (later - earlier).dt.total_minutes() / 60


def event_table(macro_events: pl.DataFrame, families: pl.DataFrame) -> pl.DataFrame:
    """One analytical event per story family (default) and per incident (drilldown),
    each with its origin country = event geography carrying most effective reports."""
    located = macro_events.filter(pl.col("event_country").is_not_null())
    dominant = (
        located.group_by("family_id", "event_country")
        .agg(pl.col("effective_reports").sum().alias("weight"))
        .sort(["family_id", "weight", "event_country"], descending=[False, True, False])
        .group_by("family_id", maintain_order=True)
        .agg(
            pl.col("event_country").first().alias("origin_country"),
            (pl.col("weight").first() / pl.col("weight").sum()).alias("origin_share"),
        )
    )
    family_types = (
        macro_events.select("family_id", "event_types", "effective_reports")
        .explode("event_types")
        .drop_nulls("event_types")
        .group_by("family_id", "event_types")
        .agg(pl.col("effective_reports").sum().alias("weight"))
        .sort(["family_id", "weight", "event_types"], descending=[False, True, False])
        .group_by("family_id", maintain_order=True)
        .agg(pl.col("event_types").head(4))
    )
    model = macro_events.group_by("family_id").agg(pl.col("resolution_model").first())
    family_level = (
        families.join(dominant, on="family_id")
        .join(family_types, on="family_id", how="left")
        .join(model, on="family_id", how="left")
        .select(
            pl.lit("family").alias("level"),
            pl.col("family_id").alias("event_id"),
            "family_id",
            "title",
            pl.col("start_time").alias("event_start"),
            pl.col("end_time").alias("event_end"),
            pl.col("event_types").fill_null(pl.lit([], dtype=pl.List(pl.String))),
            pl.col("raw_documents").alias("event_documents"),
            pl.col("effective_reports").alias("event_effective_reports"),
            "resolution_model",
            "origin_country",
            "origin_share",
        )
    )
    incident_level = located.select(
        pl.lit("incident").alias("level"),
        pl.col("macro_event_id").alias("event_id"),
        "family_id",
        "title",
        pl.col("start_time").alias("event_start"),
        pl.col("end_time").alias("event_end"),
        "event_types",
        pl.col("raw_documents").alias("event_documents"),
        pl.col("effective_reports").alias("event_effective_reports"),
        "resolution_model",
        pl.col("event_country").alias("origin_country"),
        pl.lit(1.0).alias("origin_share"),
    )
    return pl.concat([family_level, incident_level], how="vertical_relaxed")


def coverage_table(family_summary: pl.DataFrame, event_summary: pl.DataFrame) -> pl.DataFrame:
    """Per (event, publisher country) onset rows from both store levels."""
    keep = [
        "publisher_country",
        "onset",
        "world_onset",
        "suppressed",
        "raw_documents",
        "unique_domains",
        "effective_reports",
        "attention_ratio",
    ]
    fam = family_summary.select(
        pl.lit("family").alias("level"), pl.col("family_id").alias("event_id"), *keep
    )
    inc = event_summary.select(
        pl.lit("incident").alias("level"), pl.col("macro_event_id").alias("event_id"), *keep
    )
    return pl.concat([fam, inc], how="vertical_relaxed")


def build_observations(
    macro_events: pl.DataFrame,
    families: pl.DataFrame,
    family_summary: pl.DataFrame,
    event_summary: pl.DataFrame,
    baseline: pl.DataFrame,
    window_start: datetime,
    window_end: datetime,
) -> pl.DataFrame:
    events = event_table(macro_events, families)
    coverage = coverage_table(family_summary, event_summary)
    destinations = baseline.select(pl.col("publisher_country").alias("destination_country"))
    origin_onset = coverage.select(
        "level",
        "event_id",
        pl.col("publisher_country").alias("origin_country"),
        pl.col("onset").alias("origin_onset"),
    )
    world_onset = coverage.group_by("level", "event_id").agg(
        pl.col("world_onset").drop_nulls().first()
    )
    frame = (
        events.join(destinations, how="cross")
        .join(
            coverage.rename(
                {"publisher_country": "destination_country", "onset": "destination_onset"}
            ).drop("world_onset"),
            on=["level", "event_id", "destination_country"],
            how="left",
        )
        .join(origin_onset, on=["level", "event_id", "origin_country"], how="left")
        .join(world_onset, on=["level", "event_id"], how="left")
    )
    domestic = pl.col("origin_country") == pl.col("destination_country")
    covered = pl.col("destination_onset").is_not_null() & ~pl.col("suppressed").fill_null(True)
    frame = frame.with_columns(
        domestic.alias("is_domestic"),
        (pl.col("event_start") <= pl.lit(window_start)).alias("start_truncated"),
        pl.col("raw_documents").is_not_null().alias("has_documents"),
        covered.alias("covered"),
        pl.col("suppressed").fill_null(False),
        pl.when(domestic)
        .then(None)
        .otherwise(hours(pl.col("destination_onset"), pl.col("origin_onset")))
        .alias("response_hours_origin"),
        pl.when(domestic)
        .then(None)
        .otherwise(hours(pl.col("destination_onset"), pl.col("world_onset")))
        .alias("response_hours_world"),
        pl.when(domestic)
        .then(hours(pl.col("destination_onset"), pl.col("event_start")))
        .otherwise(None)
        .alias("self_response_hours"),
    )
    reference = (
        pl.when(domestic)
        .then(pl.lit("event_start"))
        .when(pl.col("origin_onset").is_not_null())
        .then(pl.lit("origin"))
        .when(pl.col("world_onset").is_not_null())
        .then(pl.lit("world_fallback"))
        .otherwise(None)
    )
    reference_time = (
        pl.when(domestic)
        .then(pl.col("event_start"))
        .when(pl.col("origin_onset").is_not_null())
        .then(pl.col("origin_onset"))
        .otherwise(pl.col("world_onset"))
    )
    frame = frame.with_columns(
        reference.alias("response_reference"),
        pl.when(domestic)
        .then(pl.col("self_response_hours"))
        .when(pl.col("origin_onset").is_not_null())
        .then(pl.col("response_hours_origin"))
        .otherwise(pl.col("response_hours_world"))
        .alias("response_hours"),
        hours(pl.lit(window_end), reference_time).alias("censor_hours"),
    )
    return frame.select(OBSERVATION_COLUMNS).sort(
        "level", "event_id", "origin_country", "destination_country"
    )


def parse_window(value: str) -> datetime:
    return datetime.fromisoformat(value.replace(" UTC", "+00:00"))


def build_from_store(store: Path) -> pl.DataFrame:
    meta = json.loads((store / "meta.json").read_text(encoding="utf-8"))
    return build_observations(
        pl.read_parquet(store / "macro_events.parquet"),
        pl.read_parquet(store / "event_families.parquet"),
        pl.read_parquet(store / "country_family_summary.parquet"),
        pl.read_parquet(store / "country_event_summary.parquet"),
        pl.read_parquet(store / "country_baseline.parquet"),
        parse_window(meta["window_start"]),
        parse_window(meta["window_end"]),
    )


def load_observations(store: Path) -> pl.DataFrame:
    path = store / OBSERVATIONS_FILE
    if path.exists():
        return pl.read_parquet(path)
    return build_from_store(store)


# --------------------------------------------------------------------------- filters


@dataclass(frozen=True)
class Filters:
    level: str = "family"
    origins: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    event_types: tuple[str, ...] = ()
    min_event_effective_reports: int = 0
    max_event_effective_reports: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    resolution_model: str | None = None
    reference: str = "origin_preferred"

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "origins": list(self.origins),
            "destinations": list(self.destinations),
            "event_types": list(self.event_types),
            "min_event_effective_reports": self.min_event_effective_reports,
            "max_event_effective_reports": self.max_event_effective_reports,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "resolution_model": self.resolution_model,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class Support:
    """Minimum evidence before a latency estimate is quoted."""

    min_covered_events: int = 5
    min_effective_reports: int = 15
    seed: int = SEED
    bootstrap_samples: int = field(default=BOOTSTRAP_SAMPLES)

    def as_dict(self) -> dict:
        return {
            "min_covered_events": self.min_covered_events,
            "min_effective_reports": self.min_effective_reports,
            "bootstrap_seed": self.seed,
            "bootstrap_samples": self.bootstrap_samples,
        }


def apply_filters(observations: pl.DataFrame, filters: Filters) -> pl.DataFrame:
    """AND-compose the event/country filters and pick the timing reference.

    ``origin_preferred`` keeps every eligible event and uses the origin country's
    onset when it exists, else the world onset (flagged ``world_fallback``).
    ``origin_only`` restricts foreign observations to events whose origin country
    reached onset. ``world`` measures every foreign observation against the world
    onset."""
    if filters.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if filters.reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    frame = observations.filter(pl.col("level") == filters.level)
    if filters.origins:
        frame = frame.filter(pl.col("origin_country").is_in(list(filters.origins)))
    if filters.destinations:
        frame = frame.filter(pl.col("destination_country").is_in(list(filters.destinations)))
    if filters.event_types:
        wanted = pl.lit(list(filters.event_types), dtype=pl.List(pl.String))
        frame = frame.filter(pl.col("event_types").list.set_intersection(wanted).list.len() > 0)
    if filters.min_event_effective_reports > 0:
        frame = frame.filter(
            pl.col("event_effective_reports") >= filters.min_event_effective_reports
        )
    if filters.max_event_effective_reports is not None:
        frame = frame.filter(
            pl.col("event_effective_reports") < filters.max_event_effective_reports
        )
    if filters.start is not None:
        frame = frame.filter(pl.col("event_start") >= pl.lit(filters.start))
    if filters.end is not None:
        frame = frame.filter(pl.col("event_start") < pl.lit(filters.end))
    if filters.resolution_model:
        frame = frame.filter(pl.col("resolution_model") == filters.resolution_model)
    foreign = ~pl.col("is_domestic")
    if filters.reference == "origin_only":
        frame = frame.filter(~foreign | pl.col("origin_onset").is_not_null())
    elif filters.reference == "world":
        frame = frame.with_columns(
            pl.when(foreign)
            .then(pl.col("response_hours_world"))
            .otherwise(pl.col("response_hours"))
            .alias("response_hours"),
            pl.when(foreign & pl.col("world_onset").is_not_null())
            .then(pl.lit("world"))
            .otherwise(pl.col("response_reference"))
            .alias("response_reference"),
        )
    return frame


def foreign(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(~pl.col("is_domestic"))


def domestic(frame: pl.DataFrame) -> pl.DataFrame:
    """Domestic observations; stories already running at the window start have no
    observed start and are not eligible."""
    return frame.filter(pl.col("is_domestic") & ~pl.col("start_truncated"))


# --------------------------------------------------------------------------- statistics


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (0, 0) when there are no trials."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_median_interval(
    values: np.ndarray, seed: int = SEED, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    """Percentile bootstrap (2.5 %, 97.5 %) of the median. The input is sorted first so
    the interval depends only on the multiset of values and the seed."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n == 0:
        raise ValueError("no values")
    if n == 1:
        return (float(ordered[0]), float(ordered[0]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(samples, n))
    medians = np.median(ordered[draws], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return (float(low), float(high))


def summarize(frame: pl.DataFrame, support: Support) -> dict:
    """Coverage + latency summary of one bag of observations (all rows are eligible)."""
    eligible = frame.height
    covered_rows = frame.filter(pl.col("covered") & pl.col("response_hours").is_not_null())
    covered = covered_rows.height
    coverage = covered / eligible if eligible else None
    low, high = wilson_interval(covered, eligible)
    effective = int(covered_rows["effective_reports"].sum()) if covered else 0
    fallback = covered_rows.filter(pl.col("response_reference") == "world_fallback").height
    supported = covered >= support.min_covered_events and effective >= support.min_effective_reports
    result: dict = {
        "eligible_events": eligible,
        "covered_events": covered,
        "uncovered_events": eligible - covered,
        "coverage_rate": coverage,
        "coverage_ci_low": low if eligible else None,
        "coverage_ci_high": high if eligible else None,
        "effective_reports": effective,
        "world_fallback_events": fallback,
        "mean_response_hours": None,
        "median_response_hours": None,
        "p25_response_hours": None,
        "p75_response_hours": None,
        "latency_ci_low": None,
        "latency_ci_high": None,
        "fastest_response_hours": None,
        "slowest_response_hours": None,
        "support": {
            "status": "ok" if supported else "insufficient",
            **support.as_dict(),
        },
    }
    if not supported:
        return result
    values = covered_rows["response_hours"].to_numpy().astype(np.float64)
    ci_low, ci_high = bootstrap_median_interval(values, support.seed, support.bootstrap_samples)
    result.update(
        {
            "mean_response_hours": float(values.mean()),
            "median_response_hours": float(np.median(values)),
            "p25_response_hours": float(np.percentile(values, 25)),
            "p75_response_hours": float(np.percentile(values, 75)),
            "latency_ci_low": ci_low,
            "latency_ci_high": ci_high,
            "fastest_response_hours": float(values.min()),
            "slowest_response_hours": float(values.max()),
        }
    )
    return result


def summarize_by(frame: pl.DataFrame, column: str, support: Support) -> list[dict]:
    """``summarize`` per distinct value of ``column``, most covered first."""
    out: list[tuple[int, int, str, dict]] = []
    for key, group in frame.group_by(column, maintain_order=True):
        row = summarize(group, support)
        out.append(
            (
                -int(row["covered_events"]),
                -int(row["eligible_events"]),
                str(key[0]),
                {column: key[0], **row},
            )
        )
    out.sort(key=lambda r: r[:3])
    return [row for *_, row in out]


def summarize_by_type(frame: pl.DataFrame, support: Support) -> list[dict]:
    exploded = frame.explode("event_types").drop_nulls("event_types")
    return summarize_by(exploded, "event_types", support)


def summarize_by_magnitude(frame: pl.DataFrame, support: Support) -> list[dict]:
    out = []
    for label, low, high in MAGNITUDE_BINS:
        cond = pl.col("event_effective_reports") >= low
        if high is not None:
            cond &= pl.col("event_effective_reports") < high
        group = frame.filter(cond)
        if group.height:
            out.append(
                {
                    "magnitude": label,
                    "min_event_effective_reports": low,
                    "max_event_effective_reports": high,
                    **summarize(group, support),
                }
            )
    return out


def pair_matrix(
    frame: pl.DataFrame, origins: list[str], destinations: list[str], support: Support
) -> list[dict]:
    """Every origin × destination cell; diagonal cells are domestic observations."""
    cells = []
    for origin in origins:
        for destination in destinations:
            group = frame.filter(
                (pl.col("origin_country") == origin)
                & (pl.col("destination_country") == destination)
            )
            group = domestic(group) if origin == destination else group
            cells.append(
                {
                    "origin_country": origin,
                    "destination_country": destination,
                    "kind": "domestic" if origin == destination else "foreign",
                    **summarize(group, support),
                }
            )
    return cells


EVENT_RECORD_COLUMNS = [
    "level",
    "event_id",
    "family_id",
    "title",
    "event_start",
    "event_types",
    "event_effective_reports",
    "origin_country",
    "destination_country",
    "origin_onset",
    "destination_onset",
    "world_onset",
    "response_hours",
    "response_reference",
    "covered",
    "suppressed",
    "has_documents",
    "censor_hours",
    "raw_documents",
    "effective_reports",
    "attention_ratio",
]


def event_records(frame: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Contributing events: covered ones first (fastest response first), then the
    right-censored remainder (largest event first)."""
    covered = frame.filter(pl.col("covered") & pl.col("response_hours").is_not_null()).sort(
        "response_hours", "event_id"
    )
    rest = frame.filter(~(pl.col("covered") & pl.col("response_hours").is_not_null())).sort(
        "event_effective_reports", "event_id", descending=[True, False]
    )
    return pl.concat([covered, rest]).select(EVENT_RECORD_COLUMNS).head(limit)


# --------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("store", type=Path, help="materialized store directory")
    args = parser.parse_args()
    frame = build_from_store(args.store)
    frame.write_parquet(args.store / OBSERVATIONS_FILE)
    fam = frame.filter(pl.col("level") == "family")
    print(
        json.dumps(
            {
                "rows": frame.height,
                "family_events": fam["event_id"].n_unique(),
                "incident_events": frame.filter(pl.col("level") == "incident")[
                    "event_id"
                ].n_unique(),
                "destinations": frame["destination_country"].n_unique(),
                "family_rows_covered": fam.filter(pl.col("covered")).height,
                "family_rows_origin_relative": fam.filter(
                    pl.col("covered") & (pl.col("response_reference") == "origin")
                ).height,
                "family_rows_world_fallback": fam.filter(
                    pl.col("covered") & (pl.col("response_reference") == "world_fallback")
                ).height,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
