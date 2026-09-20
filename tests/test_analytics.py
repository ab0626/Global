"""Country response analytics on a synthetic store where every answer is known.

Countries: A, B, C publish; D never publishes. Window 2023-02-06 00:00 → 02-08 23:45.

F1  origin A, start 06 02:00; onsets A 04:00, B 06:00, C 05:00; world 03:00
    → A→B 2 h, A→C 1 h, domestic 2 h
F2  origin A, start 06 10:00; onsets A 11:00, B 15:00, C suppressed (2 outlets)
    → A→B 4 h, domestic 1 h, C right-censored
F3  origin B, start 06 08:00; onsets B 09:00, A 08:30; world 08:00
    → B→A −0.5 h (negative kept), domestic 1 h
F4  origin A, start 07 00:00; A suppressed, B 05:00; world 02:00
    → A→B 3 h via world fallback
F5  origin B, starts at the window start; onsets B 02:00, A 03:00; world 01:00
    → truncated: no domestic observation, B→A 1 h still counts
F6  no event geography → excluded
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from attention import analytics
from attention.analytics import (
    Filters,
    Support,
    apply_filters,
    bootstrap_median_interval,
    build_observations,
    domestic,
    event_records,
    foreign,
    pair_matrix,
    summarize,
    summarize_by_type,
    wilson_interval,
)

WINDOW_START = datetime(2023, 2, 6, tzinfo=UTC)
WINDOW_END = datetime(2023, 2, 8, 23, 45, tzinfo=UTC)
LOOSE = Support(min_covered_events=1, min_effective_reports=0)


def t(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2023, 2, day, hour, minute, tzinfo=UTC)


FAMILIES = [
    # family_id, title, start, end, types, effective_reports
    (1, "quake", t(6, 2), t(8, 23), ["natural_disaster", "death"], 400),
    (2, "budget", t(6, 10), t(7, 10), ["politics_government"], 40),
    (3, "strike", t(6, 8), t(7, 8), ["protest"], 60),
    (4, "leak", t(7, 0), t(8, 0), ["politics_government"], 25),
    (5, "flood", WINDOW_START, t(7, 0), ["natural_disaster"], 80),
    (6, "orphan", t(6, 12), t(7, 12), ["sport"], 30),
]
ORIGIN = {1: "A", 2: "A", 3: "B", 4: "A", 5: "B", 6: None}
# (family, publisher, onset, world_onset, unique_domains)
COVERAGE = [
    (1, "A", t(6, 4), t(6, 3), 5),
    (1, "B", t(6, 6), t(6, 3), 4),
    (1, "C", t(6, 5), t(6, 3), 3),
    (2, "A", t(6, 11), t(6, 10, 30), 4),
    (2, "B", t(6, 15), t(6, 10, 30), 3),
    (2, "C", None, t(6, 10, 30), 2),
    (3, "B", t(6, 9), t(6, 8), 4),
    (3, "A", t(6, 8, 30), t(6, 8), 3),
    (4, "A", None, t(7, 2), 2),
    (4, "B", t(7, 5), t(7, 2), 3),
    (5, "B", t(6, 2), t(6, 1), 4),
    (5, "A", t(6, 3), t(6, 1), 3),
    (6, "A", t(6, 13), t(6, 12), 3),
]


def macro_events() -> pl.DataFrame:
    rows = []
    for fid, title, start, end, types, reports in FAMILIES:
        pieces = [(fid * 10, reports)] if fid != 1 else [(10, 300), (11, 100)]
        for mid, weight in pieces:
            rows.append(
                {
                    "macro_event_id": mid,
                    "family_id": fid,
                    "title": title,
                    "start_time": start,
                    "end_time": end,
                    "event_country": ORIGIN[fid],
                    "event_types": types,
                    "raw_documents": weight * 2,
                    "effective_reports": weight,
                    "resolution_model": "synthetic_v1",
                }
            )
    # a minority incident of the quake located in C must not flip the family origin
    rows.append({**rows[1], "macro_event_id": 12, "event_country": "C", "effective_reports": 50})
    return pl.DataFrame(rows).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC"),
        pl.col("end_time").dt.replace_time_zone("UTC"),
    )


def families() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "family_id": fid,
                "title": title,
                "start_time": start,
                "end_time": end,
                "raw_documents": reports * 2,
                "effective_reports": reports,
            }
            for fid, title, start, end, _types, reports in FAMILIES
        ]
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC"),
        pl.col("end_time").dt.replace_time_zone("UTC"),
    )


def summaries() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = [
        {
            "family_id": fid,
            "publisher_country": country,
            "onset": onset,
            "world_onset": world,
            "suppressed": domains < 3,
            "raw_documents": domains * 3,
            "unique_domains": domains,
            "effective_reports": domains * 2,
            "attention_ratio": 1.5,
        }
        for fid, country, onset, world, domains in COVERAGE
    ]
    fam = pl.DataFrame(rows).with_columns(
        pl.col("onset").dt.replace_time_zone("UTC"),
        pl.col("world_onset").dt.replace_time_zone("UTC"),
    )
    # incident-level mirror: one incident per family, two for the quake
    inc = pl.concat(
        [
            fam.with_columns((pl.col("family_id") * 10).alias("macro_event_id")),
            fam.filter(pl.col("family_id") == 1).with_columns(
                pl.lit(11, dtype=pl.Int64).alias("macro_event_id")
            ),
        ]
    ).drop("family_id")
    return fam, inc


def baseline() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "publisher_country": ["A", "B", "C", "D"],
            "country_name": ["Aland", "Bland", "Cland", "Dland"],
        }
    )


@pytest.fixture(scope="module")
def observations() -> pl.DataFrame:
    fam, inc = summaries()
    return build_observations(
        macro_events(), families(), fam, inc, baseline(), WINDOW_START, WINDOW_END
    )


def pair(frame: pl.DataFrame, origin: str, destination: str) -> pl.DataFrame:
    group = frame.filter(
        (pl.col("origin_country") == origin) & (pl.col("destination_country") == destination)
    )
    return domestic(group) if origin == destination else group


def family_level(observations: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return apply_filters(observations, Filters(**kwargs))


# ------------------------------------------------------------------ observations


def test_one_row_per_event_origin_destination(observations: pl.DataFrame) -> None:
    fam = observations.filter(pl.col("level") == "family")
    assert fam["event_id"].n_unique() == 5, "F6 has no event geography and is excluded"
    assert fam.height == 5 * 4, "every baseline country is a destination, D included"
    assert fam.filter(pl.col("event_id") == 1)["origin_country"].unique().to_list() == ["A"]
    assert fam.filter(pl.col("event_id") == 1)["origin_share"][0] == pytest.approx(400 / 450)
    assert observations.filter(pl.col("level") == "incident")["event_id"].n_unique() == 7


def test_a_to_b_and_b_to_a_are_separate(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    ab = summarize(pair(frame, "A", "B"), LOOSE)
    ba = summarize(pair(frame, "B", "A"), LOOSE)
    assert (ab["eligible_events"], ab["covered_events"]) == (3, 3)
    assert ab["mean_response_hours"] == pytest.approx(3.0)
    assert ab["median_response_hours"] == pytest.approx(3.0)
    assert ab["world_fallback_events"] == 1
    assert (ba["eligible_events"], ba["covered_events"]) == (2, 2)
    assert ba["mean_response_hours"] == pytest.approx(0.25)
    assert ba["fastest_response_hours"] == pytest.approx(-0.5), "negative responses are kept"


def test_self_response_is_measured_from_event_start(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    aa = summarize(pair(frame, "A", "A"), LOOSE)
    assert (aa["eligible_events"], aa["covered_events"]) == (3, 2)
    assert aa["mean_response_hours"] == pytest.approx(1.5)
    assert aa["median_response_hours"] == pytest.approx(1.5)
    bb = summarize(pair(frame, "B", "B"), LOOSE)
    assert (bb["eligible_events"], bb["covered_events"]) == (1, 1), "F5 started before the window"
    assert bb["median_response_hours"] == pytest.approx(1.0)
    rows = frame.filter(pl.col("is_domestic"))
    assert rows["response_hours_origin"].null_count() == rows.height
    assert (rows["response_reference"] == "event_start").all()


def test_world_fallback_when_origin_never_reaches_onset(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    row = pair(frame, "A", "B").filter(pl.col("event_id") == 4).row(0, named=True)
    assert row["origin_onset"] is None
    assert row["response_reference"] == "world_fallback"
    assert row["response_hours"] == pytest.approx(3.0)
    assert row["response_hours_origin"] is None
    origin_only = family_level(observations, reference="origin_only")
    ab = summarize(pair(origin_only, "A", "B"), LOOSE)
    assert ab["eligible_events"] == 2 and ab["mean_response_hours"] == pytest.approx(3.0)
    world = family_level(observations, reference="world")
    ab = summarize(pair(world, "A", "B"), LOOSE)
    assert ab["mean_response_hours"] == pytest.approx((3 + 4.5 + 3) / 3)


def test_coverage_denominator_and_censoring(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    ac = summarize(pair(frame, "A", "C"), LOOSE)
    assert (ac["eligible_events"], ac["covered_events"]) == (3, 1)
    assert ac["coverage_rate"] == pytest.approx(1 / 3)
    censored = pair(frame, "A", "C").filter(pl.col("event_id") == 2).row(0, named=True)
    assert censored["has_documents"] and censored["suppressed"] and not censored["covered"]
    assert censored["censor_hours"] == pytest.approx(60.75), "window end − origin onset (06 11:00)"
    to_d = summarize(foreign(frame.filter(pl.col("destination_country") == "D")), LOOSE)
    assert (to_d["eligible_events"], to_d["covered_events"]) == (5, 0)
    assert to_d["coverage_rate"] == 0 and to_d["median_response_hours"] is None
    ordered = event_records(pair(frame, "A", "C"), 10)
    assert ordered["covered"].to_list() == [True, False, False]


def test_domestic_versus_foreign_eligibility(observations: pl.DataFrame) -> None:
    frame = family_level(observations).filter(pl.col("destination_country") == "A")
    assert foreign(frame)["event_id"].sort().to_list() == [3, 5]
    assert domestic(frame)["event_id"].sort().to_list() == [1, 2, 4]


# ------------------------------------------------------------------ statistics


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert (low, high) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 10)[0] == 0.0


def test_bootstrap_is_deterministic_and_order_free() -> None:
    values = np.array([2.0, 4.0, 3.0, 7.0, 1.0, 5.0])
    first = bootstrap_median_interval(values, seed=analytics.SEED)
    second = bootstrap_median_interval(values[::-1].copy(), seed=analytics.SEED)
    assert first == second
    assert first[0] <= np.median(values) <= first[1]
    assert bootstrap_median_interval(np.array([4.0]), seed=1) == (4.0, 4.0)
    with pytest.raises(ValueError):
        bootstrap_median_interval(np.array([]))


def test_support_gate_hides_latency_but_keeps_counts(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    strict = summarize(pair(frame, "A", "B"), Support(min_covered_events=5))
    assert strict["support"]["status"] == "insufficient"
    assert (strict["eligible_events"], strict["covered_events"]) == (3, 3)
    assert strict["coverage_rate"] == 1.0 and strict["coverage_ci_high"] == 1.0
    assert strict["median_response_hours"] is None and strict["latency_ci_low"] is None
    by_reports = summarize(pair(frame, "A", "B"), Support(1, min_effective_reports=10_000))
    assert by_reports["support"]["status"] == "insufficient"
    assert summarize(pair(frame, "A", "B"), Support(3, 0))["support"]["status"] == "ok"


# ------------------------------------------------------------------ matrix & filters


def test_multi_country_matrix(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    cells = {
        (c["origin_country"], c["destination_country"]): c
        for c in pair_matrix(frame, ["A", "B"], ["A", "B", "C"], LOOSE)
    }
    assert len(cells) == 6
    assert cells[("A", "A")]["kind"] == "domestic"
    assert cells[("A", "A")]["median_response_hours"] == pytest.approx(1.5)
    assert cells[("A", "B")]["median_response_hours"] == pytest.approx(3.0)
    assert cells[("B", "A")]["median_response_hours"] == pytest.approx(0.25)
    assert cells[("B", "C")]["covered_events"] == 0
    assert cells[("A", "C")]["coverage_rate"] == pytest.approx(1 / 3)


def test_filters_compose_with_and(observations: pl.DataFrame) -> None:
    by_type = family_level(observations, event_types=("natural_disaster",))
    assert by_type["event_id"].unique().sort().to_list() == [1, 5]
    late = family_level(observations, start=t(7, 0))
    assert late["event_id"].unique().to_list() == [4]
    early = family_level(observations, end=t(6, 9))
    assert early["event_id"].unique().sort().to_list() == [1, 3, 5]
    big = family_level(observations, min_event_effective_reports=60)
    assert big["event_id"].unique().sort().to_list() == [1, 3, 5]
    both = family_level(
        observations, event_types=("natural_disaster",), min_event_effective_reports=100
    )
    assert both["event_id"].unique().to_list() == [1]
    assert family_level(observations, resolution_model="other").height == 0
    scoped = family_level(observations, origins=("B",), destinations=("A", "C"))
    assert set(scoped["destination_country"]) == {"A", "C"} and set(scoped["origin_country"]) == {
        "B"
    }
    incidents = apply_filters(observations, Filters(level="incident"))
    assert summarize(pair(incidents, "A", "B"), LOOSE)["eligible_events"] == 4
    with pytest.raises(ValueError):
        apply_filters(observations, Filters(reference="bogus"))


def test_breakdown_by_event_type(observations: pl.DataFrame) -> None:
    frame = family_level(observations)
    rows = {r["event_types"]: r for r in summarize_by_type(pair(frame, "A", "B"), LOOSE)}
    assert rows["natural_disaster"]["median_response_hours"] == pytest.approx(2.0)
    assert rows["politics_government"]["eligible_events"] == 2
    assert rows["death"]["covered_events"] == 1


def test_cli_materializes_and_loader_prefers_the_file(tmp_path: Path, monkeypatch) -> None:
    fam, inc = summaries()
    macro_events().write_parquet(tmp_path / "macro_events.parquet")
    families().write_parquet(tmp_path / "event_families.parquet")
    fam.write_parquet(tmp_path / "country_family_summary.parquet")
    inc.write_parquet(tmp_path / "country_event_summary.parquet")
    baseline().write_parquet(tmp_path / "country_baseline.parquet")
    (tmp_path / "meta.json").write_text(
        '{"window_start": "2023-02-06 00:00:00+00:00", "window_end": "2023-02-08 23:45:00+00:00"}'
    )
    derived = analytics.load_observations(tmp_path)
    monkeypatch.setattr("sys.argv", ["analytics", str(tmp_path)])
    analytics.main()
    assert (tmp_path / analytics.OBSERVATIONS_FILE).exists()
    stored = analytics.load_observations(tmp_path)
    assert stored.equals(derived)
