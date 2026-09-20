"""Unit checks for the attention pipeline (preprocess -> materialize)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from attention.cluster import ClusterSettings, leiden
from attention.materialize import coherence, event_types, label_for, onset
from attention.preprocess import canonical_url, read_raw, resolve_country


def test_canonical_url_strips_tracking_and_host_noise() -> None:
    url = "HTTPS://www.Example.com/news/story/?utm_source=x&id=3&fbclid=abc"
    assert canonical_url(url) == "https://example.com/news/story?id=3"
    assert canonical_url("http://example.com/a/") == canonical_url("http://example.com/a")


def test_read_raw_counts_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("1\ta\tb\n2\tonly-two\n3\tc\td\n", encoding="utf-8")
    frame, audit = read_raw([path], ["id", "x", "y"], ["id", "y"])
    assert frame["id"].to_list() == ["1", "3"]
    assert audit["malformed_rows"] == 1
    assert audit["files"][0]["rows"] == 3
    assert len(audit["files"][0]["sha256"]) == 64


def test_resolve_country_fallbacks() -> None:
    lookup = {"bbc.co.uk": ("UK", "BBC"), "example.com": ("US", "Example")}
    assert resolve_country("bbc.co.uk", lookup) == ("UK", 0.9, "gdelt_lookup")
    assert resolve_country("news.example.com", lookup) == ("US", 0.7, "gdelt_lookup_parent")
    assert resolve_country("zeitung.de", lookup) == ("DE", 0.5, "cctld")
    assert resolve_country("unknown.org", lookup) == (None, 0.0, "unresolved")


def ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2019, 4, 15, hour, minute, tzinfo=UTC)


def test_onset_requires_k_outlets_and_tenth_percentile() -> None:
    rows = [("a.com", ts(1))] + [(f"s{i}.com", ts(17, 15)) for i in range(20)]
    assigned = pl.DataFrame(
        {
            "macro_event_id": [0] * len(rows),
            "source_domain": [r[0] for r in rows],
            "observed_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        }
    )
    result = onset(assigned, ["macro_event_id"])
    assert result["third_source_seen"][0] == ts(17)
    assert result["onset"][0] == ts(17)
    thin = assigned.head(2)
    assert onset(thin, ["macro_event_id"]).is_empty()


def test_coherence_scores_mixed_clusters_low() -> None:
    members = pl.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1, 1, 1],
            "document_id": list(range(7)),
            "persons": [["x"], ["x"], ["x"], ["a"], ["b"], ["c"], ["d"]],
            "organizations": [[], [], [], [], [], [], []],
        }
    )
    scores = dict(coherence(members).iter_rows())
    assert scores[0] == 1.0
    assert scores[1] == 0.75


def test_event_types_from_themes_and_cameo() -> None:
    members = pl.DataFrame(
        {
            "cluster": [0, 0, 0, 0],
            "document_id": [1, 2, 3, 4],
            "themes": [
                ["DISASTER_FIRE", "TAX_FNCACT"],
                ["DISASTER_FIRE"],
                ["DISASTER_FIRE"],
                ["ECON_TAXATION"],
            ],
        }
    )
    links = pl.DataFrame(
        {
            "cluster": [0, 0, 0, 0],
            "document_id": [1, 2, 3, 4],
            "EventRootCode": ["14", "14", "14", "14"],
        }
    )
    types = event_types(members, links)
    row = types.row(0, named=True)
    assert row["event_types"][:2] == ["protest", "fire"]
    assert "economy_business" not in row["event_types"]


def test_label_prefers_frequent_specific_tokens() -> None:
    urls = [
        "https://a.com/news/notre-dame-fire-paris",
        "https://b.com/world/notre-dame-cathedral-fire",
        "https://c.com/2019/04/15/paris-notre-dame-blaze",
    ]
    assert label_for(urls).split()[:2] == ["dame", "notre"]


def test_leiden_marks_isolates_unassigned() -> None:
    pairs = pl.DataFrame({"left": [0, 1], "right": [1, 2], "gated": [0.9, 0.8]})
    membership = leiden(pairs, 5, ClusterSettings())
    assert membership[0] == membership[1] == membership[2] >= 0
    assert membership[3] == -1 and membership[4] == -1
