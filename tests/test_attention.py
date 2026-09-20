"""Unit checks for the attention pipeline (preprocess -> materialize)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import igraph as ig
import numpy as np
import polars as pl

from attention.cluster import (
    BACKGROUND_MIN_DOCS,
    BACKGROUND_PRIOR,
    ClusterSettings,
    TitleChannel,
    boilerplate_titles,
    cpm_leiden,
    leiden,
)
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


def test_cpm_leiden_edgeless_graph_returns_singletons() -> None:
    membership = cpm_leiden(ig.Graph(n=4), [], resolution=0.05, seed=1)
    assert sorted(membership.tolist()) == [0, 1, 2, 3]


def unit(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def test_title_calibration_neutralises_language_common_component() -> None:
    """A script whose random titles all share a common direction (Korean-like) must
    not score unrelated pairs above a well-tokenised script's unrelated pairs."""
    rng = np.random.default_rng(0)
    n, dim = BACKGROUND_MIN_DOCS, 32
    common = rng.normal(size=dim)
    biased = unit(common * 2.5 + rng.normal(size=(n, dim)))
    plain = unit(rng.normal(size=(n, dim)))
    vectors = np.vstack([biased, plain])
    ids = np.arange(2 * n)
    languages = np.array(["kor"] * n + ["eng"] * n)
    raw_biased = float(np.mean(biased[: n // 2] @ biased[n // 2 :].T))
    assert raw_biased > 0.5

    channel = TitleChannel(vectors, ids, 2 * n, languages, seed=1)
    left, right = ids[: n // 2], ids[n // 2 : n]
    calibrated = channel.scores(left, right, batch=1024)
    assert float(np.quantile(calibrated, 0.9)) < 0.15
    table = channel.background_table()
    kor = table.filter((pl.col("language_left") == "kor") & (pl.col("language_right") == "kor"))
    assert 0.0 < kor["quantile"][0] < 0.5

    # identical titles still score ~1 after calibration
    same = channel.scores(ids[:5], ids[:5], batch=1024)
    assert np.allclose(same, 1.0)


def test_title_channel_small_groups_use_prior_and_missing_languages() -> None:
    rng = np.random.default_rng(3)
    vectors = unit(rng.normal(size=(20, 8)))
    ids = np.arange(20)
    channel = TitleChannel(vectors, ids, 20, np.array(["eng"] * 10 + ["und"] * 10))
    assert channel.language_codes == ["other"]
    assert channel.background_table()["quantile"].to_list() == [BACKGROUND_PRIOR]
    raw = float(vectors[0] @ vectors[1])
    expected = max(0.0, (raw - BACKGROUND_PRIOR) / (1 - BACKGROUND_PRIOR))
    assert np.isclose(channel.scores(ids[:1], ids[1:2], batch=8)[0], expected, atol=1e-5)

    no_languages = TitleChannel(vectors, ids, 20)
    assert np.isclose(no_languages.scores(ids[:1], ids[1:2], batch=8)[0], expected, atol=1e-5)

    empty = TitleChannel(np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.int64), 0)
    assert empty.candidates(neighbors=5).shape == (0,)


def test_domain_centering_cancels_site_boilerplate() -> None:
    """Titles from one site sharing a suffix vector ("... | Site News") should not
    look alike after centering on that domain."""
    rng = np.random.default_rng(5)
    n, dim = BACKGROUND_MIN_DOCS, 32
    suffix = rng.normal(size=dim)
    site = unit(suffix * 2.0 + rng.normal(size=(n, dim)))
    other = unit(rng.normal(size=(n, dim)))
    vectors = np.vstack([site, other])
    ids = np.arange(2 * n)
    languages = np.array(["eng"] * (2 * n))
    domains = np.array(["site.example"] * n + [f"d{i}.example" for i in range(n)])
    without = TitleChannel(vectors, ids, 2 * n, languages, seed=1)
    with_domains = TitleChannel(vectors, ids, 2 * n, languages, seed=1, domains=domains)
    left, right = ids[: n // 2], ids[n // 2 : n]
    assert with_domains.scores(left, right, 1024).mean() < without.scores(left, right, 1024).mean()
    assert float(np.quantile(with_domains.scores(left, right, 1024), 0.9)) < 0.15


def test_boilerplate_titles_flags_repeated_domain_titles() -> None:
    documents = pl.DataFrame(
        {
            "document_id": list(range(14)),
            "domain": ["tass.ru"] * 10 + ["a.com", "a.com", "b.com", "c.com"],
            "title": ["ТАСС"] * 10 + ["Same", "Same", "ТАСС", None],
        }
    )
    assert sorted(boilerplate_titles(documents).tolist()) == list(range(10))
