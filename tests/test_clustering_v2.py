from datetime import UTC, datetime

import numpy as np
import polars as pl
from scipy import sparse

from clustering_experiment import ArticleSettings
from clustering_v2 import (
    V2Settings,
    burst_filter,
    clean_token,
    families,
    host_filter,
    label_rows,
    score_pairs,
)


def _rows(items: list[tuple[int, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(v, d, datetime.strptime(day, "%Y-%m-%d").date(), f) for v, d, day, f in items],
        schema={"vertex": pl.UInt32, "domain": pl.String, "day": pl.Date, "feature": pl.String},
        orient="row",
    )


def test_clean_token_drops_hex_fragments_and_long_numbers_but_keeps_737() -> None:
    assert clean_token("737") and clean_token("302") and clean_token("brexit")
    assert not clean_token("11e9") and not clean_token("94ab") and not clean_token("12")
    assert clean_token("idai") and clean_token("津巴布韦")


def test_burst_filter_drops_uniform_frequent_tokens_only() -> None:
    items = []
    for i in range(8):
        items.append((i, "a.com", f"2019-03-{10 + i}", "police"))
        items.append((i, "a.com", "2019-03-15", "mosque"))
    kept, dropped = burst_filter(
        _rows(items), 8, V2Settings(burst_min_articles=8, burst_entropy_max=0.95)
    )
    assert dropped["feature"].to_list() == ["police"]
    assert set(kept["feature"].to_list()) == {"mosque"}
    _, none_dropped = burst_filter(_rows(items), 8, V2Settings(burst_min_articles=9))
    assert none_dropped.height == 0


def test_host_filter_removes_cms_boilerplate_per_host() -> None:
    items = [(i, "cms.com", "2019-03-10", "articleshow") for i in range(20)]
    items += [(i, "cms.com", "2019-03-10", f"story{i}") for i in range(20)]
    items += [(99, "other.com", "2019-03-10", "articleshow")]
    kept, pairs = host_filter(_rows(items), V2Settings())
    assert pairs == 1
    assert kept.filter(pl.col("domain") == "cms.com")["feature"].str.starts_with("story").all()
    assert kept.filter(pl.col("domain") == "other.com").height == 1


def test_label_rows_prefix_actor_and_geo_and_skip_missing() -> None:
    incidence = pl.DataFrame(
        {"canonical_id": [0, 0, 1], "GlobalEventID": [1, 2, 3], "vertex": [0, 1, 2]},
        schema={"canonical_id": pl.UInt32, "GlobalEventID": pl.Int64, "vertex": pl.UInt32},
    )
    labels = pl.DataFrame(
        {
            "GlobalEventID": [1, 2],
            "Actor1Name": ["BOEING", None],
            "Actor2Name": ["", "ETHIOPIA"],
            "EventCode": ["010", "020"],
            "ActionGeo_FullName": ["Addis Ababa, Ethiopia", None],
        },
        schema_overrides={"GlobalEventID": pl.Int64},
    )
    nodes = pl.DataFrame({"canonical_id": [0, 1], "vertex": [0, 1]}).cast(
        {"canonical_id": pl.UInt32, "vertex": pl.UInt32}
    )
    articles = pl.DataFrame(
        {
            "canonical_id": pl.Series([0, 1], dtype=pl.UInt32),
            "first_seen": [datetime(2019, 3, 10, tzinfo=UTC)] * 2,
        }
    )
    rows = label_rows(incidence, labels, nodes, articles)
    assert sorted(rows["feature"].to_list()) == [
        "actor:BOEING",
        "actor:ETHIOPIA",
        "geo:Addis Ababa, Ethiopia",
    ]
    assert rows["vertex"].unique().to_list() == [0]


def test_gated_score_requires_corroboration_or_strong_single_channel() -> None:
    nodes = pl.DataFrame({"canonical_id": range(4), "vertex": range(4)})
    event = sparse.csr_matrix(np.array([[1.0, 0], [1, 0], [0, 1], [0, 0]]))
    url = sparse.csr_matrix(np.array([[1.0, 0], [0, 1], [0, 1], [0, 1]]) / 1.0)
    weak = np.array([[0.7, np.sqrt(1 - 0.49)], [0.7, np.sqrt(1 - 0.49)], [0, 1], [0, 0]])
    settings = V2Settings(single_channel_floor=0.6)
    pairs, audit = score_pairs(
        nodes, {"event": event, "url": sparse.csr_matrix(weak)}, ArticleSettings(), settings
    )
    row = pairs.filter((pl.col("left") == 0) & (pl.col("right") == 1)).row(0, named=True)
    assert row["evidence_channels"] == 2 and row["gated"] == row["combined"] == 1.0
    row = pairs.filter((pl.col("left") == 1) & (pl.col("right") == 2)).row(0, named=True)
    assert row["evidence_channels"] == 1
    assert row["combined"] == (0 + weak[1, 1]) / 2
    assert row["gated"] == row["combined"]
    pairs, _ = score_pairs(
        nodes, {"event": event, "url": url}, ArticleSettings(), V2Settings(single_channel_floor=1.1)
    )
    row = pairs.filter((pl.col("left") == 1) & (pl.col("right") == 2)).row(0, named=True)
    assert row["gated"] == 0.0 and row["available_channels"] == 2
    assert audit["candidate_recall"] is None


def test_families_link_similar_incident_clusters_and_keep_singletons() -> None:
    nodes = pl.DataFrame({"canonical_id": range(5), "vertex": range(5)})
    partition = pl.DataFrame(
        {"canonical_id": range(5), "vertex": range(5), "cluster": [0, 0, 1, 1, 2]}
    )
    url = sparse.csr_matrix(np.array([[1.0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    result, audit = families(nodes, partition, {"url": url}, V2Settings(), ArticleSettings())
    by_cluster = result.group_by("cluster").agg(pl.col("family").unique()).sort("cluster")
    fam = {c: f[0] for c, f in by_cluster.iter_rows()}
    assert fam[0] == fam[1] != fam[2]
    assert audit["families_with_multiple_incidents"] == 1
    assert result["family"].n_unique() == 2
