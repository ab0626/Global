import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from clustering_experiment import (
    ArticleSettings,
    article_features,
    channel_candidates,
    score_candidates,
    tfidf_channel,
    url_words,
)


def test_url_features_keep_unicode_and_short_story_numbers() -> None:
    words = url_words("/2019/03/15/737-max-flight-302-über-津巴布韦-12345678.html")
    assert {"737", "302", "max", "flight", "über", "津巴布韦"} <= words
    assert not {"2019", "03", "15", "12345678", "html"} & words


def test_idf_cosine_and_empty_rows() -> None:
    settings = ArticleSettings()
    matrix = sparse.csr_matrix([[1.0, 1], [1, 0], [0, 0]])
    weighted, audit = tfidf_channel(matrix, settings)
    common = np.log(4 / 3) + 1
    rare = np.log(4 / 2) + 1
    expected = common / np.sqrt(common**2 + rare**2)
    assert float((weighted @ weighted.T)[0, 1]) == pytest.approx(expected)
    assert np.isfinite(weighted.data).all()
    assert weighted[2].nnz == 0
    assert audit["articles_with_features"] == 2


def test_frequency_and_expansion_guards() -> None:
    matrix = sparse.csr_matrix(np.ones((3, 1)))
    with pytest.raises(ValueError, match="pair contributions"):
        tfidf_channel(matrix, replace(ArticleSettings(), max_pair_contributions=1))
    weighted, audit = tfidf_channel(matrix, replace(ArticleSettings(), max_feature_articles=2))
    assert weighted.nnz == 0
    assert audit["frequent_features_removed"] == 1
    with pytest.raises(ValueError, match="capacity"):
        channel_candidates(matrix, replace(ArticleSettings(), max_candidate_pairs=1))
    with pytest.raises(ValueError, match="block"):
        channel_candidates(matrix, replace(ArticleSettings(), max_block_nonzeros=1))


def test_neighbor_ties_are_deterministic_and_self_edges_are_excluded() -> None:
    matrix = sparse.csr_matrix(np.ones((4, 1)))
    settings = replace(ArticleSettings(), neighbors=1, block_size=2)
    keys = channel_candidates(matrix, settings)
    assert keys.tolist() == [1, 1, 2, 3]
    assert np.all(keys // 4 != keys % 4)


def test_union_is_scored_in_both_channels_against_dense_reference() -> None:
    settings = replace(ArticleSettings(), neighbors=3, block_size=2, scoring_batch_size=2)
    event, _ = tfidf_channel(sparse.csr_matrix([[1.0, 0], [1, 1], [0, 1], [0, 0]]), settings)
    url, _ = tfidf_channel(sparse.csr_matrix([[1.0, 0], [0, 1], [1, 0], [0, 0]]), settings)
    nodes = pl.DataFrame({"vertex": range(4)})
    pairs, audit = score_candidates(nodes, {"event": event, "url": url}, settings)
    assert set(pairs.select("left", "right").iter_rows()) == {(0, 1), (0, 2), (1, 2)}
    event_dense, url_dense = (event @ event.T).toarray(), (url @ url.T).toarray()
    for left, right, event_score, url_score, combined in pairs.iter_rows():
        assert event_score == pytest.approx(event_dense[left, right])
        assert url_score == pytest.approx(url_dense[left, right])
        assert combined == pytest.approx((event_score + url_score) / 2)
    assert audit["url_only_candidate_pairs"] == 1
    assert audit["candidate_recall"] is None


def test_wire_groups_are_nodes_and_ineligible_articles_stay_out() -> None:
    articles = pl.DataFrame(
        {
            "canonical_id": [10, 10, 20, 30],
            "slug": ["/alpha", "/beta", "/alpha-beta", "/irrelevant"],
        }
    )
    incidence = pl.DataFrame(
        {"canonical_id": [10, 20], "vertex": [0, 1], "GlobalEventID": [100, 200]}
    )
    settings = ArticleSettings()
    nodes, channels, audit = article_features(articles, incidence, settings)
    assert nodes["canonical_id"].to_list() == [10, 20]
    assert audit["active_canonical_articles"] == 2
    pairs, _ = score_candidates(nodes, channels, settings)
    assert pairs["event_score"].to_list() == [0.0]
    assert pairs["url_score"].to_list() == pytest.approx([1.0])
    assert pairs["combined"].to_list() == pytest.approx([0.5])


def test_empty_experiment_and_prepared_reuse(tmp_path: Path) -> None:
    mentions = tmp_path / "mentions"
    mentions.mkdir()
    (mentions / "20190315120000.mentions.csv").write_text(
        "1\t20190315120000\t20190315120000\t1\tpublisher.example\t"
        "https://publisher.example/low-confidence\t1\t10\t20\t30\t1\t10\t1000\t-1.5\t\t\n"
    )
    script = str(Path(__file__).parents[1] / "clustering_experiment.py")
    prepared = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            script,
            "--mentions",
            str(mentions),
            "--start-date",
            "20190315",
            "--end-date",
            "20190315",
            "--allow-partial",
            "--prepare-only",
            "--output",
            str(prepared),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "output"
    subprocess.run(
        [sys.executable, script, "--prepared", str(prepared), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = pl.read_csv(output / "comparison.csv")
    assert comparison.height == 13
    assert comparison["clusters"].sum() == 0
    assert (output / "article_pairs.parquet").exists()
