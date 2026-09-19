import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from story_clusters import (
    EVENT_COLUMNS,
    MENTION_COLUMNS,
    Settings,
    article_memberships,
    cluster_sizes,
    communities,
    input_files,
    keyword_validation,
    prepare_articles,
    project,
    read_labels,
    read_mentions,
    read_table,
)


def mention(
    url: str,
    event: int,
    *,
    offset: int = 10,
    length: int = 1000,
    confidence: int = 100,
    event_time: str = "20190315120000",
    mention_time: str = "20190315120000",
    translation: str = "",
) -> list[str]:
    return [
        str(event),
        event_time,
        mention_time,
        "1",
        "publisher.example",
        url,
        "1",
        str(offset),
        "20",
        "30",
        "1",
        str(confidence),
        str(length),
        "-1.5",
        translation,
        "",
    ]


def load(tmp_path: Path, rows: list[list[str]]) -> pl.DataFrame:
    path = tmp_path / "20190315120000.mentions.csv"
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    return read_mentions([path])[0]


def test_exact_wire_fingerprint_and_provenance(tmp_path: Path) -> None:
    rows = [
        mention("https://a.example/story", event, translation="srclc:fra;eng:GT-FRA")
        for event in [1, 2]
    ]
    rows += [
        mention("https://b.example/story", event, mention_time="20190315121500") for event in [1, 2]
    ]
    rows += [mention("https://c.example/story", event, offset=11) for event in [1, 2]]
    rows += [mention("https://d.example/story", event, length=1001) for event in [1, 2]]
    rows += [rows[0]]
    data = load(tmp_path, rows)
    _, articles, audit = prepare_articles(data, Settings())
    assert audit["urls_collapsed"] == 1
    assert audit["translated_web_urls"] == 1
    assert articles["canonical_id"].to_list() == [0, 0, 1, 2]
    assert articles["observations"].list.len().to_list() == [1, 1, 1, 1]
    assert articles["domain"].to_list() == ["a.example", "b.example", "c.example", "d.example"]
    assert articles["first_seen"].dt.minute().to_list() == [0, 15, 0, 0]


def test_no_dedup_from_filtered_or_missing_fingerprints(tmp_path: Path) -> None:
    rows = [mention("https://a.example/x", 1), mention("https://a.example/x", 2, confidence=10)]
    rows += [mention("https://b.example/x", 1), mention("https://c.example/x", 1)]
    for url in ["https://d.example/x", "https://e.example/x"]:
        for event in [1, 2]:
            row = mention(url, event)
            row[7] = ""
            rows.append(row)
    _, articles, audit = prepare_articles(load(tmp_path, rows), Settings())
    assert audit["urls_collapsed"] == 0
    assert articles["canonical_id"].n_unique() == 5


def test_projection_weights_and_singleton_events(tmp_path: Path) -> None:
    rows = [
        mention(f"https://{i}.example/x", event, length=1000 + i)
        for i, events in enumerate([[1, 2], [1, 2], [1, 3]])
        for event in events
    ]
    web, articles, _ = prepare_articles(load(tmp_path, rows), Settings())
    events, incidence, pairs, _ = project(web, articles, Settings())
    assert pairs["count"].to_list() == [2]
    assert pairs["jaccard"].to_list() == pytest.approx([2 / 3])
    for method in ["components", "louvain"]:
        membership, _ = communities(events, pairs, "jaccard", 0.5, method, 42)
        assert membership["cluster"].n_unique() == 2
        members = article_memberships(incidence, membership, articles)
        assert members["article_id"].n_unique() == 3
        assert members.height == 4


def test_count_projection_does_not_overflow_uint8(tmp_path: Path) -> None:
    rows = [
        mention(f"https://a.example/{i}", event, length=1000 + i)
        for i in range(300)
        for event in [1, 2]
    ]
    web, articles, _ = prepare_articles(load(tmp_path, rows), Settings())
    _, _, pairs, _ = project(web, articles, Settings())
    assert pairs["count"].to_list() == [300]
    assert pairs["jaccard"].to_list() == [1.0]


def test_roundups_confidence_time_and_memory_guards(tmp_path: Path) -> None:
    rows = [
        mention(
            f"https://a.example/{i}",
            event,
            length=1000 + i,
            event_time="20190310120000" if event == 1 else "20190315120000",
        )
        for i in range(2)
        for event in [1, 2]
    ]
    rows += [mention("https://roundup.example/x", e) for e in [1, 2, 3]]
    rows += [mention("https://low.example/x", 4, confidence=10)]
    settings = replace(Settings(), max_article_events=2)
    web, articles, _ = prepare_articles(load(tmp_path, rows), settings)
    events, _, pairs, audit = project(web, articles, settings)
    assert audit["roundup_urls_dropped"] == 1
    assert audit["low_confidence_rows"] == 1
    assert set(events["GlobalEventID"]) == {1, 2}
    assert pairs.is_empty()
    with pytest.raises(ValueError, match="pair contributions"):
        project(web, articles, replace(settings, max_pair_contributions=1))


def test_recurring_filter_needs_both_frequency_and_span(tmp_path: Path) -> None:
    rows = [
        mention(
            f"https://a.example/{i}",
            event,
            length=1000 + i,
            mention_time="20190315120000" if i == 0 or event == 2 else "20190318120000",
        )
        for i in range(2)
        for event in [1, 2]
    ]
    settings = replace(Settings(), recurring_articles=2)
    web, articles, _ = prepare_articles(load(tmp_path, rows), settings)
    events, _, _, audit = project(web, articles, settings)
    assert audit["recurring_events_dropped"] == 1
    assert events["GlobalEventID"].to_list() == [2]


def test_recall_includes_unassigned_and_overlap(tmp_path: Path) -> None:
    rows = [
        mention("https://a.example/brexit-one", 1),
        mention("https://b.example/brexit-two", 1),
        mention("https://b.example/brexit-two", 2),
        mention("https://c.example/brexit-three", 3, confidence=10),
        mention("https://d.example/unrelated", 1, length=1001),
    ]
    web, articles, _ = prepare_articles(load(tmp_path, rows), Settings())
    events, incidence, pairs, _ = project(web, articles, Settings())
    membership, _ = communities(events, pairs, "count", 2, "components", 42)
    members = article_memberships(incidence, membership, articles)
    sizes = cluster_sizes(members, membership)
    results, distribution = keyword_validation(articles, members, sizes)
    brexit = next(r for r in results if r["story"] == "brexit")
    assert brexit["keyword_urls"] == 3
    assert brexit["best_recall"] == pytest.approx(2 / 3)
    assert brexit["best_keyword_purity"] == pytest.approx(2 / 3)
    assert brexit["unassigned_keyword_urls"] == 1
    assert distribution.filter(pl.col("story") == "brexit")["matches"].sum() == 3
    assert next(r for r in results if r["story"] == "idai")["best_recall"] is None


def test_schema_rejects_short_rows_and_replaces_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="16 tab-separated fields"):
        read_table([path], MENTION_COLUMNS)
    row = mention("https://a.example/x", 1)
    path.write_bytes(("\t".join(row) + "\n").encode().replace(b"a.example", b"\xff.example"))
    data, audit = read_mentions([path])
    assert audit["files"][0]["replacement_characters"] == 1
    assert "\ufffd" in data["MentionIdentifier"][0]
    assert audit["null_counts"]["Extras"] == 1


def test_missing_slots_are_explicit(tmp_path: Path) -> None:
    load(tmp_path, [mention("https://a.example/x", 1)])
    date = datetime(2019, 3, 15)
    with pytest.raises(ValueError, match="95 missing"):
        input_files(tmp_path, "mentions", date, date, False)
    paths, missing = input_files(tmp_path, "mentions", date, date, True)
    assert len(paths) == 1 and len(missing) == 95


def test_event_location_is_full_name_not_country_code(tmp_path: Path) -> None:
    row = [""] * 61
    row[0], row[6], row[16], row[26] = "1", "ACTOR1", "ACTOR2", "010"
    row[52], row[53] = "Christchurch, New Zealand", "NZ"
    path = tmp_path / "20190315120000.export.csv"
    path.write_text("\t".join(row) + "\n", encoding="utf-8")
    labels, _ = read_labels([path])
    assert len(EVENT_COLUMNS) == 61
    assert labels["ActionGeo_FullName"].to_list() == ["Christchurch, New Zealand"]
    assert labels["EventCode"].to_list() == ["010"]


@pytest.mark.parametrize("confidence", [10, 100])
def test_cli_writes_all_reports_including_empty_graph(tmp_path: Path, confidence: int) -> None:
    load(tmp_path, [mention("https://a.example/brexit", 1, confidence=confidence)])
    output = tmp_path / "results"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "story_clusters.py"),
            "--mentions",
            str(tmp_path),
            "--start-date",
            "20190315",
            "--end-date",
            "20190315",
            "--allow-partial",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(list(output.glob("*/metrics.json"))) == 12
    audit = json.loads((output / "audit.json").read_text())
    assert audit["partial"] is True
    assert audit["mentions"]["rows"] == 1
    report = (output / "report.md").read_text()
    assert "Partial input: True" in report
    assert "95 expected slots missing" in report
    scores = pl.read_csv(output / "story_scores.csv").filter(pl.col("story") == "brexit")
    assert scores["unassigned_keyword_urls"].to_list() == [int(confidence == 10)] * 12
