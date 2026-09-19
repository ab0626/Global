"""End-to-end checks of the GDELT-shaped API over a synthetic two-file slice."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import dataset
from api.app import app
from api.build import build
from api.query import QueryError, parse
from api.schema import EVENT_COLUMNS

MENTIONS = [
    # event, eventtime, mentiontime, type, source, url, sentence, offsets…, confidence, len, tone
    "1\t20190315000000\t20190315000000\t1\tstuff.co.nz\thttps://stuff.co.nz/news/christchurch-attack\t1\t10\t20\t30\t1\t100\t4000\t-7.5\t\t",
    "2\t20190315000000\t20190315001500\t1\tlemonde.fr\thttps://lemonde.fr/monde/article/boeing-737-max\t2\t10\t20\t30\t1\t80\t3000\t2.5\tsrclc:fra;eng:Moses\t",
    "1\t20190315000000\t20190315001500\t1\tlemonde.fr\thttps://lemonde.fr/monde/article/boeing-737-max\t3\t10\t20\t30\t1\t60\t3000\t2.5\tsrclc:fra;eng:Moses\t",
    # A television mention must be dropped: the DOC API only serves web documents.
    "1\t20190315000000\t20190315000000\t2\tCNN\thttps://tv.example/clip\t1\t1\t2\t3\t1\t100\t10\t0.0\t\t",
]


def event_row(event_id: str, actor1: str, code: str, place: str, lat: str, lon: str) -> str:
    values = {column: "" for column in EVENT_COLUMNS}
    values.update(
        {
            "GlobalEventID": event_id,
            "Day": "20190315",
            "MonthYear": "201903",
            "Year": "2019",
            "FractionDate": "2019.2",
            "Actor1Name": actor1,
            "Actor1CountryCode": "NZL",
            "EventCode": code,
            "EventBaseCode": code,
            "EventRootCode": code[:2],
            "QuadClass": "4",
            "GoldsteinScale": "-9.0",
            "NumMentions": "5",
            "NumSources": "2",
            "NumArticles": "5",
            "AvgTone": "-7.5",
            "ActionGeo_FullName": place,
            "ActionGeo_CountryCode": "NZ",
            "ActionGeo_Lat": lat,
            "ActionGeo_Long": lon,
            "DATEADDED": "20190315000000",
            "SOURCEURL": "https://stuff.co.nz/news/christchurch-attack",
        }
    )
    return "\t".join(values[column] for column in EVENT_COLUMNS)


EVENTS = [
    event_row("1", "POLICE", "190", "Christchurch, New Zealand", "-43.5", "172.6"),
    event_row("2", "BOEING", "173", "Seattle, United States", "47.6", "-122.3"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    root = tmp_path_factory.mktemp("slice")
    mentions_dir, events_dir = root / "mentions", root / "events"
    mentions_dir.mkdir()
    events_dir.mkdir()
    (mentions_dir / "20190315000000.mentions.csv").write_text("\n".join(MENTIONS) + "\n")
    (events_dir / "20190315000000.export.csv").write_text("\n".join(EVENTS) + "\n")
    meta = build(mentions_dir, events_dir, root / "api")
    assert meta["articles"] == 2
    os.environ["GDELT_API_DATA"] = str(root / "api")
    dataset.load.cache_clear()
    return TestClient(app)


def article_urls(payload: dict) -> list[str]:
    return [article["url"] for article in payload["articles"]]


def test_artlist_has_doc_api_fields(client: TestClient) -> None:
    payload = client.get("/api/v2/doc/doc", params={"query": '"christchurch attack"'}).json()
    assert article_urls(payload) == ["https://stuff.co.nz/news/christchurch-attack"]
    article = payload["articles"][0]
    for field in ["url", "url_mobile", "title", "seendate", "socialimage", "domain", "language"]:
        assert field in article
    assert article["seendate"] == "20190315T000000Z"
    assert article["title"] == "christchurch attack"
    assert article["sourcecountry"] == "New Zealand"
    assert article["language"] == "English"


def test_search_reaches_labels_of_linked_events(client: TestClient) -> None:
    # The French article never spells "christchurch"; it mentions the event located there.
    payload = client.get("/api/v2/doc/doc", params={"query": "christchurch"}).json()
    assert sorted(article_urls(payload)) == [
        "https://lemonde.fr/monde/article/boeing-737-max",
        "https://stuff.co.nz/news/christchurch-attack",
    ]


def test_television_mentions_are_not_served(client: TestClient) -> None:
    payload = client.get("/api/v2/doc/doc", params={"query": "clip"}).json()
    assert payload["articles"] == []


def test_translation_info_sets_language(client: TestClient) -> None:
    payload = client.get("/api/v2/doc/doc", params={"query": "sourcelang:french"}).json()
    assert article_urls(payload) == ["https://lemonde.fr/monde/article/boeing-737-max"]


def test_negation_and_or_groups(client: TestClient) -> None:
    both = client.get("/api/v2/doc/doc", params={"query": "christchurch OR boeing"}).json()
    assert len(both["articles"]) == 2
    excluded = client.get(
        "/api/v2/doc/doc", params={"query": "christchurch OR boeing -domainis:lemonde.fr"}
    ).json()
    assert article_urls(excluded) == ["https://stuff.co.nz/news/christchurch-attack"]


def test_timeline_volume_is_a_share_of_monitored_articles(client: TestClient) -> None:
    payload = client.get(
        "/api/v2/doc/doc", params={"query": '"christchurch attack"', "mode": "timelinevol"}
    ).json()
    series = payload["timeline"][0]
    assert series["series"] == "Volume Intensity"
    assert series["data"][0] == {"date": "20190315T000000Z", "value": 100.0}
    assert series["data"][1]["value"] == 0.0


def test_timeline_raw_reports_matched_and_total(client: TestClient) -> None:
    payload = client.get(
        "/api/v2/doc/doc", params={"query": '"christchurch attack"', "mode": "timelinevolraw"}
    ).json()
    names = [series["series"] for series in payload["timeline"]]
    assert names == ["Article Count", "Total Monitoring Volume"]


def test_tonechart_bins_articles(client: TestClient) -> None:
    payload = client.get("/api/v2/doc/doc", params={"mode": "tonechart"}).json()
    bins = {entry["bin"]: entry["count"] for entry in payload["tonechart"]}
    assert bins == {-8: 1, 2: 1}


def test_geo_returns_geojson_points(client: TestClient) -> None:
    payload = client.get("/api/v2/geo/geo", params={"query": "christchurch"}).json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["coordinates"] == [172.6, -43.5]
    assert feature["properties"]["name"] == "Christchurch, New Zealand"


def test_event_detail_lists_its_coverage(client: TestClient) -> None:
    payload = client.get("/api/v2/ext/events/1").json()
    assert payload["event"]["rootlabel"] == "Fight"
    assert payload["event"]["quadlabel"] == "Material Conflict"
    assert "https://stuff.co.nz/news/christchurch-attack" in article_urls(payload)
    assert client.get("/api/v2/ext/events/99").status_code == 404


def test_facets_count_linked_cameo_labels(client: TestClient) -> None:
    payload = client.get("/api/v2/ext/facets", params={"query": '"christchurch attack"'}).json()
    assert payload["matched_articles"] == 1
    assert {"value": "Fight", "count": 1} in payload["themes"]


def test_unknown_mode_and_operator_are_rejected(client: TestClient) -> None:
    assert client.get("/api/v2/doc/doc", params={"mode": "wordcloud"}).status_code == 400
    assert client.get("/api/v2/doc/doc", params={"query": "tone>5"}).status_code == 400
    assert client.get("/api/v2/doc/doc", params={"query": "sentiment:happy"}).status_code == 400


def test_window_parameters_filter_by_observation_time(client: TestClient) -> None:
    payload = client.get(
        "/api/v2/doc/doc",
        params={"startdatetime": "20190315001500", "enddatetime": "20190315003000"},
    ).json()
    assert article_urls(payload) == ["https://lemonde.fr/monde/article/boeing-737-max"]


def test_meta_declares_derived_fields(client: TestClient) -> None:
    payload = client.get("/api/v2/ext/meta").json()
    assert payload["field_provenance"]["title"].startswith("DERIVED")
    assert "timelinetone" in payload["doc_modes"]


def test_query_parser_rejects_dangling_or() -> None:
    with pytest.raises(QueryError):
        parse("OR brexit")


def test_quoted_operator_values_survive_tokenisation() -> None:
    groups = parse('location:"New Zealand" brexit')
    assert [term.value for group in groups for term in group.terms] == ["New Zealand", "brexit"]


def test_build_reports_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No events files"):
        build(tmp_path, tmp_path, tmp_path / "out")
