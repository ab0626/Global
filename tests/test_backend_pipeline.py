"""End-to-end backend gate on a synthetic GDELT window.

Builds raw 15-minute GDELT files (English + translated families) for a small,
fully known world, runs preprocess -> atomic -> (fake) title embeddings -> cluster ->
materialize twice (title model and legacy fallback) and exercises the API. The world:

* ``quake``   40 documents in 6 languages / 7 publisher countries, all sharing one
  GlobalEventID and 'earthquake'-like titles; two outlets carry identical wire copy;
  publisher country BR has exactly two outlets (must be onset-suppressed).
* ``fire``    15 documents about a Chile wildfire (negative control: must not merge
  with the quake).
* ``noise``   singleton documents with unrelated titles and no event IDs.
* one URL appears in both ``gkg`` and ``translation.gkg`` (must dedupe to one doc),
  one document has no PAGE_TITLE, one domain is unresolvable, one 15-minute slot is
  missing entirely, and one file contains a malformed row.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.schema import EVENT_COLUMNS, MENTION_COLUMNS
from attention import atomic, cluster, materialize, preprocess
from attention.preprocess import GKG_COLUMNS

START = datetime(2023, 2, 6, tzinfo=UTC)
SLOTS = 12  # 3 hours of 15-minute files
MISSING_SLOT = 9
QUAKE_EVENT = 1001
FIRE_EVENT = 2002

QUAKE_OUTLETS = [
    ("cnn.com", "US"),
    ("nytimes.com", "US"),
    ("foxnews.com", "US"),
    ("hurriyet.com.tr", "TR"),
    ("sabah.com.tr", "TR"),
    ("sozcu.com.tr", "TR"),
    ("spiegel.de", "DE"),
    ("zeit.de", "DE"),
    ("faz.net", "DE"),
    ("lemonde.fr", "FR"),
    ("lefigaro.fr", "FR"),
    ("liberation.fr", "FR"),
    ("nhk.or.jp", "JP"),
    ("asahi.com", "JP"),
    ("mainichi.jp", "JP"),
    ("globo.com", "BR"),
    ("folha.uol.com.br", "BR"),
    ("elpais.com", "SP"),
    ("elmundo.es", "SP"),
    ("weird.xyz", None),
]
QUAKE_TITLES = {
    "US": ("eng", "Powerful earthquake kills hundreds in Turkey and Syria"),
    "TR": ("tur", "Kahramanmaraş'ta 7,7 büyüklüğünde deprem: yüzlerce ölü"),
    "DE": ("deu", "Erdbeben in der Türkei und Syrien: Hunderte Tote"),
    "FR": ("fra", "Séisme en Turquie et en Syrie : des centaines de morts"),
    "JP": ("jpn", "トルコ・シリア地震、死者数百人に"),
    "BR": ("por", "Terremoto na Turquia e na Síria deixa centenas de mortos"),
    "SP": ("spa", "Terremoto en Turquía y Siria deja cientos de muertos"),
    None: ("eng", "Turkey Syria earthquake death toll rises"),
}
FIRE_OUTLETS = [("latercera.com", "CI"), ("emol.com", "CI"), ("bbc.co.uk", "UK"), ("cnn.com", "US")]
NOISE_TITLES = [
    "Local bakery wins regional pastry award",
    "City council debates new parking rules",
    "Quarterly earnings beat expectations for retailer",
    "Museum opens exhibit on medieval tapestries",
    "School district announces snow day policy",
    "Startup raises funding for garden robots",
]


@dataclass
class Doc:
    url: str
    domain: str
    title: str | None
    language: str
    slot: int
    event: int | None
    persons: str
    organizations: str
    themes: str
    translated: bool
    also_english: bool = False


def slot_time(slot: int) -> datetime:
    return START + timedelta(minutes=15 * slot)


def build_world() -> list[Doc]:
    docs: list[Doc] = []
    for i in range(40):
        domain, country = QUAKE_OUTLETS[i % len(QUAKE_OUTLETS)]
        language, title = QUAKE_TITLES[country]
        slot_by_country = {"TR": 0, "US": 1, "DE": 1, "FR": 2, "JP": 4, "BR": 3, "SP": 2, None: 5}
        slot = slot_by_country[country] + (i // len(QUAKE_OUTLETS)) * 2
        docs.append(
            Doc(
                url=f"https://www.{domain}/2023/02/06/turkey-syria-earthquake-{i}?utm_source=x",
                domain=domain,
                title=None if i == 39 else f"{title} ({domain})",
                language=language,
                slot=slot,
                event=QUAKE_EVENT,
                persons="Recep Tayyip Erdogan;Fuat Oktay",
                organizations="Afad;Red Crescent",
                themes="NATURAL_DISASTER_EARTHQUAKE;CRISISLEX_T03_DEAD;TAX_FNCACT_VICTIM",
                translated=language != "eng",
                also_english=(i == 3),
            )
        )
    docs[1].title = docs[0].title  # wire copy: nytimes carries cnn's text
    docs[1].domain, docs[1].url = (
        "nytimes.com",
        "https://nytimes.com/wire/turkey-syria-earthquake-1",
    )
    for i in range(15):
        domain, country = FIRE_OUTLETS[i % len(FIRE_OUTLETS)]
        spanish = country == "CI"
        docs.append(
            Doc(
                url=f"https://{domain}/noticias/chile-incendios-forestales-{i}",
                domain=domain,
                title=(
                    f"Incendios forestales en Chile dejan {20 + i} muertos"
                    if spanish
                    else f"Chile wildfires kill {20 + i} as heatwave continues"
                ),
                language="spa" if spanish else "eng",
                slot=6 + (i % 5),
                event=FIRE_EVENT,
                persons="Gabriel Boric",
                organizations="Conaf",
                themes="DISASTER_FIRE;WILDFIRE;CRISISLEX_T03_DEAD",
                translated=spanish,
            )
        )
    for i, title in enumerate(NOISE_TITLES):
        docs.append(
            Doc(
                url=f"https://noise{i}.com/story/{i}",
                domain=f"noise{i}.com",
                title=title,
                language="eng",
                slot=i,
                event=None,
                persons="",
                organizations="",
                themes="",
                translated=False,
            )
        )
    return docs


def stamp(slot: int) -> str:
    return f"{slot_time(slot):%Y%m%d%H%M%S}"


def gkg_row(doc: Doc, slot: int) -> list[str]:
    row = {c: "" for c in GKG_COLUMNS}
    row["GKGRecordID"] = (
        f"{stamp(slot)}-{'T' if doc.translated else ''}{abs(hash(doc.url)) % 10**6}"
    )
    row["Date"] = stamp(slot)
    row["SourceCollectionIdentifier"] = "1"
    row["SourceCommonName"] = doc.domain
    row["DocumentIdentifier"] = doc.url
    row["Themes"] = doc.themes
    row["V2Locations"] = "1#Turkey#TU#TU##39#35#TU" if doc.event == QUAKE_EVENT else ""
    row["Persons"] = doc.persons
    row["Organizations"] = doc.organizations
    row["V2Tone"] = "-5.2,1.1,6.3,7.4,20,0,300"
    row["TranslationInfo"] = f"srclc:{doc.language};eng:GT" if doc.translated else ""
    row["Extras"] = f"<PAGE_TITLE>{doc.title}</PAGE_TITLE>" if doc.title else ""
    return [row[c] for c in GKG_COLUMNS]


def mention_row(doc: Doc, slot: int) -> list[str]:
    row = {c: "" for c in MENTION_COLUMNS}
    row["GlobalEventID"] = str(doc.event)
    row["EventTimeDate"] = stamp(0)
    row["MentionTimeDate"] = stamp(slot)
    row["MentionType"] = "1"
    row["MentionSourceName"] = doc.domain
    row["MentionIdentifier"] = doc.url
    row["SentenceID"] = "1"
    row["Actor1CharOffset"] = "10"
    row["Actor2CharOffset"] = "20"
    row["ActionCharOffset"] = "15"
    row["InRawText"] = "1"
    row["Confidence"] = "80"
    row["MentionDocLen"] = "3000"
    row["MentionDocTone"] = "-5.2"
    row["MentionDocTranslationInfo"] = f"srclc:{doc.language};eng:GT" if doc.translated else ""
    return [row[c] for c in MENTION_COLUMNS]


def event_row(event: int, url: str) -> list[str]:
    row = {c: "" for c in EVENT_COLUMNS}
    row["GlobalEventID"] = str(event)
    row["Day"] = "20230206"
    row["MonthYear"] = "202302"
    row["Year"] = "2023"
    row["FractionDate"] = "2023.0986"
    row["Actor1Name"] = "TURKEY" if event == QUAKE_EVENT else "CHILE"
    row["IsRootEvent"] = "1"
    row["EventCode"] = "010"
    row["EventBaseCode"] = "010"
    row["EventRootCode"] = "01"
    row["QuadClass"] = "1"
    row["GoldsteinScale"] = "0.0"
    row["NumMentions"] = "10"
    row["NumSources"] = "5"
    row["NumArticles"] = "10"
    row["AvgTone"] = "-5.0"
    row["ActionGeo_Type"] = "1"
    row["ActionGeo_FullName"] = "Turkey" if event == QUAKE_EVENT else "Chile"
    row["ActionGeo_CountryCode"] = "TU" if event == QUAKE_EVENT else "CI"
    row["ActionGeo_Lat"] = "39.0" if event == QUAKE_EVENT else "-33.4"
    row["ActionGeo_Long"] = "35.0" if event == QUAKE_EVENT else "-70.6"
    row["DATEADDED"] = stamp(0)
    row["SOURCEURL"] = url
    return [row[c] for c in EVENT_COLUMNS]


def write_zip(directory: Path, name: str, rows: list[list[str]], malformed: bool = False) -> None:
    lines = ["\t".join(r) for r in rows]
    if malformed:
        lines.append("only\ttwo")
    payload = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    with zipfile.ZipFile(directory / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)


def write_raw(directory: Path, docs: list[Doc]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(SLOTS):
        if slot == MISSING_SLOT:
            continue
        here = [d for d in docs if d.slot == slot]
        english = [d for d in here if not d.translated]
        translated = [d for d in here if d.translated]
        english_gkg = [gkg_row(d, slot) for d in english]
        english_gkg += [gkg_row(d, slot) for d in translated if d.also_english]
        write_zip(directory, f"{stamp(slot)}.gkg.csv", english_gkg, malformed=(slot == 2))
        write_zip(
            directory, f"{stamp(slot)}.translation.gkg.csv", [gkg_row(d, slot) for d in translated]
        )
        write_zip(
            directory,
            f"{stamp(slot)}.mentions.CSV",
            [mention_row(d, slot) for d in english if d.event],
        )
        write_zip(
            directory,
            f"{stamp(slot)}.translation.mentions.CSV",
            [mention_row(d, slot) for d in translated if d.event],
        )
        events = {d.event: d.url for d in here if d.event}
        write_zip(
            directory, f"{stamp(slot)}.export.CSV", [event_row(e, u) for e, u in events.items()]
        )
        write_zip(directory, f"{stamp(slot)}.translation.export.CSV", [])


def write_lookup(path: Path) -> None:
    lines = [
        f"{domain}\t{country}\t{domain}"
        for domain, country in QUAKE_OUTLETS + FIRE_OUTLETS
        if country is not None and domain != "folha.uol.com.br"
    ]
    lines.append("uol.com.br\tBR\tUOL")  # parent-domain rung for folha.uol.com.br
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake_embeddings(features: Path, output: Path, seed: int = 7) -> None:
    """Deterministic stand-in for the encoder: same-story titles land near a shared
    anchor regardless of language; noise titles are far apart."""
    output.mkdir(parents=True, exist_ok=True)
    documents = pl.read_parquet(features / "documents.parquet")
    rng = np.random.default_rng(seed)
    dim = 16
    anchors = {"quake": rng.normal(size=dim), "fire": rng.normal(size=dim)}
    ids, vectors = [], []
    for doc_id, title, url in documents.select("document_id", "title", "canonical_url").iter_rows():
        if title is None:
            continue
        if "earthquake" in url:
            base = anchors["quake"]
        elif "incendios" in url:
            base = anchors["fire"]
        else:
            base = rng.normal(size=dim) * 3
        vector = base + rng.normal(scale=0.15, size=dim)
        ids.append(doc_id)
        vectors.append(vector / np.linalg.norm(vector))
    np.save(output / "title_embeddings.npy", np.asarray(vectors, dtype=np.float32))
    np.save(output / "title_embedding_ids.npy", np.asarray(ids, dtype=np.int64))
    (output / "embed.json").write_text(json.dumps({"model": "fake", "dimension": dim}))


def run_cli(module, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("world")
    docs = build_world()
    write_raw(root / "raw", docs)
    write_lookup(root / "domains.txt")
    mp = pytest.MonkeyPatch()
    try:
        run_cli(
            preprocess,
            [
                "--raw",
                str(root / "raw"),
                "--start",
                "20230206",
                "--end",
                "20230206",
                "--domain-lookup",
                str(root / "domains.txt"),
                "--output",
                str(root / "clean"),
                "--allow-partial",
            ],
            mp,
        )
        run_cli(atomic, ["--clean", str(root / "clean"), "--output", str(root / "features")], mp)
        fake_embeddings(root / "features", root / "embeddings")
        cluster.run(
            root / "features",
            root / "embeddings",
            root / "clusters",
            cluster.ClusterSettings(),
        )
        cluster.run(root / "features", None, root / "clusters_legacy", cluster.ClusterSettings())
        for name, clusters, embeddings in [
            ("store", "clusters", str(root / "embeddings")),
            ("store_legacy", "clusters_legacy", None),
        ]:
            argv = [
                "--clean",
                str(root / "clean"),
                "--features",
                str(root / "features"),
                "--clusters",
                str(root / clusters),
                "--output",
                str(root / name),
                "--min-documents",
                "5",
                "--min-effective-reports",
                "3",
                "--min-confidence",
                "0.3",
            ]
            if embeddings:
                argv += ["--embeddings", embeddings]
            run_cli(materialize, argv, mp)
    finally:
        mp.undo()
    return {
        k: root / k
        for k in (
            "raw",
            "clean",
            "features",
            "embeddings",
            "clusters",
            "clusters_legacy",
            "store",
            "store_legacy",
        )
    }


@pytest.fixture(scope="module")
def client(world: dict[str, Path]) -> Iterator[TestClient]:
    from api import attention as api_module
    from api.app import app

    mp = pytest.MonkeyPatch()
    mp.setenv("GDELT_ATTENTION_DATA", str(world["store"]))
    api_module.load.cache_clear()
    try:
        yield TestClient(app)
    finally:
        api_module.load.cache_clear()
        mp.undo()


# --- preprocessing -----------------------------------------------------------


def test_preprocess_audit_records_missing_and_malformed(world: dict[str, Path]) -> None:
    audit = json.loads((world["clean"] / "audit.json").read_text())
    missing = audit["tables"]["gkg"]["missing_files"]["gkg.csv"]
    assert f"{stamp(MISSING_SLOT)}.gkg.csv" in missing
    assert f"{stamp(MISSING_SLOT - 1)}.gkg.csv" not in missing
    assert audit["tables"]["gkg"]["malformed_rows"] == 1
    assert audit["summary"]["gkg_languages"] >= 6
    assert all(f["sha256"] for f in audit["tables"]["gkg"]["files"])


def test_duplicate_url_across_english_and_translated_gkg_dedupes(world: dict[str, Path]) -> None:
    gkg = pl.read_parquet(world["clean"] / "gkg.parquet")
    assert gkg["canonical_url"].n_unique() == gkg.height
    dup = gkg.filter(pl.col("canonical_url").str.contains("earthquake-3$"))
    assert dup.height == 1 and not dup["translated"][0]  # English record kept


def test_canonical_url_strips_tracking(world: dict[str, Path]) -> None:
    gkg = pl.read_parquet(world["clean"] / "gkg.parquet")
    assert not gkg["canonical_url"].str.contains("utm_").any()
    assert not gkg["canonical_url"].str.starts_with("https://www.").any()


def test_sources_ladder_and_unresolved_retained(world: dict[str, Path]) -> None:
    sources = pl.read_parquet(world["clean"] / "sources.parquet")
    by = {r["domain"]: r for r in sources.to_dicts()}
    assert by["cnn.com"]["mapping_method"] == "gdelt_lookup"
    assert by["folha.uol.com.br"]["mapping_method"] == "gdelt_lookup_parent"
    assert by["folha.uol.com.br"]["country_confidence"] == 0.7
    assert by["faz.net"]["mapping_method"] == "gdelt_lookup"
    assert by["weird.xyz"]["publisher_country"] is None
    assert by["weird.xyz"]["mapping_method"] == "unresolved"
    assert by["weird.xyz"]["country_confidence"] == 0.0
    assert "country" not in sources.columns


def test_zero_document_window_is_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for kind in (
        "gkg.csv",
        "translation.gkg.csv",
        "mentions.CSV",
        "translation.mentions.CSV",
        "export.CSV",
        "translation.export.CSV",
    ):
        write_zip(raw, f"{stamp(0)}.{kind}", [])
    write_lookup(tmp_path / "domains.txt")
    run_cli(
        preprocess,
        [
            "--raw",
            str(raw),
            "--start",
            "20230206",
            "--end",
            "20230206",
            "--domain-lookup",
            str(tmp_path / "domains.txt"),
            "--output",
            str(tmp_path / "clean"),
            "--allow-partial",
        ],
        monkeypatch,
    )
    audit = json.loads((tmp_path / "clean" / "audit.json").read_text())
    assert audit["summary"]["gkg_documents"] == 0
    assert pl.read_parquet(tmp_path / "clean" / "gkg.parquet").is_empty()


# --- documents ---------------------------------------------------------------


def test_documents_carry_title_language_and_publisher_country(world: dict[str, Path]) -> None:
    docs = pl.read_parquet(world["features"] / "documents.parquet")
    assert docs["document_id"].to_list() == list(range(docs.height))
    assert docs["title"].null_count() == 1  # the one title-less quake doc
    assert docs["language"].n_unique() >= 6
    assert set(docs["publisher_country"].drop_nulls().unique()) >= {
        "US",
        "TR",
        "DE",
        "FR",
        "JP",
        "BR",
    }
    unresolved = docs.filter(pl.col("domain") == "weird.xyz")
    assert unresolved.height == 2 and unresolved["publisher_country"].null_count() == 2
    assert docs["publisher_country"].null_count() == 2 + len(NOISE_TITLES)
    assert "source_country" not in docs.columns and "country" not in docs.columns


def test_wire_copies_share_a_group_but_stay_separate_nodes(world: dict[str, Path]) -> None:
    docs = pl.read_parquet(world["features"] / "documents.parquet")
    pair = docs.filter(
        pl.col("domain").is_in(["cnn.com", "nytimes.com"])
        & (pl.col("title") == docs.filter(pl.col("domain") == "nytimes.com")["title"][0])
    )
    assert pair.height >= 2
    assert pair["wire_group"].n_unique() == 1
    assert pair["document_id"].n_unique() == pair.height


# --- clustering --------------------------------------------------------------


def quake_incident(world: dict[str, Path], key: str = "clusters") -> tuple[pl.DataFrame, int]:
    docs = pl.read_parquet(world["features"] / "documents.parquet")
    members = pl.read_parquet(world[key] / "incident_memberships.parquet").filter(
        pl.col("is_primary")
    )
    joined = docs.join(members, on="document_id")
    quake = joined.filter(
        pl.col("canonical_url").str.contains("earthquake") & (pl.col("incident_id") >= 0)
    )
    incident = int(quake["incident_id"].mode()[0])
    return joined, incident


def test_quake_forms_one_cross_lingual_incident(world: dict[str, Path]) -> None:
    joined, incident = quake_incident(world)
    members = joined.filter(pl.col("incident_id") == incident)
    quake_docs = joined.filter(pl.col("canonical_url").str.contains("earthquake"))
    recall = (
        members.filter(pl.col("canonical_url").str.contains("earthquake")).height
        / quake_docs.height
    )
    purity = (
        members.filter(pl.col("canonical_url").str.contains("earthquake")).height / members.height
    )
    assert recall >= 0.9, recall
    assert purity >= 0.9, purity
    assert members["publisher_country"].drop_nulls().n_unique() >= 5
    assert members["language"].n_unique() >= 3


def test_negative_control_fire_is_a_separate_incident(world: dict[str, Path]) -> None:
    joined, incident = quake_incident(world)
    fire = joined.filter(pl.col("canonical_url").str.contains("incendios"))
    assert (fire["incident_id"] != incident).all()
    assert fire["incident_id"].drop_nulls().mode()[0] >= 0


def test_noise_documents_are_unassigned(world: dict[str, Path]) -> None:
    joined, _ = quake_incident(world)
    noise = joined.filter(pl.col("domain").str.starts_with("noise"))
    assert (noise["incident_id"] == -1).all()
    assert (noise["assignment_score"] == 0.0).all()


def test_run_json_records_provenance(world: dict[str, Path]) -> None:
    run = json.loads((world["clusters"] / "run.json").read_text())
    assert run["resolution_model"] == "title_multilingual_v1"
    assert run["settings"]["seed"] == 2026
    assert run["settings"]["candidate_max_hours"] == 48.0
    assert run["edges"] > 0 and run["unassigned_documents"] >= len(NOISE_TITLES)
    legacy = json.loads((world["clusters_legacy"] / "run.json").read_text())
    assert legacy["resolution_model"] == "legacy_metadata_v1"


def test_legacy_fallback_still_finds_the_quake(world: dict[str, Path]) -> None:
    joined, incident = quake_incident(world, "clusters_legacy")
    members = joined.filter(pl.col("incident_id") == incident)
    quake_share = members.filter(pl.col("canonical_url").str.contains("earthquake")).height
    assert quake_share / members.height >= 0.9
    assert quake_share >= 20


def test_clustering_is_deterministic(world: dict[str, Path], tmp_path: Path) -> None:
    first = pl.read_parquet(world["clusters"] / "incident_memberships.parquet")
    cluster.run(
        world["features"], world["embeddings"], tmp_path / "again", cluster.ClusterSettings()
    )
    second = pl.read_parquet(tmp_path / "again" / "incident_memberships.parquet")
    assert first.equals(second)


def test_title_permutation_destroys_structure(world: dict[str, Path], tmp_path: Path) -> None:
    """Shuffling which title vector belongs to which document must break the
    single cross-lingual incident (title-only signal cannot survive it)."""
    vectors = np.load(world["embeddings"] / "title_embeddings.npy")
    ids = np.load(world["embeddings"] / "title_embedding_ids.npy")
    rng = np.random.default_rng(0)
    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    np.save(shuffled / "title_embeddings.npy", vectors[rng.permutation(len(vectors))])
    np.save(shuffled / "title_embedding_ids.npy", ids)
    (shuffled / "embed.json").write_text(json.dumps({"model": "shuffled"}))
    settings = cluster.ClusterSettings(channels="title")
    docs = pl.read_parquet(world["features"] / "documents.parquet")

    cluster.run(world["features"], world["embeddings"], tmp_path / "base", settings)
    cluster.run(world["features"], shuffled, tmp_path / "broken", settings)

    def purity(output: Path) -> float:
        members = pl.read_parquet(output / "incident_memberships.parquet").filter(
            pl.col("is_primary") & (pl.col("incident_id") >= 0)
        )
        largest = members.filter(pl.col("incident_id") == members["incident_id"].mode()[0])
        joined = largest.join(docs, on="document_id")
        return joined["canonical_url"].str.contains("earthquake").sum() / joined.height

    assert purity(tmp_path / "base") >= 0.9
    assert purity(tmp_path / "broken") < 0.8


def test_far_apart_pairs_are_dropped_by_temporal_gate(
    world: dict[str, Path], tmp_path: Path
) -> None:
    settings = cluster.ClusterSettings(candidate_max_hours=0.1)
    audit = cluster.run(world["features"], world["embeddings"], tmp_path / "tight", settings)
    pairs = pl.read_parquet(tmp_path / "tight" / "pair_features.parquet")
    assert (pairs["delta_hours"] <= 0.1).all()
    assert audit["pairs_in_window"] < audit["candidate_pairs"]


# --- materialisation ---------------------------------------------------------


def quake_event(store: Path) -> dict:
    events = pl.read_parquet(store / "macro_events.parquet")
    quake = events.filter(pl.col("event_country") == "TU")
    assert quake.height == 1, events.select("macro_event_id", "label", "event_country")
    return quake.to_dicts()[0]


def test_macro_event_measures_and_types(world: dict[str, Path]) -> None:
    quake = quake_event(world["store"])
    assert quake["raw_documents"] >= 36
    assert quake["unique_domains"] >= 15
    assert quake["effective_reports"] < quake["raw_documents"]  # wire copy collapsed
    assert quake["publisher_country_count"] >= 6
    assert quake["language_count"] >= 6
    assert "natural_disaster" in quake["event_types"]
    assert quake["resolution_model"] == "title_multilingual_v1"
    assert quake["title"]
    assert 0 < quake["cluster_confidence"] <= 1


def test_legacy_store_has_no_title_coherence(world: dict[str, Path]) -> None:
    quake = quake_event(world["store_legacy"])
    assert quake["resolution_model"] == "legacy_metadata_v1"
    assert quake["title_coherence"] is None
    assert quake["cluster_confidence"] == quake["entity_coherence"]


def test_country_summary_onset_ordering_and_suppression(world: dict[str, Path]) -> None:
    quake = quake_event(world["store"])
    summary = pl.read_parquet(world["store"] / "country_event_summary.parquet").filter(
        pl.col("macro_event_id") == quake["macro_event_id"]
    )
    rows = {r["publisher_country"]: r for r in summary.to_dicts()}
    assert "country" not in summary.columns
    assert rows["BR"]["unique_domains"] == 2 and rows["BR"]["suppressed"]
    assert rows["BR"]["onset"] is None and rows["BR"]["lag_hours"] is None
    assert not rows["US"]["suppressed"] and rows["US"]["onset"] is not None
    assert rows["TR"]["first_seen"] <= rows["US"]["first_seen"] <= rows["JP"]["first_seen"]
    assert rows["TR"]["onset"] <= rows["JP"]["onset"]
    assert rows["US"]["effective_share"] <= 1 and rows["US"]["raw_share"] <= 1
    assert rows["US"]["attention_ratio"] > 0
    assert "weird.xyz" not in rows and None not in rows


def test_unresolved_and_low_confidence_documents_excluded_from_attention(
    world: dict[str, Path],
) -> None:
    meta = json.loads((world["store"] / "meta.json").read_text())
    docs = pl.read_parquet(world["store"] / "macro_event_documents.parquet")
    assert meta["documents_total"] == docs.filter(pl.col("is_primary")).height
    assert meta["excluded_document_share"] > 0
    assert docs.filter(pl.col("publisher_country").is_null()).height == 2 + len(NOISE_TITLES)
    attention = pl.read_parquet(world["store"] / "country_event_attention.parquet")
    assert attention["publisher_country"].null_count() == 0
    assert meta["filters"]["min_country_confidence"] == 0.5
    assert "publisher_country" in meta and "timing" in meta


def test_memberships_keep_unassigned_documents(world: dict[str, Path]) -> None:
    docs = pl.read_parquet(world["store"] / "macro_event_documents.parquet")
    unassigned = docs.filter(pl.col("incident_id") == -1)
    assert unassigned.height >= len(NOISE_TITLES)
    assert (unassigned["macro_event_id"] == -1).all()
    assert (unassigned["family_id"] == -1).all()


# --- API ---------------------------------------------------------------------


def test_search_is_diacritic_and_cjk_tolerant(client: TestClient) -> None:
    for q in ["earthquake", "Séisme", "seisme", "地震", "Turquie", "ERDBEBEN"]:
        body = client.get("/api/v2/attention/search", params={"q": q}).json()
        assert body["total"] >= 1, q
        assert body["events"][0]["event_country"] == "TU", q
        assert body["meta"]["resolution_model"] == "title_multilingual_v1"
    assert client.get("/api/v2/attention/search", params={"q": "zzz-nothing"}).json()["total"] == 0


def test_events_list_and_detail_have_no_bare_country(client: TestClient) -> None:
    body = client.get("/api/v2/attention/events").json()
    assert body["total"] >= 2
    for event in body["events"]:
        assert "country" not in event and "publisher_country_count" in event
    detail = client.get(f"/api/v2/attention/events/{body['events'][0]['macro_event_id']}").json()
    assert detail["event"]["incident_id"] >= 0 and detail["family"] is not None
    assert detail["sample_documents"] and "publisher_country" in detail["sample_documents"][0]
    assert client.get("/api/v2/attention/events/9999").status_code == 404


def test_spread_pagination_is_stable_and_chronological(client: TestClient) -> None:
    quake = client.get("/api/v2/attention/search", params={"q": "earthquake"}).json()["events"][0]
    event_id = quake["macro_event_id"]
    full = client.get(f"/api/v2/attention/events/{event_id}/spread", params={"limit": 5000}).json()
    assert full["total"] == len(full["documents"]) >= 30
    times = [d["observed_time"] for d in full["documents"]]
    assert times == sorted(times)
    assert full["countries"][0]["publisher_country"] == "TR"
    paged: list[dict] = []
    for offset in range(0, full["total"], 7):
        page = client.get(
            f"/api/v2/attention/events/{event_id}/spread", params={"limit": 7, "offset": offset}
        ).json()
        paged.extend(page["documents"])
    assert [d["document_id"] for d in paged] == [d["document_id"] for d in full["documents"]]
    strict = client.get(
        f"/api/v2/attention/events/{event_id}/spread", params={"min_country_confidence": 0.99}
    ).json()
    assert strict["total"] == 0 and strict["documents"] == [] and strict["excluded_documents"] > 0
    assert strict["meta"]["denominators"]


def test_timeline_and_countries_endpoints(client: TestClient) -> None:
    event_id = client.get("/api/v2/attention/search", params={"q": "earthquake"}).json()["events"][
        0
    ]["macro_event_id"]
    timeline = client.get(f"/api/v2/attention/events/{event_id}/timeline").json()
    assert timeline["world"] and timeline["publisher_countries"]
    assert timeline["publisher_countries"][0]["values"]
    filtered = client.get(
        f"/api/v2/attention/events/{event_id}/timeline",
        params={"publisher_countries": "us,de", "value": "effective_reports"},
    ).json()
    assert {c["publisher_country"] for c in filtered["publisher_countries"]} == {"US", "DE"}
    countries = client.get(f"/api/v2/attention/events/{event_id}/countries").json()
    codes = {c["publisher_country"]: c for c in countries["publisher_countries"]}
    assert codes["BR"]["suppressed"] and codes["BR"]["onset"] is None
    assert countries["world_onset"] is not None
    visible = client.get(
        f"/api/v2/attention/events/{event_id}/countries", params={"include_suppressed": False}
    ).json()
    assert all(not c["suppressed"] for c in visible["publisher_countries"])


def test_type_and_country_rollups(client: TestClient) -> None:
    types = client.get("/api/v2/attention/event-types").json()["types"]
    assert any(t["type"] == "natural_disaster" for t in types)
    rollup = client.get("/api/v2/attention/event-types/natural_disaster/countries").json()
    assert rollup["publisher_countries"] and "publisher_country" in rollup["publisher_countries"][0]
    assert client.get("/api/v2/attention/event-types/nope/countries").status_code == 404
    countries = client.get("/api/v2/attention/countries").json()["publisher_countries"]
    assert countries and "country_effective_reports" in countries[0]


def test_openapi_lists_new_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for route in ("/search", "/events/{event_id}/spread", "/families/{family_id}"):
        assert f"/api/v2/attention{route}" in paths
