"""Metadata-only GDELT story clustering; see readme.md for data acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

import igraph as ig
import matplotlib
import numpy as np
import polars as pl
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MENTION_COLUMNS = [
    "GlobalEventID",
    "EventTimeDate",
    "MentionTimeDate",
    "MentionType",
    "MentionSourceName",
    "MentionIdentifier",
    "SentenceID",
    "Actor1CharOffset",
    "Actor2CharOffset",
    "ActionCharOffset",
    "InRawText",
    "Confidence",
    "MentionDocLen",
    "MentionDocTone",
    "MentionDocTranslationInfo",
    "Extras",
]
INTEGER_COLUMNS = [
    "GlobalEventID",
    "MentionType",
    "SentenceID",
    "Actor1CharOffset",
    "Actor2CharOffset",
    "ActionCharOffset",
    "InRawText",
    "Confidence",
    "MentionDocLen",
]
FINGERPRINT = [
    "GlobalEventID",
    "SentenceID",
    "Actor1CharOffset",
    "Actor2CharOffset",
    "ActionCharOffset",
    "MentionDocLen",
]
EVENT_LABELS = ["GlobalEventID", "Actor1Name", "Actor2Name", "EventCode", "ActionGeo_FullName"]
EVENT_COLUMNS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    *[
        f"{actor}{suffix}"
        for actor in ["Actor1", "Actor2"]
        for suffix in [
            "Code",
            "Name",
            "CountryCode",
            "KnownGroupCode",
            "EthnicCode",
            "Religion1Code",
            "Religion2Code",
            "Type1Code",
            "Type2Code",
            "Type3Code",
        ]
    ],
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    *[
        f"{geo}_{suffix}"
        for geo in ["Actor1Geo", "Actor2Geo", "ActionGeo"]
        for suffix in [
            "Type",
            "FullName",
            "CountryCode",
            "ADM1Code",
            "ADM2Code",
            "Lat",
            "Long",
            "FeatureID",
        ]
    ],
    "DATEADDED",
    "SOURCEURL",
]
STORIES = {
    "crash_grounding": r"ethiopian|boeing|737",
    "christchurch": r"christchurch|mosque",
    "idai": r"idai|cyclone|mozambique",
    "brexit": r"brexit",
}
STOPWORDS = set(
    "the and for that with from this are was has have after into over amid says "
    "news world article articles story stories html htm com www index amp live "
    "a an of to in on at by as is it be its us uk".split()
)


@dataclass(frozen=True)
class Settings:
    confidence: int = 40
    max_article_events: int = 25
    recurring_articles: int = 1000
    recurring_days: float = 2
    event_gap_days: float = 3
    min_fingerprint_events: int = 2
    min_shared_articles: int = 2
    max_pair_contributions: int = 50_000_000
    seed: int = 2026


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def max_count(series: pl.Series) -> int:
    value = series.max()
    if value is None:
        return 0
    if not isinstance(value, int):
        raise TypeError(f"Expected integer counts in {series.name}")
    return value


def input_files(
    directory: Path, suffix: str, start: datetime, end: datetime, allow_partial: bool
) -> tuple[list[Path], list[str]]:
    expected = []
    stamp = start
    while stamp < end + timedelta(days=1):
        expected.append(f"{stamp:%Y%m%d%H%M%S}.{suffix}.csv")
        stamp += timedelta(minutes=15)
    paths = [directory / name for name in expected if (directory / name).is_file()]
    missing = sorted(set(expected) - {p.name for p in paths})
    if not paths:
        raise ValueError(f"No {suffix} files for requested dates in {directory}")
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} missing {suffix} files; use --allow-partial explicitly")
    return paths, missing


def read_table(paths: list[Path], columns: list[str]) -> tuple[pl.DataFrame, dict]:
    frames = []
    manifest = []
    for path in paths:
        digest = hashlib.sha256()
        replacements = 0
        rows = 0
        with path.open("rb") as stream:
            for rows, line in enumerate(stream, 1):
                digest.update(line)
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                replacements += text.count("\ufffd")
                if len(text.split("\t")) != len(columns):
                    raise ValueError(f"{path}:{rows}: expected {len(columns)} tab-separated fields")
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": digest.hexdigest(),
                "replacement_characters": replacements,
            }
        )
        if rows:
            frames.append(
                pl.read_csv(
                    path,
                    separator="\t",
                    has_header=False,
                    quote_char=None,
                    schema={c: pl.String for c in columns},
                    encoding="utf8-lossy",
                )
            )
    if not frames:
        raise ValueError("Input files contain no records")
    data = pl.concat(frames)
    null_counts = data.null_count().row(0, named=True)
    return data, {
        "files": manifest,
        "rows": data.height,
        "column_order": columns,
        "null_counts": null_counts,
        "null_rates": {c: n / data.height for c, n in null_counts.items()},
    }


def read_mentions(paths: list[Path]) -> tuple[pl.DataFrame, dict]:
    data, audit = read_table(paths, MENTION_COLUMNS)
    data = data.with_columns(
        pl.col(INTEGER_COLUMNS).cast(pl.Int64),
        pl.col("MentionDocTone").cast(pl.Float64),
        pl.col("EventTimeDate", "MentionTimeDate")
        .str.to_datetime("%Y%m%d%H%M%S", strict=True)
        .dt.replace_time_zone("UTC"),
    )
    invalid = data.filter(
        pl.any_horizontal(
            pl.col(
                [
                    "GlobalEventID",
                    "EventTimeDate",
                    "MentionTimeDate",
                    "MentionType",
                    "Confidence",
                ]
            ).is_null()
        )
        | ~pl.col("Confidence").is_between(10, 100)
        | (pl.col("GlobalEventID") <= 0)
    )
    if invalid.height:
        raise ValueError(
            f"{invalid.height} rows have missing/invalid required IDs, dates or confidence"
        )
    translated = pl.col("MentionDocTranslationInfo").fill_null("").str.strip_chars() != ""
    audit.update(
        {
            "mention_types": data.group_by("MentionType").len().sort("MentionType").to_dicts(),
            "translated_rows": data.filter(translated).height,
            "translated_identifiers": data.filter(translated)["MentionIdentifier"].n_unique(),
            "source_languages": data.filter(translated)
            .select(
                pl.col("MentionDocTranslationInfo")
                .str.extract(r"(?i)srclc:([^; ]+)", 1)
                .alias("language")
            )
            .group_by("language")
            .len()
            .sort("len", descending=True)
            .to_dicts(),
            "mention_time_min": str(data["MentionTimeDate"].min()),
            "mention_time_max": str(data["MentionTimeDate"].max()),
            "event_time_min": str(data["EventTimeDate"].min()),
            "event_time_max": str(data["EventTimeDate"].max()),
            "inconsistent_event_timestamps": data.group_by("GlobalEventID")
            .agg(pl.col("EventTimeDate").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height,
            "confidence": data["Confidence"].describe().to_dicts(),
        }
    )
    return data, audit


def hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def slug(url: str) -> str:
    try:
        return unquote(urlsplit(url).path).lower()
    except ValueError:
        return ""


def prepare_articles(
    data: pl.DataFrame, settings: Settings
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    web = data.filter(
        (pl.col("MentionType") == 1) & pl.col("MentionIdentifier").str.contains(r"(?i)^https?://")
    )
    articles = (
        web.group_by("MentionIdentifier")
        .agg(
            pl.struct(FINGERPRINT).unique().sort().alias("fingerprint"),
            pl.any_horizontal(pl.col(FINGERPRINT).is_null()).any().alias("incomplete_fingerprint"),
            pl.col("GlobalEventID").n_unique().alias("original_degree"),
            pl.col("MentionTimeDate").min().alias("first_seen"),
            pl.col("MentionTimeDate").max().alias("last_seen"),
            pl.struct(["MentionSourceName", "MentionTimeDate", "MentionDocTranslationInfo"])
            .unique()
            .sort()
            .alias("observations"),
        )
        .sort("MentionIdentifier")
        .with_row_index("article_id")
        .with_columns(
            pl.col("MentionIdentifier")
            .map_elements(hostname, return_dtype=pl.String)
            .alias("domain"),
            pl.col("MentionIdentifier").map_elements(slug, return_dtype=pl.String).alias("slug"),
        )
        .with_columns(
            pl.when(
                (pl.col("original_degree") >= settings.min_fingerprint_events)
                & ~pl.col("incomplete_fingerprint")
            )
            .then(pl.lit(""))
            .otherwise(pl.col("MentionIdentifier"))
            .alias("salt")
        )
    )
    if not articles.height:
        raise ValueError("No WEB articles with HTTP(S) URLs")
    groups = (
        articles.group_by("fingerprint", "salt")
        .agg(pl.col("article_id").sort())
        .sort(pl.col("article_id").list.first())
        .with_row_index("canonical_id")
    )
    mapping = groups.select("article_id", "canonical_id").explode("article_id")
    articles = (
        articles.join(mapping, on="article_id")
        .drop("fingerprint", "salt", "incomplete_fingerprint")
        .sort("article_id")
    )
    for name, pattern in STORIES.items():
        articles = articles.with_columns(pl.col("slug").str.contains(pattern).alias(name))
    return (
        web,
        articles,
        {
            "web_mention_rows": web.height,
            "excluded_non_web_or_non_url_rows": data.height - web.height,
            "distinct_urls": articles.height,
            "canonical_articles": groups.height,
            "urls_collapsed": articles.height - groups.height,
            "multi_url_groups": groups.filter(pl.col("article_id").list.len() > 1).height,
            "unparseable_domains": articles.filter(pl.col("domain") == "").height,
            "translated_web_urls": web.filter(
                pl.col("MentionDocTranslationInfo").fill_null("").str.strip_chars() != ""
            )["MentionIdentifier"].n_unique(),
            "url_event_pairs": web.select("MentionIdentifier", "GlobalEventID").unique().height,
        },
    )


def project(
    web: pl.DataFrame, articles: pl.DataFrame, settings: Settings
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    eligible = articles.filter(pl.col("original_degree") <= settings.max_article_events)
    edges = web.filter(pl.col("Confidence") >= settings.confidence).join(
        eligible.select("MentionIdentifier", "canonical_id"), on="MentionIdentifier"
    )
    recurring = (
        edges.group_by("GlobalEventID")
        .agg(
            pl.col("canonical_id").n_unique().alias("n"),
            (pl.col("MentionTimeDate").max() - pl.col("MentionTimeDate").min())
            .dt.total_seconds()
            .alias("span"),
        )
        .filter(
            (pl.col("n") >= settings.recurring_articles)
            & (pl.col("span") >= settings.recurring_days * 86400)
        )
    )
    edges = edges.join(recurring.select("GlobalEventID"), on="GlobalEventID", how="anti")
    events = (
        edges.group_by("GlobalEventID")
        .agg(pl.col("EventTimeDate").min().dt.epoch("s").alias("event_time"))
        .sort("GlobalEventID")
        .with_row_index("vertex")
    )
    incidence = (
        edges.select("canonical_id", "GlobalEventID")
        .unique()
        .join(events.select("GlobalEventID", "vertex"), on="GlobalEventID")
        .sort("canonical_id", "vertex")
    )
    degrees = incidence.group_by("canonical_id").len()["len"].cast(pl.Int64)
    contributions = int((degrees * (degrees - 1) // 2).sum())
    if contributions > settings.max_pair_contributions:
        raise ValueError(
            f"{contributions:,} pair contributions exceed --max-pair-contributions; "
            "start with March 15–16 or explicitly raise the memory guard"
        )
    matrix = sparse.csr_matrix(
        (
            np.ones(incidence.height, dtype=np.int64),
            (incidence["canonical_id"].to_numpy(), incidence["vertex"].to_numpy()),
        ),
        shape=(articles["canonical_id"].n_unique(), events.height),
        dtype=np.int64,
    )
    common = sparse.triu(matrix.T @ matrix, k=1, format="coo")
    times = events["event_time"].to_numpy()
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    mask = (common.data >= settings.min_shared_articles) & (
        np.abs(times[common.row] - times[common.col]) <= settings.event_gap_days * 86400
    )
    left, right, weights = common.row[mask], common.col[mask], common.data[mask]
    pairs = pl.DataFrame(
        {
            "left": left,
            "right": right,
            "count": weights,
            "jaccard": weights / (counts[left] + counts[right] - weights),
        }
    )
    return (
        events,
        incidence,
        pairs,
        {
            "roundup_urls_dropped": articles.height - eligible.height,
            "low_confidence_rows": web.filter(pl.col("Confidence") < settings.confidence).height,
            "recurring_events_dropped": recurring.height,
            "retained_events": events.height,
            "retained_canonical_event_edges": incidence.height,
            "retained_canonical_articles": incidence["canonical_id"].n_unique(),
            "pair_contributions": contributions,
            "projected_pairs": len(common.data),
            "pairs_after_time_and_min_count_filters": pairs.height,
        },
    )


def communities(
    events: pl.DataFrame,
    pairs: pl.DataFrame,
    metric: str,
    threshold: float,
    method: str,
    seed: int,
    resolution: float = 1.0,
) -> tuple[pl.DataFrame, int]:
    if method not in {"components", "louvain", "leiden"}:
        raise ValueError(f"Unknown community method: {method}")
    selected = pairs.filter(pl.col(metric) >= threshold)
    graph = ig.Graph(n=events.height, edges=selected.select("left", "right").iter_rows())
    ig.set_random_number_generator(random.Random(seed))
    if method == "components" or not graph.ecount():
        membership = graph.connected_components().membership
    elif method == "louvain":
        membership = graph.community_multilevel(
            weights=selected[metric].to_list(), resolution=resolution
        ).membership
    else:
        membership = graph.community_leiden(
            weights=selected[metric].to_list(),
            objective_function="modularity",
            resolution=resolution,
            n_iterations=-1,
        ).membership
    result = events.with_columns(pl.Series("cluster", membership, dtype=pl.Int64))
    return result, graph.ecount()


def article_memberships(
    incidence: pl.DataFrame, membership: pl.DataFrame, articles: pl.DataFrame
) -> pl.DataFrame:
    return (
        incidence.join(membership.select("GlobalEventID", "cluster"), on="GlobalEventID")
        .select("canonical_id", "cluster")
        .unique()
        .join(articles, on="canonical_id")
        .sort("cluster", "article_id")
    )


def cluster_sizes(members: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    sizes = members.group_by("cluster").agg(
        pl.len().alias("articles"),
        pl.col("canonical_id").n_unique().alias("canonical_articles"),
        pl.col("domain").filter(pl.col("domain") != "").n_unique().alias("domains"),
        pl.col("first_seen").min().alias("first_seen"),
        pl.col("last_seen").max().alias("last_seen"),
    )
    return sizes.join(
        membership.group_by("cluster").len().rename({"len": "events"}), on="cluster"
    ).sort(["articles", "cluster"], descending=[True, False])


def keyword_validation(
    articles: pl.DataFrame, members: pl.DataFrame, sizes: pl.DataFrame
) -> tuple[list[dict], pl.DataFrame]:
    results = []
    distributions = []
    for story in STORIES:
        matches = articles.filter(pl.col(story))
        hits = members.filter(pl.col(story)).group_by("cluster").len().rename({"len": "matches"})
        distribution = (
            hits.join(sizes.select("cluster", "articles"), on="cluster")
            .sort(["matches", "cluster"], descending=[True, False])
            .with_columns(
                pl.lit(story).alias("story"),
                pl.lit(matches.height).alias("all_keyword_urls"),
                (
                    pl.col("matches") / matches.height
                    if matches.height
                    else pl.lit(None, dtype=pl.Float64)
                ).alias("fraction_of_keyword_urls"),
                (pl.col("matches") / pl.col("articles")).alias("keyword_purity"),
            )
        )
        distributions.append(distribution)
        best = distribution.row(0, named=True) if distribution.height else None
        assigned = members.filter(pl.col(story))["article_id"].n_unique()
        best_hits = best["matches"] if best else 0
        results.append(
            {
                "story": story,
                "regex_on_decoded_url_path": STORIES[story],
                "keyword_urls": matches.height,
                "assigned_keyword_urls": assigned,
                "unassigned_keyword_urls": matches.height - assigned,
                "best_cluster": best["cluster"] if best else None,
                "best_matches": best_hits,
                "best_cluster_articles": best["articles"] if best else 0,
                "best_recall": best_hits / matches.height if matches.height else None,
                "best_keyword_purity": best_hits / best["articles"] if best else None,
                "elsewhere_not_in_best": assigned - best_hits,
                "clusters_with_matches": distribution.height,
            }
        )
    return results, pl.concat(distributions)


def tokens(urls: list[str]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for url in urls:
        counts.update(
            sorted(
                {
                    t
                    for t in re.findall(r"[^\W\d_]{3,}|737", slug(url), flags=re.UNICODE)
                    if t not in STOPWORDS
                }
            )
        )
    return counts.most_common(15)


def label_counts(labels: pl.DataFrame, ids: pl.Series, column: str) -> list[tuple[str, int]]:
    values = labels.filter(pl.col("GlobalEventID").is_in(ids.implode()))[column].drop_nulls()
    return Counter(v for v in values.to_list() if v).most_common(10)


def sample_urls(frame: pl.DataFrame, seed: int) -> list[str]:
    return random.Random(seed).sample(
        frame["MentionIdentifier"].sort().to_list(), min(10, frame.height)
    )


def cluster_report(
    sizes: pl.DataFrame,
    members: pl.DataFrame,
    membership: pl.DataFrame,
    labels: pl.DataFrame,
    validation: list[dict],
    seed: int,
) -> str:
    lines = ["# Cluster inspection", "", "Manual cohesion judgments are NOT automated.", ""]
    best_ids = {r["best_cluster"] for r in validation if r["best_cluster"] is not None}
    inspect_ids = list(dict.fromkeys(sizes.head(20)["cluster"].to_list() + sorted(best_ids)))
    for cid in inspect_ids:
        info = sizes.filter(pl.col("cluster") == cid).row(0, named=True)
        rows = members.filter(pl.col("cluster") == cid)
        ids = membership.filter(pl.col("cluster") == cid)["GlobalEventID"]
        lines.extend(
            [
                f"## Cluster {cid}",
                f"{info['events']} events; {info['articles']} URLs; "
                f"{info['canonical_articles']} canonical articles; {info['domains']} hostnames.",
                f"Observed GDELT span: {info['first_seen']} to {info['last_seen']}.",
                "Top URL-path tokens (document frequency): "
                f"{tokens(rows['MentionIdentifier'].to_list())}",
            ]
        )
        found = labels.filter(pl.col("GlobalEventID").is_in(ids.implode())).height
        lines.append(f"Event labels available: {found}/{len(ids)}.")
        for column in EVENT_LABELS:
            if column != "GlobalEventID":
                lines.append(f"{column}: {label_counts(labels, ids, column)}")
        lines.extend(
            ["", "Seeded random URLs:", *[f"- {u}" for u in sample_urls(rows, seed + cid)]]
        )
        for result in validation:
            if result["best_cluster"] == cid:
                nonmatching = rows.filter(~pl.col(result["story"]))
                lines.extend(
                    [
                        "",
                        f"Nonmatching URLs for {result['story']} ({nonmatching.height} total):",
                        *[f"- {u}" for u in sample_urls(nonmatching, seed + cid)],
                    ]
                )
        lines.extend(["", "Human judgment: PENDING (one story / mixed / unclear; rationale).", ""])
    return "\n".join(lines)


def evaluate(
    output: Path,
    name: str,
    articles: pl.DataFrame,
    events: pl.DataFrame,
    incidence: pl.DataFrame,
    pairs: pl.DataFrame,
    labels: pl.DataFrame,
    metric: str,
    threshold: float,
    method: str,
    seed: int,
) -> tuple[dict, list[dict], pl.DataFrame]:
    membership, edge_count = communities(events, pairs, metric, threshold, method, seed)
    members = article_memberships(incidence, membership, articles)
    return save_evaluation(
        output,
        name,
        articles,
        membership,
        members,
        labels,
        metric,
        threshold,
        method,
        seed,
        edge_count,
    )


def save_evaluation(
    output: Path,
    name: str,
    articles: pl.DataFrame,
    membership: pl.DataFrame,
    members: pl.DataFrame,
    labels: pl.DataFrame,
    metric: str,
    threshold: float,
    method: str,
    seed: int,
    edge_count: int,
) -> tuple[dict, list[dict], pl.DataFrame]:
    destination = output / name
    destination.mkdir()
    sizes = cluster_sizes(members, membership)
    validation, distribution = keyword_validation(articles, members, sizes)
    assigned = members["article_id"].n_unique()
    overlap = members.group_by("article_id").len().filter(pl.col("len") > 1).height
    best_ids = [r["best_cluster"] for r in validation]
    valid_ids = [cid for cid in best_ids if cid is not None]
    category_counts = distribution.group_by("cluster").agg(pl.col("story").n_unique().alias("n"))
    summary = {
        "run": name,
        "metric": metric,
        "threshold": threshold,
        "method": method,
        "graph_edges": edge_count,
        "clusters": sizes.height,
        "assigned_urls": assigned,
        "unassigned_urls": articles.height - assigned,
        "multiply_assigned_urls": overlap,
        "largest_cluster_urls": max_count(sizes["articles"]),
        "largest_fraction_all_urls": max_count(sizes["articles"]) / articles.height,
        "largest_fraction_assigned_urls": max_count(sizes["articles"]) / assigned
        if assigned
        else 0,
        "largest_fraction_canonical_articles": (
            max_count(sizes["canonical_articles"]) / articles["canonical_id"].n_unique()
        ),
        "best_clusters_all_distinct": len(valid_ids) == 4 and len(set(valid_ids)) == 4,
        "clusters_matching_all_four_keyword_sets": category_counts.filter(pl.col("n") == 4).height,
        "single_event_clusters": sizes.filter(pl.col("events") == 1).height,
        "single_canonical_article_clusters": sizes.filter(pl.col("canonical_articles") == 1).height,
        "singleton_only_assigned_urls": assigned
        - members.join(
            sizes.filter(pl.col("canonical_articles") > 1).select("cluster"), on="cluster"
        )["article_id"].n_unique(),
        "canonical_article_count_quantiles": {
            str(q): sizes["canonical_articles"].quantile(q)
            for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
        },
        "article_count_quantiles": {
            str(q): sizes["articles"].quantile(q) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
        },
    }
    membership.write_parquet(destination / "event_clusters.parquet")
    members.select("article_id", "canonical_id", "cluster").write_parquet(
        destination / "article_clusters.parquet"
    )
    sizes.write_csv(destination / "cluster_sizes.csv")
    distribution.write_csv(destination / "keyword_distribution.csv")
    write_json(destination / "metrics.json", {"summary": summary, "expected_stories": validation})
    (destination / "clusters.md").write_text(
        cluster_report(sizes, members, membership, labels, validation, seed), encoding="utf-8"
    )
    return summary, validation, sizes


def read_labels(paths: list[Path]) -> tuple[pl.DataFrame, dict]:
    if not paths:
        schema = pl.Schema({name: pl.String for name in EVENT_LABELS})
        schema["GlobalEventID"] = pl.Int64
        return pl.DataFrame(schema=schema), {"provided": False}
    data, audit = read_table(paths, EVENT_COLUMNS)
    labels = data.select(EVENT_LABELS).with_columns(pl.col("GlobalEventID").cast(pl.Int64))
    audit["duplicate_event_ids"] = labels.height - labels["GlobalEventID"].n_unique()
    return labels.unique("GlobalEventID", keep="first", maintain_order=True), audit


def render_summary(
    output: Path, summaries: list[dict], validation: list[dict], audit: dict
) -> None:
    lines = [
        "# GDELT metadata clustering validation",
        "",
        f"Requested dates: {audit['start_date']}–{audit['end_date_inclusive']} inclusive "
        f"({audit['requested_days']} days). **Partial input: {audit['partial']}.**",
        f"Loaded {len(audit['mentions']['files'])} Mentions files / "
        f"{audit['mentions']['rows']:,} rows; "
        f"{len(audit['missing_mention_files'])} expected slots missing.",
        f"Observed mentions: {audit['mentions']['mention_time_min']} to "
        f"{audit['mentions']['mention_time_max']}.",
        f"Distinct URLs: {audit['deduplication']['distinct_urls']:,}; "
        f"wire candidates collapsed: {audit['deduplication']['urls_collapsed']:,}.",
        f"Nonempty translation metadata: {audit['mentions']['translated_rows']:,} rows.",
        "",
        "This is an automated measurement report, not a manual cohesion verdict.",
        "See audit.json for scope, missing files, null rates, translation provenance and hashes.",
        "All URL counts expand deduplicated wire candidates back to their observed copies.",
        "Articles may belong to multiple clusters; distribution fractions need not sum to one.",
        "Keyword recall is against URL-path matches only, not against all true story coverage.",
        "Keyword purity is noisy: missing words reduce it; false-positive words inflate it.",
        "Blank translation metadata is not proof of English; human translation is unmarked.",
        "GDELT timestamps measure observation, not publication or real event occurrence.",
        "",
        "## Threshold sweep",
        "",
        "| Run | Clusters | Largest / all URLs | Assigned URLs | Multi-cluster URLs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['clusters']} | {row['largest_fraction_all_urls']:.2%} "
            f"| {row['assigned_urls']} | {row['multiply_assigned_urls']} |"
        )
    lines.extend(
        [
            "",
            "## Expected-story URL proxies",
            "",
            "| Run | Story | Keyword URLs | Best cluster | Recall | Keyword purity | Unassigned |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation:
        recall = f"{row['best_recall']:.2%}" if row["best_recall"] is not None else "N/A"
        purity = (
            f"{row['best_keyword_purity']:.2%}" if row["best_keyword_purity"] is not None else "N/A"
        )
        lines.append(
            f"| {row['run']} | {row['story']} | {row['keyword_urls']} | {row['best_cluster']} "
            f"| {recall} | {purity} | {row['unassigned_keyword_urls']} |"
        )
    lines.extend(
        [
            "",
            "## Required manual follow-up",
            "",
            "Read clusters.md: top 20 plus expected-story winners and nonmatching samples.",
            "keyword_distribution.csv gives the full, overlapping distribution beyond the winner.",
            "Check metrics.json for separation and size quantiles; inspect mixed keyword clusters.",
            "Record human judgments before declaring go/no-go. No automatic story-quality verdict.",
            "",
            "![Cluster size survival distribution](cluster_sizes.svg)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--start-date", default="20190310")
    parser.add_argument("--end-date", default="20190317", help="Inclusive UTC day")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["components", "louvain", "leiden"],
        default=["components", "louvain"],
    )
    defaults = Settings()
    for field in asdict(defaults):
        value = asdict(defaults)[field]
        parser.add_argument(
            "--" + field.replace("_", "-"),
            type=float if field.endswith("_days") else int,
            default=value,
        )
    args = parser.parse_args()
    settings = Settings(**{field: vars(args)[field] for field in asdict(defaults)})
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d")
    if not datetime(2015, 2, 18) <= start <= end <= datetime(2019, 4, 16):
        parser.error("Choose ordered dates within the stale S3 v2 archive")
    if not 0 <= settings.confidence <= 100 or any(
        value <= 0 for key, value in asdict(settings).items() if key not in {"confidence", "seed"}
    ):
        parser.error("Thresholds must be positive; confidence must be 0–100")
    mention_paths, missing_mentions = input_files(
        args.mentions, "mentions", start, end, args.allow_partial
    )
    event_paths, missing_events = (
        input_files(args.events, "export", start, end, args.allow_partial)
        if args.events
        else ([], [])
    )
    args.output.mkdir(parents=True, exist_ok=False)
    print("Reading and auditing mentions...", flush=True)
    data, mention_audit = read_mentions(mention_paths)
    out_of_range = data.filter(
        (pl.col("MentionTimeDate").dt.date() < start.date())
        | (pl.col("MentionTimeDate").dt.date() > end.date())
    ).height
    if out_of_range:
        raise ValueError(f"{out_of_range} mention rows outside requested date range")
    labels, event_audit = read_labels(event_paths)
    print("Deduplicating article fingerprints...", flush=True)
    web, articles, dedup_audit = prepare_articles(data, settings)
    articles.write_parquet(args.output / "articles.parquet")
    audit = {
        "command": sys.argv,
        "versions": {p: version(p) for p in ["polars", "igraph", "numpy", "scipy", "matplotlib"]},
        "settings": asdict(settings),
        "start_date": args.start_date,
        "end_date_inclusive": args.end_date,
        "requested_days": (end - start).days + 1,
        "partial": bool(missing_mentions or missing_events),
        "missing_mention_files": missing_mentions,
        "missing_event_files": missing_events,
        "mentions": mention_audit,
        "events": event_audit,
        "deduplication": dedup_audit,
    }
    write_json(args.output / "audit.json", audit)
    print("Projecting the sparse event graph...", flush=True)
    events, incidence, pairs, graph_audit = project(web, articles, settings)
    audit["graph"] = graph_audit
    write_json(args.output / "audit.json", audit)
    del data, web
    summaries, all_validation = [], []
    fig, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    for metric, thresholds in [("jaccard", [0.05, 0.1, 0.2]), ("count", [2, 3, 5])]:
        for threshold in thresholds:
            for method in args.methods:
                name = f"{metric}-{threshold}-{method}"
                print(f"Evaluating {name}...", flush=True)
                summary, validation, sizes = evaluate(
                    args.output,
                    name,
                    articles,
                    events,
                    incidence,
                    pairs,
                    labels,
                    metric,
                    threshold,
                    method,
                    settings.seed,
                )
                summaries.append(summary)
                all_validation.extend({"run": name, **row} for row in validation)
                values = np.sort(sizes["articles"].to_numpy())
                if len(values):
                    axis.step(
                        values,
                        np.arange(len(values), 0, -1) / len(values),
                        where="post",
                        label=name,
                        linestyle="--" if method == "components" else "-",
                    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="URLs per cluster (wire copies expanded)",
        ylabel="Fraction of clusters with at least this many URLs",
        title="Story candidate sizes across fixed thresholds",
    )
    axis.legend(fontsize=7, ncol=2)
    fig.savefig(args.output / "cluster_sizes.svg")
    plt.close(fig)
    write_json(args.output / "sweep.json", {"summaries": summaries, "stories": all_validation})
    pl.DataFrame(all_validation).write_csv(args.output / "story_scores.csv")
    render_summary(args.output, summaries, all_validation, audit)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
