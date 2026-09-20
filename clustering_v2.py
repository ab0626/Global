"""Second article-graph iteration: bursty-token filtering, an Events label channel,
evidence-gated edges and an incident -> story-family layer, on a prepared checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from scipy import sparse

from clustering_experiment import (
    ArticleSettings,
    channel_candidates,
    evaluate_articles,
    tfidf_channel,
    url_words,
)
from story_clusters import (
    STORIES,
    Settings,
    communities,
    max_count,
    render_summary,
    save_evaluation,
    write_json,
)

BENCHMARK = {
    **STORIES,
    "idai_strict": r"(^|[^a-z])idai([^a-z]|$)",
    "border_veto": r"veto.*(border|emergency|national)|(border|emergency).*veto",
    "college_admissions": r"loughlin|huffman|admissions?-(scandal|scam|bribery|cheating)",
    "manafort": r"manafort",
    "pell": r"(^|[^a-z])pell([^a-z]|$)",
    "smollett": r"smollett",
    "bomb_cyclone": r"bomb-?cyclone",
    "pelosi_impeachment": r"pelosi.*impeach|impeach.*pelosi",
    "venezuela_blackout": r"venezuela.*(blackout|power|outage|electricity)",
    "r_kelly": r"r-kelly|rkelly",
    "gmail_outage": r"(gmail|google).*(outage|down|disruption)",
}

HEX_JUNK = re.compile(r"^(?=.*\d)(?=.*[a-f])[0-9a-f]+$")


@dataclass(frozen=True)
class V2Settings:
    burst_entropy_max: float = 0.95
    burst_min_articles: int = 100
    host_min_articles: int = 20
    host_boilerplate_share: float = 0.2
    corroboration_min_channels: int = 2
    single_channel_floor: float = 0.6
    family_threshold: float = 0.25
    family_min_canonical: int = 2
    family_neighbors: int = 10


def clean_token(word: str) -> bool:
    if HEX_JUNK.match(word):
        return False
    if word.isdecimal():
        return len(word) == 3
    return True


def token_rows(articles: pl.DataFrame, nodes: pl.DataFrame, clean: bool) -> pl.DataFrame:
    frame = articles.join(nodes, on="canonical_id").select(
        "vertex", "domain", "slug", pl.col("first_seen").dt.date().alias("day")
    )
    rows = [
        (vertex, domain, day, word)
        for vertex, domain, path, day in frame.iter_rows()
        for word in url_words(path)
        if not clean or clean_token(word)
    ]
    return pl.DataFrame(
        rows,
        schema={"vertex": pl.UInt32, "domain": pl.String, "day": pl.Date, "feature": pl.String},
        orient="row",
    ).unique()


def label_rows(
    incidence: pl.DataFrame, labels: pl.DataFrame, nodes: pl.DataFrame, articles: pl.DataFrame
) -> pl.DataFrame:
    days = (
        articles.group_by("canonical_id")
        .agg(pl.col("first_seen").min().dt.date().alias("day"))
        .join(nodes, on="canonical_id")
        .select("vertex", "day")
    )
    joined = (
        incidence.drop("vertex").join(labels, on="GlobalEventID").join(nodes, on="canonical_id")
    )
    parts = [
        joined.filter(pl.col(column).fill_null("") != "").select(
            "vertex", (pl.lit(prefix) + pl.col(column)).alias("feature")
        )
        for prefix, column in [
            ("actor:", "Actor1Name"),
            ("actor:", "Actor2Name"),
            ("geo:", "ActionGeo_FullName"),
        ]
    ]
    return (
        pl.concat(parts)
        .unique()
        .join(days, on="vertex")
        .with_columns(pl.lit("", dtype=pl.String).alias("domain"))
        .select("vertex", "domain", "day", "feature")
    )


def burst_filter(
    rows: pl.DataFrame, days: int, settings: V2Settings
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop features whose daily first-seen distribution is near uniform across the
    window (normalised entropy >= burst_entropy_max) once they are frequent enough."""
    per_day = rows.unique(["vertex", "feature", "day"]).group_by("feature", "day").len()
    stats = per_day.group_by("feature").agg(
        pl.col("len").sum().alias("article_days"),
        (
            -(
                (pl.col("len") / pl.col("len").sum()) * (pl.col("len") / pl.col("len").sum()).log()
            ).sum()
            / np.log(max(days, 2))
        ).alias("entropy"),
    )
    frequency = rows.unique(["vertex", "feature"]).group_by("feature").len().rename({"len": "df"})
    stats = stats.join(frequency, on="feature")
    dropped = stats.filter(
        (pl.col("entropy") >= settings.burst_entropy_max)
        & (pl.col("df") >= settings.burst_min_articles)
    )
    return rows.join(dropped.select("feature"), on="feature", how="anti"), dropped


def host_filter(rows: pl.DataFrame, settings: V2Settings) -> tuple[pl.DataFrame, int]:
    """Drop (host, token) pairs where the token appears in a large share of that host's
    articles: CMS path boilerplate such as `articleshow`, `php`, `newsdisplay`."""
    hosts = (
        rows.unique(["vertex", "domain"]).group_by("domain").len().rename({"len": "host_articles"})
    )
    shares = (
        rows.unique(["vertex", "domain", "feature"])
        .group_by("domain", "feature")
        .len()
        .join(hosts, on="domain")
        .filter(
            (pl.col("host_articles") >= settings.host_min_articles)
            & (pl.col("len") / pl.col("host_articles") >= settings.host_boilerplate_share)
            & (pl.col("domain") != "")
        )
        .select("domain", "feature")
    )
    return rows.join(shares, on=["domain", "feature"], how="anti"), shares.height


def binary_matrix(rows: pl.DataFrame, count: int) -> tuple[sparse.csr_matrix, list[str]]:
    pairs = rows.select("vertex", "feature").unique().sort("feature", "vertex")
    vocabulary = pairs["feature"].unique(maintain_order=True).to_list()
    index = {feature: i for i, feature in enumerate(vocabulary)}
    columns = np.fromiter(
        (index[f] for f in pairs["feature"].to_list()), dtype=np.int64, count=pairs.height
    )
    matrix = sparse.csr_matrix(
        (np.ones(pairs.height, dtype=np.float64), (pairs["vertex"].to_numpy(), columns)),
        shape=(count, len(vocabulary)),
    )
    return matrix, vocabulary


def build_channels(
    articles: pl.DataFrame,
    incidence: pl.DataFrame,
    labels: pl.DataFrame,
    days: int,
    settings: V2Settings,
    article_settings: ArticleSettings,
    v1_url_channel: bool,
    use_labels: bool,
) -> tuple[pl.DataFrame, dict[str, sparse.csr_matrix], dict]:
    nodes = incidence.select("canonical_id").unique().sort("canonical_id").with_row_index("vertex")
    mapped = incidence.rename({"vertex": "event_vertex"}).join(nodes, on="canonical_id")
    event_matrix = sparse.csr_matrix(
        (
            np.ones(mapped.height, dtype=np.float64),
            (mapped["vertex"].to_numpy(), mapped["event_vertex"].to_numpy()),
        ),
        shape=(nodes.height, max_count(incidence["vertex"]) + 1 if incidence.height else 0),
    )
    audit: dict = {"active_canonical_articles": nodes.height}
    raw: dict[str, sparse.csr_matrix] = {"event": event_matrix}
    uncapped = replace(article_settings, max_feature_articles=max(nodes.height, 1))
    per_channel_settings = {"event": uncapped}
    url_rows = token_rows(articles, nodes, clean=not v1_url_channel)
    if v1_url_channel:
        raw["url"], _ = binary_matrix(url_rows, nodes.height)
        per_channel_settings["url"] = article_settings
        audit["url_rule"] = "v1: raw url_words, frequency cap max_feature_articles"
    else:
        url_rows, url_dropped = burst_filter(url_rows, days, settings)
        url_rows, host_pairs = host_filter(url_rows, settings)
        raw["url"], _ = binary_matrix(url_rows, nodes.height)
        per_channel_settings["url"] = uncapped
        audit["url_rule"] = (
            "v2: url_words minus hex/numeric junk, minus near-uniform daily-entropy tokens, "
            "minus per-host boilerplate; no global frequency cap"
        )
        audit["url_burst_dropped"] = url_dropped.sort("df", descending=True).head(200).to_dicts()
        audit["url_burst_dropped_count"] = url_dropped.height
        audit["url_host_boilerplate_pairs_dropped"] = host_pairs
    if use_labels:
        rows = label_rows(incidence, labels, nodes, articles)
        rows, label_dropped = burst_filter(rows, days, settings)
        raw["label"], _ = binary_matrix(rows, nodes.height)
        per_channel_settings["label"] = uncapped
        audit["label_rule"] = "actor:/geo: features from Events, burst-filtered, no cap"
        audit["label_burst_dropped"] = (
            label_dropped.sort("df", descending=True).head(100).to_dicts()
        )
        audit["label_burst_dropped_count"] = label_dropped.height
    channels = {}
    for name, matrix in raw.items():
        channels[name], audit[name] = tfidf_channel(matrix, per_channel_settings[name])
    return nodes, channels, audit


def score_pairs(
    nodes: pl.DataFrame,
    channels: dict[str, sparse.csr_matrix],
    article_settings: ArticleSettings,
    settings: V2Settings,
) -> tuple[pl.DataFrame, dict]:
    retrieved, counts = [], {}
    for name, matrix in channels.items():
        print(f"Retrieving neighbors: {name}...", flush=True)
        keys = channel_candidates(matrix, article_settings)
        retrieved.append(keys)
        counts[name] = len(keys)
    keys = np.unique(np.concatenate(retrieved))
    if len(keys) > article_settings.max_candidate_pairs:
        raise ValueError("Candidate union exceeds --max-candidate-pairs")
    left, right = keys // nodes.height, keys % nodes.height
    pairs = pl.DataFrame({"left": left, "right": right})
    available = np.zeros(len(keys), dtype=np.int64)
    evidence = np.zeros(len(keys), dtype=np.int64)
    total = np.zeros(len(keys), dtype=np.float64)
    best = np.zeros(len(keys), dtype=np.float64)
    for name, matrix in channels.items():
        print(f"Scoring: {name}...", flush=True)
        has = np.asarray(matrix.getnnz(axis=1) > 0).ravel()
        scores = np.zeros(len(keys), dtype=np.float64)
        batch = article_settings.scoring_batch_size
        for start in range(0, len(keys), batch):
            stop = start + batch
            scores[start:stop] = np.asarray(
                matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
            ).ravel()
        scores = np.clip(scores, 0, 1)
        pairs = pairs.with_columns(pl.Series(f"{name}_score", scores))
        available += (has[left] & has[right]).astype(np.int64)
        evidence += (scores > 0).astype(np.int64)
        total += scores
        best = np.maximum(best, scores)
    combined = total / np.maximum(available, 1)
    gated = np.where(
        (evidence >= settings.corroboration_min_channels) | (best >= settings.single_channel_floor),
        combined,
        0.0,
    )
    pairs = pairs.with_columns(
        pl.Series("available_channels", available),
        pl.Series("evidence_channels", evidence),
        pl.Series("combined", combined),
        pl.Series("gated", gated),
    )
    return pairs, {
        "directed_neighbor_counts": counts,
        "candidate_pairs": pairs.height,
        "pairs_with_single_channel_evidence": int((evidence == 1).sum()),
        "combined_rule": "mean over channels where both articles have features",
        "gated_rule": (
            f"combined if evidence channels >= {settings.corroboration_min_channels} "
            f"or any channel >= {settings.single_channel_floor}, else 0"
        ),
        "candidate_recall": None,
    }


def families(
    nodes: pl.DataFrame,
    partition: pl.DataFrame,
    channels: dict[str, sparse.csr_matrix],
    settings: V2Settings,
    article_settings: ArticleSettings,
) -> tuple[pl.DataFrame, dict]:
    """Link incident clusters whose centroid vectors are similar into story families.
    Returns one row per canonical article with `cluster` (incident) and `family`."""
    sizes = partition.group_by("cluster").len()
    eligible = sizes.filter(pl.col("len") >= settings.family_min_canonical).sort("cluster")
    incidents = eligible.with_row_index("incident_vertex").select("cluster", "incident_vertex")
    membership = partition.join(incidents, on="cluster")
    assign = sparse.csr_matrix(
        (
            np.ones(membership.height),
            (membership["incident_vertex"].to_numpy(), membership["vertex"].to_numpy()),
        ),
        shape=(incidents.height, nodes.height),
    )
    retrieval = replace(article_settings, neighbors=settings.family_neighbors)
    centroids = []
    keys_all = []
    for matrix in channels.values():
        centroid = (assign @ matrix).tocsr()
        norms = np.sqrt(np.asarray(centroid.multiply(centroid).sum(axis=1)).ravel())
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        centroid = centroid.multiply(inverse[:, None]).tocsr()
        centroids.append(centroid)
        if incidents.height:
            keys_all.append(channel_candidates(centroid, retrieval))
    keys = np.unique(np.concatenate(keys_all)) if keys_all else np.array([], dtype=np.int64)
    width = max(incidents.height, 1)
    left, right = keys // width, keys % width
    total = np.zeros(len(keys))
    for centroid in centroids:
        total += np.asarray(centroid[left].multiply(centroid[right]).sum(axis=1)).ravel()
    links = pl.DataFrame(
        {"left": left, "right": right, "family_score": total / max(len(centroids), 1)}
    )
    incident_nodes = incidents.rename({"cluster": "incident", "incident_vertex": "vertex"})
    family_partition, edges = communities(
        incident_nodes, links, "family_score", settings.family_threshold, "components", 0
    )
    family_of_incident = family_partition.select(
        pl.col("incident").alias("cluster"), pl.col("cluster").alias("family")
    )
    offset = family_of_incident["family"].max() if family_of_incident.height else -1
    offset = int(offset) + 1 if offset is not None else 0
    result = partition.join(family_of_incident, on="cluster", how="left").with_columns(
        pl.col("family").fill_null(pl.col("cluster") + offset)
    )
    return result, {
        "family_edges": edges,
        "eligible_incident_clusters": incidents.height,
        "families_with_multiple_incidents": family_of_incident.group_by("family")
        .len()
        .filter(pl.col("len") > 1)
        .height,
    }


def evaluate_partition(
    output: Path,
    name: str,
    articles: pl.DataFrame,
    incidence: pl.DataFrame,
    partition: pl.DataFrame,
    labels: pl.DataFrame,
    seed: int,
    stories: dict[str, str],
) -> tuple[dict, list[dict]]:
    members = partition.select("canonical_id", "cluster").join(articles, on="canonical_id")
    event_membership = (
        incidence.join(partition.select("canonical_id", "cluster"), on="canonical_id")
        .select("GlobalEventID", "cluster")
        .unique()
    )
    summary, validation, _ = save_evaluation(
        output,
        name,
        articles,
        event_membership,
        members,
        labels,
        "family",
        0.0,
        "components",
        seed,
        0,
        stories,
    )
    partition.write_parquet(output / name / "canonical_clusters.parquet")
    return summary, validation


def load_prepared(source: Path, settings: Settings) -> tuple[list[pl.DataFrame], dict]:
    audit = json.loads((source / "audit.json").read_text())
    if audit["settings"] != asdict(settings):
        raise ValueError("Prepared input used different baseline settings")
    frames = [
        pl.read_parquet(source / f"{name}.parquet") for name in ["articles", "incidence", "labels"]
    ]
    return frames, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-reference", action="store_true")
    for field, value in asdict(ArticleSettings()).items():
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=value)
    args = parser.parse_args()
    article_settings = ArticleSettings(
        **{field: vars(args)[field] for field in asdict(ArticleSettings())}
    )
    settings = V2Settings()
    baseline = replace(Settings(), min_shared_articles=1)
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    (articles, incidence, labels), audit = load_prepared(args.prepared, baseline)
    for story, pattern in BENCHMARK.items():
        if story not in articles.columns:
            articles = articles.with_columns(pl.col("slug").str.contains(pattern).alias(story))
    days = int(audit["requested_days"])
    audit.update(
        command=sys.argv,
        prepared_source=str(args.prepared.resolve()),
        v2_settings=asdict(settings),
        article_settings=asdict(article_settings),
        benchmark=BENCHMARK,
        benchmark_note=(
            "Story regexes are URL-path proxies chosen from URL inspection of the week, "
            "not independent labels; circular for URL-featured models."
        ),
    )
    summaries: list[dict] = []
    validations: list[dict] = []

    def record(name: str, summary: dict, validation: list[dict], run_started: float) -> None:
        summary["runtime_seconds"] = perf_counter() - run_started
        summaries.append(summary)
        validations.extend({"run": name, **row} for row in validation)
        write_json(args.output / "sweep.json", {"summaries": summaries, "stories": validations})

    configs = [
        ("v1-reference", True, False),
        ("v2-url", False, False),
        ("v2-url-label", False, True),
    ]
    if args.skip_reference:
        configs = configs[1:]
    for config, v1_url, use_labels in configs:
        print(f"Building channels: {config}...", flush=True)
        nodes, channels, feature_audit = build_channels(
            articles, incidence, labels, days, settings, article_settings, v1_url, use_labels
        )
        pairs, pair_audit = score_pairs(nodes, channels, article_settings, settings)
        audit[f"{config}_features"] = feature_audit
        audit[f"{config}_pairs"] = pair_audit
        write_json(args.output / "audit.json", audit)
        pairs.write_parquet(args.output / f"{config}-pairs.parquet")
        metrics = [("combined", 0.3)]
        if use_labels:
            metrics = [("combined", 0.3), ("gated", 0.2), ("gated", 0.3), ("gated", 0.4)]
        for metric, threshold in metrics:
            name = f"{config}-{metric}-{threshold}-leiden"
            print(f"Evaluating {name}...", flush=True)
            run_started = perf_counter()
            summary, validation, _ = evaluate_articles(
                args.output,
                name,
                articles,
                nodes,
                incidence,
                pairs,
                labels,
                metric,
                threshold,
                "leiden",
                baseline.seed,
                BENCHMARK,
            )
            partition = pl.read_parquet(args.output / name / "canonical_clusters.parquet")
            best = [r["best_cluster"] for r in validation if r["best_cluster"] is not None]
            summary["distinct_best_clusters"] = len(set(best))
            record(name, summary, validation, run_started)
            if metric == "gated" and threshold == 0.3:
                print("Linking incident clusters into families...", flush=True)
                run_started = perf_counter()
                family_partition, family_audit = families(
                    nodes, partition, channels, settings, article_settings
                )
                audit[f"{config}_families"] = family_audit
                family_name = f"{name}-families"
                summary, validation = evaluate_partition(
                    args.output,
                    family_name,
                    articles,
                    incidence,
                    family_partition.select("canonical_id", pl.col("family").alias("cluster")),
                    labels,
                    baseline.seed,
                    BENCHMARK,
                )
                family_partition.write_parquet(
                    args.output / family_name / "incident_families.parquet"
                )
                record(family_name, summary, validation, run_started)
        del pairs, channels
    audit["runtime_seconds"] = perf_counter() - started
    audit["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(args.output / "audit.json", audit)
    pl.DataFrame(summaries).drop(
        "article_count_quantiles", "canonical_article_count_quantiles"
    ).write_csv(args.output / "comparison.csv")
    pl.DataFrame(validations).write_csv(args.output / "story_scores.csv")
    render_summary(args.output, summaries, validations, audit)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
