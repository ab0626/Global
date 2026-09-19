"""Controlled event-projection and sparse article-graph comparisons."""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from scipy import sparse

from story_clusters import (
    STOPWORDS,
    Settings,
    communities,
    evaluate,
    input_files,
    max_count,
    prepare_articles,
    project,
    read_labels,
    read_mentions,
    render_summary,
    save_evaluation,
    write_json,
)


@dataclass(frozen=True)
class ArticleSettings:
    neighbors: int = 50
    block_size: int = 128
    max_feature_articles: int = 2000
    max_pair_contributions: int = 200_000_000
    max_candidate_pairs: int = 50_000_000
    max_block_nonzeros: int = 5_000_000
    scoring_batch_size: int = 20_000


def url_words(path: str) -> set[str]:
    path = re.sub(r"(?<=/)(?:\d{1,2}|(?:19|20)\d{2})(?=/)", "", path)
    return {
        word
        for word in re.findall(r"[^\W_]+", path.casefold(), flags=re.UNICODE)
        if len(word) >= 2 and word not in STOPWORDS and not (word.isdecimal() and len(word) >= 5)
    }


def tfidf_channel(
    matrix: sparse.csr_matrix, settings: ArticleSettings
) -> tuple[sparse.csr_matrix, dict]:
    frequencies = np.asarray(matrix.getnnz(axis=0), dtype=np.int64)
    keep = (frequencies > 0) & (frequencies <= settings.max_feature_articles)
    retained = frequencies[keep]
    contributions = int((retained * (retained - 1) // 2).sum())
    if contributions > settings.max_pair_contributions:
        raise ValueError(
            f"{contributions:,} feature pair contributions exceed the article graph guard "
            f"({settings.max_pair_contributions:,}); use a smaller date slice or explicitly "
            "raise --max-pair-contributions"
        )
    weighted = matrix[:, keep].multiply(np.log((matrix.shape[0] + 1) / (retained + 1)) + 1).tocsr()
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    weighted = weighted.multiply(inverse[:, None]).tocsr()
    weighted.eliminate_zeros()
    return weighted, {
        "features_before_frequency_cap": int(np.count_nonzero(frequencies)),
        "features_retained": int(keep.sum()),
        "frequent_features_removed": int((frequencies > settings.max_feature_articles).sum()),
        "nonzero_values": weighted.nnz,
        "articles_with_features": int(np.count_nonzero(norms)),
        "feature_pair_contributions": contributions,
    }


def article_features(
    articles: pl.DataFrame,
    incidence: pl.DataFrame,
    settings: ArticleSettings,
) -> tuple[pl.DataFrame, dict[str, sparse.csr_matrix], dict]:
    nodes = incidence.select("canonical_id").unique().sort("canonical_id").with_row_index("vertex")
    mapped = incidence.rename({"vertex": "event_vertex"}).join(nodes, on="canonical_id")
    event_columns = max_count(incidence["vertex"]) + 1 if incidence.height else 0
    event_matrix = sparse.csr_matrix(
        (
            np.ones(mapped.height, dtype=np.float64),
            (mapped["vertex"].to_numpy(), mapped["event_vertex"].to_numpy()),
        ),
        shape=(nodes.height, event_columns),
    )
    groups = (
        articles.join(nodes, on="canonical_id")
        .group_by("vertex")
        .agg(pl.col("slug").unique().sort())
        .sort("vertex")
    )
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    columns: list[int] = []
    for vertex, paths in groups.iter_rows():
        words = set().union(*(url_words(path) for path in paths))
        for word in sorted(words):
            index = vocabulary.setdefault(word, len(vocabulary))
            rows.append(vertex)
            columns.append(index)
    url_matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(nodes.height, len(vocabulary)),
    )
    channels = {}
    audit = {}
    for name, matrix in [("event", event_matrix), ("url", url_matrix)]:
        channels[name], audit[name] = tfidf_channel(matrix, settings)
    audit["active_canonical_articles"] = nodes.height
    audit["url_feature_rule"] = (
        "Union of decoded URL-path words across each wire group; no host/query features. "
        "Drop numeric date path segments, long numeric IDs, short tokens and fixed stopwords."
    )
    return nodes, channels, audit


def channel_candidates(matrix: sparse.csr_matrix, settings: ArticleSettings) -> np.ndarray:
    count = matrix.shape[0]
    capacity = count * min(settings.neighbors, max(count - 1, 0))
    if capacity > settings.max_candidate_pairs:
        raise ValueError("Directed neighbor capacity exceeds --max-candidate-pairs")
    keys = np.empty(capacity, dtype=np.int64)
    used = 0
    transpose = matrix.T.tocsc()
    for start in range(0, count, settings.block_size):
        scores = (matrix[start : start + settings.block_size] @ transpose).tocsr()
        if scores.nnz > settings.max_block_nonzeros:
            raise ValueError(
                "Article similarity block exceeds --max-block-nonzeros; reduce --block-size"
            )
        for offset in range(scores.shape[0]):
            node = start + offset
            begin, end = scores.indptr[offset : offset + 2]
            indices = scores.indices[begin:end]
            values = scores.data[begin:end]
            valid = (indices != node) & (values > 0)
            indices, values = indices[valid], values[valid]
            order = np.lexsort((indices, -values))[: settings.neighbors]
            neighbors = indices[order].astype(np.int64)
            size = len(neighbors)
            keys[used : used + size] = np.minimum(node, neighbors) * count + np.maximum(
                node, neighbors
            )
            used += size
    return keys[:used]


def score_candidates(
    nodes: pl.DataFrame,
    channels: dict[str, sparse.csr_matrix],
    settings: ArticleSettings,
) -> tuple[pl.DataFrame, dict]:
    retrieved = []
    retrieval_counts = {}
    for name, matrix in channels.items():
        print(f"Retrieving article neighbors: {name}...", flush=True)
        keys = channel_candidates(matrix, settings)
        retrieved.append(keys)
        retrieval_counts[name] = len(keys)
    keys = np.unique(np.concatenate(retrieved))
    if len(keys) > settings.max_candidate_pairs:
        raise ValueError("Candidate union exceeds --max-candidate-pairs")
    if len(keys):
        left, right = keys // nodes.height, keys % nodes.height
    else:
        left = right = np.array([], dtype=np.int64)
    pairs = pl.DataFrame({"left": left, "right": right})
    for name, matrix in channels.items():
        print(f"Scoring candidate union: {name}...", flush=True)
        scores = np.zeros(len(keys), dtype=np.float64)
        for start in range(0, len(keys), settings.scoring_batch_size):
            stop = start + settings.scoring_batch_size
            scores[start:stop] = np.asarray(
                matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
            ).ravel()
        pairs = pairs.with_columns(pl.Series(f"{name}_score", np.clip(scores, 0, 1)))
    pairs = pairs.with_columns(
        ((pl.col("event_score") + pl.col("url_score")) / 2).alias("combined")
    )
    return pairs, {
        "directed_neighbor_counts": retrieval_counts,
        "candidate_pairs": pairs.height,
        "url_only_candidate_pairs": pairs.filter(
            (pl.col("url_score") > 0) & (pl.col("event_score") == 0)
        ).height,
        "candidate_rule": "Union of per-channel top-k; ties use vertex order; no time cutoff.",
        "score_rule": "Equal channel weights; absent features contribute zero.",
        "candidate_recall": None,
        "candidate_recall_note": "No independently labeled same-story pairs available.",
    }


def evaluate_articles(
    output: Path,
    name: str,
    articles: pl.DataFrame,
    nodes: pl.DataFrame,
    incidence: pl.DataFrame,
    pairs: pl.DataFrame,
    labels: pl.DataFrame,
    metric: str,
    threshold: float,
    method: str,
    seed: int,
) -> tuple[dict, list[dict], pl.DataFrame]:
    partition, edge_count = communities(nodes, pairs, metric, threshold, method, seed)
    members = partition.select("canonical_id", "cluster").join(articles, on="canonical_id")
    event_membership = (
        incidence.join(partition.select("canonical_id", "cluster"), on="canonical_id")
        .select("GlobalEventID", "cluster")
        .unique()
    )
    result = save_evaluation(
        output,
        name,
        articles,
        event_membership,
        members,
        labels,
        metric,
        threshold,
        method,
        seed,
        edge_count,
    )
    partition.write_parquet(output / name / "canonical_clusters.parquet")
    return result


def prepare_input(
    args: argparse.Namespace, settings: Settings
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    if args.prepared:
        source = args.prepared
        audit = json.loads((source / "audit.json").read_text())
        if audit["settings"] != asdict(settings):
            raise ValueError("Prepared input used different baseline settings")
        frames = [
            pl.read_parquet(source / f"{name}.parquet")
            for name in ["articles", "events", "incidence", "event_pairs", "labels"]
        ]
        audit["prepared_source"] = str(source.resolve())
        articles, events, incidence, pairs, labels = frames
        return articles, events, incidence, pairs, labels, audit
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d")
    if not datetime(2015, 2, 18) <= start <= end <= datetime(2019, 4, 16):
        raise ValueError("Choose ordered dates within the stale S3 v2 archive")
    paths, missing = input_files(args.mentions, "mentions", start, end, args.allow_partial)
    event_paths, missing_events = (
        input_files(args.events, "export", start, end, args.allow_partial)
        if args.events
        else ([], [])
    )
    print("Reading and auditing Mentions...", flush=True)
    data, mention_audit = read_mentions(paths)
    if data.filter(
        (pl.col("MentionTimeDate").dt.date() < start.date())
        | (pl.col("MentionTimeDate").dt.date() > end.date())
    ).height:
        raise ValueError("Mentions outside requested date range")
    print("Deduplicating wire fingerprints...", flush=True)
    web, articles, dedup_audit = prepare_articles(data, settings)
    print("Projecting event pairs with minimum support one...", flush=True)
    events, incidence, pairs, graph_audit = project(web, articles, settings)
    del data, web
    labels, event_audit = read_labels(event_paths)
    labeled = incidence.join(labels.select("GlobalEventID"), on="GlobalEventID")
    audit = {
        "settings": asdict(settings),
        "start_date": args.start_date,
        "end_date_inclusive": args.end_date,
        "requested_days": (end - start).days + 1,
        "partial": bool(missing or missing_events),
        "missing_mention_files": missing,
        "missing_event_files": missing_events,
        "mentions": mention_audit,
        "events": event_audit,
        "deduplication": dedup_audit,
        "graph": graph_audit,
        "label_coverage": {
            "retained_events": events.height,
            "labeled_retained_events": labeled["GlobalEventID"].n_unique(),
            "active_canonical_articles": incidence["canonical_id"].n_unique(),
            "canonical_articles_with_any_label": labeled["canonical_id"].n_unique(),
        },
    }
    return articles, events, incidence, pairs, labels, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mentions", type=Path)
    source.add_argument("--prepared", type=Path, help="Reuse a prior experiment's prepared tables")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--start-date", default="20190310")
    parser.add_argument("--end-date", default="20190317")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    for field, value in asdict(ArticleSettings()).items():
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=value)
    args = parser.parse_args()
    article_settings = ArticleSettings(
        **{field: vars(args)[field] for field in asdict(ArticleSettings())}
    )
    if any(value <= 0 for value in asdict(article_settings).values()):
        parser.error("Article graph bounds must be positive")
    if args.prepared and args.events:
        parser.error("--prepared includes Events labels; do not also pass --events")
    started = perf_counter()
    settings = replace(Settings(), min_shared_articles=1)
    args.output.mkdir(parents=True, exist_ok=False)
    articles, events, incidence, event_pairs, labels, audit = prepare_input(args, settings)
    audit["command"] = sys.argv
    audit["versions"] = {
        package: version(package) for package in ["polars", "igraph", "numpy", "scipy"]
    }
    audit["article_settings"] = asdict(article_settings)
    for name, frame in [
        ("articles", articles),
        ("events", events),
        ("incidence", incidence),
        ("event_pairs", event_pairs),
        ("labels", labels),
    ]:
        frame.write_parquet(args.output / f"{name}.parquet")
    write_json(args.output / "audit.json", audit)
    if args.prepare_only:
        print(f"Prepared input: {args.output}", flush=True)
        return
    summaries, validations = [], []
    for support in [1, 2]:
        pairs = event_pairs.filter(pl.col("count") >= support)
        for method in ["components", "louvain", "leiden"]:
            name = f"event-support-{support}-{method}"
            print(f"Evaluating {name}...", flush=True)
            run_started = perf_counter()
            summary, validation, _ = evaluate(
                args.output,
                name,
                articles,
                events,
                incidence,
                pairs,
                labels,
                "jaccard",
                0.1,
                method,
                settings.seed,
            )
            summary.update(minimum_support=support, runtime_seconds=perf_counter() - run_started)
            summaries.append(summary)
            validations.extend({"run": name, **row} for row in validation)
    del pairs, event_pairs
    nodes, channels, feature_audit = article_features(articles, incidence, article_settings)
    pairs, candidate_audit = score_candidates(nodes, channels, article_settings)
    audit["article_features"] = feature_audit
    audit["article_candidates"] = candidate_audit
    write_json(args.output / "audit.json", audit)
    pairs.write_parquet(args.output / "article_pairs.parquet")
    comparisons = [
        ("event_score", 0.3, "leiden"),
        ("url_score", 0.3, "leiden"),
        ("combined", 0.15, "leiden"),
        ("combined", 0.3, "leiden"),
        ("combined", 0.45, "leiden"),
        ("combined", 0.3, "louvain"),
        ("combined", 0.3, "components"),
    ]
    for metric, threshold, method in comparisons:
        name = f"article-{metric}-{threshold}-{method}"
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
            method,
            settings.seed,
        )
        summary["runtime_seconds"] = perf_counter() - run_started
        summaries.append(summary)
        validations.extend({"run": name, **row} for row in validation)
    audit["runtime_seconds"] = perf_counter() - started
    audit["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(args.output / "audit.json", audit)
    write_json(args.output / "sweep.json", {"summaries": summaries, "stories": validations})
    pl.DataFrame(summaries).drop(
        "article_count_quantiles", "canonical_article_count_quantiles"
    ).write_csv(args.output / "comparison.csv")
    pl.DataFrame(validations).write_csv(args.output / "story_scores.csv")
    render_summary(args.output, summaries, validations, audit)
    notes = (
        "# Controlled article-graph experiment\n\n"
        "All 13 runs share the same input, wire groups and eligible canonical articles. "
        "Event runs may overlap; article runs assign one cluster per active canonical article. "
        "URL keyword scores are circular for URL-based models and cannot establish accuracy. "
        "Parameters are fixed exploratory settings; no held-out story labels were available. "
        "Leiden and Louvain both use weighted modularity, resolution 1, seed 2026. "
        "The article models use no time cutoff; event projection retains the three-day guard. "
        "Frequency caps and top-k retrieval can miss relevant pairs; candidate recall is unknown. "
        "Isolated articles remain visible as singleton candidates, not certified stories. "
        "Events labels are for inspection only, not similarity features. "
        "In article runs, event_clusters.parquet is an overlapping event/cluster relation; "
        "canonical_clusters.parquet is the disjoint article partition.\n\n"
    )
    report = args.output / "report.md"
    report.write_text(notes + report.read_text(), encoding="utf-8")
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
