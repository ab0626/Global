import json
import resource
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import polars as pl

from clustering_experiment import (
    ArticleSettings,
    article_features,
    evaluate_articles,
    score_candidates,
)
from story_clusters import Settings, evaluate, render_summary, write_json

output = Path("/home/ubuntu/repos/Global/results/week-comparison-2")
settings = replace(Settings(), min_shared_articles=1)
article_settings = replace(ArticleSettings(), max_pair_contributions=1_000_000_000)
audit = json.loads((output / "audit.json").read_text())
assert audit["settings"] == asdict(settings)
assert audit["article_settings"] == asdict(article_settings)
articles, events, incidence, event_pairs, labels = (
    pl.read_parquet(output / f"{name}.parquet")
    for name in ["articles", "events", "incidence", "event_pairs", "labels"]
)
started = perf_counter()
summaries, validations = [], []
required = [
    "metrics.json",
    "clusters.md",
    "cluster_sizes.csv",
    "keyword_distribution.csv",
    "article_clusters.parquet",
    "event_clusters.parquet",
]
for support in [1, 2]:
    pairs = event_pairs.filter(pl.col("count") >= support)
    for method in ["components", "louvain", "leiden"]:
        name = f"event-support-{support}-{method}"
        destination = output / name
        if destination.exists():
            assert all((destination / filename).exists() for filename in required)
            saved = json.loads((destination / "metrics.json").read_text())
            summary, validation = saved["summary"], saved["expected_stories"]
            summary["runtime_seconds"] = None
            print(f"Reusing complete result: {name}", flush=True)
        else:
            print(f"Evaluating {name}...", flush=True)
            run_started = perf_counter()
            summary, validation, _ = evaluate(
                output,
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
            summary["runtime_seconds"] = perf_counter() - run_started
        summary["minimum_support"] = support
        summaries.append(summary)
        validations.extend({"run": name, **row} for row in validation)
del event_pairs, pairs
if (output / "article_pairs.parquet").exists():
    nodes = incidence.select("canonical_id").unique().sort("canonical_id").with_row_index("vertex")
    pairs = pl.read_parquet(output / "article_pairs.parquet")
else:
    nodes, channels, feature_audit = article_features(articles, incidence, article_settings)
    pairs, candidate_audit = score_candidates(nodes, channels, article_settings)
    audit["article_features"] = feature_audit
    audit["article_candidates"] = candidate_audit
    write_json(output / "audit.json", audit)
    pairs.write_parquet(output / "article_pairs.parquet")
    del channels
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
    destination = output / name
    if destination.exists():
        assert all((destination / filename).exists() for filename in required)
        assert (destination / "canonical_clusters.parquet").exists()
        saved = json.loads((destination / "metrics.json").read_text())
        summary, validation = saved["summary"], saved["expected_stories"]
        summary["runtime_seconds"] = None
        print(f"Reusing complete result: {name}", flush=True)
    else:
        print(f"Evaluating {name}...", flush=True)
        run_started = perf_counter()
        summary, validation, _ = evaluate_articles(
            output,
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
audit["resumed_after_process_restart"] = True
audit["timing_note"] = (
    "Elapsed/RSS below cover only this resumed process; reused run times are null."
)
audit["resumed_process_runtime_seconds"] = perf_counter() - started
audit["resumed_process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
write_json(output / "audit.json", audit)
write_json(output / "sweep.json", {"summaries": summaries, "stories": validations})
pl.DataFrame(summaries).drop(
    "article_count_quantiles", "canonical_article_count_quantiles"
).write_csv(output / "comparison.csv")
pl.DataFrame(validations).write_csv(output / "story_scores.csv")
render_summary(output, summaries, validations, audit)
report = output / "report.md"
report.write_text(
    "# Controlled article-graph experiment\n\n"
    "URL keyword scores are circular for URL-based models; no independent accuracy claim. "
    "All runs use the same input and eligibility. Article partitions are disjoint, with "
    "isolates; event-based URL memberships overlap. Frequency caps and top-k retrieval "
    "have unknown recall. Louvain/Leiden use weighted modularity at resolution 1. "
    "There is no time gate in article models; event projection retains its three-day guard. "
    "The original 200-million article pair-contribution budget was raised to one billion "
    "after the URL channel required 723,230,788 contributions; similarity features and "
    "thresholds were unchanged. A process restart interrupted the second execution; this "
    "driver reused completed outputs and prepared tables. Runtime is not total elapsed time.\n\n"
    + report.read_text()
)
print(f"Report: {report}", flush=True)
