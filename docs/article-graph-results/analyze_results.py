import argparse
import json
from pathlib import Path

import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root, output = args.input, args.output
output.mkdir(exist_ok=True, parents=True)
audit = json.loads((root / "audit.json").read_text())
sweep = json.loads((root / "sweep.json").read_text())
assert len(sweep["summaries"]) == 13
articles = pl.read_parquet(root / "articles.parquet").with_columns(
    pl.col("slug").str.contains(r"(^|[^a-z])idai([^a-z]|$)").alias("idai_specific")
)
stories = ["crash_grounding", "christchurch", "idai", "brexit", "idai_specific"]
totals = {
    story: (
        articles.filter(story).height,
        articles.filter(story)["canonical_id"].n_unique(),
    )
    for story in stories
}
flags = articles.select("article_id", *stories)
comparisons, story_rows, strata = [], [], []
for summary in sweep["summaries"]:
    name = summary["run"]
    members = pl.read_parquet(root / name / "article_clusters.parquet")
    sizes = pl.read_csv(root / name / "cluster_sizes.csv")
    assert members["article_id"].n_unique() == summary["assigned_urls"]
    assert summary["assigned_urls"] + summary["unassigned_urls"] == articles.height
    if name.startswith("article-"):
        assert members.height == summary["assigned_urls"]
        partition = pl.read_parquet(root / name / "canonical_clusters.parquet")
        assert partition["canonical_id"].n_unique() == partition.height == 475501
        assert members["canonical_id"].n_unique() == 475501
    comparisons.append(
        {
            "run": name,
            "clusters": summary["clusters"],
            "singleton_cluster_percent": 100
            * summary["single_canonical_article_clusters"]
            / summary["clusters"],
            "singleton_only_percent_assigned_urls": 100
            * summary["singleton_only_assigned_urls"]
            / summary["assigned_urls"],
            "largest_urls": summary["largest_cluster_urls"],
            "largest_percent_all_urls": 100 * summary["largest_fraction_all_urls"],
            "overlap_urls": summary["multiply_assigned_urls"],
            "unassigned_urls": summary["unassigned_urls"],
            "canonical_size_median": summary["canonical_article_count_quantiles"]["0.5"],
            "canonical_size_p90": summary["canonical_article_count_quantiles"]["0.9"],
            "canonical_size_p99": summary["canonical_article_count_quantiles"]["0.99"],
        }
    )
    matched = members.join(flags, on="article_id")
    canonical = matched.group_by("canonical_id", "cluster").agg(pl.col(stories).any())
    for story in stories:
        for unit, frame, key, denominator in [
            ("url", matched, "article_id", totals[story][0]),
            ("canonical_any_matching_alias", canonical, "canonical_id", totals[story][1]),
        ]:
            hits = frame.filter(story)
            distribution = (
                hits.group_by("cluster").len().sort(["len", "cluster"], descending=[True, False])
            )
            assert distribution.height > 0
            best = distribution.row(0, named=True)
            top_ids = distribution.head(5)["cluster"]
            cluster_size = frame.filter(pl.col("cluster") == best["cluster"]).height
            story_rows.append(
                {
                    "run": name,
                    "story": story,
                    "unit": unit,
                    "all_matching_units": denominator,
                    "assigned_matching_units": hits[key].n_unique(),
                    "clusters_with_matches": distribution.height,
                    "best_cluster": best["cluster"],
                    "best_matches": best["len"],
                    "best_cluster_size": cluster_size,
                    "best_match_fraction": best["len"] / denominator,
                    "best_keyword_purity": best["len"] / cluster_size,
                    "top5_union_match_fraction": hits.filter(
                        pl.col("cluster").is_in(top_ids.implode())
                    )[key].n_unique()
                    / denominator,
                }
            )
            if story == "idai_specific":
                distribution.write_csv(output / f"{name}-idai-{unit}.csv")
    non_singleton_ids = (
        members.join(sizes.filter(pl.col("canonical_articles") > 1).select("cluster"), on="cluster")
        .select("article_id")
        .unique()
        .with_columns(pl.lit(True).alias("multi"))
    )
    assigned_ids = (
        members.select("article_id").unique().with_columns(pl.lit(True).alias("assigned"))
    )
    by_domain = (
        articles.select("article_id", "domain")
        .join(assigned_ids, on="article_id", how="left")
        .join(non_singleton_ids, on="article_id", how="left")
        .group_by("domain")
        .agg(
            pl.len().alias("urls"),
            pl.col("assigned").fill_null(False).sum().alias("assigned_urls"),
            pl.col("multi").fill_null(False).sum().alias("multi_article_cluster_urls"),
        )
        .sort(["urls", "domain"], descending=[True, False])
        .head(50)
        .with_columns(pl.lit(name).alias("run"))
    )
    strata.append(by_domain)
    print(name, comparisons[-1], flush=True)
pl.DataFrame(comparisons).write_csv(output / "comparison-derived.csv")
pl.DataFrame(story_rows).write_csv(output / "story-diagnostics.csv")
pl.concat(strata).write_csv(output / "domain-strata-top50.csv")
pairs = pl.scan_parquet(root / "article_pairs.parquet")
checks = (
    pairs.select(
        pl.len().alias("pairs"),
        (pl.col("left") < pl.col("right")).all().alias("ordered_no_self_edges"),
        pl.all_horizontal(
            pl.col("event_score", "url_score", "combined").is_finite()
            & pl.col("event_score", "url_score", "combined").is_between(0, 1)
        )
        .all()
        .alias("scores_valid"),
        ((pl.col("combined") - (pl.col("event_score") + pl.col("url_score")) / 2).abs() < 1e-12)
        .all()
        .alias("equal_weights_valid"),
        ((pl.col("url_score") >= 0.6) & (pl.col("event_score") == 0))
        .sum()
        .alias("combined_03_edges_with_no_event_overlap"),
    )
    .collect()
    .row(0, named=True)
)
assert checks["ordered_no_self_edges"] and checks["scores_valid"] and checks["equal_weights_valid"]
assert checks["pairs"] == audit["article_candidates"]["candidate_pairs"]
for summary in sweep["summaries"]:
    if summary["run"].startswith("article-"):
        edge_count = (
            pairs.filter(pl.col(summary["metric"]) >= summary["threshold"])
            .select(pl.len())
            .collect()
            .item()
        )
        assert edge_count == summary["graph_edges"]
metadata = {
    "keyword_totals": totals,
    "idai_specific_regex": r"(^|[^a-z])idai([^a-z]|$)",
    "specificity_note": "Post hoc diagnostic, not independent labels or a retuned model.",
    "pair_checks": checks,
    "translation_metadata_nonempty_rows": audit["mentions"]["translated_rows"],
    "language_stratification": (
        "Unavailable: all translation fields blank; language not inferred from host."
    ),
    "accuracy_metrics": {
        "candidate_recall": None,
        "pair_precision": None,
        "pair_recall": None,
        "bcubed": None,
        "ceafe": None,
        "reason": "No independent and sufficiently complete same-story labels.",
    },
    "canonical_weighting": (
        "A wire group matches if any URL alias matches; it is counted once per cluster."
    ),
    "all_13_assignment_and_edge_checks_passed": True,
}
(output / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
