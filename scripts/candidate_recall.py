"""Measure candidate-retrieval (blocking) recall independently of the candidate set.

``attention.evaluate`` reports candidate recall over the labelled pairs, but those
pairs were drawn *from* ``candidate_pairs`` so the number is 1.0 by construction.
This script scores the blocker against positives that never depended on it:

* the neighbourhood documents with a ``gold_story`` label (per-document labels, so
  candidate membership was not a selection criterion);
* the keyword proxy: every titled document whose title matches a story regex from
  ``sample_eval_neighborhoods.NEIGHBORHOODS`` (noisy, includes off-story hits).

Retrieval is top-k (``neighbors=30``) so a 16k-document story has ~128M same-story
pairs but at most ~500k candidates; all-pairs recall is bounded near zero by design
and says nothing about whether the story can be assembled. Leiden only needs the
story to be *connected*, so we report, per story:

* ``doc_reach``: share of gold documents with >= 1 candidate edge to any document
  matching the story regex (overall / same-language / cross-language partner);
* ``component``: over the induced candidate graph on regex-matching documents, the
  share of those documents (and of the gold documents) inside the largest connected
  component -- the ceiling on recall for any clustering run on this candidate set;
* ``pairwise``: all-pairs recall for gold pairs within ``--max-hours``, for reference.

    uv run python scripts/candidate_recall.py --clusters data/clusters/20230206_fam \
        --features data/features/20230206 --neighborhoods eval/neighborhoods_20230206.jsonl \
        --output eval/candidate_recall_20230206.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_eval_neighborhoods import NEIGHBORHOODS  # noqa: E402


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def components(nodes: set[int], edges: pl.DataFrame) -> dict[int, int]:
    """Union-find over ``edges`` restricted to ``nodes``; returns node -> root."""
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in edges.select("left", "right").iter_rows():
        ra, rb = find(left), find(right)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--neighborhoods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=48.0)
    parser.add_argument("--missed-examples", type=int, default=15)
    args = parser.parse_args()

    docs = pl.read_parquet(
        args.features / "documents.parquet",
        columns=["document_id", "title", "language", "first_seen"],
    ).filter(pl.col("title").is_not_null())
    candidates = pl.scan_parquet(args.clusters / "candidate_pairs.parquet").select("left", "right")
    labelled = [
        json.loads(line)
        for line in args.neighborhoods.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_by_story: dict[str, list[dict]] = defaultdict(list)
    for d in labelled:
        if d.get("gold_story") and d["gold_story"] != "other":
            gold_by_story[d["gold_story"]].append(d)

    language = dict(zip(docs["document_id"], docs["language"], strict=True))
    first_seen = dict(zip(docs["document_id"], docs["first_seen"], strict=True))
    report: dict = {
        "clusters": str(args.clusters),
        "max_hours": args.max_hours,
        "stories": {},
    }
    totals: Counter = Counter()

    for story, pattern in NEIGHBORHOODS.items():
        regex_ids = set(docs.filter(pl.col("title").str.contains(pattern))["document_id"])
        gold = gold_by_story.get(story, [])
        gold_ids = {d["document_id"] for d in gold}
        universe = regex_ids | gold_ids
        universe_frame = pl.DataFrame({"node": sorted(universe)}, schema={"node": pl.Int64})
        edges = (
            candidates.join(universe_frame.lazy(), left_on="left", right_on="node")
            .join(universe_frame.lazy(), left_on="right", right_on="node")
            .collect()
        )

        partners: dict[int, set[int]] = defaultdict(set)
        for left, right in edges.iter_rows():
            partners[left].add(right)
            partners[right].add(left)

        reach = Counter()
        unreached: list[dict] = []
        for d in gold:
            doc = d["document_id"]
            hits = partners.get(doc, set()) & regex_ids
            reach["gold"] += 1
            if hits:
                reach["reached"] += 1
            else:
                unreached.append(
                    {"document_id": doc, "title": d["title"], "language": d["language"]}
                )
            if any(language.get(p) == language.get(doc) for p in hits):
                reach["same_language_partner"] += 1
            if any(language.get(p) != language.get(doc) for p in hits):
                reach["cross_language_partner"] += 1

        roots = components(universe, edges)
        sizes = Counter(roots.values())
        largest_root, largest_size = sizes.most_common(1)[0] if sizes else (None, 0)
        gold_in_largest = sum(roots[g] == largest_root for g in gold_ids)

        pair_total = pair_found = pair_cross = pair_cross_found = 0
        pair_set = set(map(tuple, edges.select("left", "right").iter_rows()))
        for a, b in combinations(gold_ids, 2):
            hours = abs((first_seen[a] - first_seen[b]).total_seconds()) / 3600
            if hours > args.max_hours:
                continue
            found = (a, b) in pair_set or (b, a) in pair_set
            cross = language.get(a) != language.get(b)
            pair_total += 1
            pair_found += found
            pair_cross += cross
            pair_cross_found += found and cross

        report["stories"][story] = {
            "regex_documents": len(regex_ids),
            "gold_documents": len(gold_ids),
            "candidate_edges_within_story": edges.height,
            "doc_reach": {
                "reached": ratio(reach["reached"], reach["gold"]),
                "same_language_partner": ratio(reach["same_language_partner"], reach["gold"]),
                "cross_language_partner": ratio(reach["cross_language_partner"], reach["gold"]),
                "unreached_examples": unreached[: args.missed_examples],
            },
            "component": {
                "components": len(sizes),
                "largest_share_of_regex_docs": ratio(largest_size, len(universe)),
                "gold_in_largest": ratio(gold_in_largest, len(gold_ids)),
                "gold_components": len({roots[g] for g in gold_ids}),
            },
            "pairwise": {
                "pairs": pair_total,
                "recall": ratio(pair_found, pair_total),
                "cross_language_recall": ratio(pair_cross_found, pair_cross),
            },
        }
        totals["gold"] += reach["gold"]
        totals["reached"] += reach["reached"]
        totals["gold_in_largest"] += gold_in_largest
        totals["universe"] += len(universe)
        totals["largest"] += largest_size

    report["overall"] = {
        "doc_reach": ratio(totals["reached"], totals["gold"]),
        "gold_in_largest_component": ratio(totals["gold_in_largest"], totals["gold"]),
        "regex_docs_in_largest_component": ratio(totals["largest"], totals["universe"]),
        "note": "pairwise recall is bounded by top-k retrieval; component/reach are the "
        "quantities that limit clustering recall",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for story, s in report["stories"].items():
        print(
            f"{story:20s} regex={s['regex_documents']:6d} gold={s['gold_documents']:3d}"
            f" reach={s['doc_reach']['reached']:.3f}"
            f" largest_cc={s['component']['largest_share_of_regex_docs']:.3f}"
            f" gold_in_cc={s['component']['gold_in_largest']:.3f}"
            f" pairwise={s['pairwise']['recall']:.3f}"
        )
    print("overall", report["overall"])


if __name__ == "__main__":
    main()
