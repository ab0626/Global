"""Qualitative audit of a clustering run's flagship stories.

For each story (found by /search-style title regex over macro-event titles, largest
matching family wins) print: family size, incident count, publisher countries,
languages; a random 25-title sample; the 25 lowest-assignment-score documents;
the riskiest *bridge* documents (in-family documents ranked by ``bridge_risk``:
title-less / generic-title nodes with many cross-incident edges that show little
title similarity or shared-event support — the structure that let boilerplate
pages glue unrelated articles together); the
share of in-family edges that are URL-only / entity-only; and cross-language
assignments. Precision failures hide in the bottom samples, not in the metrics.

    uv run python scripts/inspect_flagship.py \
        --store data/store/20230206_v3 --clusters data/clusters/20230206_v3 \
        --output eval/flagship_20230206_v3.md
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention.cluster import boilerplate_titles  # noqa: E402

STORIES = {
    "Turkey–Syria earthquake": (
        r"(?i)earthquake|quake|erdbeben|terremoto|séisme|deprem|زلزال|地震|землетряс"
    ),
    "Chinese balloon": r"(?i)balloon|ballon|globo|气球|pallone|balão",
    "State of the Union": r"(?i)state of the union|sotu",
    "Grammys": r"(?i)grammy",
    "Ohio train derailment (small control)": r"(?i)east palestine|derail",
}
SAMPLE = 25
BRIDGES = 10
# bridge_risk weights (heuristic ranking, not tuned against labels)
RISK_TITLELESS = 1.0
RISK_GENERIC = 1.0
RISK_DEGREE = 0.25
RISK_SPECIFICITY = 1.0
RISK_CORROBORATION = 1.0


def pick_family(events: pl.DataFrame, pattern: str) -> int | None:
    hits = events.filter(pl.col("title").str.contains(pattern))
    if hits.is_empty():
        return None
    return (
        hits.group_by("family_id")
        .agg(pl.col("raw_documents").sum())
        .sort("raw_documents", descending=True)["family_id"][0]
    )


def titles(frame: pl.DataFrame, n: int, rng: random.Random | None = None) -> list[str]:
    rows = frame.select("title", "language", "source_domain", "assignment_score").to_dicts()
    if rng is not None:
        rows = rng.sample(rows, min(n, len(rows)))
    else:
        rows = rows[:n]
    return [
        f"- `{r['assignment_score']:.2f}` [{r['language'] or '?'}] {r['source_domain']}: "
        f"{(r['title'] or '(no title)')[:110]}"
        for r in rows
    ]


def audit(
    name: str,
    family_id: int,
    docs: pl.DataFrame,
    pairs: pl.LazyFrame,
    rng: random.Random,
    generic: pl.Series,
) -> list[str]:
    fam = docs.filter((pl.col("family_id") == family_id) & pl.col("is_primary"))
    ids = fam["document_id"].implode()
    inc = fam.select("document_id", "incident_id")
    edges = (
        pairs.filter((pl.col("gated") > 0) & pl.col("left").is_in(ids) & pl.col("right").is_in(ids))
        .collect()
        .join(inc.rename({"document_id": "left", "incident_id": "inc_l"}), on="left")
        .join(inc.rename({"document_id": "right", "incident_id": "inc_r"}), on="right")
    )
    cross = edges.filter(pl.col("inc_l") != pl.col("inc_r"))
    url_only = cross.filter(
        (pl.col("url_score") > 0)
        & (pl.col("title_score") == 0)
        & (pl.col("event_score") == 0)
        & (pl.col("entity_score") == 0)
    )
    entity_only = cross.filter(
        (pl.col("entity_score") > 0)
        & (pl.col("title_score") == 0)
        & (pl.col("event_score") == 0)
        & (pl.col("url_score") == 0)
    )
    one_channel = cross.filter(pl.col("evidence_channels") == 1)
    degree = (
        pl.concat(
            [
                cross.select(pl.col("left").alias("document_id"), "title_score", "event_score"),
                cross.select(pl.col("right").alias("document_id"), "title_score", "event_score"),
            ]
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("cross_incident_edges"),
            pl.col("title_score").mean().alias("title_specificity"),
            (pl.col("event_score") > 0).mean().alias("event_corroboration"),
        )
        .join(fam, on="document_id")
        .with_columns(
            pl.col("title").is_null().alias("titleless"),
            pl.col("document_id").is_in(generic).alias("generic_title"),
        )
        .with_columns(
            (
                RISK_TITLELESS * pl.col("titleless").cast(pl.Float64)
                + RISK_GENERIC * pl.col("generic_title").cast(pl.Float64)
                + RISK_DEGREE * (1 + pl.col("cross_incident_edges")).log()
                - RISK_SPECIFICITY * pl.col("title_specificity")
                - RISK_CORROBORATION * pl.col("event_corroboration")
            ).alias("bridge_risk")
        )
        .sort("bridge_risk", descending=True)
    )
    langs = fam.group_by("language").len().sort("len", descending=True)
    out = [
        f"## {name} — family {family_id}",
        "",
        f"- documents {fam.height:,} · incidents {fam['incident_id'].n_unique():,} · "
        f"publisher countries {fam['publisher_country'].drop_nulls().n_unique()} · "
        f"languages {langs.height} (top: "
        + ", ".join(f"{r['language']} {r['len']}" for r in langs.head(6).to_dicts())
        + ")",
        f"- in-family gated edges {edges.height:,}; cross-incident {cross.height:,}; of those "
        f"single-channel {one_channel.height:,} ({one_channel.height / max(cross.height, 1):.1%}), "
        f"URL-only {url_only.height:,}, entity-only {entity_only.height:,}",
        "",
        f"### Random {SAMPLE}",
        *titles(fam, SAMPLE, rng),
        "",
        f"### Lowest assignment score {SAMPLE}",
        *titles(fam.sort("assignment_score"), SAMPLE),
        "",
        f"### Bridge documents (top {BRIDGES} by bridge_risk)",
        f"`bridge_risk = {RISK_TITLELESS}·titleless + {RISK_GENERIC}·generic_title + "
        f"{RISK_DEGREE}·log(1+cross_degree) - {RISK_SPECIFICITY}·mean title_score - "
        f"{RISK_CORROBORATION}·share with shared event` over the document's cross-incident "
        "edges; heuristic ranking, weights untuned. "
        f"{degree['bridge_risk'].gt(0).sum()} of {degree.height:,} bridge documents have risk > 0.",
        *[
            f"- risk {r['bridge_risk']:.2f} · degree {r['cross_incident_edges']} · "
            f"title {r['title_specificity']:.2f} · event {r['event_corroboration']:.0%}"
            f"{' · GENERIC' if r['generic_title'] else ''} · "
            f"[{r['language'] or '?'}] {r['source_domain']}: {(r['title'] or '(no title)')[:100]}"
            for r in degree.head(BRIDGES).to_dicts()
        ],
        "",
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    events = pl.read_parquet(args.store / "macro_events.parquet")
    docs = pl.read_parquet(args.store / "macro_event_documents.parquet")
    pairs = pl.scan_parquet(args.clusters / "pair_features.parquet")
    generic = pl.Series(
        boilerplate_titles(docs.rename({"source_domain": "domain"}).unique("document_id"))
    ).implode()
    rng = random.Random(args.seed)
    lines = [f"# Flagship audit — {args.store}", ""]
    for name, pattern in STORIES.items():
        family_id = pick_family(events, pattern)
        if family_id is None:
            lines += [f"## {name}", "", "- no macro-event title matched", ""]
            continue
        lines += audit(name, family_id, docs, pairs, rng, generic)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
