"""Draw a stratified sample of *hard* document pairs for manual labelling.

Strata are chosen where the system is most likely to be wrong: cross-language
pairs the clusterer merged, near-gate merges, sibling incidents inside one family,
strong-title pairs it kept apart, shared-GlobalEventID pairs it kept apart, and a
small easy-negative control. Output is JSONL with one pair per line and an empty
``label`` to fill in with one of ``same_event`` / ``related`` / ``different``.

    uv run python scripts/sample_eval_pairs.py --clusters data/clusters/20230206_fam \
        --features data/features/20230206 --output eval/pairs_20230206.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

STRATA: list[tuple[str, pl.Expr, int]] = [
    (
        "merged_cross_language",
        (pl.col("incident_l") == pl.col("incident_r"))
        & (pl.col("language_l") != pl.col("language_r"))
        & (pl.col("title_score") < 0.6),
        60,
    ),
    (
        "merged_near_gate",
        (pl.col("incident_l") == pl.col("incident_r")) & (pl.col("gated") < 0.4),
        50,
    ),
    (
        "sibling_incidents",
        (pl.col("incident_l") != pl.col("incident_r")) & (pl.col("family_l") == pl.col("family_r")),
        60,
    ),
    (
        "split_strong_title",
        (pl.col("family_l") != pl.col("family_r")) & (pl.col("title_score") >= 0.5),
        60,
    ),
    (
        "split_shared_event",
        (pl.col("family_l") != pl.col("family_r")) & pl.col("from_event"),
        40,
    ),
    (
        "easy_negative",
        (pl.col("family_l") != pl.col("family_r"))
        & pl.col("from_title")
        & (pl.col("title_score") < 0.2),
        30,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    docs = pl.read_parquet(
        args.features / "documents.parquet",
        columns=["document_id", "title", "language", "domain", "publisher_country", "first_seen"],
    ).filter(pl.col("title").is_not_null())
    members = pl.read_parquet(args.clusters / "incident_memberships.parquet").filter(
        pl.col("is_primary")
    )
    docs = docs.join(members.select("document_id", "incident_id", "family_id"), on="document_id")

    def side(suffix: str) -> pl.DataFrame:
        return docs.rename({c: f"{c}_{suffix}" for c in docs.columns})

    pairs = (
        pl.scan_parquet(args.clusters / "pair_features.parquet")
        .join(side("l").lazy(), left_on="left", right_on="document_id_l")
        .join(side("r").lazy(), left_on="right", right_on="document_id_r")
        .filter(pl.col("title_l") != pl.col("title_r"))
        .rename({"incident_id_l": "incident_l", "incident_id_r": "incident_r"})
        .rename({"family_id_l": "family_l", "family_id_r": "family_r"})
    )
    frames = []
    for name, expr, n in STRATA:
        frame = pairs.filter(expr).collect()
        frames.append(
            frame.sample(min(n, frame.height), seed=args.seed).with_columns(
                pl.lit(name).alias("stratum")
            )
        )
        print(f"{name}: {frame.height} eligible, {min(n, frame.height)} drawn")
    sample = pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(sample.iter_rows(named=True)):
            record = {
                "pair_id": i,
                "stratum": row["stratum"],
                "label": "",
                "left": {
                    "document_id": row["left"],
                    "title": row["title_l"],
                    "language": row["language_l"],
                    "domain": row["domain_l"],
                    "publisher_country": row["publisher_country_l"],
                    "observed": str(row["first_seen_l"]),
                    "incident_id": row["incident_l"],
                    "family_id": row["family_l"],
                },
                "right": {
                    "document_id": row["right"],
                    "title": row["title_r"],
                    "language": row["language_r"],
                    "domain": row["domain_r"],
                    "publisher_country": row["publisher_country_r"],
                    "observed": str(row["first_seen_r"]),
                    "incident_id": row["incident_r"],
                    "family_id": row["family_r"],
                },
                "system": {
                    "same_incident": row["incident_l"] == row["incident_r"],
                    "same_family": row["family_l"] == row["family_r"],
                    "title_score": row["title_score"],
                    "event_score": row["event_score"],
                    "url_score": row["url_score"],
                    "entity_score": row["entity_score"],
                    "gated": row["gated"],
                    "evidence_channels": row["evidence_channels"],
                    "delta_hours": row["delta_hours"],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {sample.height} pairs to {args.output}")


if __name__ == "__main__":
    main()
