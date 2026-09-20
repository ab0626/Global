"""Draw documents around a few big stories for full manual labelling.

For each neighbourhood we take the story family that best matches its keywords, then
sample (a) members of that family and (b) keyword-matching documents the system put
elsewhere or left unassigned. Each document gets an empty ``gold_story`` (story-level
label, e.g. ``turkey_quake`` or ``other``) and ``gold_incident`` (finer sub-event tag)
to fill in. Documents without a title are skipped: they cannot be judged.

    uv run python scripts/sample_eval_neighborhoods.py --clusters data/clusters/20230206_fam \
        --features data/features/20230206 --output eval/neighborhoods_20230206.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

NEIGHBORHOODS: dict[str, str] = {
    "turkey_quake": (
        r"(?i)earthquake|séisme|seisme|erdbeben|terremoto|deprem|زلزال|землетряс|σεισμ|지진|地震"
    ),
    "china_balloon": (
        r"(?i)balloon|ballon|globo|balão|balon|аэростат|шар-шпион|منطاد|풍선|气球|氣球|μπαλόνι"
    ),
    "state_of_the_union": (
        r"(?i)state of the union|sotu|estado de la unión|état de l'union|zur lage der nation"
    ),
    "grammys": r"(?i)grammy|грэмми|그래미|格莱美|غرامي",
    "zelensky_visit": (
        r"(?i)(zelensk|zelensky|зеленськ|зеленск|zelenski).*"
        r"(london|londres|londra|лондон|brussel|bruxelles|брюссел|paris|parís|париж"
        r"|sunak|parliament|parlement)"
    ),
}
FAMILY_SAMPLE = 40
OUTSIDE_SAMPLE = 20


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

    frames = []
    for name, pattern in NEIGHBORHOODS.items():
        matching = docs.filter(pl.col("title").str.contains(pattern))
        assigned = matching.filter(pl.col("family_id") >= 0)
        if assigned.is_empty():
            print(f"{name}: no keyword hits, skipped")
            continue
        family_id = assigned["family_id"].mode()[0]
        inside = docs.filter(pl.col("family_id") == family_id)
        outside = matching.filter(pl.col("family_id") != family_id)
        drawn = pl.concat(
            [
                inside.sample(min(FAMILY_SAMPLE, inside.height), seed=args.seed).with_columns(
                    pl.lit("family_member").alias("origin")
                ),
                outside.sample(min(OUTSIDE_SAMPLE, outside.height), seed=args.seed).with_columns(
                    pl.lit("keyword_outside_family").alias("origin")
                ),
            ]
        ).with_columns(
            pl.lit(name).alias("neighborhood"), pl.lit(family_id).alias("seed_family_id")
        )
        print(
            f"{name}: family {family_id} ({inside.height} docs); keyword hits:"
            f" {matching.height - outside.height} inside, {outside.height} outside;"
            f" drew {drawn.height}"
        )
        frames.append(drawn)

    sample = pl.concat(frames).sort("neighborhood", "first_seen")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sample.iter_rows(named=True):
            record = {
                "neighborhood": row["neighborhood"],
                "origin": row["origin"],
                "document_id": row["document_id"],
                "title": row["title"],
                "language": row["language"],
                "domain": row["domain"],
                "publisher_country": row["publisher_country"],
                "observed": str(row["first_seen"]),
                "system": {
                    "incident_id": row["incident_id"],
                    "family_id": row["family_id"],
                    "seed_family_id": row["seed_family_id"],
                },
                "gold_story": "",
                "gold_incident": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {sample.height} documents to {args.output}")


if __name__ == "__main__":
    main()
