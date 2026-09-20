"""Export the hardest first-pass labelled pairs as a CSV for a human second pass.

Hardness ranks pairs where a second opinion changes the evaluation most:
``related`` labels (the ambiguous middle class), pairs where the system's
family decision disagrees with the label, cross-language pairs, and pairs whose
gated score sits near the merge threshold. Reviewers fill ``review_label`` with
``same_event`` / ``related`` / ``different`` and optionally ``review_note``.

    uv run python scripts/export_review_csv.py --pairs eval/pairs_20230206.jsonl \
        --output eval/review_20230206.csv --limit 150
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = ("same_event", "related", "different")


def hardness(row: dict) -> float:
    label = row["label"]
    system = row["system"]
    score = 0.0
    if label == "related":
        score += 3.0
    system_same = bool(system["same_family"])
    if (label == "same_event") != system_same:
        score += 2.0
    if row["left"]["language"] != row["right"]["language"]:
        score += 1.0
    score += max(0.0, 1.0 - abs(float(system["gated"]) - 0.4) / 0.4)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("label") in LABELS]
    rows.sort(key=lambda r: (-hardness(r), r["pair_id"]))
    chosen = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pair_id",
                "stratum",
                "first_pass_label",
                "review_label",
                "review_note",
                "left_title",
                "left_language",
                "left_domain",
                "left_publisher_country",
                "left_observed",
                "right_title",
                "right_language",
                "right_domain",
                "right_publisher_country",
                "right_observed",
                "system_same_incident",
                "system_same_family",
                "title_score",
                "gated_score",
                "delta_hours",
            ]
        )
        for r in chosen:
            left, right, system = r["left"], r["right"], r["system"]
            writer.writerow(
                [
                    r["pair_id"],
                    r["stratum"],
                    r["label"],
                    "",
                    "",
                    left["title"],
                    left["language"],
                    left["domain"],
                    left["publisher_country"],
                    left["observed"],
                    right["title"],
                    right["language"],
                    right["domain"],
                    right["publisher_country"],
                    right["observed"],
                    system["same_incident"],
                    system["same_family"],
                    f"{float(system['title_score']):.3f}",
                    f"{float(system['gated']):.3f}",
                    f"{float(system['delta_hours']):.2f}",
                ]
            )
    print(f"wrote {len(chosen)} pairs to {args.output}")


if __name__ == "__main__":
    main()
