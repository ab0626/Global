"""Sweep gate / Leiden / family settings on saved pair features and score each
configuration against the hand-labelled pairs and neighbourhoods.

Retrieval and pair scoring (the expensive part) are reused from ``--pairs-from``;
only gating, Leiden and family linkage are recomputed per configuration.

    uv run python scripts/eval_sweep.py --features data/features/20230206 \
        --embeddings data/embeddings/20230206 --pairs-from data/clusters/20230206_v3 \
        --labels-pairs eval/pairs_20230206.jsonl \
        --labels-neighborhoods eval/neighborhoods_20230206.jsonl \
        --output data/sweeps/20230206 --stage gate
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention.cluster import ClusterSettings, apply_gate, load_inputs, partition  # noqa: E402
from attention.evaluate import evaluate  # noqa: E402

STAGES: dict[str, list[dict]] = {
    "gate": [
        {"single_channel_title_veto": v, "single_channel_min_features": m, "threshold": t}
        for v in (0.0, 0.15, 0.25)
        for m in (1, 2, 3)
        for t in (0.3, 0.4)
    ],
    "leiden": [{"resolution": r} for r in (0.02, 0.05, 0.1, 0.2)],
    "family": [
        {
            "family_title_threshold": ft,
            "family_entity_threshold": fe,
            "family_strong_title": fs,
            "family_resolution": fr,
        }
        for ft in (0.3, 0.4, 0.5)
        for fe in (0.2, 0.3)
        for fs in (0.6, 0.7)
        for fr in (0.5,)
    ],
}


MAX_UNASSIGNED = 0.12


def objective(row: dict) -> float:
    """Precision-first selection score: false merges are more damaging to Ripple than
    false splits, so family precision dominates and B³ (which rewards keeping huge
    neighbourhoods together) is only one term. Runs above ``MAX_UNASSIGNED`` are
    disqualified (``-inf``)."""
    if row["unassigned"] >= MAX_UNASSIGNED:
        return float("-inf")
    return (
        0.35 * row["pair_fam_p"]
        + 0.20 * row["pair_fam_f1"]
        + 0.20 * row["b3_fam_f1"]
        + 0.15 * row["pair_inc_p"]
        + 0.10 * row["pair_inc_f1"]
    )


def summarize(report: dict) -> dict:
    pairs, hoods = report["pairs"], report["neighborhoods"]
    row = {
        "pair_inc_p": pairs["incident"]["all"]["precision"],
        "pair_inc_r": pairs["incident"]["all"]["recall"],
        "pair_inc_f1": pairs["incident"]["all"]["f1"],
        "pair_fam_p": pairs["family"]["all"]["precision"],
        "pair_fam_r": pairs["family"]["all"]["recall"],
        "pair_fam_f1": pairs["family"]["all"]["f1"],
        "b3_inc_f1": hoods["incident"]["b_cubed"]["f1"],
        "ceafe_inc_f1": hoods["incident"]["ceaf_e"]["f1"],
        "b3_fam_f1": hoods["family"]["b_cubed"]["f1"],
        "ceafe_fam_f1": hoods["family"]["ceaf_e"]["f1"],
        "fam_precision_min": min(
            v["family_precision"] or 0.0 for v in hoods["per_neighborhood"].values()
        ),
        "fam_recall_mean": sum(
            v["family_recall_within_sample"] or 0.0 for v in hoods["per_neighborhood"].values()
        )
        / len(hoods["per_neighborhood"]),
        "unassigned": report["unassigned_rate"],
    }
    row["objective"] = objective(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--pairs-from", type=Path, required=True)
    parser.add_argument("--labels-pairs", type=Path, required=True)
    parser.add_argument("--labels-neighborhoods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="gate")
    parser.add_argument(
        "--base", type=Path, help="JSON of settings overrides applied under every configuration"
    )
    args = parser.parse_args()

    base = json.loads(args.base.read_text()) if args.base else {}
    settings0 = ClusterSettings(**base)
    documents, links, title, _ = load_inputs(args.features, args.embeddings, settings0)
    candidates = pl.read_parquet(args.pairs_from / "candidate_pairs.parquet")
    raw_pairs = pl.read_parquet(args.pairs_from / "pair_features.parquet")
    channels = json.loads((args.pairs_from / "run.json").read_text())["channels"]

    rows = []
    for i, overrides in enumerate(STAGES[args.stage]):
        settings = dataclasses.replace(settings0, **overrides)
        out = args.output / args.stage / f"{i:02d}"
        out.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        pairs = apply_gate(raw_pairs, settings)
        audit = {
            "command": sys.argv,
            "settings": dataclasses.asdict(settings),
            "documents": documents.height,
            "channels": channels,
            "reused_pairs_from": str(args.pairs_from),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            partition(
                candidates,
                pairs,
                documents,
                links,
                title,
                settings,
                out,
                audit,
                started,
                write_pairs=False,
            )
            report = evaluate(out, args.labels_pairs, args.labels_neighborhoods)
        (out / "eval.json").write_text(json.dumps(report, indent=2))
        row = {"run": f"{args.stage}/{i:02d}", **overrides, **summarize(report)}
        row["seconds"] = perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / "graph_edges.parquet").unlink(missing_ok=True)

    table = pl.DataFrame(rows).sort("objective", descending=True)
    table.write_csv(args.output / f"{args.stage}.csv")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3):
        print(table)


if __name__ == "__main__":
    main()
