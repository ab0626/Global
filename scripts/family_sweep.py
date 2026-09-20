"""Incident -> family sweep on a finished cluster run.

The document gate and incident Leiden are *not* recomputed: the incident assignment
of ``--clusters`` is fixed and only ``link_families`` runs per configuration, so each
configuration takes seconds instead of the ~8 min full partition. Every configuration
is scored on the labelled pairs / neighbourhoods with the same precision-first objective
as ``eval_sweep.py`` (incident-level metrics are identical across rows by construction).

    uv run python scripts/family_sweep.py --features data/features/20230206 \
        --embeddings data/embeddings/20230206 --clusters data/clusters/20230206_v5 \
        --labels-pairs eval/pairs_20230206.jsonl \
        --labels-neighborhoods eval/neighborhoods_20230206.jsonl \
        --output data/sweeps/20230206/family
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_sweep import summarize  # noqa: E402

from attention.cluster import ClusterSettings, link_families, load_inputs  # noqa: E402
from attention.evaluate import evaluate  # noqa: E402

GRID: list[dict] = [
    {
        "family_title_threshold": ft,
        "family_strong_title": fs,
        "family_max_hours": fh,
        "family_corroboration": fc,
        "family_resolution": fr,
    }
    for ft in (0.3, 0.4)
    for fs in (0.6, 0.7)
    for fh in (48.0, 96.0)
    for fc in ("entity,event", "entity,event,geo")
    for fr in (0.3, 0.5)
]


def incident_vector(clusters: Path, count: int) -> np.ndarray:
    members = pl.read_parquet(clusters / "incident_memberships.parquet").filter(
        pl.col("is_primary")
    )
    incident = np.full(count, -1, dtype=np.int64)
    incident[members["document_id"].to_numpy()] = members["incident_id"].to_numpy()
    return incident


def write_run(
    source: Path, out: Path, family_of_incident: np.ndarray, settings: ClusterSettings
) -> None:
    """``source`` with family ids replaced: memberships, incidents and run.json."""
    out.mkdir(parents=True, exist_ok=True)
    families = pl.DataFrame(
        {
            "incident_id": np.arange(len(family_of_incident), dtype=np.int64),
            "new_family": family_of_incident,
        }
    )
    for name in ("incident_memberships", "incidents"):
        frame = pl.read_parquet(source / f"{name}.parquet")
        frame.join(families, on="incident_id", how="left").with_columns(
            pl.col("new_family").fill_null(-1).alias("family_id")
        ).drop("new_family").select(frame.columns).write_parquet(out / f"{name}.parquet")
    audit = json.loads((source / "run.json").read_text())
    audit["settings"] = dataclasses.asdict(settings)
    audit["families"] = int(len(np.unique(family_of_incident[family_of_incident >= 0])))
    audit["family_stage_only_from"] = str(source)
    (out / "run.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    link = out / "candidate_pairs.parquet"
    link.unlink(missing_ok=True)
    link.symlink_to((source / "candidate_pairs.parquet").resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True, help="finished cluster run")
    parser.add_argument("--labels-pairs", type=Path, required=True)
    parser.add_argument("--labels-neighborhoods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads((args.clusters / "run.json").read_text())["settings"]
    known = ClusterSettings.__dataclass_fields__
    settings0 = ClusterSettings(**{k: v for k, v in base.items() if k in known})
    documents, links, title, _ = load_inputs(args.features, args.embeddings, settings0)
    incident = incident_vector(args.clusters, documents.height)

    rows = []
    for i, overrides in enumerate(GRID):
        settings = dataclasses.replace(settings0, **overrides)
        started = perf_counter()
        family_of_incident = link_families(incident, documents, title, settings, links)
        out = args.output / f"{i:02d}"
        write_run(args.clusters, out, family_of_incident, settings)
        report = evaluate(out, args.labels_pairs, args.labels_neighborhoods)
        (out / "eval.json").write_text(json.dumps(report, indent=2))
        (out / "candidate_pairs.parquet").unlink()
        row = {
            "run": f"family/{i:02d}",
            **overrides,
            "families": int(len(np.unique(family_of_incident[family_of_incident >= 0]))),
            **summarize(report),
        }
        row["seconds"] = perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)

    table = pl.DataFrame(rows).sort("objective", descending=True)
    args.output.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.output / "family.csv")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3):
        print(table)


if __name__ == "__main__":
    main()
