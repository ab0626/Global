"""Re-rank finished eval_sweep runs (each ``<output>/<stage>/NN/eval.json``) with the
precision-first objective, so a sweep that is still running (or that was started
before the objective existed) can be summarised at any time. With ``--reeval`` the
metrics are recomputed from each run's ``incident_memberships.parquet`` (needed
when ``attention.evaluate`` changed after the sweep started).

    uv run python scripts/rank_sweep.py data/sweeps/20230206_v3 gate \
        --reeval eval/pairs_20230206.jsonl eval/neighborhoods_20230206.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from eval_sweep import STAGES, evaluate, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--reeval", nargs=2, type=Path, metavar=("PAIRS", "NEIGHBORHOODS"))
    args = parser.parse_args()
    rows = []
    for i, overrides in enumerate(STAGES[args.stage]):
        run = args.root / args.stage / f"{i:02d}"
        path = run / "eval.json"
        if not path.exists():
            continue
        if args.reeval:
            report = evaluate(run, *args.reeval)
            path.write_text(json.dumps(report, indent=2))
        else:
            report = json.loads(path.read_text())
        rows.append({"run": f"{args.stage}/{i:02d}", **overrides, **summarize(report)})
    table = pl.DataFrame(rows).sort("objective", descending=True)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3):
        print(table)
    print(f"{len(rows)}/{len(STAGES[args.stage])} runs finished")


if __name__ == "__main__":
    main()
