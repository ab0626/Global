"""Quality proxy for a real clustering run: earthquake keyword purity/recall at the
incident and family level, plus title samples from the largest families.

    uv run python scripts/inspect_run.py data/clusters/<run> data/features/<window>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

QUAKE = (
    r"(?i)earthquake|quake|erdbeben|s[ée]isme|terremoto|sismo|deprem"
    r"|地震|землетрясен|zemětřesení|tremblement|지진|زلزال"
)


def report(frame: pl.DataFrame, column: str) -> None:
    total = int(frame["q"].sum())
    per = (
        frame.filter(pl.col(column) >= 0)
        .group_by(column)
        .agg(pl.len().alias("size"), pl.col("q").sum().alias("hits"))
        .sort("hits", descending=True)
    )
    top = per.row(0, named=True)
    print(
        f"[{column}] groups={per.height} largest={per['size'].max()} "
        f"ge50={int((per['size'] >= 50).sum())} quake-top size={top['size']} "
        f"hits={top['hits']} purity={top['hits'] / top['size']:.2f} "
        f"recall={top['hits'] / total:.2f}"
    )
    print("   quake groups:", per.head(5).to_dicts())
    largest = per.sort("size", descending=True).head(12)
    print("   largest groups:", largest.to_dicts())
    return None


def main() -> None:
    clusters, features = Path(sys.argv[1]), Path(sys.argv[2])
    docs = pl.read_parquet(features / "documents.parquet")
    members = pl.read_parquet(clusters / "incident_memberships.parquet").filter("is_primary")
    frame = docs.join(members, on="document_id", how="left").with_columns(
        pl.col("incident_id").fill_null(-1),
        pl.col("family_id").fill_null(-1),
        pl.col("title").str.contains(QUAKE).fill_null(False).alias("q"),
    )
    report(frame, "incident_id")
    report(frame, "family_id")
    big = (
        frame.filter(pl.col("family_id") >= 0)
        .group_by("family_id")
        .len()
        .sort("len", descending=True)
        .head(int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    )
    for fid, size in big.iter_rows():
        sub = frame.filter(pl.col("family_id") == fid)
        print(
            f"== family {fid} size={size} incidents={sub['incident_id'].n_unique()} "
            f"langs={sub['language'].n_unique()} quake_hits={int(sub['q'].sum())}"
        )
        for row in sub.select("language", "domain", "title").sample(6, seed=1).iter_rows():
            print("     ", row)


if __name__ == "__main__":
    main()
