"""Score a clustering run against hand-labelled real data.

Two label files, both JSONL:

* pairs -- one document pair per line with ``label`` in
  ``same_event`` / ``related`` / ``different`` (see scripts/sample_eval_pairs.py).
  Incident-level positives are ``same_event``; family-level positives are
  ``same_event`` or ``related``.
* neighborhoods -- one document per line with ``gold_story`` (story family label,
  ``other`` for off-topic) and ``gold_incident`` (finer sub-event tag, see
  scripts/sample_eval_neighborhoods.py). ``other`` documents are singletons.

Reported: pairwise precision/recall/F1 at incident and family level (overall and
per sampling stratum), candidate recall, B-cubed and CEAF-e over the labelled
neighbourhood documents, per-neighbourhood family precision/recall, and the
unassigned rate. The sampling is deliberately adversarial (strata where the
system is likeliest to be wrong) so the pair numbers are pessimistic, not
population estimates.

    uv run python -m attention.evaluate --clusters data/clusters/20230206_fam \
        --pairs eval/pairs_20230206.jsonl --neighborhoods eval/neighborhoods_20230206.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

INCIDENT_POSITIVE = {"same_event"}
FAMILY_POSITIVE = {"same_event", "related"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Assignment:
    incident: dict[int, int]
    family: dict[int, int]
    unassigned_rate: float

    @classmethod
    def load(cls, clusters: Path) -> Assignment:
        members = pl.read_parquet(clusters / "incident_memberships.parquet").filter(
            pl.col("is_primary")
        )
        incident = dict(zip(members["document_id"], members["incident_id"], strict=True))
        family = dict(zip(members["document_id"], members["family_id"], strict=True))
        unassigned = (members["incident_id"] < 0).sum() / max(members.height, 1)
        return cls(incident, family, float(unassigned))

    def same(self, level: str, a: int, b: int) -> bool:
        table = self.incident if level == "incident" else self.family
        x, y = table.get(a, -1), table.get(b, -1)
        return x >= 0 and x == y


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pair_metrics(pairs: list[dict], assignment: Assignment, candidates: set[tuple[int, int]]):
    labelled = [p for p in pairs if p.get("label")]
    out: dict = {
        "labelled_pairs": len(labelled),
        "label_counts": dict(Counter(p["label"] for p in labelled)),
    }
    for level, positive in (("incident", INCIDENT_POSITIVE), ("family", FAMILY_POSITIVE)):
        counts: dict[str, Counter] = defaultdict(Counter)
        for p in labelled:
            a, b = p["left"]["document_id"], p["right"]["document_id"]
            gold = p["label"] in positive
            pred = assignment.same(level, a, b)
            key = "tp" if gold and pred else "fp" if pred else "fn" if gold else "tn"
            counts["all"][key] += 1
            counts[p["stratum"]][key] += 1
        out[level] = {
            name: {**prf(c["tp"], c["fp"], c["fn"]), "tn": c["tn"], "n": sum(c.values())}
            for name, c in counts.items()
        }
    positives = [p for p in labelled if p["label"] in FAMILY_POSITIVE]
    found = sum(
        (p["left"]["document_id"], p["right"]["document_id"]) in candidates
        or (p["right"]["document_id"], p["left"]["document_id"]) in candidates
        for p in positives
    )
    out["candidate_recall"] = {
        "positives": len(positives),
        "in_candidates": found,
        "recall": found / len(positives) if positives else None,
        "note": "pairs were drawn from candidate_pairs, so this is 1.0 by construction "
        "unless the run being scored used different retrieval settings",
    }
    return out


def b_cubed(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    n = len(gold)
    precision = recall = 0.0
    for i in range(n):
        same_pred = pred == pred[i]
        same_gold = gold == gold[i]
        both = np.sum(same_pred & same_gold)
        precision += both / same_pred.sum()
        recall += both / same_gold.sum()
    precision /= n
    recall /= n
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def ceaf_e(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Entity-based CEAF (Luo 2005) with phi4 = 2|K∩R| / (|K|+|R|)."""
    gold_sets = [set(np.flatnonzero(gold == g)) for g in np.unique(gold)]
    pred_sets = [set(np.flatnonzero(pred == p)) for p in np.unique(pred)]
    sim = np.zeros((len(gold_sets), len(pred_sets)))
    for i, k in enumerate(gold_sets):
        for j, r in enumerate(pred_sets):
            sim[i, j] = 2 * len(k & r) / (len(k) + len(r))
    rows, cols = linear_sum_assignment(-sim)
    total = sim[rows, cols].sum()
    precision = total / len(pred_sets)
    recall = total / len(gold_sets)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def pairwise_docs(gold: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    iu = np.triu_indices(len(gold), 1)
    g = gold[iu[0]] == gold[iu[1]]
    p = pred[iu[0]] == pred[iu[1]]
    return prf(int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum()))


def encode(values: list) -> np.ndarray:
    codes = {v: i for i, v in enumerate(dict.fromkeys(values))}
    return np.array([codes[v] for v in values])


def neighborhood_metrics(docs: list[dict], assignment: Assignment) -> dict:
    labelled = [d for d in docs if d.get("gold_story")]
    ids = [d["document_id"] for d in labelled]

    def system(level: str) -> np.ndarray:
        table = assignment.incident if level == "incident" else assignment.family
        # unassigned documents are singletons: give each a unique negative id
        return encode(
            [table.get(i, -1) if table.get(i, -1) >= 0 else -(k + 1) for k, i in enumerate(ids)]
        )

    def gold(level: str) -> np.ndarray:
        key = "gold_incident" if level == "incident" else "gold_story"
        return encode(
            [
                d[key] if d["gold_story"] != "other" else f"other:{d['document_id']}"
                for d in labelled
            ]
        )

    out: dict = {"labelled_documents": len(labelled)}
    for level in ("incident", "family"):
        g, p = gold(level), system(level)
        out[level] = {
            "gold_clusters": int(len(np.unique(g))),
            "system_clusters": int(len(np.unique(p))),
            "b_cubed": b_cubed(g, p),
            "ceaf_e": ceaf_e(g, p),
            "pairwise": pairwise_docs(g, p),
        }
    per: dict = {}
    for name in sorted({d["neighborhood"] for d in labelled}):
        rows = [d for d in labelled if d["neighborhood"] == name]
        seed = rows[0]["system"]["seed_family_id"]
        in_family = [d for d in rows if assignment.family.get(d["document_id"], -1) == seed]
        in_story = [d for d in rows if d["gold_story"] == name]
        hit = [d for d in in_family if d["gold_story"] == name]
        per[name] = {
            "seed_family_id": seed,
            "sampled": len(rows),
            "family_precision": len(hit) / len(in_family) if in_family else None,
            "family_recall_within_sample": len(hit) / len(in_story) if in_story else None,
            "gold_incidents": len({d["gold_incident"] for d in in_story}),
            "system_incidents_in_story": len(
                {assignment.incident.get(d["document_id"], -1) for d in in_story}
            ),
        }
    out["per_neighborhood"] = per
    return out


def evaluate(clusters: Path, pairs: Path | None, neighborhoods: Path | None) -> dict:
    assignment = Assignment.load(clusters)
    report: dict = {"clusters": str(clusters), "unassigned_rate": assignment.unassigned_rate}
    if pairs is not None:
        candidate_file = clusters / "candidate_pairs.parquet"
        candidates: set[tuple[int, int]] = set()
        if candidate_file.exists():
            cand = pl.read_parquet(candidate_file, columns=["left", "right"])
            candidates = set(zip(cand["left"], cand["right"], strict=True))
        report["pairs"] = pair_metrics(read_jsonl(pairs), assignment, candidates)
        if not candidate_file.exists():
            report["pairs"]["candidate_recall"] = None
    if neighborhoods is not None:
        report["neighborhoods"] = neighborhood_metrics(read_jsonl(neighborhoods), assignment)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--neighborhoods", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.clusters, args.pairs, args.neighborhoods)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
