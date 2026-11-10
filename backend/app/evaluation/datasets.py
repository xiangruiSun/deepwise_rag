"""Evaluation datasets: queries, graded relevance labels, candidate pooling and
annotation adjudication (spec section 21)."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import REPO_ROOT
from app.core.hashing import stable_hash

QREL_REQUIRED_FIELDS = (
    "query_id", "candidate_case_id", "relevance_grade", "query_category", "relevant_modalities",
    "reviewer_id", "annotation_version", "confidence", "evidence_references", "adjudication_status",
)
VALID_GRADES = {0, 1, 2, 3}
QUERY_CATEGORIES = {
    "A": "Radiology semantics",
    "B": "Demographic and diagnosis constraints",
    "C": "Laboratory characteristics",
    "D": "Multimodal case retrieval",
    "E": "Difficult retrieval cases",
    "F": "Insufficient-evidence queries",
}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"evaluation file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


@dataclass
class EvalQuery:
    query_id: str
    stage: str  # "dev" (validation patients) or "test" (held-out patients)
    category: str
    query_mode: str
    case_id: str | None = None
    query_text: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    expects_abstention: bool = False
    note: str | None = None


@dataclass
class QuerySet:
    queries: list[EvalQuery]
    version: str

    @classmethod
    def load(cls, path: str | Path) -> QuerySet:
        rows = read_jsonl(resolve_path(path))
        queries = []
        for row in rows:
            if row.get("stage") not in ("dev", "test"):
                raise ValueError(f"query {row.get('query_id')}: stage must be 'dev' or 'test'")
            if row.get("category") not in QUERY_CATEGORIES:
                raise ValueError(f"query {row.get('query_id')}: unknown category {row.get('category')}")
            queries.append(
                EvalQuery(
                    query_id=row["query_id"], stage=row["stage"], category=row["category"],
                    query_mode=row["query_mode"], case_id=row.get("case_id"),
                    query_text=row.get("query_text"), filters=row.get("filters") or {},
                    expects_abstention=bool(row.get("expects_abstention")), note=row.get("note"),
                )
            )
        ids = [q.query_id for q in queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query ids in query set")
        return cls(queries, "queries-" + stable_hash(rows, 16))

    def for_stage(self, stage: str) -> list[EvalQuery]:
        return [q for q in self.queries if q.stage == stage]

    def composition(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for q in self.queries:
            out.setdefault(q.stage, Counter())[q.category] += 1  # type: ignore[index]
        return {stage: dict(sorted(c.items())) for stage, c in out.items()}


@dataclass
class Qrels:
    """``by_query[query_id][case_id] -> grade`` for the selected label sources only."""

    by_query: dict[str, dict[str, int]]
    version: str
    label_sources: list[str]
    n_rows: int
    n_excluded: int  # rows from other label sources (e.g. weak labels)

    @classmethod
    def load(cls, path: str | Path, label_sources: list[str]) -> Qrels:
        rows = read_jsonl(resolve_path(path))
        by_query: dict[str, dict[str, int]] = defaultdict(dict)
        excluded = 0
        for i, row in enumerate(rows, start=1):
            missing = [f for f in QREL_REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"qrels row {i}: missing fields {missing}")
            if row.get("label_source", "clinician") not in label_sources:
                excluded += 1  # weak labels never mix silently with reviewed labels
                continue
            grade = row["relevance_grade"]
            if grade is None or grade == "unjudged":
                continue  # explicit "unjudged" rows stay unjudged
            if grade not in VALID_GRADES:
                raise ValueError(f"qrels row {i}: invalid relevance grade {grade!r}")
            previous = by_query[row["query_id"]].get(row["candidate_case_id"])
            if previous is not None and previous != grade:
                raise ValueError(
                    f"qrels row {i}: conflicting grades for ({row['query_id']}, {row['candidate_case_id']}); "
                    "adjudicate duplicate judgments before evaluation"
                )
            by_query[row["query_id"]][row["candidate_case_id"]] = int(grade)
        return cls(dict(by_query), "qrels-" + stable_hash(rows, 16), list(label_sources), len(rows), excluded)

    def for_query(self, query_id: str) -> dict[str, int]:
        return self.by_query.get(query_id, {})


# --------------------------------------------------------------------------- #
# Candidate pooling for blinded annotation
# --------------------------------------------------------------------------- #
def build_annotation_pool(
    runs: dict[str, dict[str, list[str]]],
    depth: int = 20,
    seed: int = 13,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pool the top ``depth`` cases of several systems per query.

    Returns ``(annotation_tasks, provenance, procedure)``.  Tasks are de-duplicated by
    case, shuffled deterministically and carry NO information about which system
    returned a candidate; provenance is written to a separate file that annotators
    must not see.
    """
    tasks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    query_ids = sorted({q for system in runs.values() for q in system})
    for query_id in query_ids:
        contributors: dict[str, list[str]] = defaultdict(list)
        for system in sorted(runs):
            for case_id in runs[system].get(query_id, [])[:depth]:
                contributors[case_id].append(system)
        candidates = sorted(contributors)
        random.Random(f"{seed}:{query_id}").shuffle(candidates)
        for position, case_id in enumerate(candidates, start=1):
            tasks.append(
                {"query_id": query_id, "candidate_case_id": case_id, "presentation_order": position,
                 "relevance_grade": None, "confidence": None, "reviewer_id": None,
                 "evidence_references": [], "adjudication_status": "pending"}
            )
            provenance.append({"query_id": query_id, "candidate_case_id": case_id,
                               "systems": contributors[case_id]})
    procedure = {
        "method": "top-k pooling", "depth": depth, "seed": seed, "systems": sorted(runs),
        "n_queries": len(query_ids), "n_tasks": len(tasks),
        "blinding": "tasks are shuffled per query and contain no system attribution",
    }
    return tasks, provenance, procedure


def adjudicate(annotations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge independent judgments.  Agreement → accepted; disagreement → needs adjudication
    (never averaged silently).  Returns ``(merged_rows, agreement_statistics)``."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        if row.get("relevance_grade") is not None:
            groups[(row["query_id"], row["candidate_case_id"])].append(row)
    merged: list[dict[str, Any]] = []
    pairs: list[tuple[int, int]] = []
    disagreements = 0
    for (_query_id, _case_id), rows in sorted(groups.items()):
        grades = [int(r["relevance_grade"]) for r in rows]
        if len(rows) >= 2:
            pairs.append((grades[0], grades[1]))
        agreed = len(set(grades)) == 1
        disagreements += 0 if agreed else 1
        base = dict(rows[0])
        base.update(
            relevance_grade=grades[0] if agreed else None,
            reviewer_id="+".join(sorted({str(r.get("reviewer_id")) for r in rows})),
            confidence=min(float(r.get("confidence") or 0.0) for r in rows),
            adjudication_status=("agreed" if agreed and len(rows) > 1 else "single_reviewer" if agreed
                                 else "needs_adjudication"),
            individual_grades=grades,
        )
        merged.append(base)
    stats = {
        "n_pairs_double_judged": len(pairs),
        "n_disagreements": disagreements,
        "raw_agreement": (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None,
        "cohen_kappa": _cohen_kappa(pairs),
    }
    return merged, stats


def _cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[g] * second[g] for g in VALID_GRADES) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
