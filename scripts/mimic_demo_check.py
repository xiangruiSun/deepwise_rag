"""Sanity check of structured retrieval on the MIMIC-IV demo (run from backend/).

    INSPECT_RAG_APP_CONFIG=mimic_demo.yaml python ../scripts/mimic_demo_check.py [output.json]

The demo has no relevance labels, so this is NOT a retrieval benchmark.  It reports
(1) how often laboratory similarity is available, and (2) a weak proxy: do the stays
retrieved by laboratory similarity ALONE share more discharge-diagnosis categories with
the query stay than an average corpus stay does?  Diagnosis codes are not used for the
laboratory-only ranking, so the proxy is not circular — but shared ICD categories are a
crude stand-in for clinical similarity, and 37 query stays give wide uncertainty.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict

from sqlalchemy import select

from app.core.container import build_context
from app.core.security import AccessPolicy, Principal, Role
from app.db.models import ClinicalCase, ClinicalEvent
from app.schemas.search import SearchRequest

TOP_K = 5


def category(code: str) -> str:
    system, _, value = code.partition("/")
    return f"{system}/{value.replace('.', '')[:3]}"


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    ctx = build_context()
    if ctx.config.dataset.name != "mimic_iv_demo":
        raise SystemExit("run with INSPECT_RAG_APP_CONFIG=mimic_demo.yaml")
    policy = AccessPolicy(Principal("check", Role.RESEARCHER, frozenset({"research"})))
    with ctx.session_factory() as session:
        cases = list(session.scalars(select(ClinicalCase)))
        dx: dict[str, set[str]] = defaultdict(set)
        rows = session.execute(select(ClinicalEvent.patient_id, ClinicalEvent.event_time, ClinicalEvent.concept_code)
                               .where(ClinicalEvent.event_type == "diagnosis"))
        by_stay: dict[tuple[str, object], set[str]] = defaultdict(set)
        for patient_id, event_time, code in rows:
            by_stay[(patient_id, event_time)].add(category(code))
        for case in cases:  # diagnoses recorded at this stay's discharge
            dx[case.case_id] = by_stay.get((case.patient_id, case.procedure_time), set())
        splits = {c.case_id: c.dataset_split for c in cases}
        patients = {c.case_id: c.patient_id for c in cases}
    corpus = [cid for cid, split in splits.items() if split == "train"]
    queries = sorted(cid for cid, split in splits.items() if split in ("valid", "test"))

    out: dict[str, object] = {"dataset": "mimic_iv_demo", "corpus_stays": len(corpus), "query_stays": len(queries),
                              "top_k": TOP_K, "recipes": {}}
    rng = random.Random(13)
    for recipe in ("structured_lab", "structured_dx", "structured_lab_dx"):
        with_results, lab_available, overlaps, retrieved_j, corpus_j, random_j = 0, 0, [], [], [], []
        same_patient = 0
        for case_id in queries:
            request = SearchRequest(query_mode="case", case_id=case_id, top_k=TOP_K, retrieval_recipe_id=recipe,
                                    candidate_splits=["train"])
            response = ctx.retrieval.search(request, policy)
            if not response.results:
                continue
            with_results += 1
            lab_available += any(c.name == "laboratory" and c.available for c in response.results[0].components)
            overlaps.extend(len(r.laboratory_comparison or []) for r in response.results)
            same_patient += sum(patients[r.case_id] == patients[case_id] for r in response.results)
            retrieved_j.append(statistics.mean(jaccard(dx[case_id], dx[r.case_id]) for r in response.results))
            corpus_j.append(statistics.mean(jaccard(dx[case_id], dx[c]) for c in corpus))
            random_j.append(statistics.mean(jaccard(dx[case_id], dx[c]) for c in rng.sample(corpus, TOP_K)))
        out["recipes"][recipe] = {  # type: ignore[index]
            "queries_with_results": with_results,
            "queries_with_laboratory_similarity": lab_available,
            "mean_overlapping_lab_concepts_in_results": round(statistics.mean(overlaps), 2) if overlaps else None,
            "results_from_the_query_patient": same_patient,
            "mean_dx_category_jaccard_top_k": round(statistics.mean(retrieved_j), 4) if retrieved_j else None,
            "mean_dx_category_jaccard_random_k": round(statistics.mean(random_j), 4) if random_j else None,
            "mean_dx_category_jaccard_all_corpus": round(statistics.mean(corpus_j), 4) if corpus_j else None,
            "queries_where_top_k_beats_corpus_mean": sum(r > c for r, c in zip(retrieved_j, corpus_j, strict=True)),
        }
    out["note"] = ("Sanity check only. No relevance labels exist for this dataset; shared ICD categories are a "
                   "crude proxy. structured_dx and structured_lab_dx rank BY diagnosis overlap, so their "
                   "diagnosis agreement is circular and shown only for reference.")
    text = json.dumps(out, indent=2)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
