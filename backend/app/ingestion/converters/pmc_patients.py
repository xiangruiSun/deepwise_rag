"""PMC-Patients (patient-to-patient retrieval, PPR) -> canonical CSV layout.

Source: PMC-Patients-ReCDS, https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS
(CC BY-NC-SA 4.0).  Zhao et al., "A large-scale dataset of patient summaries for
retrieval-based clinical decision support systems", Scientific Data 10, 909 (2023).

The benchmark ships a corpus of patient summaries (one per line, ``_id`` / ``text``),
dev and test query patients, and qrels linking each query patient to similar corpus
patients (derived by the dataset authors from the PubMed citation graph).

What this converter does — and what it deliberately does not do:

* corpus patients become the candidate corpus (split ``train``); dev / test query
  patients become cases of the ``valid`` / ``test`` splits, so the split-isolation
  gate of the evaluation applies unchanged;
* it can draw a **subset**: a seeded sample of query patients, every corpus patient
  judged relevant to one of them, and seeded random distractors up to the requested
  corpus size.  Numbers measured on a subset are NOT comparable with results on the
  full 155k-patient benchmark (fewer distractors make retrieval easier);
* the summaries carry no dates, so every case gets the same placeholder index time and
  no temporal feature is available;
* age and sex are not columns of the benchmark files; they are read from the opening
  of the summary with a conservative pattern and left empty when it does not match;
* nothing is downloaded here.  The raw files must already be on disk.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.hashing import make_case_id

CONVERTER_VERSION = "pmc-patients-1"
PLACEHOLDER_TIME = "2000-01-01 00:00:00"
LABEL_SOURCE = "pmc_patients_ppr"
RAW_FILES = {
    "corpus": "PPR/corpus.jsonl",
    "qrels_dev": "PPR/qrels_dev.tsv",
    "qrels_test": "PPR/qrels_test.tsv",
    "queries_dev": "queries/dev_queries.jsonl",
    "queries_test": "queries/test_queries.jsonl",
}

_AGE_YEARS = re.compile(r"\b(\d{1,3})[- ]?(?:years?|yrs?|y)[- ]?old\b", re.IGNORECASE)
_AGE_MONTHS = re.compile(r"\b(\d{1,2})[- ]?months?[- ]?old\b", re.IGNORECASE)
_FEMALE = re.compile(r"\b(female|woman|girl|lady)\b", re.IGNORECASE)
_MALE = re.compile(r"\b(male|man|boy|gentleman)\b", re.IGNORECASE)


def parse_demographics(text: str) -> tuple[float | None, str | None]:
    """Age (years) and sex from the opening of a summary; ``None`` when not stated plainly."""
    head = text[:300]
    age: float | None = None
    years = _AGE_YEARS.search(head)
    if years and 0 < int(years.group(1)) <= 120:
        age = float(years.group(1))
    else:
        months = _AGE_MONTHS.search(head)
        if months:
            age = round(int(months.group(1)) / 12.0, 2)
    female, male = _FEMALE.search(head), _MALE.search(head)
    gender: str | None = None
    if female and (not male or female.start() <= male.start()):
        gender = "F"
    elif male:
        gender = "M"
    return age, gender


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        if [h.strip().lower() for h in header] != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"{path.name}: unexpected qrels header {header}")
        for query_id, corpus_id, score in reader:
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def prepare_pmc_patients(
    raw_dir: Path,
    output_dir: Path,
    dev_queries: int | None = 200,
    test_queries: int | None = 200,
    corpus_size: int | None = 20_000,
    generation_items: int = 6,
    seed: int = 13,
) -> dict[str, Any]:
    """Write cohort.csv, impressions.csv, eval/ and DATASET.json.  ``None`` keeps everything."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    for key, rel in RAW_FILES.items():
        if not (raw_dir / rel).is_file():
            raise FileNotFoundError(f"PMC-Patients file is missing ({key}): {raw_dir / rel}")
    rng = random.Random(seed)
    qrels = {"dev": _read_qrels(raw_dir / RAW_FILES["qrels_dev"]),
             "test": _read_qrels(raw_dir / RAW_FILES["qrels_test"])}
    limits = {"dev": dev_queries, "test": test_queries}
    chosen: dict[str, list[str]] = {}
    for stage in ("dev", "test"):
        judged = sorted(qrels[stage])
        limit = limits[stage]
        chosen[stage] = judged if limit is None or limit >= len(judged) else sorted(rng.sample(judged, limit))
    relevant = {doc for stage in chosen for q in chosen[stage] for doc in qrels[stage][q]}

    corpus_path = raw_dir / RAW_FILES["corpus"]
    all_ids = [row["_id"] for row in _read_jsonl(corpus_path)]
    missing_relevant = relevant - set(all_ids)
    if corpus_size is None or corpus_size >= len(all_ids):
        keep = set(all_ids)
    else:
        if corpus_size < len(relevant):
            raise ValueError(f"corpus_size {corpus_size} is smaller than the {len(relevant)} judged-relevant patients")
        others = sorted(set(all_ids) - relevant)
        keep = relevant | set(rng.sample(others, corpus_size - len(relevant & set(all_ids))))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval").mkdir(exist_ok=True)
    counts = {"corpus": 0, "dev": 0, "test": 0, "age_parsed": 0, "gender_parsed": 0}
    case_of: dict[str, str] = {}
    with (output_dir / "cohort.csv").open("w", encoding="utf-8", newline="") as cohort_fh, \
            (output_dir / "impressions.csv").open("w", encoding="utf-8", newline="") as text_fh:
        cohort = csv.writer(cohort_fh)
        texts = csv.writer(text_fh)
        cohort.writerow(["patient_id", "procedure_time", "split", "impression_id", "age", "gender"])
        texts.writerow(["impression_id", "impressions"])

        def emit(uid: str, text: str, split: str, bucket: str) -> None:
            age, gender = parse_demographics(text)
            counts[bucket] += 1
            counts["age_parsed"] += age is not None
            counts["gender_parsed"] += gender is not None
            cohort.writerow([uid, PLACEHOLDER_TIME, split, f"pmc_{uid}", "" if age is None else age, gender or ""])
            texts.writerow([f"pmc_{uid}", text])
            case_of[uid] = make_case_id(uid, PLACEHOLDER_TIME)

        for row in _read_jsonl(corpus_path):
            if row["_id"] in keep:
                emit(row["_id"], row["text"], "train", "corpus")
        for stage, split in (("dev", "valid"), ("test", "test")):
            wanted = set(chosen[stage])
            for row in _read_jsonl(raw_dir / RAW_FILES[f"queries_{stage}"]):
                if row["_id"] in wanted:
                    emit(row["_id"], row["text"], split, stage)

    queries: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    for stage in ("dev", "test"):
        for n, uid in enumerate(chosen[stage]):
            if uid not in case_of:
                continue  # a judged query id without a query record
            query_id = f"{stage}_{uid}"
            queries.append({"query_id": query_id, "stage": stage, "category": "G", "query_mode": "case",
                            "case_id": case_of[uid], "filters": {}, "expects_abstention": False})
            for doc, score in sorted(qrels[stage][uid].items()):
                if doc not in case_of:
                    continue
                qrel_rows.append({
                    "query_id": query_id, "query_case_id": case_of[uid], "query_text": None,
                    "candidate_case_id": case_of[doc], "relevance_grade": score, "query_category": "G",
                    "relevant_modalities": ["text"], "reviewer_id": LABEL_SOURCE,
                    "annotation_version": "PMC-Patients-ReCDS/PPR", "confidence": 1.0,
                    "evidence_references": [], "adjudication_status": "published", "label_source": LABEL_SOURCE,
                })
            if n < generation_items:
                generation.append({
                    "item_id": f"gen_{query_id}", "query_id": query_id, "stage": stage, "kind": "comparison",
                    "question": "How does the selected patient compare with the retrieved patients?",
                    "expected_terms": [], "expected_numeric_facts": [], "expects_abstention": False,
                })
            elif n < generation_items + 2:
                generation.append({
                    "item_id": f"gen_{query_id}", "query_id": query_id, "stage": stage, "kind": "insufficient",
                    "question": "Which treatment should this patient receive next?",
                    "abstention_reason": "treatment recommendation", "expected_terms": [],
                    "expected_numeric_facts": [], "expects_abstention": True,
                })
    for name, rows in (("queries.jsonl", queries), ("qrels.jsonl", qrel_rows), ("generation_eval.jsonl", generation)):
        with (output_dir / "eval" / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    subset = corpus_size is not None and counts["corpus"] < len(all_ids)
    info = {
        "dataset": "pmc_patients",
        "prepared": True,
        "converter_version": CONVERTER_VERSION,
        "source": "PMC-Patients-ReCDS, patient-to-patient retrieval (PPR)",
        "source_url": "https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS",
        "license": "CC BY-NC-SA 4.0",
        "citation": "Zhao Z, Jin Q, Chen F, Peng T, Yu S. A large-scale dataset of patient summaries for "
                    "retrieval-based clinical decision support systems. Scientific Data 10, 909 (2023).",
        "seed": seed,
        "full_corpus_size": len(all_ids),
        "corpus_patients": counts["corpus"],
        "dev_query_patients": counts["dev"],
        "test_query_patients": counts["test"],
        "full_dev_queries_with_labels": len(qrels["dev"]),
        "full_test_queries_with_labels": len(qrels["test"]),
        "judged_relevant_patients_in_corpus": len(relevant & set(all_ids)),
        "judged_relevant_patients_missing_from_corpus": len(missing_relevant),
        "qrel_rows": len(qrel_rows),
        "age_parsed": counts["age_parsed"],
        "gender_parsed": counts["gender_parsed"],
        "is_subset": subset,
        "notice": (
            "PMC-Patients (published case-report summaries), not INSPECT. "
            + ("SUBSET of the benchmark: all judged-relevant patients of the sampled queries plus random "
               "distractors; results are not comparable with the full 155k-patient benchmark. " if subset else "")
            + "Relevance labels come from the dataset's PubMed citation graph, not from clinician review. "
              "No dates, laboratory values or images; age and sex are parsed from the summary text."
        ),
    }
    (output_dir / "DATASET.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
