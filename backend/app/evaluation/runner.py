"""ExperimentRunner (spec section 29).

For every experiment of a suite the runner: validates the dataset fingerprint,
resolves model revisions, verifies (or builds) compatible indexes, executes all
evaluation queries, stores ranked case lists, computes retrieval metrics,
optionally runs the generation evaluation, records performance, evaluates the
safety gates and persists configuration + results (files, database, optional
MLflow).  Runs are keyed by a configuration hash, so resuming never mixes
incompatible configurations.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import (
    REPO_ROOT,
    AppConfig,
    EvaluationConfig,
    ExperimentRunSpec,
    ExperimentSuite,
    RetrievalRecipe,
    resolve_suite,
)
from app.core.container import AppContext
from app.core.errors import AppError, ConfigurationError, IndexCompatibilityError
from app.core.hashing import file_sha256, sha256_text, stable_hash
from app.core.security import AccessPolicy, system_principal
from app.db.base import session_scope
from app.db.models import ClinicalCase, ExperimentArtifact, ExperimentMetric, ExperimentRun, utcnow
from app.evaluation.datasets import EvalQuery, Qrels, QuerySet, read_jsonl, resolve_path, write_jsonl
from app.evaluation.generation_metrics import aggregate_generation, evaluate_answer
from app.evaluation.retrieval_metrics import (
    METRIC_DEFINITIONS,
    aggregate,
    evaluate_query,
    percentile_summary,
)
from app.indexing.indexer import LEXICAL_KEY, Indexer, report_index_name
from app.observability.metrics import (
    get_logger,
    gpu_memory_mb,
    hardware_description,
    process_memory_mb,
)
from app.rag.pipeline import RagPipeline
from app.retrieval.features import CaseFeatureService
from app.schemas.rag import RagRequest
from app.schemas.search import LabFilter, SearchRequest, SearchResponse

log = get_logger("experiments")


@dataclass
class RunResult:
    run_id: str
    experiment_id: str
    status: str
    config_hash: str
    run_dir: str
    stage: str
    data_kind: str
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def git_commit() -> str | None:
    """Current commit hash read directly from ``.git`` (no subprocess needed)."""
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


class ExperimentRunner:
    def __init__(self, ctx: AppContext, output_dir: Path | None = None) -> None:
        self.ctx = ctx
        self.output_dir = Path(output_dir) if output_dir else ctx.settings.experiment_dir
        scope = ctx.config.dataset.default_access_scope
        self.policy = AccessPolicy(system_principal([scope]))
        self._case_meta: dict[str, tuple[str, str | None, str]] | None = None

    # ------------------------------------------------------------------ helpers
    def _fingerprint(self) -> str:
        if self.ctx.dataset_fingerprint:
            return self.ctx.dataset_fingerprint
        manifest = (self.ctx.settings.artifact_dir / "ingestion" / self.ctx.config.dataset.name
                    / "data_manifest.json")
        if not manifest.exists():
            raise ConfigurationError(
                "no data manifest found: ingest the dataset before running experiments", {"expected": str(manifest)}
            )
        fingerprint = json.loads(manifest.read_text(encoding="utf-8")).get("dataset_fingerprint")
        if not fingerprint:
            raise ConfigurationError("data manifest has no dataset fingerprint")
        self.ctx.dataset_fingerprint = fingerprint
        return fingerprint

    def _cases(self) -> dict[str, tuple[str, str | None, str]]:
        if self._case_meta is None:
            with self.ctx.session_factory() as session:
                self._case_meta = {
                    c.case_id: (c.patient_id, c.dataset_split, c.access_scope)
                    for c in session.scalars(select(ClinicalCase))
                }
        return self._case_meta

    def _ensure_indexes(self, recipe: RetrievalRecipe, build_missing: bool) -> dict[str, Any]:
        """Verify that the indexes a recipe needs exist and were built with the configured
        embedding identity; optionally build missing ones."""
        prefix = self.ctx.config.index.report_index_prefix
        indexer = Indexer(self.ctx.session_factory, self.ctx.backend, self.ctx.config, self.ctx.embeddings,
                          "experiment-runner", self._fingerprint())
        needed: list[tuple[str | None, str]] = []
        if "bm25" in recipe.branches:
            needed.append((None, recipe.chunking))
        if "dense" in recipe.branches:
            needed.append((recipe.embedding, recipe.chunking))
        versions: dict[str, Any] = {}
        for embedding_key, chunking_key in needed:
            name = report_index_name(prefix, embedding_key, chunking_key)
            if not self.ctx.backend.index_exists(name):
                if not build_missing:
                    raise IndexCompatibilityError(f"required index '{name}' does not exist")
                indexer.sync_report_index(embedding_key, chunking_key)
            spec = self.ctx.backend.index_spec(name)
            expected = (sha256_text(self.ctx.embeddings.provider(embedding_key).identity())
                        if embedding_key else LEXICAL_KEY)
            if spec.meta.get("embedding_identity") != expected:
                raise IndexCompatibilityError(
                    f"index '{name}' was built with a different embedding identity than the configured model"
                )
            stats = self.ctx.backend.stats(name)
            versions[name] = {"documents": stats.get("documents"), "size_bytes": stats.get("size_bytes"),
                              "embedding_identity": expected, "mapping_version": spec.mapping_version}
        return versions

    def _request(self, query: EvalQuery, recipe: RetrievalRecipe, top_k: int) -> SearchRequest:
        f = query.filters
        return SearchRequest(
            query_mode=query.query_mode,  # type: ignore[arg-type]
            case_id=query.case_id,
            query_text=query.query_text,
            age_min=f.get("age_min"),
            age_max=f.get("age_max"),
            gender=f.get("gender"),
            diagnosis_codes=f.get("diagnosis_codes", []),
            lab_filters=[LabFilter(**lab) for lab in f.get("lab_filters", [])],
            # the candidate corpus is the training split; query patients are always excluded
            candidate_splits=[self.ctx.config.dataset.corpus_split],  # type: ignore[list-item]
            top_k=top_k,
            retrieval_recipe_id=recipe.recipe_id,
            exclude_same_patient=True,
        )

    def _gate_checks(self, query: EvalQuery, response: SearchResponse) -> dict[str, bool]:
        cases = self._cases()
        corpus = self.ctx.config.dataset.corpus_split
        query_patient = cases[query.case_id][0] if query.case_id and query.case_id in cases else None
        split_ok, auth_ok = True, True
        if query.case_id and query.case_id in cases and cases[query.case_id][1] == corpus:
            split_ok = False  # a query patient must never come from the candidate corpus
        for result in response.results:
            patient_id, split, scope = cases[result.case_id]
            if split != corpus or (query_patient is not None and patient_id == query_patient):
                split_ok = False
            if not self.policy.allows(scope):
                auth_ok = False
        return {"split_isolation": split_ok, "authorization_isolation": auth_ok}

    def _temporal_gate(self, queries: list[EvalQuery], recipe: RetrievalRecipe) -> bool:
        """No query-time feature may originate after the query case's procedure time."""
        lab_cfg = recipe.multimodal.laboratory
        case_ids = [q.case_id for q in queries if q.case_id]
        with self.ctx.session_factory() as session:
            contexts = CaseFeatureService(session).contexts(case_ids, lab_cfg.lookback_days, lab_cfg.aggregate,
                                                            with_reports=False)
            for ctx in contexts.values():
                cutoff = ctx.case.procedure_time
                if any(e.event_time > cutoff for e in ctx.diagnosis_events):
                    return False
                if ctx.labs and any(f.event_time > cutoff for f in ctx.labs.features.values()):
                    return False
        return True

    # ---------------------------------------------------------------- suite run
    def run_suite(self, suite: ExperimentSuite, resume: bool = True, force: bool = False,
                  build_missing_indexes: bool = True, allow_test: bool = False,
                  only: list[str] | None = None) -> dict[str, Any]:
        app_config, eval_cfg = resolve_suite(suite, self.ctx.settings.config_dir)
        if app_config.dataset.name != self.ctx.config.dataset.name:
            raise ConfigurationError("suite base_config uses a different dataset than the application context")
        if not eval_cfg.query_set or not eval_cfg.qrels:
            raise ConfigurationError("evaluation.query_set and evaluation.qrels must be configured")
        fingerprint = self._fingerprint()
        query_set = QuerySet.load(eval_cfg.query_set)
        qrels = Qrels.load(eval_cfg.qrels, eval_cfg.label_sources)
        stage = "dev" if suite.stage == "validation" else "test"
        suite_dir = self.output_dir / suite.suite_id
        suite_dir.mkdir(parents=True, exist_ok=True)
        if stage == "test":
            self._guard_test_stage(suite, suite_dir, allow_test, stable_hash(suite.model_dump(mode="json")))
        queries = query_set.for_stage(stage)
        rag = RagPipeline(self.ctx.session_factory, self.ctx.retrieval, app_config, suite.generation_variants)

        results: list[RunResult] = []
        # Free-text query vectors are always computed during evaluation so that query-encoding
        # latency is measured, not served from the cache of an earlier run.
        cache_queries_before = self.ctx.embeddings.cache_queries
        self.ctx.embeddings.cache_queries = False
        try:
            results = self._run_all(suite, app_config, eval_cfg, queries, query_set, qrels, fingerprint,
                                    stage, suite_dir, rag, resume, force, build_missing_indexes, only)
        finally:
            self.ctx.embeddings.cache_queries = cache_queries_before
        leaderboard = build_leaderboard(results)
        summary = {
            "suite_id": suite.suite_id,
            "description": suite.description,
            "stage": suite.stage,
            "data_kind": self.ctx.data_kind,
            "baseline_experiment": suite.baseline_experiment,
            "dataset_fingerprint": fingerprint,
            "query_set_version": query_set.version,
            "qrels_version": qrels.version,
            "label_sources": qrels.label_sources,
            "weak_or_other_label_rows_excluded": qrels.n_excluded,
            "query_composition": query_set.composition(),
            "metric_definitions": METRIC_DEFINITIONS,
            "generated_at": utcnow().isoformat(timespec="seconds"),
            "runs": [r.__dict__ for r in results],
            "leaderboard": leaderboard,
        }
        (suite_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (suite_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2, default=str), encoding="utf-8")
        return summary

    def _run_all(self, suite: ExperimentSuite, app_config: AppConfig, eval_cfg: EvaluationConfig,
                 queries: list[EvalQuery], query_set: QuerySet, qrels: Qrels, fingerprint: str, stage: str,
                 suite_dir: Path, rag: RagPipeline, resume: bool, force: bool, build_missing_indexes: bool,
                 only: list[str] | None) -> list[RunResult]:
        results: list[RunResult] = []
        for spec in suite.runs:
            if only and spec.experiment_id not in only:
                continue
            recipe = app_config.recipes[spec.recipe]
            if (recipe.query_enhancement.method in ("llm_rewrite", "hyde")
                    and self.ctx.retrieval.enhancer.completer is None):
                results.append(RunResult(
                    run_id=f"{spec.experiment_id}-skipped", experiment_id=spec.experiment_id, status="skipped",
                    config_hash="", run_dir="", stage=suite.stage, data_kind=self.ctx.data_kind,
                    error=f"query enhancement '{recipe.query_enhancement.method}' requires a local LLM backend "
                          "(generation.backend); none is configured"))
                continue
            try:
                result = self._run_one(suite, spec, recipe, app_config, eval_cfg, queries, query_set, qrels,
                                       fingerprint, stage, suite_dir, rag, resume, force, build_missing_indexes)
            except AppError as exc:
                log.error("experiment %s failed: %s", spec.experiment_id, exc.message)
                result = RunResult(run_id=f"{spec.experiment_id}-failed", experiment_id=spec.experiment_id,
                                   status="failed", config_hash="", run_dir="", stage=suite.stage,
                                   data_kind=self.ctx.data_kind, error=f"{exc.code}: {exc.message}")
            results.append(result)
        return results

    def _guard_test_stage(self, suite: ExperimentSuite, suite_dir: Path, allow_test: bool, suite_hash: str) -> None:
        """The held-out test set is evaluated ONCE with a frozen configuration."""
        if not allow_test:
            raise ConfigurationError(
                "this suite evaluates the held-out TEST set; re-run with the explicit final-test flag "
                "only after the configuration has been frozen on validation data"
            )
        lock = suite_dir / "TEST_EVALUATION_LOCK.json"
        if lock.exists():
            previous = json.loads(lock.read_text(encoding="utf-8"))
            if previous.get("suite_hash") != suite_hash:
                raise ConfigurationError(
                    "the held-out test set was already evaluated with a different configuration; "
                    "tuning against test results is not permitted", {"lock_file": str(lock)}
                )
        else:
            lock.write_text(json.dumps({"suite_hash": suite_hash, "locked_at": utcnow().isoformat()}, indent=2),
                            encoding="utf-8")

    # ----------------------------------------------------------------- one run
    def _run_one(self, suite: ExperimentSuite, spec: ExperimentRunSpec, recipe: RetrievalRecipe,
                 app_config: AppConfig, eval_cfg: EvaluationConfig, queries: list[EvalQuery],
                 query_set: QuerySet, qrels: Qrels, fingerprint: str, stage: str, suite_dir: Path,
                 rag: RagPipeline, resume: bool, force: bool, build_missing: bool) -> RunResult:
        index_versions = self._ensure_indexes(recipe, build_missing)
        model_revisions: dict[str, Any] = {}
        if recipe.embedding and "dense" in recipe.branches:
            model_revisions["embedding"] = self.ctx.embeddings.provider(recipe.embedding).model_metadata()
        reranker = self.ctx.rerankers.get(recipe.reranker)
        if reranker is not None:
            model_revisions["reranker"] = reranker.metadata()
        generation_cfg = suite.generation_variants.get(spec.generation) if spec.generation else None
        run_config = {
            "suite_id": suite.suite_id, "experiment_id": spec.experiment_id, "stage": suite.stage,
            "recipe": recipe.model_dump(mode="json"), "evaluation": eval_cfg.model_dump(mode="json"),
            "generation": generation_cfg.model_dump(mode="json") if generation_cfg else None,
            "dataset_fingerprint": fingerprint, "query_set_version": query_set.version,
            "qrels_version": qrels.version, "model_revisions": model_revisions,
            "index_versions": {k: v["embedding_identity"] for k, v in index_versions.items()},
            "search_backend": self.ctx.backend.name,
        }
        config_hash = stable_hash(run_config)
        run_id = f"{spec.experiment_id}-{config_hash[:12]}"
        run_dir = suite_dir / spec.experiment_id / config_hash[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_file, per_query_file = run_dir / "metrics.json", run_dir / "per_query.jsonl"
        if metrics_file.exists() and resume and not force:
            saved = json.loads(metrics_file.read_text(encoding="utf-8"))
            if saved.get("status") == "completed" and saved.get("config_hash") == config_hash:
                return RunResult(run_id, spec.experiment_id, "completed", config_hash, str(run_dir),
                                 suite.stage, self.ctx.data_kind, saved["summary"])
        if force:
            per_query_file.unlink(missing_ok=True)
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
        self._db_start(run_id, suite, spec, run_config, config_hash, fingerprint, query_set, qrels,
                       model_revisions, index_versions, eval_cfg)

        # resume: rows already written under the SAME configuration hash are reused
        done: dict[str, dict[str, Any]] = {}
        if per_query_file.exists() and resume:
            done = {row["query_id"]: row for row in read_jsonl(per_query_file)}
        applicable = [q for q in queries if self._applicable(q, recipe)]
        for query in applicable[: eval_cfg.warmup_queries]:  # warm-up (not measured)
            self.ctx.retrieval.search(self._request(query, recipe, eval_cfg.top_k), self.policy, recipe)

        gates: dict[str, bool] = {"split_isolation": True, "authorization_isolation": True}
        started = time.perf_counter()
        with per_query_file.open("a", encoding="utf-8") as fh:
            for query in queries:
                if query.query_id in done:
                    continue
                if not self._applicable(query, recipe):
                    # the recipe cannot represent this query (e.g. image-only recipe, free-text
                    # query): excluded from metric means, counted in applicability coverage
                    row = {"query_id": query.query_id, "category": query.category,
                           "query_mode": query.query_mode, "applicable": False, "ranked_case_ids": [],
                           "scores": [], "n_eligible": None, "n_judged_candidates": 0,
                           "has_judgments": False, "expects_abstention": query.expects_abstention,
                           "metrics": {}, "latency_ms": {}, "modality_coverage": {}, "warnings": [
                               "query not applicable to this recipe"], "gates": {}}
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    done[query.query_id] = row
                    continue
                response = self.ctx.retrieval.search(
                    self._request(query, recipe, eval_cfg.top_k), self.policy, recipe, query.query_id
                )
                ranked = [r.case_id for r in response.results]
                judgments = qrels.for_query(query.query_id)
                checks = self._gate_checks(query, response)
                row = {
                    "query_id": query.query_id, "category": query.category, "query_mode": query.query_mode,
                    "applicable": True,
                    "ranked_case_ids": ranked,
                    "scores": [r.final_score for r in response.results],
                    "n_eligible": response.total_eligible_cases,
                    "n_judged_candidates": len(judgments),
                    "has_judgments": bool(judgments),
                    "expects_abstention": query.expects_abstention,
                    "metrics": evaluate_query(ranked, judgments, eval_cfg.k_values,
                                              eval_cfg.relevance_threshold, eval_cfg.unjudged_policy)
                    if judgments else {"zero_result": 1.0 if not ranked else 0.0},
                    "latency_ms": response.latency_ms,
                    "modality_coverage": response.modality_coverage,
                    "warnings": response.warnings,
                    "gates": checks,
                }
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fh.flush()
                done[query.query_id] = row
        wall_seconds = time.perf_counter() - started
        rows = [done[q.query_id] for q in queries if q.query_id in done]
        for row in rows:
            row_gates: dict[str, bool] = dict(row["gates"])  # type: ignore[call-overload]
            for gate, ok in row_gates.items():
                gates[gate] = gates[gate] and bool(ok)
        gates["temporal_cutoff"] = self._temporal_gate(queries, recipe)
        gates["reproducible_execution"] = self._reproducible(applicable, recipe, eval_cfg, done)

        generation_summary: dict[str, Any] | None = None
        if generation_cfg is not None and spec.generation is not None:
            generation_summary, citation_ok = self._generation_eval(
                run_dir, rag, spec.generation, recipe, eval_cfg, query_set, stage
            )
            gates["citation_validity"] = citation_ok

        summary = self._summarise(spec, recipe, rows, gates, index_versions, model_revisions,
                                  generation_summary, wall_seconds)
        payload = {"status": "completed", "run_id": run_id, "config_hash": config_hash, "summary": summary,
                   "git_commit": git_commit(), "hardware": hardware_description(),
                   "seed": eval_cfg.seed, "finished_at": utcnow().isoformat(timespec="seconds")}
        metrics_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._db_finish(run_id, summary, run_dir)
        self._mlflow(run_id, suite, spec, run_config, summary)
        return RunResult(run_id, spec.experiment_id, "completed", config_hash, str(run_dir), suite.stage,
                         self.ctx.data_kind, summary)

    @staticmethod
    def _applicable(query: EvalQuery, recipe: RetrievalRecipe) -> bool:
        return bool(recipe.branches) or query.query_mode != "text"

    def _reproducible(self, queries: list[EvalQuery], recipe: RetrievalRecipe, eval_cfg: EvaluationConfig,
                      done: dict[str, dict[str, Any]]) -> bool:
        for query in queries[:3]:
            again = self.ctx.retrieval.search(self._request(query, recipe, eval_cfg.top_k), self.policy, recipe)
            if [r.case_id for r in again.results] != done[query.query_id]["ranked_case_ids"]:
                return False
        return True

    def _generation_eval(self, run_dir: Path, rag: RagPipeline, generation_key: str, recipe: RetrievalRecipe,
                         eval_cfg: EvaluationConfig, query_set: QuerySet, stage: str
                         ) -> tuple[dict[str, Any], bool]:
        if not eval_cfg.generation_set:
            raise ConfigurationError("evaluation.generation_set is required for generation experiments")
        items = [i for i in read_jsonl(resolve_path(eval_cfg.generation_set)) if i.get("stage") == stage]
        by_id = {q.query_id: q for q in query_set.queries}
        gen_cfg = rag.variants[generation_key]
        rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, float | None]] = []
        invalid = 0
        for item in items:
            query = by_id[item["query_id"]]
            request = self._request(query, recipe, gen_cfg.max_cases)
            # retrieval is frozen: the same search response feeds every generation variant
            response = self.ctx.retrieval.search(request, self.policy, recipe, query.query_id)
            rag_request = RagRequest(question=item["question"], search=request, generation_key=generation_key)
            answer = rag.answer(rag_request, self.policy, search_response=response)
            bundle = rag.build_bundle(rag_request, self.policy, response) if response.results else None
            metrics = evaluate_answer(answer, bundle, item)
            report = answer.generation_metadata.get("citation_report") or {}
            invalid += len(report.get("invalid_citations") or [])
            metric_rows.append(metrics)
            rows.append({"item_id": item["item_id"], "query_id": item["query_id"], "kind": item.get("kind"),
                         "status": answer.status, "expects_abstention": item.get("expects_abstention"),
                         "n_claims": len(answer.claims()), "n_citations": len(answer.citations),
                         "removed_claims": report.get("removed_claims"), "metrics": metrics})
        write_jsonl(run_dir / "generation.jsonl", rows)
        summary = aggregate_generation(metric_rows)
        summary["invalid_citations"] = invalid
        summary["generation_key"] = generation_key
        summary["backend"] = gen_cfg.backend
        summary["model"] = gen_cfg.model
        summary["prompt_variant"] = gen_cfg.prompt_variant
        summary["post_verification"] = gen_cfg.post_verification
        # with post-verification enabled invalid references are removed before the answer is returned
        final_invalid = sum(1 for r in metric_rows if r.get("citation_correctness") not in (None, 1.0))
        return summary, final_invalid == 0

    # --------------------------------------------------------------- summaries
    def _summarise(self, spec: ExperimentRunSpec, recipe: RetrievalRecipe, rows: list[dict[str, Any]],
                   gates: dict[str, bool], index_versions: dict[str, Any], model_revisions: dict[str, Any],
                   generation: dict[str, Any] | None, wall_seconds: float) -> dict[str, Any]:
        not_applicable = [r for r in rows if not r.get("applicable", True)]
        rows = [r for r in rows if r.get("applicable", True)]
        judged = [r for r in rows if r["has_judgments"]]
        overall = aggregate([r["metrics"] for r in judged])
        by_category: dict[str, Any] = {}
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in judged:
            groups[row["category"]].append(row["metrics"])
        for category, metric_rows in sorted(groups.items()):
            by_category[category] = aggregate(metric_rows)
        stages: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for stage, value in row["latency_ms"].items():
                stages[stage].append(float(value))
        latency = {stage: percentile_summary(values) for stage, values in sorted(stages.items())}
        retrieval_only = [
            float(r["latency_ms"]["total"]) - float(r["latency_ms"].get("result_assembly", 0.0)) for r in rows
        ]
        coverage: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for modality, value in (row.get("modality_coverage") or {}).items():
                coverage[modality].append(float(value))
        mm = recipe.multimodal
        abstain = [r for r in rows if r.get("expects_abstention")]
        return {
            "experiment_id": spec.experiment_id,
            "description": spec.description or recipe.description,
            "recipe_id": recipe.recipe_id,
            "embedding": recipe.embedding if "dense" in recipe.branches else None,
            "retrieval_method": "+".join(recipe.branches) if recipe.branches else "structured",
            "fusion": recipe.fusion.method if len(recipe.branches) > 1 else None,
            "reranker": recipe.reranker.key if recipe.reranker.enabled else None,
            "rerank_depth": recipe.reranker.depth if recipe.reranker.enabled else None,
            "chunking": recipe.chunking,
            "aggregation": recipe.aggregation.method,
            "modalities": list(mm.modalities) if mm.enabled else ["text"],
            "query_enhancement": recipe.query_enhancement.method,
            "n_queries": len(rows),
            "n_queries_not_applicable": len(not_applicable),
            "query_applicability": len(rows) / (len(rows) + len(not_applicable)) if (rows or not_applicable) else None,
            "n_queries_with_judgments": len(judged),
            "n_queries_without_judgments": len(rows) - len(judged),
            "metrics": overall,
            "metrics_by_category": by_category,
            "zero_result_rate": (sum(1 for r in rows if not r["ranked_case_ids"]) / len(rows)) if rows else None,
            "abstention_queries_with_zero_results": (
                sum(1 for r in abstain if not r["ranked_case_ids"]) / len(abstain) if abstain else None
            ),
            "modality_coverage": {m: sum(v) / len(v) for m, v in sorted(coverage.items())},
            "latency_ms": latency,
            "p50_latency_ms": latency.get("total", {}).get("p50"),
            "p95_latency_ms": latency.get("total", {}).get("p95"),
            "p95_retrieval_latency_ms": percentile_summary(retrieval_only)["p95"],
            "wall_seconds": round(wall_seconds, 3),
            "index_versions": index_versions,
            "index_size_bytes": sum(int(v.get("size_bytes") or 0) for v in index_versions.values()),
            "model_revisions": model_revisions,
            "resources": {"process_peak_memory_mb": process_memory_mb(), "gpu_peak_memory_mb": gpu_memory_mb()},
            "gates": gates,
            "generation": generation,
            "faithfulness": generation.get("faithfulness") if generation else None,
            "citation_coverage": generation.get("citation_coverage") if generation else None,
            "tags": spec.tags,
        }

    # ------------------------------------------------------------- persistence
    def _db_start(self, run_id: str, suite: ExperimentSuite, spec: ExperimentRunSpec, run_config: dict[str, Any],
                  config_hash: str, fingerprint: str, query_set: QuerySet, qrels: Qrels,
                  model_revisions: dict[str, Any], index_versions: dict[str, Any],
                  eval_cfg: EvaluationConfig) -> None:
        with session_scope(self.ctx.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                run = ExperimentRun(run_id=run_id, experiment_id=spec.experiment_id, config_hash=config_hash)
                session.add(run)
            run.suite_id = suite.suite_id
            run.status = "running"
            run.stage = suite.stage
            run.data_kind = self.ctx.data_kind
            run.dataset_fingerprint = fingerprint
            run.git_commit = git_commit()
            run.query_set_version = query_set.version
            run.qrels_version = qrels.version
            run.config = run_config
            run.model_revisions = model_revisions
            run.index_versions = index_versions
            run.hardware = hardware_description()
            run.seed = eval_cfg.seed
            run.error = None
            run.started_at = utcnow()
            run.finished_at = None

    def _db_finish(self, run_id: str, summary: dict[str, Any], run_dir: Path) -> None:
        with session_scope(self.ctx.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                return
            run.status = "completed"
            run.finished_at = utcnow()
            run.metrics.clear()
            run.artifacts.clear()
            session.flush()
            for name, value in summary["metrics"].items():
                if isinstance(value, (int, float)) and not name.endswith("__n"):
                    run.metrics.append(ExperimentMetric(name=name, value=float(value), group="overall"))
            for category, metrics in summary["metrics_by_category"].items():
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and not name.endswith("__n"):
                        run.metrics.append(ExperimentMetric(name=name, value=float(value), group=f"category:{category}"))
            for name in ("p50_latency_ms", "p95_latency_ms", "p95_retrieval_latency_ms", "index_size_bytes",
                         "faithfulness", "citation_coverage"):
                if isinstance(summary.get(name), (int, float)):
                    run.metrics.append(ExperimentMetric(name=name, value=float(summary[name]), group="performance"))
            for gate, ok in summary["gates"].items():
                run.metrics.append(ExperimentMetric(name=f"gate:{gate}", value=1.0 if ok else 0.0, group="gates"))
            scope = self.ctx.config.dataset.default_access_scope
            for kind, file_name in (("config", "config.json"), ("metrics", "metrics.json"),
                                    ("per_query", "per_query.jsonl"), ("generation", "generation.jsonl")):
                path = run_dir / file_name
                if path.exists():
                    run.artifacts.append(ExperimentArtifact(kind=kind, path=str(path),
                                                            content_hash=file_sha256(path), access_scope=scope))

    def _mlflow(self, run_id: str, suite: ExperimentSuite, spec: ExperimentRunSpec,
                run_config: dict[str, Any], summary: dict[str, Any]) -> None:
        uri = self.ctx.settings.mlflow_tracking_uri
        if not uri:
            return
        try:
            import mlflow
        except ImportError:
            log.warning("MLflow tracking URI configured but mlflow is not installed")
            return
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(suite.suite_id)
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params({"experiment_id": spec.experiment_id, "recipe": spec.recipe,
                                   "embedding": summary["embedding"], "retrieval_method": summary["retrieval_method"],
                                   "reranker": summary["reranker"], "chunking": summary["chunking"],
                                   "data_kind": self.ctx.data_kind, "config_hash": run_id.rsplit("-", 1)[-1]})
                for name, value in summary["metrics"].items():
                    if isinstance(value, (int, float)) and not name.endswith("__n"):
                        mlflow.log_metric(name.replace("@", "_at_"), float(value))
                for name in ("p50_latency_ms", "p95_latency_ms"):
                    if isinstance(summary.get(name), (int, float)):
                        mlflow.log_metric(name, float(summary[name]))
                mlflow.log_dict(run_config, "config.json")
        except Exception as exc:  # tracking must never fail an experiment
            log.warning("MLflow logging failed: %s", type(exc).__name__)


def build_leaderboard(results: list[RunResult]) -> list[dict[str, Any]]:
    """Machine-readable leaderboard, best primary metric first; failed runs last."""
    rows: list[dict[str, Any]] = []
    for result in results:
        s = result.summary
        metrics = s.get("metrics", {}) if s else {}
        rows.append(
            {
                "experiment_id": result.experiment_id,
                "run_id": result.run_id,
                "evaluation_status": result.status,
                "stage": result.stage,
                "data_kind": result.data_kind,
                "embedding_model": s.get("embedding") if s else None,
                "retrieval_method": s.get("retrieval_method") if s else None,
                "fusion_strategy": s.get("fusion") if s else None,
                "reranker": s.get("reranker") if s else None,
                "chunking": s.get("chunking") if s else None,
                "modality_combination": "+".join(s.get("modalities", [])) if s else None,
                "recall@10": metrics.get("recall@10"),
                "precision@10": metrics.get("precision@10"),
                "ndcg@10": metrics.get("ndcg@10"),
                "judged_coverage@10": metrics.get("judged_coverage@10"),
                "p50_latency_ms": s.get("p50_latency_ms") if s else None,
                "p95_retrieval_latency_ms": s.get("p95_retrieval_latency_ms") if s else None,
                "faithfulness": s.get("faithfulness") if s else None,
                "citation_coverage": s.get("citation_coverage") if s else None,
                "index_size_bytes": s.get("index_size_bytes") if s else None,
                "gates_passed": all(s.get("gates", {}).values()) if s else None,
                "error": result.error,
            }
        )
    rows.sort(key=lambda r: (r["evaluation_status"] != "completed", -(r["ndcg@10"] or -1.0), r["experiment_id"]))
    return rows
