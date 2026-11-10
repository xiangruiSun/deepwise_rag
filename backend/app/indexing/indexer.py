"""Versioned, incremental index construction (spec sections 7F and 10).

* one report index per (embedding, chunking) pair — embeddings of different
  models never share an index;
* unchanged content is not re-embedded (content hash + embedding cache);
* changed or removed sources are propagated: obsolete documents are deleted
  from the search index and from ``retrieval_documents``, and their cached
  embeddings are invalidated;
* every update is recorded in ``index_manifests`` and the audit log.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import AppConfig
from app.core.errors import IndexCompatibilityError
from app.core.hashing import make_id, sha256_text, stable_hash
from app.db.base import session_scope
from app.db.models import (
    AuditEvent,
    ClinicalCase,
    ImageFeature,
    IndexManifest,
    RadiologyReport,
    RetrievalDocument,
    utcnow,
)
from app.embeddings.base import WhitespaceTokenizer
from app.embeddings.registry import EmbeddingRegistry
from app.imaging.providers import ImageEmbeddingProvider
from app.indexing.backend import VECTOR_FIELD, IndexSpec, SearchBackend
from app.indexing.chunking import Chunk, Chunker, SourceDocument, chunk_statistics
from app.retrieval.features import CaseFeatureService, case_metadata_fields

LEXICAL_KEY = "lexical"
LEXICAL_TOKEN_LIMIT = 100_000


def report_index_name(prefix: str, embedding_key: str | None, chunking_key: str) -> str:
    base = f"{prefix}_{embedding_key or LEXICAL_KEY}"
    return base if chunking_key == "whole_impression" else f"{base}__{chunking_key}"


class Indexer:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        backend: SearchBackend,
        config: AppConfig,
        embeddings: EmbeddingRegistry,
        actor: str = "system",
        dataset_fingerprint: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.backend = backend
        self.config = config
        self.embeddings = embeddings
        self.actor = actor
        self.dataset_fingerprint = dataset_fingerprint

    # ------------------------------------------------------------------ helpers
    def _all_case_metadata(self, session: Session) -> dict[str, dict[str, Any]]:
        service = CaseFeatureService(session)
        case_ids = list(session.scalars(select(ClinicalCase.case_id).order_by(ClinicalCase.case_id)))
        out: dict[str, dict[str, Any]] = {}
        lookback = self.config.dataset.default_lookback_days
        for start in range(0, len(case_ids), 500):
            contexts = service.contexts(case_ids[start : start + 500], lookback)
            for case_id, ctx in contexts.items():
                out[case_id] = case_metadata_fields(ctx)
        return out

    def _manifest(self, session: Session, spec: IndexSpec, stats: dict[str, Any], **fields: Any) -> None:
        backend_stats = self.backend.stats(spec.name)
        manifest = session.get(IndexManifest, spec.name)
        if manifest is None:
            manifest = IndexManifest(index_name=spec.name, kind=spec.kind,
                                     mapping_version=spec.mapping_version)
            session.add(manifest)
        manifest.kind = spec.kind
        manifest.mapping_version = spec.mapping_version
        manifest.dimension = spec.dimension
        manifest.dataset_fingerprint = self.dataset_fingerprint
        manifest.document_count = int(backend_stats.get("documents", 0))
        manifest.size_bytes = backend_stats.get("size_bytes")
        manifest.stats = {**stats, "backend": backend_stats}
        manifest.built_at = utcnow()
        for key, value in fields.items():
            setattr(manifest, key, value)
        session.add(
            AuditEvent(actor=self.actor, action="index.updated", resource_type="index",
                       resource_id=spec.name,
                       details={k: v for k, v in stats.items() if isinstance(v, (int, float, str))})
        )

    # -------------------------------------------------------------- cases index
    def sync_cases_index(self) -> dict[str, Any]:
        name = self.config.index.cases_index
        spec = IndexSpec(name=name, kind="cases", mapping_version=self.config.index.mapping_version)
        with session_scope(self.session_factory) as session:
            metadata = self._all_case_metadata(session)
            self.backend.create_index(spec, recreate=True)
            docs = [{"_id": case_id, **fields} for case_id, fields in metadata.items()]
            for start in range(0, len(docs), 1000):
                self.backend.bulk_index(name, docs[start : start + 1000])
            self.backend.refresh(name)
            stats = {"documents": len(docs)}
            self._manifest(session, spec, stats)
        return {"index": name, **stats}

    # ------------------------------------------------------------ report index
    def sync_report_index(self, embedding_key: str | None, chunking_key: str,
                          recreate: bool = False) -> dict[str, Any]:
        chunk_cfg = self.config.chunking[chunking_key]
        name = report_index_name(self.config.index.report_index_prefix, embedding_key, chunking_key)
        provider = self.embeddings.provider(embedding_key) if embedding_key else None
        embedder = self.embeddings.embedder(embedding_key) if embedding_key else None
        if provider is not None:
            tokenizer = provider.tokenizer()
            prefix_tokens = len(tokenizer.token_spans(provider.config.document_prefix))
            encoder_limit = provider.max_document_content_tokens - prefix_tokens
            identity = sha256_text(provider.identity())
            revision = provider.resolved_revision()
        else:
            tokenizer, encoder_limit, identity, revision = WhitespaceTokenizer(), LEXICAL_TOKEN_LIMIT, LEXICAL_KEY, {}
        spec = IndexSpec(
            name=name,
            kind="reports",
            dimension=provider.embedding_dimension() if provider else None,
            similarity=provider.config.similarity if provider else "cosine",
            mapping_version=self.config.index.mapping_version,
            meta={"embedding_key": embedding_key or LEXICAL_KEY, "embedding_identity": identity,
                  "chunking_key": chunking_key, "chunking_version": chunk_cfg.version},
        )
        if self.backend.index_exists(name) and not recreate:
            existing_identity = self.backend.index_spec(name).meta.get("embedding_identity")
            if existing_identity not in (None, identity):
                raise IndexCompatibilityError(
                    f"index '{name}' was built with a different embedding model/revision/"
                    "preprocessing; rebuild it with recreate=True instead of mixing embeddings"
                )
        self.backend.create_index(spec, recreate=recreate)
        chunker = Chunker(tokenizer, encoder_limit)
        started = time.perf_counter()

        with session_scope(self.session_factory) as session:
            if recreate:
                session.execute(delete(RetrievalDocument).where(RetrievalDocument.index_version == name))
            metadata = self._all_case_metadata(session)
            reports = list(session.scalars(select(RadiologyReport).order_by(RadiologyReport.report_id)))
            desired: dict[str, dict[str, Any]] = {}
            all_chunks: list[Chunk] = []
            for report in reports:
                meta = metadata.get(report.case_id)
                if meta is None:
                    continue
                chunks = chunker.split(SourceDocument(report.report_id, report.impression_text), chunk_cfg)
                chunker.validate_chunks(chunks)
                all_chunks.extend(chunks)
                meta_hash = stable_hash(meta)
                for chunk in chunks:
                    doc_id = make_id("doc", name, report.report_id, chunk.chunk_index)
                    desired[doc_id] = {
                        "report": report, "chunk": chunk, "meta": meta, "meta_hash": meta_hash,
                        "content_hash": sha256_text(chunk.text),
                    }
            existing = {
                d.document_id: d
                for d in session.scalars(
                    select(RetrievalDocument).where(RetrievalDocument.index_version == name)
                )
            }
            changed = [
                doc_id for doc_id, d in desired.items()
                if doc_id not in existing
                or existing[doc_id].content_hash != d["content_hash"]
                or existing[doc_id].metadata_hash != d["meta_hash"]
            ]
            obsolete = [doc_id for doc_id in existing if doc_id not in desired]
            stale_hashes = {existing[i].content_hash for i in obsolete} | {
                existing[i].content_hash for i in changed
                if i in existing and existing[i].content_hash != desired[i]["content_hash"]
            }

            embed_seconds = 0.0
            for start in range(0, len(changed), 256):
                batch_ids = changed[start : start + 256]
                vectors = None
                if embedder is not None:
                    t0 = time.perf_counter()
                    vectors = embedder.encode_documents([desired[i]["chunk"].text for i in batch_ids])
                    embed_seconds += time.perf_counter() - t0
                docs = []
                for row, doc_id in enumerate(batch_ids):
                    d = desired[doc_id]
                    report, chunk = d["report"], d["chunk"]
                    doc = {
                        "_id": doc_id,
                        **d["meta"],
                        "document_id": doc_id,
                        "source_type": "radiology_report",
                        "source_id": report.report_id,
                        "parent_document_id": report.report_id,
                        "chunk_index": chunk.chunk_index,
                        "chunking": chunking_key,
                        "content": chunk.text,
                        "content_hash": d["content_hash"],
                        "source_timestamp": report.report_time.isoformat() if report.report_time else None,
                        "embedding_model_id": provider.config.model_id if provider else None,
                        "embedding_revision": revision.get("document"),
                        "source_version": self.config.dataset.release,
                    }
                    if vectors is not None:
                        doc[VECTOR_FIELD] = vectors[row]
                    docs.append(doc)
                    row_obj = existing.get(doc_id)
                    if row_obj is None:
                        row_obj = RetrievalDocument(document_id=doc_id, case_id=report.case_id,
                                                    patient_id=report.patient_id, source_type="radiology_report",
                                                    source_id=report.report_id, content=chunk.text,
                                                    content_hash=d["content_hash"],
                                                    chunking_strategy=chunking_key,
                                                    chunking_version=chunk_cfg.version)
                        session.add(row_obj)
                    row_obj.case_id = report.case_id
                    row_obj.patient_id = report.patient_id
                    row_obj.parent_document_id = report.report_id
                    row_obj.section = chunk.section
                    row_obj.chunk_index = chunk.chunk_index
                    row_obj.char_start = chunk.char_start
                    row_obj.char_end = chunk.char_end
                    row_obj.content = chunk.text
                    row_obj.content_hash = d["content_hash"]
                    row_obj.metadata_hash = d["meta_hash"]
                    row_obj.token_count = chunk.token_count
                    row_obj.source_timestamp = report.report_time
                    row_obj.embedding_model_id = provider.config.model_id if provider else None
                    row_obj.index_version = name
                    row_obj.access_scope = d["meta"]["access_scope"]
                self.backend.bulk_index(name, docs)
            if obsolete:
                self.backend.delete_documents(name, obsolete)
                session.execute(delete(RetrievalDocument).where(RetrievalDocument.document_id.in_(obsolete)))
            self.backend.refresh(name)
            session.flush()
            invalidated = 0
            if stale_hashes:
                still_used = set(
                    session.scalars(
                        select(RetrievalDocument.content_hash)
                        .where(RetrievalDocument.content_hash.in_(stale_hashes)).distinct()
                    )
                )
                invalidated = self.embeddings.cache.invalidate_content(sorted(stale_hashes - still_used))

            stats: dict[str, Any] = {
                "reports": len(reports),
                "indexed_or_updated": len(changed),
                "unchanged": len(desired) - len(changed),
                "deleted": len(obsolete),
                "cache_entries_invalidated": invalidated,
                "embedding_seconds": round(embed_seconds, 4),
                "indexing_seconds": round(time.perf_counter() - started, 4),
                "documents_per_second": round(len(changed) / max(time.perf_counter() - started, 1e-9), 2),
                **chunk_statistics(all_chunks, len(reports)),
            }
            if provider is not None:
                stats["encoder_truncation"] = dict(provider.truncation)
            self._manifest(
                session, spec, stats,
                embedding_key=embedding_key or LEXICAL_KEY,
                embedding_model_id=provider.config.model_id if provider else None,
                embedding_revision=str(revision.get("document")) if revision else None,
                chunking_key=chunking_key, chunking_version=chunk_cfg.version,
            )
        return {"index": name, **stats}

    # ------------------------------------------------------------- image index
    def sync_image_index(self, provider: ImageEmbeddingProvider) -> dict[str, Any]:
        name = self.config.index.image_index
        if not provider.enabled:
            return {"index": name, "disabled": True, "documents": 0}
        spec = IndexSpec(name=name, kind="images", dimension=provider.dimension(), similarity="cosine",
                         mapping_version=self.config.index.mapping_version,
                         meta={"provider_id": provider.provider_id, "provider_version": provider.version})
        with session_scope(self.session_factory) as session:
            metadata = self._all_case_metadata(session)
            features = list(
                session.scalars(
                    select(ImageFeature).where(ImageFeature.provider_id == provider.provider_id,
                                               ImageFeature.provider_version == provider.version)
                )
            )
            self.backend.create_index(spec, recreate=True)
            docs = [
                {"_id": f.study_id, **metadata[f.case_id], "study_id": f.study_id,
                 "provider_id": f.provider_id, "provider_version": f.provider_version,
                 VECTOR_FIELD: f.vector}
                for f in features if f.case_id in metadata
            ]
            for start in range(0, len(docs), 500):
                self.backend.bulk_index(name, docs[start : start + 500])
            self.backend.refresh(name)
            stats = {"documents": len(docs), "provider": provider.provider_id, "version": provider.version}
            self._manifest(session, spec, stats)
        return {"index": name, **stats}

    # ----------------------------------------------------------------- aliases
    def promote(self, kind: str, index_name: str) -> str:
        """Point ``<alias_prefix>_<kind>_current`` at ``index_name`` (previous indexes are kept)."""
        alias = f"{self.config.index.alias_prefix}_{kind}_current"
        self.backend.put_alias(alias, index_name)
        with session_scope(self.session_factory) as session:
            manifest = session.get(IndexManifest, index_name)
            if manifest is not None:
                manifest.alias = alias
            session.add(AuditEvent(actor=self.actor, action="index.promoted", resource_type="index",
                                   resource_id=index_name, details={"alias": alias}))
        return alias

    def required_report_indexes(self) -> list[tuple[str | None, str]]:
        """(embedding, chunking) pairs needed by the configured recipes and index lists."""
        pairs: set[tuple[str | None, str]] = set()
        for recipe in self.config.recipes.values():
            if "bm25" in recipe.branches:
                pairs.add((None, recipe.chunking))
            if "dense" in recipe.branches:
                pairs.add((recipe.embedding, recipe.chunking))
        for emb in self.config.index_embeddings:
            for chunk in self.config.index_chunkings:
                pairs.add((emb, chunk))
        return sorted(pairs, key=lambda p: (p[0] or "", p[1]))
