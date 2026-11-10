"""Versioned hybrid chunk indexes: build, validate, activate, roll back, sync, verify.

One index holds the chunk text (BM25) and the chunk vector (kNN) of every live
document, so lexical and dense retrieval always see the same population and the
same pre-filters.  An index is identified by its embedding identity and its
chunking identity; vectors of different models can never share an index.

Readers go through an alias.  A new index version is built next to the active
one, validated, and only then becomes the alias target; the previous version is
kept for rollback.

The registry (``corpus_documents``) records which document version each index
holds, so incremental sync touches only documents that changed, deletion always
propagates, and drift between registry and index is detectable and repairable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import ConfigurationError, NotFoundError
from app.core.hashing import sha256_text
from app.corpus.chunks import chunk_document
from app.corpus.config import CorpusConfig
from app.corpus.schema import make_chunk_id, to_epoch
from app.db.models import CorpusDocument, CorpusIndexVersion
from app.embeddings.base import CachedEmbedder
from app.indexing.backend import TEXT_FIELD, VECTOR_FIELD, FilterSpec, IndexSpec, SearchBackend
from app.indexing.mappings import OPEN_ENDED_TS

# extension fields copied to the index as filterable metadata (prefixed with ``x_``)
INDEXED_EXTENSION_FIELDS = ("patient_pseudo_id", "encounter_id", "department")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class IndexManager:
    def __init__(self, config: CorpusConfig, session_factory: sessionmaker[Session], backend: SearchBackend,
                 embedder: CachedEmbedder) -> None:
        self.config, self.session_factory, self.backend, self.embedder = config, session_factory, backend, embedder
        self.provider = embedder.provider
        # called with the tenants whose searchable content changed (cache invalidation hook)
        self.on_change: Callable[[set[str]], None] | None = None

    # ---------------------------------------------------------------- identity
    @property
    def embedding_identity(self) -> str:
        return sha256_text(self.provider.identity())[:16]

    def _spec(self, name: str) -> IndexSpec:
        meta: dict[str, Any] = {"embedding_identity": self.embedding_identity,
                                "chunking_identity": self.config.chunking.identity}
        if self.config.vector_quantization != "none":
            meta["vector_quantization"] = self.config.vector_quantization
        return IndexSpec(name=name, kind="chunks", dimension=self.provider.embedding_dimension(),
                         similarity=self.provider.config.similarity, mapping_version=self.config.mapping_version,
                         meta=meta)

    def active(self) -> str | None:
        return self.backend.get_alias(self.config.alias)

    def require_active(self) -> str:
        name = self.active()
        if name is None:
            raise NotFoundError("no active corpus index; build and activate one first",
                                {"alias": self.config.alias})
        return name

    # --------------------------------------------------------------- lifecycle
    def create_version(self) -> str:
        """Create an empty index version (status ``building``) and return its name."""
        with self.session_factory() as session:
            n = session.scalar(select(func.count()).select_from(CorpusIndexVersion)
                               .where(CorpusIndexVersion.alias == self.config.alias)) or 0
            name = f"{self.config.index_prefix}_v{n + 1:04d}_{uuid.uuid4().hex[:6]}"
            self.backend.create_index(self._spec(name))
            session.add(CorpusIndexVersion(
                index_name=name, alias=self.config.alias, status="building",
                embedding_identity=self.embedding_identity, chunking_identity=self.config.chunking.identity,
                mapping_version=self.config.mapping_version, dimension=self.provider.embedding_dimension()))
            session.commit()
        return name

    def build(self, name: str, batch_size: int | None = None,
              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Index every live document into ``name``.  Resumable: documents already staged at
        their current version are skipped, so a second call continues where the first stopped."""
        batch_size = batch_size or self.config.index_batch_size
        stats: dict[str, Any] = {"documents": 0, "chunks": 0, "embedding_seconds": 0.0, "indexing_seconds": 0.0}
        started = time.perf_counter()
        cursor = ""
        while True:
            with self.session_factory() as session:
                rows = list(session.scalars(
                    select(CorpusDocument).where(
                        CorpusDocument.document_id > cursor, CorpusDocument.is_deleted.is_(False),
                        or_(CorpusDocument.staged_in.is_(None), CorpusDocument.staged_in != name,
                            CorpusDocument.staged_version != CorpusDocument.document_version))
                    .order_by(CorpusDocument.document_id).limit(batch_size)))
                if not rows:
                    break
                cursor = rows[-1].document_id
                written = self._write(name, rows, stats, replace_stale=False)
                session.execute(update(CorpusDocument).where(CorpusDocument.document_id.in_(list(written)))
                                .values(staged_in=name, staged_version=CorpusDocument.document_version))
                session.commit()
            if progress:
                progress(dict(stats))
        self.backend.refresh(name)
        stats["seconds"] = round(time.perf_counter() - started, 3)
        stats["documents_per_second"] = round(stats["documents"] / stats["seconds"], 1) if stats["seconds"] else None
        with self.session_factory() as session:
            version = session.get(CorpusIndexVersion, name)
            if version is not None:
                merged = dict(version.stats or {})
                for key in ("documents", "chunks", "embedding_seconds", "indexing_seconds"):
                    merged[key] = round(merged.get(key, 0) + stats[key], 3)
                version.stats = merged
                session.commit()
        return stats

    def validate(self, name: str, sample: int = 50) -> dict[str, Any]:
        """Checks that must pass before ``name`` may serve traffic."""
        problems: list[str] = []
        spec = self.backend.index_spec(name)
        if spec.dimension != self.provider.embedding_dimension():
            problems.append("vector dimension differs from the configured embedding model")
        if spec.meta.get("embedding_identity") not in (None, self.embedding_identity):
            problems.append("index was built with a different embedding model identity")
        with self.session_factory() as session:
            live = session.scalar(select(func.count()).select_from(CorpusDocument)
                                  .where(CorpusDocument.is_deleted.is_(False))) or 0
            staged = session.scalar(select(func.count()).select_from(CorpusDocument).where(
                CorpusDocument.is_deleted.is_(False), CorpusDocument.staged_in == name,
                CorpusDocument.staged_version == CorpusDocument.document_version)) or 0
            rows = list(session.scalars(select(CorpusDocument).where(
                CorpusDocument.is_deleted.is_(False), CorpusDocument.staged_in == name)
                .order_by(CorpusDocument.content_hash).limit(sample)))
        if staged != live:
            problems.append(f"{live - staged} live documents are not indexed at their current version")
        missing = 0
        for row in rows:  # exact-match lookup of the first chunk of sampled documents
            if not row.content.strip():
                continue
            source = self.backend.get_document(name, make_chunk_id(row.document_id, 0))
            if source is None or int(source.get("document_version", -1)) != row.staged_version:
                missing += 1
        if missing:
            problems.append(f"{missing} of {len(rows)} sampled documents are missing or stale in the index")
        health = self.backend.stats(name).get("health")
        if health == "red":
            problems.append("index health is red")
        result = {"index": name, "valid": not problems, "problems": problems, "live_documents": live,
                  "indexed_documents": staged, "index_documents": self.backend.count(name), "health": health,
                  "sampled": len(rows)}
        with self.session_factory() as session:
            version = session.get(CorpusIndexVersion, name)
            if version is not None and version.status in ("building", "failed", "validated"):
                version.status = "validated" if not problems else "failed"
                version.stats = {**(version.stats or {}), "validation": result}
                session.commit()
        return result

    def activate(self, name: str, force: bool = False) -> dict[str, Any]:
        """Point the read alias at ``name``.  Refused unless the index validated (or ``force``)."""
        with self.session_factory() as session:
            version = session.get(CorpusIndexVersion, name)
            if version is None:
                raise NotFoundError(f"unknown index version '{name}'")
            if version.status not in ("validated", "retired", "active") and not force:
                raise ConfigurationError(
                    f"index '{name}' has status '{version.status}'; validate it before activating it")
            previous = self.active()
            self.backend.put_alias(self.config.alias, name)  # atomic switch
            if previous and previous != name:
                old = session.get(CorpusIndexVersion, previous)
                if old is not None:
                    old.status, old.retired_at = "retired", _now()
            version.status, version.activated_at = "active", _now()
            # the registry now describes the new index: promote what the build staged
            session.execute(update(CorpusDocument).where(CorpusDocument.staged_in == name).values(
                indexed_in=name, indexed_version=CorpusDocument.staged_version,
                indexed_hash=CorpusDocument.content_hash))
            session.execute(update(CorpusDocument).where(
                or_(CorpusDocument.staged_in.is_(None), CorpusDocument.staged_in != name),
                CorpusDocument.indexed_in.is_not(None), CorpusDocument.is_deleted.is_(False))
                .values(indexed_in=None, indexed_version=None, indexed_hash=None))
            session.commit()
        return {"alias": self.config.alias, "active": name, "previous": previous}

    def rollback(self) -> dict[str, Any]:
        """Re-activate the most recently retired index version.

        The retired index missed every change made while the other version served, so the
        registry is compared with it document by document and the differences are re-synced."""
        with self.session_factory() as session:
            previous = session.scalars(select(CorpusIndexVersion).where(
                CorpusIndexVersion.alias == self.config.alias, CorpusIndexVersion.status == "retired")
                .order_by(CorpusIndexVersion.retired_at.desc())).first()
            if previous is None:
                raise NotFoundError("there is no retired index version to roll back to")
            name = previous.index_name
            if not self.backend.index_exists(name):
                raise NotFoundError(f"retired index '{name}' no longer exists in the search backend")
            current = self.active()
            self.backend.put_alias(self.config.alias, name)  # atomic switch back
            if current and current != name:
                failed = session.get(CorpusIndexVersion, current)
                if failed is not None:
                    failed.status, failed.retired_at = "retired", _now()
            previous.status, previous.activated_at = "active", _now()
            # assume the old index matches, then let the consistency check find what does not
            session.execute(update(CorpusDocument).values(
                indexed_in=name, indexed_version=CorpusDocument.document_version,
                indexed_hash=CorpusDocument.content_hash, staged_in=None, staged_version=None))
            session.commit()
        check = self.consistency_check(repair=True)
        return {"alias": self.config.alias, "active": name, "previous": current,
                "repaired": check["repaired"], "resync": self.sync()}

    def retire_old(self, keep: int = 1) -> list[str]:
        """Delete retired index versions except the ``keep`` most recent (rollback targets)."""
        removed: list[str] = []
        with self.session_factory() as session:
            retired = list(session.scalars(select(CorpusIndexVersion).where(
                CorpusIndexVersion.alias == self.config.alias, CorpusIndexVersion.status == "retired")
                .order_by(CorpusIndexVersion.retired_at.desc())))
            for version in retired[keep:]:
                self.backend.delete_index(version.index_name)
                version.status = "deleted"
                removed.append(version.index_name)
            session.commit()
        return removed

    # ------------------------------------------------------------ incremental
    def sync(self, limit: int | None = None, batch_size: int | None = None) -> dict[str, Any]:
        """Bring the ACTIVE index up to date with the registry (changed and deleted documents)."""
        name = self.require_active()
        batch_size = batch_size or self.config.index_batch_size
        stats: dict[str, Any] = {"documents": 0, "chunks": 0, "deleted_documents": 0, "embedding_seconds": 0.0,
                                 "indexing_seconds": 0.0}
        touched: set[str] = set()
        started = time.perf_counter()
        pending_upsert = and_(CorpusDocument.is_deleted.is_(False), or_(
            CorpusDocument.indexed_in.is_(None), CorpusDocument.indexed_in != name,
            CorpusDocument.indexed_version.is_(None),
            CorpusDocument.indexed_version != CorpusDocument.document_version))
        pending_delete = and_(CorpusDocument.is_deleted.is_(True), CorpusDocument.indexed_version.is_not(None),
                              CorpusDocument.indexed_version != CorpusDocument.document_version)
        while limit is None or stats["documents"] + stats["deleted_documents"] < limit:
            with self.session_factory() as session:
                doomed = list(session.scalars(select(CorpusDocument).where(pending_delete).limit(batch_size)))
                if doomed:
                    ids = [row.document_id for row in doomed]
                    touched.update(row.tenant_id for row in doomed)
                    self.backend.delete_by_filter(name, FilterSpec(terms={"document_id": ids}))
                    session.execute(update(CorpusDocument).where(CorpusDocument.document_id.in_(ids)).values(
                        indexed_version=CorpusDocument.document_version, indexed_in=name, indexed_hash=None))
                    session.commit()
                    stats["deleted_documents"] += len(ids)
                    continue
                rows = list(session.scalars(select(CorpusDocument).where(pending_upsert)
                                            .order_by(CorpusDocument.document_id).limit(batch_size)))
                if not rows:
                    break
                written = self._write(name, rows, stats, replace_stale=True)
                touched.update(row.tenant_id for row in rows)
                session.execute(update(CorpusDocument).where(CorpusDocument.document_id.in_(list(written))).values(
                    indexed_in=name, indexed_version=CorpusDocument.document_version,
                    indexed_hash=CorpusDocument.content_hash))
                session.commit()
        self.backend.refresh(name)
        if touched and self.on_change is not None:
            self.on_change(touched)  # results cached for these tenants are now stale
        stats["seconds"] = round(time.perf_counter() - started, 3)
        stats["index"] = name
        return stats

    def pending(self) -> dict[str, int]:
        """How many documents the active index is behind the registry."""
        name = self.active()
        with self.session_factory() as session:
            upserts = session.scalar(select(func.count()).select_from(CorpusDocument).where(
                CorpusDocument.is_deleted.is_(False), or_(
                    CorpusDocument.indexed_in.is_(None), CorpusDocument.indexed_in != (name or ""),
                    CorpusDocument.indexed_version != CorpusDocument.document_version))) or 0
            deletes = session.scalar(select(func.count()).select_from(CorpusDocument).where(
                CorpusDocument.is_deleted.is_(True), CorpusDocument.indexed_version.is_not(None),
                CorpusDocument.indexed_version != CorpusDocument.document_version)) or 0
        return {"upserts": int(upserts), "deletes": int(deletes)}

    # ---------------------------------------------------------------- writing
    def _write(self, name: str, rows: list[CorpusDocument], stats: dict[str, Any],
               replace_stale: bool) -> set[str]:
        """Chunk, embed and index ``rows``; with ``replace_stale`` remove what older versions left."""
        write_id = uuid.uuid4().hex[:12]
        docs: list[dict[str, Any]] = []
        texts: list[str] = []
        limit = self.provider.max_document_content_tokens
        embed = self.embedder.encode_documents if self.config.chunking.strategy == "semantic" else None
        tokenizer = self.provider.tokenizer()
        for row in rows:
            pieces = chunk_document(row.document_id, row.content, self.config.chunking, tokenizer, limit, embed)
            extension = row.extension or {}
            for index, piece in enumerate(pieces):
                indexed_text = f"{row.title}\n{piece.text}" if row.title else piece.text
                texts.append(indexed_text)
                doc = {
                    "_id": make_chunk_id(row.document_id, index), "tenant_id": row.tenant_id,
                    "document_id": row.document_id, "document_version": row.document_version,
                    "document_type": row.document_type, "source_system": row.source_system,
                    "source_record_id": row.source_record_id, "acl_groups": list(row.access_policy or []),
                    "chunk_index": index, "chunk_count": len(pieces), "token_count": piece.token_count,
                    "section_path": piece.section_path, "title": row.title, TEXT_FIELD: indexed_text,
                    "chunk_text": piece.text, "content_hash": row.content_hash, "write_id": write_id,
                    "event_ts": to_epoch(row.event_time) or 0.0,
                    "valid_from_ts": to_epoch(row.valid_from) or 0.0,
                    "valid_to_ts": to_epoch(row.valid_to) or OPEN_ENDED_TS,
                    "event_time": row.event_time.isoformat() if row.event_time else None,
                    "source_updated_at": row.source_updated_at.isoformat() if row.source_updated_at else None,
                    "chunking_version": self.config.chunking.identity,
                    "embedding_model_version": self.embedding_identity,
                }
                for key in INDEXED_EXTENSION_FIELDS:
                    if extension.get(key) is not None:
                        doc[f"x_{key}"] = str(extension[key])
                docs.append(doc)
        t0 = time.perf_counter()
        vectors: Any = self.embedder.encode_documents(texts) if texts else []
        stats["embedding_seconds"] += time.perf_counter() - t0
        for doc, vector in zip(docs, vectors, strict=True):
            doc[VECTOR_FIELD] = vector
        t1 = time.perf_counter()
        if docs:
            self.backend.bulk_index(name, docs)
        if replace_stale:
            previously = [row.document_id for row in rows if row.indexed_version is not None]
            if previously:  # chunks of older versions that the new version did not overwrite
                self.backend.delete_by_filter(name, FilterSpec(terms={"document_id": previously},
                                                               must_not={"write_id": [write_id]}))
        stats["indexing_seconds"] += time.perf_counter() - t1
        stats["documents"] += len(rows)
        stats["chunks"] += len(docs)
        return {row.document_id for row in rows}

    # ----------------------------------------------------------- consistency
    def consistency_check(self, repair: bool = False, limit: int | None = None) -> dict[str, Any]:
        """Compare the registry with the active index document by document.

        Reports documents that are missing from the index, indexed at an older version, or still
        present after deletion.  With ``repair`` the registry marks them for the next ``sync``."""
        name = self.require_active()
        result: dict[str, Any] = {"index": name, "checked": 0, "missing": 0, "stale": 0,
                                  "deleted_but_indexed": 0, "repaired": 0}
        cursor = ""
        while limit is None or result["checked"] < limit:
            with self.session_factory() as session:
                rows = list(session.scalars(select(CorpusDocument).where(CorpusDocument.document_id > cursor)
                                            .order_by(CorpusDocument.document_id).limit(500)))
                if not rows:
                    break
                cursor = rows[-1].document_id
                broken: list[str] = []
                for row in rows:
                    result["checked"] += 1
                    source = self.backend.get_document(name, make_chunk_id(row.document_id, 0))
                    if row.is_deleted:
                        if source is not None:
                            result["deleted_but_indexed"] += 1
                            broken.append(row.document_id)
                    elif not row.content.strip():
                        continue
                    elif source is None:
                        result["missing"] += 1
                        broken.append(row.document_id)
                    elif int(source.get("document_version", -1)) != row.document_version:
                        result["stale"] += 1
                        broken.append(row.document_id)
                if repair and broken:
                    # an "indexed at version 0" marker makes both pending queries of sync() match
                    session.execute(update(CorpusDocument).where(CorpusDocument.document_id.in_(broken))
                                    .values(indexed_version=0, indexed_hash=None))
                    session.commit()
                    result["repaired"] += len(broken)
        result["consistent"] = not (result["missing"] or result["stale"] or result["deleted_but_indexed"])
        return result

    def health(self) -> dict[str, Any]:
        name = self.active()
        if name is None:
            return {"active": None, "healthy": False, "reason": "no active index"}
        stats = self.backend.stats(name)
        pending = self.pending()
        with self.session_factory() as session:
            versions = [{"index": v.index_name, "status": v.status, "created_at": v.created_at.isoformat()}
                        for v in session.scalars(select(CorpusIndexVersion).where(
                            CorpusIndexVersion.alias == self.config.alias)
                            .order_by(CorpusIndexVersion.created_at))]
        return {"active": name, "healthy": stats.get("health") != "red", "stats": stats, "pending": pending,
                "embedding_identity": self.embedding_identity, "chunking_identity": self.config.chunking.identity,
                "versions": versions}
