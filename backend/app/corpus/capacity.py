"""Capacity planning for the corpus platform.

``estimate`` turns a workload description into storage and memory figures, component
by component.  Every input is explicit and every output states its formula, because
the estimate is a *starting point for a benchmark*, not a promise: real sizes depend
on the analyser, the mapping, compression, the shard layout and the data itself.
``calibrate`` replaces the modelled per-chunk index size with one measured on an
actual index, which is the recommended way to use this tool:

1. index a representative sample (for example one million chunks);
2. run ``corpus capacity --calibrate`` to read the measured bytes per chunk;
3. extrapolate, then verify with a larger sample before buying hardware.

Shard counts are never hardcoded; the tool suggests one from a target shard size.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GB = 1e9


class CapacityInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_records: int = Field(default=100_000_000, ge=1)
    avg_document_bytes: float = Field(default=1_200, gt=0, description="UTF-8 bytes of content per document")
    versions_retained: float = Field(default=1.0, ge=1.0, description="document versions kept in raw storage")
    chunks_per_document: float = Field(default=1.3, gt=0)
    avg_chunk_bytes: float = Field(default=900, gt=0, description="indexed text bytes per chunk (title + chunk)")
    dimension: int = Field(default=1024, ge=1)
    vector_encoding: Literal["float32", "float16", "int8"] = "float32"
    quantized_index: bool = Field(default=False, description="int8_hnsw: quantised copy searched, floats kept")
    hnsw_m: int = Field(default=16, ge=2)
    replicas: int = Field(default=1, ge=0)
    metadata_bytes_per_chunk: float = Field(default=350, ge=0)
    inverted_index_ratio: float = Field(default=0.35, ge=0, description="inverted index / indexed text bytes")
    stored_source_ratio: float = Field(default=0.5, ge=0, description="compressed _source / raw source bytes")
    parquet_ratio: float = Field(default=0.3, gt=0, description="compressed Parquet / raw text bytes")
    registry_row_overhead_bytes: float = Field(default=300, ge=0)
    shard_target_gb: float = Field(default=40, gt=0)
    embedding_chunks_per_second_per_gpu: float | None = Field(default=None, gt=0)
    gpus: int = Field(default=1, ge=1)


_COMPONENT_BYTES = {"float32": 4, "float16": 2, "int8": 1}


def estimate(inputs: CapacityInputs, measured_index_bytes_per_chunk: float | None = None) -> dict[str, Any]:
    n_docs = inputs.source_records
    n_chunks = n_docs * inputs.chunks_per_document
    raw_text = n_docs * inputs.avg_document_bytes
    component = _COMPONENT_BYTES[inputs.vector_encoding]
    vectors = n_chunks * inputs.dimension * component
    quantized = n_chunks * inputs.dimension if inputs.quantized_index else 0.0
    # layer 0 keeps up to 2*M neighbours of 4 bytes each; upper layers add about 1/M more
    graph = n_chunks * inputs.hnsw_m * 2 * 4 * (1 + 1 / inputs.hnsw_m)
    indexed_text = n_chunks * inputs.avg_chunk_bytes
    inverted = indexed_text * inputs.inverted_index_ratio
    stored_source = (indexed_text * 2 + n_chunks * inputs.metadata_bytes_per_chunk) * inputs.stored_source_ratio
    doc_values = n_chunks * inputs.metadata_bytes_per_chunk * 0.5
    modelled_primary = vectors + quantized + graph + inverted + stored_source + doc_values
    primary = n_chunks * measured_index_bytes_per_chunk if measured_index_bytes_per_chunk else modelled_primary
    with_replicas = primary * (1 + inputs.replicas)
    registry = n_docs * (inputs.avg_document_bytes * 0.6 + inputs.registry_row_overhead_bytes)
    searched_vectors = quantized if inputs.quantized_index else vectors
    out: dict[str, Any] = {
        "inputs": inputs.model_dump(),
        "documents": n_docs,
        "chunks": int(n_chunks),
        "components_gb": {
            "raw_documents_parquet": round(raw_text * inputs.versions_retained * inputs.parquet_ratio / GB, 2),
            "normalized_parquet": round(raw_text * inputs.parquet_ratio * 1.1 / GB, 2),
            "registry_database": round(registry / GB, 2),
            "vectors_raw": round(vectors / GB, 2),
            "vectors_quantized_copy": round(quantized / GB, 2),
            "ann_graph": round(graph / GB, 2),
            "inverted_index": round(inverted / GB, 2),
            "stored_source": round(stored_source / GB, 2),
            "doc_values_metadata": round(doc_values / GB, 2),
        },
        "index_primary_gb": round(primary / GB, 2),
        "index_primary_source": "measured bytes per chunk" if measured_index_bytes_per_chunk else "model",
        "index_with_replicas_gb": round(with_replicas / GB, 2),
        # a rebuild runs next to the serving index until the alias switches
        "temporary_reindex_gb": round(with_replicas / GB, 2),
        "index_peak_gb": round(2 * with_replicas / GB, 2),
        "search_memory_gb": {
            "vectors_and_graph_page_cache_per_copy": round((searched_vectors + graph) / GB, 2),
            "note": "kNN is fast only while the searched vectors and the graph fit in the page cache of the "
                    "nodes holding each copy; heap is additional",
        },
        "suggested_primary_shards": max(1, math.ceil(primary / GB / inputs.shard_target_gb)),
        "formulas": {
            "vectors_raw": "chunks * dimension * bytes_per_component",
            "ann_graph": "chunks * 2*M * 4 bytes * (1 + 1/M)",
            "index_peak": "2 * primary * (1 + replicas)  (serving index + rebuild)",
        },
        "caveats": [
            "Model output, not a measurement: calibrate with a real index before sizing hardware.",
            "Shard count is a starting point; benchmark indexing and query latency at the chosen size.",
        ],
    }
    if inputs.embedding_chunks_per_second_per_gpu:
        hours = n_chunks / (inputs.embedding_chunks_per_second_per_gpu * inputs.gpus) / 3600
        out["embedding_hours"] = round(hours, 1)
    return out


def calibrate(index_stats: dict[str, Any]) -> float | None:
    """Measured primary bytes per chunk of an existing index (``SearchBackend.stats``)."""
    documents, size = index_stats.get("documents"), index_stats.get("size_bytes")
    return float(size) / float(documents) if documents and size else None
