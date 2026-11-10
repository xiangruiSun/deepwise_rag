"""Tenant-aware document corpus platform.

A generic layer next to the clinical case-retrieval code: canonical documents and
chunks, streaming / distributed ETL, versioned hybrid indexes, online and batch RAG,
evaluation.  It reuses the existing search backends, embedding cache, fusion,
rerankers, security model and metric functions instead of duplicating them.
"""
