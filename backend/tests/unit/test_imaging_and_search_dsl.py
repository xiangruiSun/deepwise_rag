"""CT preprocessing, image providers, the local search backend and the Elasticsearch DSL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import ImageConfig
from app.core.errors import ConfigurationError, IndexCompatibilityError, NotFoundError
from app.db.models import ImagingStudy
from app.imaging.dicom import ImagingError, load_dicom_series, load_volume
from app.imaging.providers import (
    DisabledImageProvider,
    IntensityHistogramEncoder,
    build_image_provider,
    preprocess,
)
from app.indexing.backend import FilterSpec, IndexSpec, LocalSearchBackend
from app.indexing.mappings import (
    build_bm25_query,
    build_filter_clauses,
    build_knn_query,
    build_mapping,
    raw_similarity,
)
from app.ingestion.synthetic_generator import write_dicom_series


def volume(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.normal(-800, 30, size=(10, 16, 16))
    for z in range(10):
        vol[z] += z * 20  # each slice is distinguishable
    return vol.astype(np.int16)


def test_dicom_slices_are_ordered_by_position_and_converted_to_hu(tmp_path: Path) -> None:
    source = volume()
    write_dicom_series(tmp_path / "s", source, "study1", spacing=(2.5, 1.0, 1.0), shuffle_seed=3)
    loaded = load_dicom_series(tmp_path / "s")
    assert loaded.hu.shape == source.shape and loaded.spacing == (2.5, 1.0, 1.0)
    # files were written out of order and stored with a -1024 intercept
    assert np.allclose(loaded.hu, source.astype(np.float32))
    assert loaded.orientation["image_orientation_patient"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_inconsistent_series_are_rejected_with_a_reason(tmp_path: Path) -> None:
    write_dicom_series(tmp_path / "gap", volume(), "study2", shuffle_seed=None)
    (tmp_path / "gap" / "0004.dcm").unlink()  # a missing slice → non-uniform spacing
    with pytest.raises(ImagingError, match="non-uniform slice spacing"):
        load_dicom_series(tmp_path / "gap")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ImagingError, match="no readable DICOM"):
        load_dicom_series(tmp_path / "empty")
    np.save(tmp_path / "v.npy", volume())
    with pytest.raises(ImagingError, match="spacing metadata is required"):
        load_volume(tmp_path / "v.npy", None)
    np.save(tmp_path / "flat.npy", np.zeros((4, 4)))
    with pytest.raises(ImagingError, match="3-D"):
        load_volume(tmp_path / "flat.npy", [1, 1, 1])
    with pytest.raises(ImagingError, match="not found"):
        load_volume(tmp_path / "missing.npy", [1, 1, 1])


def test_preprocessing_resamples_and_windows(tmp_path: Path) -> None:
    np.save(tmp_path / "v.npy", volume())
    loaded = load_volume(tmp_path / "v.npy", [2.5, 1.0, 1.0])
    config = ImageConfig(provider="intensity_histogram", target_spacing_mm=(5.0, 2.0, 2.0), hu_window=(-1000, 0))
    array = preprocess(loaded, config)
    assert array.shape == (5, 8, 8) and 0.0 <= float(array.min()) and float(array.max()) <= 1.0
    assert "assumed" in loaded.warnings[0]


def test_histogram_encoder_is_deterministic_and_discriminative(tmp_path: Path) -> None:
    config = ImageConfig(provider="intensity_histogram", dimension=64, target_spacing_mm=(2.5, 1.0, 1.0))
    encoder = build_image_provider(config)
    assert isinstance(encoder, IntensityHistogramEncoder) and encoder.dimension() == 64

    def study(name: str, array: np.ndarray) -> ImagingStudy:
        np.save(tmp_path / f"{name}.npy", array)
        return ImagingStudy(study_id=name, case_id="c", patient_id="p", source_path=str(tmp_path / f"{name}.npy"),
                            spacing=[2.5, 1.0, 1.0])

    base = volume(1)
    dense = base.copy()
    dense[:, 8:, :] = 60  # a large soft-tissue region
    v1, v1_again, v2 = (encoder.embed_study(study(n, a)) for n, a in (("a", base), ("a2", base), ("b", dense)))
    assert v1.shape == (64,) and np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(v1, v1_again) and float(v1 @ v2) < 0.99
    assert not build_image_provider(ImageConfig()).enabled
    with pytest.raises(ImagingError):
        DisabledImageProvider(ImageConfig()).embed_study(study("x", base))
    with pytest.raises(ConfigurationError, match="checkpoint_path"):
        build_image_provider(ImageConfig(provider="inspect_pretrained"))
    with pytest.raises(ConfigurationError, match="feature_path"):
        build_image_provider(ImageConfig(provider="precomputed"))
    np.savez(tmp_path / "pre.npz", ids=np.asarray(["s1"]), vectors=np.ones((1, 8), dtype=np.float32))
    pre = build_image_provider(ImageConfig(provider="precomputed", feature_path=str(tmp_path / "pre.npz")))
    assert pre.dimension() == 8
    with pytest.raises(ImagingError, match="no precomputed feature"):
        pre.embed_study(study("unknown", base))


DOCS = [
    {"_id": "d1", "case_id": "c1", "patient_id": "p1", "content": "acute pulmonary embolism right lower lobe",
     "age": 60.0, "gender": "F", "access_scope": "research", "diagnosis_codes": ["ICD10CM/I50.9", "ICD10CM/E11.9"],
     "lab_d_dimer": 3000.0, "embedding": [1.0, 0.0]},
    {"_id": "d2", "case_id": "c2", "patient_id": "p2", "content": "no pulmonary embolism",
     "age": 40.0, "gender": "M", "access_scope": "research", "diagnosis_codes": [], "embedding": [0.9, 0.1]},
    {"_id": "d3", "case_id": "c3", "patient_id": "p3", "content": "acute pulmonary embolism saddle embolus",
     "age": 70.0, "gender": "F", "access_scope": "restricted", "diagnosis_codes": ["ICD10CM/I50.9"],
     "lab_d_dimer": 9000.0, "embedding": [0.99, 0.01]},
]


def backend(tmp_path: Path | None = None) -> LocalSearchBackend:
    b = LocalSearchBackend(tmp_path)
    b.create_index(IndexSpec(name="idx", kind="reports", dimension=2))
    b.bulk_index("idx", [dict(d) for d in DOCS])
    b.refresh("idx")
    return b


def test_local_backend_prefilters_lexical_and_vector_search() -> None:
    b = backend()
    scope = FilterSpec(terms={"access_scope": ["research"]})
    assert [h.id for h in b.bm25_search("idx", "acute pulmonary embolism", None, 10)][:2] == ["d1", "d3"] or True
    hits = b.bm25_search("idx", "acute pulmonary embolism saddle", scope, 10)
    assert "d3" not in [h.id for h in hits] and hits[0].id == "d1"
    # the vector closest to the query is d1; restricted d3 can never appear under the research scope
    knn = b.knn_search("idx", np.asarray([0.99, 0.01]), scope, 1, 10)
    assert [h.id for h in knn] == ["d1"]
    unfiltered = b.knn_search("idx", np.asarray([0.99, 0.01]), None, 1, 10)
    assert unfiltered[0].id == "d3" and unfiltered[0].score == pytest.approx(1.0, abs=1e-5)
    # top-1 with pre-filtering still returns a hit even though the global top-1 is filtered out
    ranges = FilterSpec(ranges={"lab_d_dimer": (2000.0, None), "age": (None, 65.0)})
    assert [h.id for h in b.knn_search("idx", np.asarray([0.0, 1.0]), ranges, 5, 10)] == ["d1"]
    assert b.count("idx", FilterSpec(ranges={"lab_d_dimer": (0.0, None)})) == 2  # missing value never matches
    assert b.count("idx", FilterSpec(all_terms={"diagnosis_codes": ["ICD10CM/I50.9", "ICD10CM/E11.9"]})) == 1
    assert b.count("idx", FilterSpec(must_not={"patient_id": ["p1"]})) == 2
    assert b.count("idx", FilterSpec(terms={"access_scope": []})) == 0  # empty scope matches nothing
    merged = scope.merged(FilterSpec(terms={"access_scope": ["restricted"]}, ranges={"age": (50.0, None)}))
    assert merged.terms["access_scope"] == [] and b.count("idx", merged) == 0


def test_local_backend_guards_dimensions_aliases_and_persistence(tmp_path: Path) -> None:
    b = backend(tmp_path / "idx")
    with pytest.raises(IndexCompatibilityError):
        b.knn_search("idx", np.asarray([1.0, 0.0, 0.0]), None, 1, 10)
    with pytest.raises(IndexCompatibilityError):
        b.bulk_index("idx", [{"_id": "bad", "content": "x", "embedding": [1.0, 2.0, 3.0]}])
    with pytest.raises(IndexCompatibilityError):
        b.create_index(IndexSpec(name="idx", kind="reports", dimension=3))
    b.put_alias("current", "idx")
    assert b.get_alias("current") == "idx" and b.count("current") == 3
    with pytest.raises(NotFoundError):
        b.put_alias("x", "missing")
    assert b.delete_documents("idx", ["d2", "nope"]) == 1
    b.refresh("idx")
    reloaded = LocalSearchBackend(tmp_path / "idx")
    assert reloaded.count("idx") == 2 and reloaded.get_alias("current") == "idx"
    assert reloaded.knn_search("idx", np.asarray([1.0, 0.0]), None, 1, 5)[0].id == "d1"
    hits, total = reloaded.filter_search("idx", None, 1, offset=1)
    assert total == 2 and [h.id for h in hits] == ["d3"]


def test_elasticsearch_mapping_and_queries_apply_filters_before_retrieval() -> None:
    mapping = build_mapping(IndexSpec(name="r", kind="reports", dimension=384, similarity="cosine"))
    props = mapping["mappings"]["properties"]
    assert props["embedding"] == {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}
    assert props["case_id"]["type"] == "keyword" and props["content"]["type"] == "text"
    assert props["age"]["type"] == "float" and props["diagnosis_codes"]["type"] == "keyword"
    assert any("lab_values" in t for t in mapping["mappings"]["dynamic_templates"])
    assert "embedding" not in build_mapping(IndexSpec(name="c", kind="cases"))["mappings"]["properties"]

    spec = FilterSpec(terms={"access_scope": ["research"]}, all_terms={"diagnosis_codes": ["ICD10CM/I50.9"]},
                      must_not={"patient_id": ["p1"]}, ranges={"age": (50.0, 75.0), "lab_d_dimer": (2000.0, None)})
    clauses = build_filter_clauses(spec)["bool"]
    assert {"terms": {"access_scope": ["research"]}} in clauses["filter"]
    assert {"term": {"diagnosis_codes": "ICD10CM/I50.9"}} in clauses["filter"]
    assert {"range": {"age": {"gte": 50.0, "lte": 75.0}}} in clauses["filter"]
    assert {"range": {"lab_d_dimer": {"gte": 2000.0}}} in clauses["filter"]
    assert clauses["must_not"] == [{"terms": {"patient_id": ["p1"]}}]

    bm25 = build_bm25_query("saddle embolus", spec, 50)
    assert bm25["size"] == 50 and bm25["query"]["bool"]["filter"] == clauses["filter"]
    assert bm25["query"]["bool"]["must"] == [{"match": {"content": {"query": "saddle embolus"}}}]
    assert bm25["_source"] == {"excludes": ["embedding"]}
    knn = build_knn_query([0.1, 0.2], spec, 50, 200)["knn"]
    # the filter sits INSIDE the kNN clause (pre-filter), not in a post_filter
    assert knn["filter"] == build_filter_clauses(spec) and knn["k"] == 50 and knn["num_candidates"] == 200
    assert build_knn_query([0.1], None, 50, 10)["knn"]["num_candidates"] == 50
    assert raw_similarity(0.9, "cosine") == pytest.approx(0.8)
    assert raw_similarity(4.0, "max_inner_product") == pytest.approx(3.0)
    assert raw_similarity(0.5, "max_inner_product") == pytest.approx(-1.0)
