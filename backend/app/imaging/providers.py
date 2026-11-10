"""Image preprocessing, embedding providers and the feature pipeline (spec section 14).

Image similarity is an EXPERIMENTAL representation score.  None of the providers
here is a validated lesion-similarity model, and no CT finding is ever generated
from vector similarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import ImageConfig
from app.core.errors import ConfigurationError, DependencyUnavailable
from app.db.models import ImageFeature, ImagingStudy
from app.imaging.dicom import ImagingError, Volume, load_volume

PREPROCESSING_VERSION = "ct-prep-v1"


def preprocess(volume: Volume, config: ImageConfig) -> np.ndarray:
    """Resample to the configured spacing, clip to the HU window, scale to [0, 1]."""
    if any(s <= 0 for s in volume.spacing):
        raise ImagingError("non-positive voxel spacing")
    if volume.hu.ndim != 3 or min(volume.hu.shape) < 2:
        raise ImagingError(f"volume too small or not 3-D: shape {volume.hu.shape}")
    factors = [s / t for s, t in zip(volume.spacing, config.target_spacing_mm, strict=True)]
    array = volume.hu
    if not np.allclose(factors, 1.0, atol=1e-3):
        from scipy.ndimage import zoom

        array = zoom(array, zoom=factors, order=1, mode="nearest")
    if min(array.shape) < 2:
        raise ImagingError("volume collapses below 2 voxels after resampling")
    lo, hi = config.hu_window
    array = np.clip(array, lo, hi)
    return ((array - lo) / (hi - lo)).astype(np.float32)


class ImageEmbeddingProvider(ABC):
    provider_id: str = "base"

    def __init__(self, config: ImageConfig) -> None:
        self.config = config

    @property
    def version(self) -> str:
        return f"{self.config.version}+{PREPROCESSING_VERSION}"

    @property
    def enabled(self) -> bool:
        return True

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_study(self, study: ImagingStudy) -> np.ndarray:
        """Return the study-level vector; raise ``ImagingError`` to reject the study."""


class DisabledImageProvider(ImageEmbeddingProvider):
    provider_id = "disabled"

    @property
    def enabled(self) -> bool:
        return False

    def dimension(self) -> int:
        return 0

    def embed_study(self, study: ImagingStudy) -> np.ndarray:
        raise ImagingError("image modality is disabled")


class IntensityHistogramEncoder(ImageEmbeddingProvider):
    """Explicitly configured, non-learned 3-D CT descriptor: a Hounsfield histogram
    concatenated with a coarse spatial grid of mean intensities.

    It is a transparent engineering baseline that makes the multimodal code path
    runnable; it has no demonstrated relationship to clot burden or any other
    clinical property and must be judged by the ablation study.
    """

    provider_id = "intensity_histogram"
    GRID = (2, 4, 4)

    def dimension(self) -> int:
        return self.config.dimension

    def embed_study(self, study: ImagingStudy) -> np.ndarray:
        if not study.source_path:
            raise ImagingError("study has no source path")
        volume = load_volume(Path(study.source_path), study.spacing)
        array = preprocess(volume, self.config)
        grid_dim = int(np.prod(self.GRID))
        bins = self.config.dimension - grid_dim
        if bins < 4:
            raise ConfigurationError("image.dimension too small for the histogram encoder")
        hist, _ = np.histogram(array, bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
        pooled = np.zeros(self.GRID, dtype=np.float32)
        edges = [np.linspace(0, n, g + 1).astype(int) for n, g in zip(array.shape, self.GRID, strict=True)]
        for z in range(self.GRID[0]):
            for y in range(self.GRID[1]):
                for x in range(self.GRID[2]):
                    block = array[edges[0][z]:edges[0][z + 1], edges[1][y]:edges[1][y + 1],
                                  edges[2][x]:edges[2][x + 1]]
                    pooled[z, y, x] = float(block.mean()) if block.size else 0.0
        vector = np.concatenate([np.sqrt(hist), pooled.ravel() - pooled.mean()])
        norm = float(np.linalg.norm(vector))
        self._last_volume = volume
        return (vector / norm if norm > 0 else vector).astype(np.float32)


class PrecomputedFeatureProvider(ImageEmbeddingProvider):
    """Study-level vectors computed elsewhere (e.g. in the isolated legacy INSPECT
    environment).  File: ``.npz`` with arrays ``ids`` (study ids) and ``vectors``."""

    provider_id = "precomputed"

    def __init__(self, config: ImageConfig) -> None:
        super().__init__(config)
        if not config.feature_path:
            raise ConfigurationError("precomputed image provider requires image.feature_path")
        path = Path(config.feature_path)
        if not path.exists():
            raise ConfigurationError(f"precomputed image feature file not found: {path}")
        data = np.load(path, allow_pickle=False)
        if "ids" not in data or "vectors" not in data:
            raise ConfigurationError("precomputed features must contain 'ids' and 'vectors' arrays")
        self._vectors = {str(i): v.astype(np.float32) for i, v in zip(data["ids"], data["vectors"], strict=True)}
        dims = {v.shape[0] for v in self._vectors.values()}
        if len(dims) != 1:
            raise ConfigurationError("precomputed image vectors have inconsistent dimensions")
        self._dim = dims.pop()

    def dimension(self) -> int:
        return self._dim

    def embed_study(self, study: ImagingStudy) -> np.ndarray:
        vector = self._vectors.get(study.study_id)
        if vector is None:
            raise ImagingError("no precomputed feature for this study")
        return vector


class InspectPretrainedProvider(ImageEmbeddingProvider):
    """Wrapper for a compatible pretrained 3-D CT encoder exported as TorchScript.

    NOT VALIDATED: the released INSPECT image models are classification
    backbones; whether their features are useful for case similarity is an open
    experimental question.  This provider only defines the integration contract:
    a TorchScript module mapping a ``(1, 1, Z, Y, X)`` float tensor in [0, 1] to a
    ``(1, D)`` feature tensor.  Export the model in the isolated legacy
    environment and point ``image.checkpoint_path`` at it.
    """

    provider_id = "inspect_pretrained"

    def __init__(self, config: ImageConfig) -> None:
        super().__init__(config)
        if not config.checkpoint_path or not Path(config.checkpoint_path).exists():
            raise ConfigurationError(
                "inspect_pretrained image provider requires image.checkpoint_path pointing to an "
                "exported TorchScript encoder; none is bundled with this repository"
            )
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailable("PyTorch is required for the pretrained image encoder") from exc
        self._torch = torch
        self._model = torch.jit.load(config.checkpoint_path, map_location="cpu").eval()

    def dimension(self) -> int:
        return self.config.dimension

    def embed_study(self, study: ImagingStudy) -> np.ndarray:
        if not study.source_path:
            raise ImagingError("study has no source path")
        array = preprocess(load_volume(Path(study.source_path), study.spacing), self.config)
        with self._torch.no_grad():
            out = self._model(self._torch.from_numpy(array)[None, None])
        vector = out.reshape(-1).numpy().astype(np.float32)
        if vector.shape[0] != self.config.dimension:
            raise ConfigurationError(
                f"encoder returned {vector.shape[0]} features; image.dimension is {self.config.dimension}"
            )
        return vector


def build_image_provider(config: ImageConfig) -> ImageEmbeddingProvider:
    providers: dict[str, type[ImageEmbeddingProvider]] = {
        "disabled": DisabledImageProvider,
        "intensity_histogram": IntensityHistogramEncoder,
        "precomputed": PrecomputedFeatureProvider,
        "inspect_pretrained": InspectPretrainedProvider,
    }
    return providers[config.provider](config)


class ImageFeaturePipeline:
    """Computes and caches study-level features; records rejected studies."""

    def __init__(self, session: Session, provider: ImageEmbeddingProvider) -> None:
        self.session = session
        self.provider = provider

    def run(self, recompute: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {"provider": self.provider.provider_id, "version": self.provider.version,
                                   "computed": 0, "cached": 0, "rejected": 0, "missing_source": 0,
                                   "rejections": {}}
        if not self.provider.enabled:
            summary["disabled"] = True
            return summary
        existing = {
            f.study_id: f
            for f in self.session.scalars(
                select(ImageFeature).where(ImageFeature.provider_id == self.provider.provider_id,
                                           ImageFeature.provider_version == self.provider.version)
            )
        }
        for study in self.session.scalars(select(ImagingStudy).order_by(ImagingStudy.study_id)):
            if study.feature_status == "missing_source":
                summary["missing_source"] += 1
                continue
            cached = existing.get(study.study_id)
            if cached is not None and cached.source_hash == study.source_hash and not recompute:
                summary["cached"] += 1
                continue
            self.provider._last_volume = None  # type: ignore[attr-defined]
            try:
                vector = self.provider.embed_study(study)
            except ImagingError as exc:
                study.feature_status = "rejected"
                study.feature_error = str(exc)[:500]
                summary["rejected"] += 1
                summary["rejections"][str(exc)] = summary["rejections"].get(str(exc), 0) + 1
                continue
            volume: Volume | None = getattr(self.provider, "_last_volume", None)
            if volume is not None:
                study.slice_count = volume.slice_count
                study.spacing = [float(s) for s in volume.spacing]
                study.orientation_metadata = {**(study.orientation_metadata or {}), **volume.orientation,
                                              "warnings": volume.warnings}
            study.preprocessing_version = PREPROCESSING_VERSION
            study.feature_status = "ready"
            study.feature_error = None
            if cached is None:
                self.session.add(
                    ImageFeature(study_id=study.study_id, provider_id=self.provider.provider_id,
                                 provider_version=self.provider.version, case_id=study.case_id,
                                 dimension=int(vector.shape[0]), vector=[float(v) for v in vector],
                                 source_hash=study.source_hash)
                )
            else:
                cached.vector = [float(v) for v in vector]
                cached.dimension = int(vector.shape[0])
                cached.source_hash = study.source_hash
                cached.case_id = study.case_id
            summary["computed"] += 1
        self.session.flush()
        return summary
