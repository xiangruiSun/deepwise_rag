"""CT volume loading (spec section 14.1).

DICOM series are ordered by *spatial position* (projection of
``ImagePositionPatient`` on the slice normal), never by file name or instance
number alone, and are converted to Hounsfield units with the per-slice rescale
slope/intercept.  Studies with inconsistent geometry are rejected with a
reason instead of being silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class ImagingError(ValueError):
    """The study cannot be used; the message is stored as the rejection reason."""


@dataclass
class Volume:
    hu: np.ndarray  # (Z, Y, X) float32 in Hounsfield units
    spacing: tuple[float, float, float]  # (z, y, x) millimetres
    orientation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def slice_count(self) -> int:
        return int(self.hu.shape[0])


def discover_series(directory: Path) -> dict[str, list[Path]]:
    """Group DICOM files below ``directory`` by SeriesInstanceUID."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    series: dict[str, list[Path]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
        except (InvalidDicomError, OSError):
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "")) or "unknown"
        series.setdefault(uid, []).append(path)
    return series


def load_dicom_series(directory: Path, series_uid: str | None = None,
                      spacing_tolerance: float = 0.05) -> Volume:
    import pydicom

    all_series = discover_series(directory)
    if not all_series:
        raise ImagingError("no readable DICOM files found")
    if series_uid is None:
        # deterministic choice: the series with the most slices, ties by UID
        series_uid = sorted(all_series, key=lambda u: (-len(all_series[u]), u))[0]
    if series_uid not in all_series:
        raise ImagingError(f"series {series_uid} not found")
    warnings: list[str] = []
    if len(all_series) > 1:
        warnings.append(f"{len(all_series)} series present; using the largest")
    datasets = [pydicom.dcmread(p) for p in all_series[series_uid]]
    if len(datasets) < 2:
        raise ImagingError("series has fewer than 2 slices")

    first = datasets[0]
    for required in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
        if any(not hasattr(ds, required) for ds in datasets):
            raise ImagingError(f"missing required spatial attribute {required}")
    orientation = [float(v) for v in first.ImageOrientationPatient]
    for ds in datasets:
        if not np.allclose([float(v) for v in ds.ImageOrientationPatient], orientation, atol=1e-3):
            raise ImagingError("inconsistent slice orientation within series")
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ImagingError("inconsistent slice dimensions within series")
    row_dir, col_dir = np.asarray(orientation[:3]), np.asarray(orientation[3:])
    normal = np.cross(row_dir, col_dir)
    positions = np.asarray([[float(v) for v in ds.ImagePositionPatient] @ normal for ds in datasets])
    order = np.argsort(positions, kind="stable")
    sorted_pos = positions[order]
    gaps = np.diff(sorted_pos)
    if np.any(gaps <= 1e-6):
        raise ImagingError("duplicate slice positions within series")
    z_spacing = float(np.median(gaps))
    if np.any(np.abs(gaps - z_spacing) > spacing_tolerance * z_spacing):
        raise ImagingError("non-uniform slice spacing (missing or irregular slices)")

    slices = []
    for idx in order:
        ds = datasets[int(idx)]
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)
    pixel_spacing = [float(v) for v in first.PixelSpacing]
    return Volume(
        hu=np.stack(slices, axis=0),
        spacing=(z_spacing, pixel_spacing[0], pixel_spacing[1]),
        orientation={"image_orientation_patient": orientation, "series_uid": series_uid,
                     "slice_normal": [float(v) for v in normal]},
        warnings=warnings,
    )


def load_volume(path: Path, spacing: list[float] | None = None) -> Volume:
    """Load a study from a DICOM directory, a ``.npy`` volume or a NIfTI file."""
    if not path.exists():
        raise ImagingError("imaging source not found")
    if path.is_dir():
        return load_dicom_series(path)
    name = path.name.lower()
    if name.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3:
            raise ImagingError(f"expected a 3-D volume, got shape {array.shape}")
        if spacing is None or len(spacing) != 3:
            raise ImagingError("NumPy volumes carry no geometry: spacing metadata is required")
        return Volume(
            hu=array.astype(np.float32),
            spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            orientation={"source": "numpy", "assumed_axes": "ZYX"},
            warnings=["NumPy volume: Hounsfield units and ZYX axis order are assumed, not verified"],
        )
    if name.endswith((".nii", ".nii.gz")):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImagingError("SimpleITK is required to read NIfTI volumes") from exc
        image = sitk.ReadImage(str(path))
        sx, sy, sz = image.GetSpacing()
        return Volume(
            hu=sitk.GetArrayFromImage(image).astype(np.float32),
            spacing=(float(sz), float(sy), float(sx)),
            orientation={"direction": [float(v) for v in image.GetDirection()], "source": "nifti"},
        )
    raise ImagingError(f"unsupported imaging format: {path.suffix}")
