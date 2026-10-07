"""Bounded OpenZGY reads and survey-coordinate conversion helpers."""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ZgyGeometry:
  """Subset of ZGY geometry needed to preserve survey coordinates."""

  size: tuple[int, int, int]
  zstart: float
  zinc: float
  annotstart: tuple[float, float]
  annotinc: tuple[float, float]
  corners: tuple[tuple[float, float], ...]
  zunitname: str
  hunitname: str

  @classmethod
  def from_reader(cls, reader: Any) -> 'ZgyGeometry':
    return cls(
        size=tuple(int(value) for value in reader.size),
        zstart=float(reader.zstart),
        zinc=float(reader.zinc),
        annotstart=tuple(float(value) for value in reader.annotstart),
        annotinc=tuple(float(value) for value in reader.annotinc),
        corners=tuple(
            tuple(float(value) for value in corner)
            for corner in reader.corners
        ),
        zunitname=str(reader.meta.get('zunitname', '')),
        hunitname=str(reader.meta.get('hunitname', '')),
    )


@contextlib.contextmanager
def open_zgy_reader(path: str | Path) -> Iterator[Any]:
  """Opens one local ZGY reader and closes it on every exit path."""
  try:
    from openzgy.api import ZgyReader
  except ImportError as error:
    raise ImportError(
        'Reading .zgy files requires pyzgy. Install '
        'requirements_seismic_torch.txt in the active environment.'
    ) from error
  with ZgyReader(str(path)) as reader:
    yield reader


def read_window(
    reader: Any,
    start: Sequence[int],
    shape: Sequence[int],
) -> np.ndarray:
  """Performs one bounded float32 read in [inline, crossline, sample] order."""
  start = tuple(int(value) for value in start)
  shape = tuple(int(value) for value in shape)
  if len(start) != 3 or len(shape) != 3:
    raise ValueError('ZGY reads require three start and shape coordinates.')
  if any(value < 0 for value in start) or any(value < 1 for value in shape):
    raise ValueError('ZGY read starts must be nonnegative and shapes positive.')
  limit = tuple(int(value) for value in reader.size)
  stop = tuple(origin + length for origin, length in zip(start, shape))
  if any(end > bound for end, bound in zip(stop, limit)):
    raise ValueError(f'Requested ZGY window {start}..{stop} exceeds {limit}.')
  values = np.empty(shape, dtype=np.float32)
  reader.read(start, values)
  return values


def annotation_to_index(
    geometry: ZgyGeometry,
    inline: float,
    crossline: float,
    z: float,
) -> tuple[float, float, float]:
  """Converts inline/crossline annotations and physical Z to voxel indices."""
  increments = (*geometry.annotinc, geometry.zinc)
  if any(value == 0.0 for value in increments):
    raise ValueError('ZGY annotation increments must be nonzero.')
  return (
      (float(inline) - geometry.annotstart[0]) / geometry.annotinc[0],
      (float(crossline) - geometry.annotstart[1]) / geometry.annotinc[1],
      (float(z) - geometry.zstart) / geometry.zinc,
  )


def index_to_annotation(
    geometry: ZgyGeometry,
    inline_index: float,
    crossline_index: float,
    z_index: float,
) -> tuple[float, float, float]:
  """Converts fractional voxel indices to line annotations and physical Z."""
  return (
      geometry.annotstart[0] + float(inline_index) * geometry.annotinc[0],
      geometry.annotstart[1]
      + float(crossline_index) * geometry.annotinc[1],
      geometry.zstart + float(z_index) * geometry.zinc,
  )


def index_to_world(
    geometry: ZgyGeometry,
    inline_index: float | np.ndarray,
    crossline_index: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Interpolates X/Y from ordered ZGY corners for fractional indices."""
  if geometry.size[0] < 2 or geometry.size[1] < 2:
    raise ValueError('ZGY world conversion requires two horizontal samples.')
  u = np.asarray(inline_index, dtype=np.float64) / (geometry.size[0] - 1)
  v = np.asarray(crossline_index, dtype=np.float64) / (
      geometry.size[1] - 1
  )
  p00, p10, p01, p11 = (
      np.asarray(corner, dtype=np.float64) for corner in geometry.corners
  )
  points = (
      (1.0 - u)[..., None] * (1.0 - v)[..., None] * p00
      + u[..., None] * (1.0 - v)[..., None] * p10
      + (1.0 - u)[..., None] * v[..., None] * p01
      + u[..., None] * v[..., None] * p11
  )
  return points[..., 0], points[..., 1]


def centered_window_start(center: float, length: int, limit: int) -> int:
  """Returns a centered, boundary-clamped start for a fixed-size window."""
  if length < 1 or length > limit:
    raise ValueError(f'Window length {length} is invalid for axis size {limit}.')
  return min(max(int(round(center)) - length // 2, 0), limit - length)


def validate_index(
    index: Sequence[float], geometry: ZgyGeometry
) -> tuple[float, float, float]:
  """Checks that one fractional [inline, crossline, sample] index is in-cube."""
  index = tuple(float(value) for value in index)
  if len(index) != 3:
    raise ValueError('A ZGY query requires inline, crossline, and sample.')
  if any(not np.isfinite(value) for value in index):
    raise ValueError('ZGY query coordinates must be finite.')
  if any(value < 0.0 or value > size - 1 for value, size in zip(index, geometry.size)):
    raise ValueError(f'Query index {index} is outside ZGY size {geometry.size}.')
  return index
