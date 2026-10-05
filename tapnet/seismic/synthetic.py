"""Procedural seismic examples in TAPIR's point-tracking format.

This module intentionally depends only on NumPy.  TensorFlow wrapping for the
training pipeline lives in ``tapnet.seismic.dataset`` so generation and label
semantics can be tested without installing the full TAPIR training stack.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Mapping, Union

import numpy as np


@dataclasses.dataclass(frozen=True)
class SyntheticSeismicConfig:
  """Controls one synthetic inline/crossline sweep.

  Axis convention is ``[frame, depth, lateral]``.  The generated TAPIR video
  therefore has shape ``[num_frames, height, width, 3]``.
  """

  num_frames: int = 24
  height: int = 256
  width: int = 256
  num_horizons: int = 16
  num_queries: int = 64
  wavelet_length: int = 33
  min_wavelet_frequency: float = 0.06
  max_wavelet_frequency: float = 0.14
  noise_std: float = 0.12
  max_fault_throw: float = 12.0
  fault_probability: float = 0.7
  termination_probability: float = 0.25
  reverse_probability: float = 0.5

  def validate(self) -> None:
    """Raises ``ValueError`` when dimensions cannot make valid examples."""
    if self.num_frames < 2:
      raise ValueError('num_frames must be at least 2.')
    if self.height < 64 or self.width < 8:
      raise ValueError('height must be >= 64 and width must be >= 8.')
    if self.num_horizons < 2:
      raise ValueError('num_horizons must be at least 2.')
    if self.num_queries < 1:
      raise ValueError('num_queries must be positive.')
    if self.wavelet_length < 3 or self.wavelet_length % 2 == 0:
      raise ValueError('wavelet_length must be an odd integer >= 3.')
    if not 0 < self.min_wavelet_frequency < self.max_wavelet_frequency < 0.5:
      raise ValueError('Wavelet frequencies must satisfy 0 < min < max < 0.5.')
    for name in (
        'fault_probability',
        'termination_probability',
        'reverse_probability',
    ):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1].')
    margin = _depth_margin(self)
    usable_depth = self.height - 2 * margin
    if usable_depth <= self.num_horizons:
      raise ValueError(
          'height is too small for num_horizons and max_fault_throw.'
      )


def _depth_margin(config: SyntheticSeismicConfig) -> int:
  return max(20, int(np.ceil(config.max_fault_throw)) + 18)


def _ricker_wavelet(length: int, frequency: float) -> np.ndarray:
  sample = np.arange(length, dtype=np.float32) - (length - 1) / 2
  phase = np.pi * frequency * sample
  wavelet = (1.0 - 2.0 * phase**2) * np.exp(-(phase**2))
  wavelet /= np.max(np.abs(wavelet))
  return wavelet.astype(np.float32)


def _convolve_depth(reflectivity: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
  """Applies a zero-phase wavelet along the depth axis using an FFT."""
  full_length = reflectivity.shape[1] + wavelet.size - 1
  fft_length = 1 << (full_length - 1).bit_length()
  reflectivity_spectrum = np.fft.rfft(
      reflectivity, n=fft_length, axis=1
  )
  wavelet_spectrum = np.fft.rfft(wavelet, n=fft_length)
  convolved = np.fft.irfft(
      reflectivity_spectrum * wavelet_spectrum[None, :, None],
      n=fft_length,
      axis=1,
  )
  start = (wavelet.size - 1) // 2
  stop = start + reflectivity.shape[1]
  return convolved[:, start:stop].astype(np.float32)


def _make_horizons(
    config: SyntheticSeismicConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Returns depth surfaces, visibility masks, and whether a fault was used."""
  frame = np.linspace(-1.0, 1.0, config.num_frames, dtype=np.float32)
  lateral = np.linspace(-1.0, 1.0, config.width, dtype=np.float32)
  frame_grid, lateral_grid = np.meshgrid(frame, lateral, indexing='ij')

  dip = rng.uniform(-10.0, 10.0) * frame_grid
  cross_dip = rng.uniform(-7.0, 7.0) * lateral_grid
  curvature = rng.uniform(-6.0, 6.0) * (frame_grid**2 - 1.0 / 3.0)
  fold = rng.uniform(2.0, 8.0) * np.sin(
      rng.uniform(0.8, 1.8) * np.pi * frame_grid
      + rng.uniform(-np.pi, np.pi)
      + rng.uniform(-1.0, 1.0) * lateral_grid
  )
  shared_structure = dip + cross_dip + curvature + fold

  faulted = bool(rng.random() < config.fault_probability)
  if faulted:
    fault_slope = rng.uniform(-0.7, 0.7)
    fault_offset = rng.uniform(-0.45, 0.45)
    fault_side = frame_grid - fault_slope * lateral_grid > fault_offset
    fault_throw = rng.uniform(
        -config.max_fault_throw, config.max_fault_throw
    )
    shared_structure = shared_structure + fault_side * fault_throw

  margin = _depth_margin(config)
  bases = np.linspace(
      margin,
      config.height - margin - 1,
      config.num_horizons,
      dtype=np.float32,
  )
  horizon_phase = rng.uniform(-np.pi, np.pi, size=config.num_horizons)
  surfaces = []
  for horizon_index, base in enumerate(bases):
    local_relief = rng.uniform(0.3, 1.3) * np.sin(
        np.pi * frame_grid
        + horizon_phase[horizon_index]
        + rng.uniform(-0.8, 0.8) * lateral_grid
    )
    surfaces.append(base + shared_structure + local_relief)
  surface_array = np.stack(surfaces).astype(np.float32)
  surface_array = np.clip(surface_array, 1.0, config.height - 2.0)

  visibility = np.ones(surface_array.shape, dtype=bool)
  for horizon_index in range(config.num_horizons):
    if rng.random() >= config.termination_probability:
      continue
    boundary = (
        frame_grid
        + rng.uniform(-0.7, 0.7) * lateral_grid
        - rng.uniform(-0.5, 0.5)
    )
    if rng.random() < 0.5:
      visibility[horizon_index] = boundary <= 0.0
    else:
      visibility[horizon_index] = boundary >= 0.0

  return surface_array, visibility, faulted


def _render_amplitudes(
    surfaces: np.ndarray,
    visibility: np.ndarray,
    config: SyntheticSeismicConfig,
    rng: np.random.Generator,
) -> np.ndarray:
  reflectivity = np.zeros(
      (config.num_frames, config.height, config.width), dtype=np.float32
  )
  frame_index, lateral_index = np.indices(
      (config.num_frames, config.width)
  )
  base_amplitudes = rng.uniform(0.45, 1.0, size=config.num_horizons)
  polarities = rng.choice(np.array([-1.0, 1.0]), size=config.num_horizons)

  for horizon_index in range(config.num_horizons):
    depths = surfaces[horizon_index]
    low = np.floor(depths).astype(np.int32)
    fraction = depths - low
    lateral_gain = 1.0 + 0.2 * np.sin(
        np.linspace(0, rng.uniform(1.0, 3.0) * np.pi, config.width)
        + rng.uniform(-np.pi, np.pi)
    )
    values = (
        base_amplitudes[horizon_index]
        * polarities[horizon_index]
        * lateral_gain[None, :]
        * visibility[horizon_index]
    )
    np.add.at(
        reflectivity,
        (frame_index, low, lateral_index),
        values * (1.0 - fraction),
    )
    np.add.at(
        reflectivity,
        (frame_index, low + 1, lateral_index),
        values * fraction,
    )

  frequency = rng.uniform(
      config.min_wavelet_frequency, config.max_wavelet_frequency
  )
  amplitudes = _convolve_depth(
      reflectivity, _ricker_wavelet(config.wavelet_length, frequency)
  )

  noise = rng.normal(0.0, config.noise_std, size=amplitudes.shape).astype(
      np.float32
  )
  coherent_noise = (
      0.55 * noise
      + 0.2 * np.roll(noise, 1, axis=1)
      + 0.15 * np.roll(noise, -1, axis=1)
      + 0.05 * np.roll(noise, 1, axis=2)
      + 0.05 * np.roll(noise, -1, axis=2)
  )
  depth_gain = np.linspace(
      rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.35), config.height
  )[None, :, None]
  amplitudes = amplitudes * depth_gain + coherent_noise

  scale = float(np.percentile(np.abs(amplitudes), 99.5))
  if scale <= np.finfo(np.float32).eps:
    raise RuntimeError('Synthetic amplitude normalization received zero scale.')
  return np.clip(amplitudes / scale, -1.0, 1.0).astype(np.float32)


def _sample_tracks(
    surfaces: np.ndarray,
    visibility: np.ndarray,
    config: SyntheticSeismicConfig,
    rng: np.random.Generator,
) -> Mapping[str, np.ndarray]:
  valid_pairs = np.argwhere(np.any(visibility, axis=1))
  if valid_pairs.size == 0:
    raise RuntimeError('No visible horizon/lateral pair was generated.')
  selected_pairs = valid_pairs[
      rng.integers(0, len(valid_pairs), size=config.num_queries)
  ]
  horizon_ids = selected_pairs[:, 0].astype(np.int32)
  lateral_positions = selected_pairs[:, 1].astype(np.int32)
  query_points = np.empty((config.num_queries, 3), dtype=np.float32)
  target_points = np.empty(
      (config.num_queries, config.num_frames, 2), dtype=np.float32
  )
  occluded = np.empty(
      (config.num_queries, config.num_frames), dtype=bool
  )

  for query_index, (horizon_id, lateral_position) in enumerate(
      zip(horizon_ids, lateral_positions)
  ):
    track_visibility = visibility[horizon_id, :, lateral_position]
    visible_frames = np.flatnonzero(track_visibility)
    query_frame = int(rng.choice(visible_frames))
    track_depth = surfaces[horizon_id, :, lateral_position]
    query_points[query_index] = (
        query_frame,
        track_depth[query_frame],
        lateral_position,
    )
    target_points[query_index, :, 0] = lateral_position
    target_points[query_index, :, 1] = track_depth
    occluded[query_index] = ~track_visibility

  return {
      'query_points': query_points,
      'target_points': target_points,
      'occluded': occluded,
      'label_valid': np.ones_like(occluded, dtype=bool),
      'trackgroup': horizon_ids,
  }


def generate_synthetic_sample(
    config: SyntheticSeismicConfig,
    rng: Union[np.random.Generator, int, None] = None,
) -> Mapping[str, np.ndarray]:
  """Generates one deterministic TAPIR-compatible seismic example.

  Args:
    config: Geometry and simulation controls.
    rng: NumPy generator or seed.  ``None`` requests a non-deterministic seed.

  Returns:
    A mapping with video, query/target coordinates, occlusion, supervision
    validity, horizon grouping, and two scalar generation flags.
  """
  config.validate()
  if not isinstance(rng, np.random.Generator):
    rng = np.random.default_rng(rng)

  surfaces, visibility, faulted = _make_horizons(config, rng)
  reversed_sweep = bool(rng.random() < config.reverse_probability)
  if reversed_sweep:
    surfaces = surfaces[:, ::-1]
    visibility = visibility[:, ::-1]

  amplitudes = _render_amplitudes(surfaces, visibility, config, rng)
  tracks = _sample_tracks(surfaces, visibility, config, rng)
  video = np.repeat(amplitudes[..., None], 3, axis=-1)

  return {
      'video': video.astype(np.float32),
      **tracks,
      'faulted': np.asarray(faulted, dtype=bool),
      'sweep_reversed': np.asarray(reversed_sweep, dtype=bool),
  }


def iter_synthetic_samples(
    config: SyntheticSeismicConfig, seed: int = 0
) -> Iterator[Mapping[str, np.ndarray]]:
  """Yields a reproducible infinite stream of independent examples."""
  stream_rng = np.random.default_rng(seed)
  while True:
    sample_seed = int(
        stream_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
    )
    yield generate_synthetic_sample(config, sample_seed)
