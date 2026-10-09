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
  min_fault_throw: float = 0.0
  max_fault_throw: float = 12.0
  fault_probability: float = 0.7
  max_faults: int = 1
  max_fault_offset: float = 0.45
  divide_fault_throw_by_count: bool = True
  min_fault_damage_width: int = 0
  max_fault_damage_width: int = 0
  fault_query_probability: float = 0.0
  termination_probability: float = 0.25
  reverse_probability: float = 0.5
  frame_strides: tuple[int, ...] = (1,)
  center_aligned_views: bool = False

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
    if self.max_faults < 1:
      raise ValueError('max_faults must be positive.')
    if not 0.0 <= self.min_fault_throw <= self.max_fault_throw:
      raise ValueError(
          'Fault throws must satisfy 0 <= min_fault_throw <= '
          'max_fault_throw.'
      )
    if not 0.0 <= self.max_fault_offset < 1.0:
      raise ValueError('max_fault_offset must be in [0, 1).')
    if (
        self.divide_fault_throw_by_count
        and self.min_fault_throw * self.max_faults > self.max_fault_throw
    ):
      raise ValueError(
          'min_fault_throw is incompatible with divided multi-fault throws.'
      )
    if not (
        0 <= self.min_fault_damage_width <= self.max_fault_damage_width
    ):
      raise ValueError(
          'Fault damage widths must satisfy 0 <= min <= max.'
      )
    if not self.frame_strides:
      raise ValueError('frame_strides must not be empty.')
    if any(stride < 1 for stride in self.frame_strides):
      raise ValueError('frame_strides must contain positive integers.')
    if len(set(self.frame_strides)) != len(self.frame_strides):
      raise ValueError('frame_strides must not contain duplicates.')
    if self.wavelet_length < 3 or self.wavelet_length % 2 == 0:
      raise ValueError('wavelet_length must be an odd integer >= 3.')
    if not 0 < self.min_wavelet_frequency < self.max_wavelet_frequency < 0.5:
      raise ValueError('Wavelet frequencies must satisfy 0 < min < max < 0.5.')
    for name in (
        'fault_probability',
        'fault_query_probability',
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


def _make_horizons_and_damage(
    config: SyntheticSeismicConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
  """Returns surfaces, visibility, fault count, and the fault-core mask."""
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
  fault_damage = np.zeros(frame_grid.shape, dtype=bool)

  fault_count = 0
  if rng.random() < config.fault_probability:
    fault_count = (
        1
        if config.max_faults == 1
        else int(rng.integers(1, config.max_faults + 1))
    )
  for _ in range(fault_count):
    fault_slope = rng.uniform(-0.7, 0.7)
    fault_offset = rng.uniform(-config.max_fault_offset, config.max_fault_offset)
    signed_fault_distance = (
        frame_grid - fault_slope * lateral_grid - fault_offset
    )
    fault_side = signed_fault_distance > 0.0
    throw_limit = (
        config.max_fault_throw / fault_count
        if config.divide_fault_throw_by_count
        else config.max_fault_throw
    )
    if config.min_fault_throw == 0.0:
      fault_throw = rng.uniform(-throw_limit, throw_limit)
    else:
      throw_magnitude = rng.uniform(config.min_fault_throw, throw_limit)
      fault_throw = throw_magnitude * rng.choice((-1.0, 1.0))
    shared_structure = shared_structure + fault_side * fault_throw
    if config.max_fault_damage_width > 0:
      damage_width = int(rng.integers(
          config.min_fault_damage_width,
          config.max_fault_damage_width + 1,
      ))
      if damage_width > 0:
        half_width_normalized = damage_width / max(
            config.num_frames - 1, 1
        )
        fault_damage |= (
            np.abs(signed_fault_distance) <= half_width_normalized
        )

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

  visibility &= ~fault_damage[None, :, :]

  return surface_array, visibility, fault_count, fault_damage


def _make_horizons(
    config: SyntheticSeismicConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
  """Returns depth surfaces, visibility masks, and generated fault count."""
  surfaces, visibility, fault_count, _ = _make_horizons_and_damage(config, rng)
  return surfaces, visibility, fault_count


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
    fault_damage: np.ndarray | None = None,
) -> Mapping[str, np.ndarray]:
  valid_pairs = np.argwhere(np.any(visibility, axis=1))
  if valid_pairs.size == 0:
    raise RuntimeError('No visible horizon/lateral pair was generated.')
  if config.fault_query_probability > 0.0 and fault_damage is not None:
    crossing_pairs = []
    for horizon_id, lateral_position in valid_pairs:
      damaged_frames = np.flatnonzero(fault_damage[:, lateral_position])
      if damaged_frames.size == 0:
        continue
      track_visibility = visibility[horizon_id, :, lateral_position]
      if (
          np.any(track_visibility[:damaged_frames[0]])
          and np.any(track_visibility[damaged_frames[-1] + 1:])
      ):
        crossing_pairs.append((horizon_id, lateral_position))
    crossing_pairs = np.asarray(crossing_pairs, dtype=np.int32)
  else:
    crossing_pairs = np.empty((0, 2), dtype=np.int32)
  if crossing_pairs.size:
    selected_pairs = np.asarray([
        (
            crossing_pairs[rng.integers(0, len(crossing_pairs))]
            if rng.random() < config.fault_query_probability
            else valid_pairs[rng.integers(0, len(valid_pairs))]
        )
        for _ in range(config.num_queries)
    ])
  else:
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
    *,
    frame_stride: int | None = None,
) -> Mapping[str, np.ndarray]:
  """Generates one deterministic TAPIR-compatible seismic example.

  Args:
    config: Geometry and simulation controls.
    rng: NumPy generator or seed.  ``None`` requests a non-deterministic seed.
    frame_stride: Optional explicit member of ``config.frame_strides``. When
      omitted, one configured stride is selected uniformly.

  Returns:
    A mapping with video, query/target coordinates, occlusion, supervision
    validity, horizon grouping, and two scalar generation flags.
  """
  config.validate()
  if not isinstance(rng, np.random.Generator):
    rng = np.random.default_rng(rng)

  if frame_stride is not None and frame_stride not in config.frame_strides:
    raise ValueError(
        f'frame_stride {frame_stride} is not in configured frame_strides '
        f'{config.frame_strides!r}.'
    )

  scene_num_frames = (
      (config.num_frames - 1) * max(config.frame_strides) + 1
  )
  scene_config = dataclasses.replace(
      config, num_frames=scene_num_frames, frame_strides=(1,)
  )
  surfaces, visibility, fault_count, fault_damage = (
      _make_horizons_and_damage(scene_config, rng)
  )
  reversed_sweep = bool(rng.random() < config.reverse_probability)
  if reversed_sweep:
    surfaces = surfaces[:, ::-1]
    visibility = visibility[:, ::-1]
    fault_damage = fault_damage[::-1]

  amplitudes = _render_amplitudes(surfaces, visibility, scene_config, rng)
  if frame_stride is None:
    frame_stride = int(rng.choice(config.frame_strides))
  if config.center_aligned_views:
    center = (config.num_frames // 2) * max(config.frame_strides)
    frame_indices = center + (
        np.arange(config.num_frames, dtype=np.int32)
        - config.num_frames // 2
    ) * frame_stride
  else:
    frame_indices = (
        np.arange(config.num_frames, dtype=np.int32) * frame_stride
    )
  surfaces = surfaces[:, frame_indices]
  visibility = visibility[:, frame_indices]
  fault_damage = fault_damage[frame_indices]
  amplitudes = amplitudes[frame_indices]
  tracks = _sample_tracks(
      surfaces, visibility, config, rng, fault_damage=fault_damage
  )
  video = np.repeat(amplitudes[..., None], 3, axis=-1)

  return {
      'video': video.astype(np.float32),
      **tracks,
      'faulted': np.asarray(fault_count > 0, dtype=bool),
      'sweep_reversed': np.asarray(reversed_sweep, dtype=bool),
      'frame_stride': np.asarray(frame_stride, dtype=np.int32),
      'scene_num_frames': np.asarray(scene_num_frames, dtype=np.int32),
      'frame_indices': frame_indices,
  }


def iter_synthetic_samples(
    config: SyntheticSeismicConfig, seed: int = 0
) -> Iterator[Mapping[str, np.ndarray]]:
  """Yields a reproducible infinite stream of independent examples."""
  stream_rng = np.random.default_rng(seed)
  sample_index = 0
  while True:
    sample_seed = int(
        stream_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
    )
    frame_stride = config.frame_strides[
        sample_index % len(config.frame_strides)
    ]
    yield generate_synthetic_sample(
        config, sample_seed, frame_stride=frame_stride
    )
    sample_index += 1
