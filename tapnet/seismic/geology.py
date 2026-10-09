"""Paper-inspired 3D stratigraphy with paired seismic, RGT, and TAP labels.

Volumes use [frame, depth, lateral]. Reflectivity and RGT share Wu et al.'s
structural inverse maps; erosion removes ages rather than hiding reflectors.
This is a convolutional seismic model, not a wave-equation simulator.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from tapnet.seismic import synthetic
from tapnet.seismic import structure


SCENARIOS = (
    'layered', 'dipping', 'folded', 'faulted', 'unconformity', 'clinoform', 'mixed'
)


@dataclasses.dataclass(frozen=True)
class GeologicalSeismicConfig:
  """Controls the geological suite independently of the legacy generator."""

  num_frames: int = 256
  height: int = 256
  width: int = 256
  scene_num_frames: int | None = None
  num_horizons: int = 20
  num_queries: int = 24
  wavelet_length: int = 25
  min_wavelet_frequency: float = 0.06
  max_wavelet_frequency: float = 0.14
  noise_std: float = 0.12
  max_fault_throw: float = 80.0
  max_faults: int = 3
  fault_label_width: int = 1
  generator_version: int = 2
  reverse_probability: float = 0.5
  frame_strides: tuple[int, ...] = (1,)
  scenarios: tuple[str, ...] = SCENARIOS

  def tracking_config(self) -> synthetic.SyntheticSeismicConfig:
    """Shares rendering and coordinate semantics with the existing pipeline."""
    return synthetic.SyntheticSeismicConfig(
        num_frames=self.num_frames,
        height=self.height,
        width=self.width,
        num_horizons=self.num_horizons,
        num_queries=self.num_queries,
        wavelet_length=self.wavelet_length,
        min_wavelet_frequency=self.min_wavelet_frequency,
        max_wavelet_frequency=self.max_wavelet_frequency,
        noise_std=self.noise_std,
        max_fault_throw=0.0,
        termination_probability=0.0,
    )

  def validate(self) -> None:
    self.tracking_config().validate()
    integer_fields = (
        'num_frames', 'height', 'width', 'num_horizons', 'num_queries',
        'wavelet_length', 'max_faults', 'fault_label_width', 'generator_version',
    )
    if any(
        isinstance(getattr(self, name), bool)
        or not isinstance(getattr(self, name), (int, np.integer))
        for name in integer_fields
    ):
      raise ValueError('Dimensions, counts, and widths must be integers.')
    if not np.isfinite(self.noise_std) or self.noise_std < 0:
      raise ValueError('noise_std must be finite and non-negative.')
    if not np.isfinite(self.max_fault_throw) or not (
        0 <= self.max_fault_throw < self.height / 2
    ):
      raise ValueError('max_fault_throw must be in [0, height / 2).')
    if self.max_faults < 1 or self.fault_label_width < 0:
      raise ValueError('max_faults must be positive and fault_label_width >= 0.')
    if self.generator_version != 2:
      raise ValueError('This implementation requires generator_version=2.')
    if not 0 <= self.reverse_probability <= 1:
      raise ValueError('reverse_probability must be in [0, 1].')
    if not self.frame_strides or any(
        isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1
        for s in self.frame_strides
    ) or len(set(self.frame_strides)) != len(self.frame_strides):
      raise ValueError('frame_strides must be distinct positive integers.')
    if self.scene_num_frames is not None and (
        isinstance(self.scene_num_frames, bool)
        or not isinstance(self.scene_num_frames, (int, np.integer))
        or self.scene_num_frames < (self.num_frames - 1) * max(self.frame_strides) + 1
    ):
      raise ValueError('scene_num_frames must fit every strided training view.')
    if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
      raise ValueError('scenarios must be non-empty and distinct.')
    if any(name not in SCENARIOS for name in self.scenarios):
      raise ValueError(f'scenarios must be selected from {SCENARIOS!r}.')


def _make_structure(config, scenario, rng):
  """Restores a regular volume through sequential faults, then folding."""
  extent = config.height - 1
  frame_positions = np.linspace(-extent / 2, extent / 2, config.num_frames)
  lateral_positions = np.linspace(-extent / 2, extent / 2, config.width)
  shape = (config.num_frames, config.height, config.width)
  dip, gaussians = structure.sample_fold(config.height, scenario, rng)
  faults = structure.sample_faults(
      config.height, config.max_faults, config.max_fault_throw, rng
  ) if scenario in ('faulted', 'mixed') else []
  clinoform_amplitude = rng.uniform(0.15, 0.3) * extent if scenario in (
      'clinoform', 'mixed'
  ) else 0.0
  eroded = scenario in ('unconformity', 'mixed')
  if eroded:
    erosion_depth = rng.uniform(0.35, 0.55)
    age_gap = rng.uniform(0.06, 0.14) * extent
    cover_slope = rng.uniform(0.6, 1.0)
  fault_mask = np.zeros(shape, dtype=bool)
  discontinuity = np.zeros(shape, dtype=bool)
  unconformity = np.zeros(shape, dtype=bool)
  age = np.empty(shape, dtype=np.float64)
  covered = np.zeros(shape, dtype=bool) if eroded else None
  contact_min = np.inf

  def depositional_age(coordinates):
    x, y, z = coordinates
    material_age = structure.restore_fold(coordinates, config.height, dip, gaussians)
    if clinoform_amplitude:
      front = 2 * y / extent - 1.3 * (material_age / extent - 0.5) + 0.3 * x / extent
      material_age -= clinoform_amplitude / (1 + np.exp(-5 * front))
    return material_age

  # Full scene arrays persist, but expensive inverse-map temporaries stay small.
  # Each slab includes a preceding frame so fault rasterization is seam-free.
  for start in range(0, config.num_frames, 16):
    stop = min(start + 16, config.num_frames)
    halo_start = max(0, start - 1)
    keep = slice(start - halo_start, None)
    slab_shape = (stop - halo_start, config.height, config.width)
    coordinates = (
        frame_positions[halo_start:stop, None, None],
        lateral_positions[None, None, :],
        np.arange(config.height)[None, :, None],
    )
    slab_fault = np.zeros(slab_shape, dtype=bool)
    slab_jump = np.zeros(slab_shape, dtype=bool)
    for fault in reversed(faults):
      distance, active = fault.fault_coordinates(coordinates)
      sheet = (np.abs(distance) <= config.fault_label_width) & active
      for axis in range(3):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        low, high = tuple(low), tuple(high)
        crossing = ((distance[low] >= 0) != (distance[high] >= 0))
        crossing &= active[low] | active[high]
        sheet[high] |= crossing
        if axis == 1:
          slab_jump[high] |= crossing
      slab_fault |= sheet
      coordinates = fault.restore(coordinates)
    x, y, z = coordinates
    slab_age = depositional_age(coordinates)
    if eroded:
      erosion = extent * (erosion_depth + 0.12 * y / extent + 0.16 * (x / extent)**2)
      below = np.broadcast_to(z >= erosion, slab_shape)
      contact_min = min(contact_min, float(depositional_age((x, y, erosion)).min()))
      slab_age = np.where(below, slab_age, (z - erosion) * cover_slope)
      covered[start:stop] = ~below[keep]
      unconformity[start:stop, 1:] = (
          (below[:, 1:] != below[:, :-1]) & ~slab_jump[:, 1:]
      )[keep]
    age[start:stop] = np.broadcast_to(slab_age, slab_shape)[keep]
    fault_mask[start:stop] = slab_fault[keep]
    discontinuity[start:stop] = slab_jump[keep]
  if eroded:
    for start in range(0, config.num_frames, 16):
      stop = min(start + 16, config.num_frames)
      slab = age[start:stop]
      slab[covered[start:stop]] += contact_min - age_gap
  bounds = np.asarray((age.min(), age.max()), dtype=np.float32)
  rgt = np.empty(shape, dtype=np.float32)
  for start in range(0, config.num_frames, 16):
    slab = age[start:start + 16]
    rgt[start:start + 16] = 2 * (slab - bounds[0]) / (bounds[1] - bounds[0]) - 1
  parameters = np.zeros((config.max_faults, len(structure.FAULT_PARAMETER_NAMES)), np.float32)
  controls = np.zeros((config.max_faults, structure.SURFACE_POINT_COUNT, 3), np.float32)
  for index, fault in enumerate(faults):
    parameters[index] = fault.parameters()
    controls[index] = fault.surface_controls
  return {
      'rgt': rgt, 'fault_mask': fault_mask,
      'fault_discontinuity_mask': discontinuity,
      'unconformity_mask': unconformity,
      'fault_count': np.asarray(len(faults), dtype=np.int32),
      'fault_parameters': parameters, 'fault_surface_controls': controls,
      'age_bounds': bounds,
  }


def _extract_horizons(rgt, levels, discontinuity, unconformity):
  """Finds genuine age roots, rejecting jumps and flagging repeated ages."""
  frames, height, width = rgt.shape
  frame_index, lateral_index = np.indices((frames, width))
  surfaces, visibility, validity, root_counts = [], [], [], []
  low_age, high_age = rgt[:, :-1], rgt[:, 1:]
  continuous = ~(discontinuity[:, 1:] | unconformity[:, 1:])
  for level in levels:
    crossing = (
        ((low_age <= level) & (high_age > level))
        | ((low_age > level) & (high_age <= level))
    ) & continuous
    count = np.sum(crossing, axis=1)
    low = np.argmax(crossing, axis=1)
    a = low_age[frame_index, low, lateral_index]
    b = high_age[frame_index, low, lateral_index]
    fraction = np.divide(level - a, b - a, out=np.zeros_like(a), where=b != a)
    depth = low + np.clip(fraction, 0, 1)
    # Absent/ambiguous depths are bounded placeholders, interpreted via masks.
    surfaces.append(depth)
    visibility.append(count > 0)
    validity.append(count <= 1)
    root_counts.append(count)
  return (
      np.asarray(surfaces, dtype=np.float32), np.asarray(visibility),
      np.asarray(validity), np.asarray(root_counts, dtype=np.int32),
  )


def _render_reflectivity(reflectivity, config, rng):
  """Convolves the deformed reflectivity, then applies acquisition noise/gain."""
  frequency = rng.uniform(config.min_wavelet_frequency, config.max_wavelet_frequency)
  wavelet = synthetic._ricker_wavelet(config.wavelet_length, frequency)
  amplitudes = np.empty_like(reflectivity)
  for start in range(0, config.num_frames, 16):
    amplitudes[start:start + 16] = synthetic._convolve_depth(
        reflectivity[start:start + 16], wavelet
    )
  noise = rng.normal(0, config.noise_std, amplitudes.shape).astype(np.float32)
  coherent_noise = (
      0.55 * noise + 0.2 * np.roll(noise, 1, axis=1)
      + 0.15 * np.roll(noise, -1, axis=1)
      + 0.05 * np.roll(noise, 1, axis=2) + 0.05 * np.roll(noise, -1, axis=2)
  )
  gain = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.35), config.height)
  amplitudes = amplitudes * gain[None, :, None] + coherent_noise
  scale = float(np.percentile(np.abs(amplitudes), 99.5))
  seismic = np.clip(amplitudes / scale, -1, 1).astype(np.float32)
  return seismic, wavelet, gain.astype(np.float32), np.asarray(scale, np.float32)


def generate_geological_volume(
    config: GeologicalSeismicConfig,
    rng: np.random.Generator | int | None = None,
    *,
    scenario: str | None = None,
) -> Mapping[str, np.ndarray]:
  """Pairs seismic/RGT by restoring Wu's folded, curved dip-slip structures.

  A shared 1D reflectivity profile is evaluated at the restored material age,
  then convolved in depth. Fault segmentation never controls amplitudes or
  visibility. Repeated horizon intersections have horizon_valid=False.
  """
  config.validate()
  if not isinstance(rng, np.random.Generator):
    rng = np.random.default_rng(rng)
  if scenario is None:
    scenario = str(rng.choice(config.scenarios))
  if scenario not in config.scenarios:
    raise ValueError(f'Scenario {scenario!r} is not in config.scenarios.')
  volume = _make_structure(config, scenario, rng)
  rgt = volume['rgt']
  edges = np.linspace(-0.9, 0.9, config.num_horizons + 1)
  levels = rng.uniform(edges[:-1], edges[1:]).astype(np.float32)
  surfaces, visible, valid, counts = _extract_horizons(
      rgt, levels, volume['fault_discontinuity_mask'], volume['unconformity_mask']
  )
  coefficients = (
      rng.uniform(0.45, 1.0, config.num_horizons)
      * rng.choice((-1, 1), config.num_horizons)
  ).astype(np.float32)
  # Linear resampling of irregular unit-width impulses in the original flat
  # stratigraphy. Both reflectivity and RGT use this same restored coordinate.
  age_scale = (volume['age_bounds'][1] - volume['age_bounds'][0]) / 2
  reflectivity = np.zeros_like(rgt)
  for level, coefficient in zip(levels, coefficients):
    reflectivity += coefficient * np.maximum(1 - np.abs(rgt - level) * age_scale, 0)
  if volume['unconformity_mask'].any():
    reflectivity += rng.uniform(0.45, 1.0) * volume['unconformity_mask']
  seismic, wavelet, gain, scale = _render_reflectivity(reflectivity, config, rng)
  return {
      **volume, 'seismic': seismic, 'reflectivity': reflectivity,
      'wavelet': wavelet, 'depth_gain': gain, 'amplitude_scale': scale,
      'horizon_depths': surfaces, 'horizon_visible': visible,
      'horizon_valid': valid, 'horizon_root_count': counts,
      'horizon_rgt': levels, 'horizon_reflectivity': coefficients,
      'scenario_id': np.asarray(SCENARIOS.index(scenario), dtype=np.int32),
  }


def generate_geological_sample(
    config: GeologicalSeismicConfig,
    rng: np.random.Generator | int | None = None,
    *,
    frame_stride: int | None = None,
    scenario: str | None = None,
    include_volume: bool = False,
) -> Mapping[str, np.ndarray]:
  """Makes TAPIR labels from the same geology as the paired dense volumes."""
  config.validate()
  if not isinstance(rng, np.random.Generator):
    rng = np.random.default_rng(rng)
  if frame_stride is not None and frame_stride not in config.frame_strides:
    raise ValueError('frame_stride must belong to config.frame_strides.')
  scene_frames = config.scene_num_frames or (
      (config.num_frames - 1) * max(config.frame_strides) + 1
  )
  scene_config = dataclasses.replace(
      config, num_frames=scene_frames, scene_num_frames=None, frame_strides=(1,)
  )
  volume = dict(generate_geological_volume(scene_config, rng, scenario=scenario))
  reversed_sweep = bool(rng.random() < config.reverse_probability)
  if frame_stride is None:
    frame_stride = int(rng.choice(config.frame_strides))
  left_margin = (config.num_frames // 2) * max(config.frame_strides)
  right_margin = (config.num_frames - 1 - config.num_frames // 2) * max(config.frame_strides)
  center = int(np.clip(scene_frames // 2, left_margin, scene_frames - 1 - right_margin))
  indices = center + (
      np.arange(config.num_frames, dtype=np.int32) - config.num_frames // 2
  ) * frame_stride
  for name in ('seismic', 'reflectivity', 'rgt', 'fault_mask',
               'fault_discontinuity_mask', 'unconformity_mask'):
    array = volume[name][::-1] if reversed_sweep else volume[name]
    volume[name] = array[indices]
  for name in ('horizon_depths', 'horizon_visible', 'horizon_valid', 'horizon_root_count'):
    array = volume[name][:, ::-1] if reversed_sweep else volume[name]
    volume[name] = array[:, indices]
  tracks = synthetic._sample_tracks(
      volume['horizon_depths'], volume['horizon_visible'] & volume['horizon_valid'],
      config.tracking_config(), rng,
  )
  for query, horizon in enumerate(tracks['trackgroup']):
    lateral = int(tracks['query_points'][query, 2])
    tracks['occluded'][query] = ~volume['horizon_visible'][horizon, :, lateral]
    tracks['label_valid'][query] = volume['horizon_valid'][horizon, :, lateral]
  sample = {
      'video': np.repeat(volume['seismic'][..., None], 3, axis=-1),
      **tracks,
      'faulted': np.asarray(volume['fault_count'] > 0),
      'sweep_reversed': np.asarray(reversed_sweep),
      'frame_stride': np.asarray(frame_stride, dtype=np.int32),
      'scene_num_frames': np.asarray(scene_frames, dtype=np.int32),
      'frame_indices': indices,
      'scenario_id': volume['scenario_id'],
  }
  if include_volume:
    sample.update(volume)
  return sample
