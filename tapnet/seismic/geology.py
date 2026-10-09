"""Paper-inspired 3D stratigraphy with paired seismic, RGT, and TAP labels.

Volumes use [frame, depth, lateral]. RGT increases with depth within each
fault block; erosion removes ages rather than arbitrarily hiding reflectors.
This is a convolutional seismic model, not a wave-equation simulator.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from tapnet.seismic import synthetic


SCENARIOS = (
    'layered', 'dipping', 'folded', 'faulted', 'unconformity', 'clinoform', 'mixed'
)


@dataclasses.dataclass(frozen=True)
class GeologicalSeismicConfig:
  """Controls the geological suite independently of the legacy generator."""

  num_frames: int = 32
  height: int = 128
  width: int = 128
  num_horizons: int = 20
  num_queries: int = 24
  wavelet_length: int = 25
  min_wavelet_frequency: float = 0.06
  max_wavelet_frequency: float = 0.14
  noise_std: float = 0.12
  max_fault_throw: float = 20.0
  max_faults: int = 3
  fault_damage_width: int = 1
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
        'wavelet_length', 'max_faults', 'fault_damage_width',
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
    if self.max_faults < 1 or self.fault_damage_width < 0:
      raise ValueError('max_faults must be positive and damage width >= 0.')
    if self.fault_damage_width > (self.num_frames - 1) / 2:
      raise ValueError('fault_damage_width must be less than half the sweep length.')
    if not 0 <= self.reverse_probability <= 1:
      raise ValueError('reverse_probability must be in [0, 1].')
    if not self.frame_strides or any(
        isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1
        for s in self.frame_strides
    ) or len(set(self.frame_strides)) != len(self.frame_strides):
      raise ValueError('frame_strides must be distinct positive integers.')
    if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
      raise ValueError('scenarios must be non-empty and distinct.')
    if any(name not in SCENARIOS for name in self.scenarios):
      raise ValueError(f'scenarios must be selected from {SCENARIOS!r}.')


def _make_rgt(config, scenario, rng):
  """Makes a monotone stratigraphic coordinate and geological boundaries."""
  frame = np.linspace(-1, 1, config.num_frames, dtype=np.float32)[:, None]
  lateral = np.linspace(-1, 1, config.width, dtype=np.float32)[None, :]
  depth = np.arange(config.height, dtype=np.float32)[None, :, None]
  scale = config.height - 1
  structure = np.zeros((config.num_frames, config.width), dtype=np.float32)
  if scenario != 'layered':
    dip = 0.25 if scenario == 'dipping' else 0.08
    structure += scale * (
        rng.uniform(-dip, dip) * frame
        + rng.uniform(-dip, dip) * lateral
    )
  if scenario in ('folded', 'faulted', 'unconformity', 'mixed'):
    for _ in range(3):
      distance = (
          (frame - rng.uniform(-0.8, 0.8))**2
          + (lateral - rng.uniform(-0.8, 0.8))**2
      )
      structure += scale * rng.uniform(-0.12, 0.12) * np.exp(
          -distance / rng.uniform(0.1, 0.7)
      )

  displacement = np.zeros_like(structure)
  fault_core = np.zeros_like(structure, dtype=bool)
  fault_count = 0
  if scenario in ('faulted', 'mixed') and config.max_fault_throw > 0:
    fault_count = int(rng.integers(1, config.max_faults + 1))
  for _ in range(fault_count):
    distance = (
        frame - rng.uniform(-0.65, 0.65) * lateral - rng.uniform(-0.4, 0.4)
    )
    throw = rng.uniform(0.4, 1.0) * config.max_fault_throw / fault_count
    displacement += (distance > 0) * throw * rng.choice((-1, 1))
    # Include a rasterized boundary even when no damage zone is requested.
    fault_core |= np.abs(distance) <= (
        max(0.5, config.fault_damage_width) / (config.num_frames - 1)
    )
  material_depth = depth - displacement[:, None, :]
  # A depth-dependent deformation changes layer thickness without crossings.
  age = (
      material_depth
      - structure[:, None, :] * (0.7 + 0.3 * material_depth / scale)
  ) / scale
  if scenario in ('clinoform', 'mixed'):
    front = lateral[:, None, :] - 1.3 * (material_depth / scale - 0.5)
    front += 0.15 * frame[:, None, :]
    age -= rng.uniform(0.15, 0.3) / (1 + np.exp(-5 * front))

  unconformity = np.zeros(age.shape, dtype=bool)
  if scenario in ('unconformity', 'mixed'):
    erosion = scale * (
        rng.uniform(0.35, 0.55) + 0.06 * lateral + 0.04 * frame**2
    )
    below = material_depth >= erosion[:, None, :]
    boundary_index = np.argmax(below, axis=1)
    frame_index, lateral_index = np.indices(structure.shape)
    old_contact_age = age[frame_index, boundary_index, lateral_index]
    # All cover ages precede every preserved age at the erosional contact.
    cover_contact_age = old_contact_age.min() - rng.uniform(0.06, 0.14)
    cover_age = (
        cover_contact_age
        + (material_depth - erosion[:, None, :]) / scale * rng.uniform(0.6, 1.0)
    )
    age = np.where(below, age, cover_age)
    unconformity[frame_index, boundary_index, lateral_index] = True

  age_min, age_max = float(age.min()), float(age.max())
  rgt = (2 * (age - age_min) / (age_max - age_min) - 1).astype(np.float32)
  fault_mask = np.broadcast_to(fault_core[:, None, :], rgt.shape).copy()
  return rgt, fault_mask, unconformity, fault_count


def _extract_horizons(rgt, levels, fault_mask, unconformity_mask):
  """Inverts each depth trace; ages erased by erosion have no visible horizon."""
  frames, height, width = rgt.shape
  frame_index, lateral_index = np.indices((frames, width))
  surfaces, visibility = [], []
  for level in levels:
    high = np.clip(np.sum(rgt < level, axis=1), 1, height - 1)
    low_age = rgt[frame_index, high - 1, lateral_index]
    high_age = rgt[frame_index, high, lateral_index]
    fraction = np.clip((level - low_age) / (high_age - low_age), 0, 1)
    surfaces.append(high - 1 + fraction)
    visibility.append(
        (level >= rgt[:, 0]) & (level <= rgt[:, -1])
        & ~unconformity_mask[frame_index, high, lateral_index]
        & ~fault_mask[frame_index, high, lateral_index]
    )
  return np.asarray(surfaces, dtype=np.float32), np.asarray(visibility)


def generate_geological_volume(
    config: GeologicalSeismicConfig,
    rng: np.random.Generator | int | None = None,
    *,
    scenario: str | None = None,
) -> Mapping[str, np.ndarray]:
  """Returns paired 3D seismic/RGT and the horizons used for rendering.

  Horizon levels are globally shared across traces and fault blocks. No
  per-trace normalization or independent, unlabelled reflector termination is
  applied. Coordinates for absent horizons are bounded surrogate positions;
  ``horizon_visible`` must be used to interpret them.
  """
  config.validate()
  if not isinstance(rng, np.random.Generator):
    rng = np.random.default_rng(rng)
  if scenario is None:
    scenario = str(rng.choice(config.scenarios))
  if scenario not in config.scenarios:
    raise ValueError(f'Scenario {scenario!r} is not in config.scenarios.')
  rgt, fault_mask, unconformity, fault_count = _make_rgt(config, scenario, rng)
  # Irregular, ordered ages produce variable bed spacing and thin beds.
  edges = np.linspace(-0.9, 0.9, config.num_horizons + 1)
  levels = rng.uniform(edges[:-1], edges[1:]).astype(np.float32)
  surfaces, visible = _extract_horizons(rgt, levels, fault_mask, unconformity)
  render_surfaces, render_visible = surfaces, visible
  render_config = config.tracking_config()
  if unconformity.any():
    contact = np.argmax(unconformity, axis=1).astype(np.float32)
    contact_visible = np.any(unconformity, axis=1) & ~fault_mask[:, 0, :]
    render_surfaces = np.concatenate((surfaces, contact[None]))
    render_visible = np.concatenate((visible, contact_visible[None]))
    render_config = dataclasses.replace(
        render_config, num_horizons=config.num_horizons + 1
    )
  seismic = synthetic._render_amplitudes(
      render_surfaces, render_visible, render_config, rng
  )
  return {
      'seismic': seismic,
      'rgt': rgt,
      'fault_mask': fault_mask,
      'unconformity_mask': unconformity,
      'horizon_depths': surfaces,
      'horizon_visible': visible,
      'horizon_rgt': levels,
      'scenario_id': np.asarray(SCENARIOS.index(scenario), dtype=np.int32),
      'fault_count': np.asarray(fault_count, dtype=np.int32),
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
  scene_frames = (config.num_frames - 1) * max(config.frame_strides) + 1
  scene_config = dataclasses.replace(config, num_frames=scene_frames)
  volume = dict(generate_geological_volume(scene_config, rng, scenario=scenario))
  reversed_sweep = bool(rng.random() < config.reverse_probability)
  if frame_stride is None:
    frame_stride = int(rng.choice(config.frame_strides))
  center = (config.num_frames // 2) * max(config.frame_strides)
  indices = center + (
      np.arange(config.num_frames, dtype=np.int32) - config.num_frames // 2
  ) * frame_stride
  for name in ('seismic', 'rgt', 'fault_mask', 'unconformity_mask'):
    array = volume[name][::-1] if reversed_sweep else volume[name]
    volume[name] = array[indices]
  for name in ('horizon_depths', 'horizon_visible'):
    array = volume[name][:, ::-1] if reversed_sweep else volume[name]
    volume[name] = array[:, indices]
  tracks = synthetic._sample_tracks(
      volume['horizon_depths'], volume['horizon_visible'],
      config.tracking_config(), rng,
  )
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
