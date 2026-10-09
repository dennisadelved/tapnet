"""Geological topology, label alignment, split isolation, and reproducibility."""

import json

import numpy as np
import pytest

from tapnet.seismic import geology
from tapnet.seismic import generate_suite
from tapnet.seismic import synthetic


def _config(**overrides):
  values = dict(
      num_frames=12, height=96, width=24, num_horizons=12,
      num_queries=16, wavelet_length=17, max_fault_throw=12,
  )
  values.update(overrides)
  return geology.GeologicalSeismicConfig(**values)


@pytest.mark.parametrize('scenario', geology.SCENARIOS)
def test_volume_topology_and_labels_come_from_the_same_stratigraphy(scenario):
  config = _config()
  volume = geology.generate_geological_volume(config, 7, scenario=scenario)
  rgt = volume['rgt']
  assert rgt.shape == (12, 96, 24)
  assert rgt.dtype == np.float32
  assert np.isfinite(rgt).all()
  assert np.min(rgt) == pytest.approx(-1)
  assert np.max(rgt) == pytest.approx(1)
  assert np.all(np.diff(rgt, axis=1) > 0)
  assert np.all(np.diff(volume['horizon_rgt']) > 0)
  assert volume['seismic'].dtype == np.float32
  assert np.isfinite(volume['seismic']).all()
  assert np.max(np.abs(volume['seismic'])) <= 1
  frame, lateral = np.indices((12, 24))
  for index, level in enumerate(volume['horizon_rgt']):
    depths = volume['horizon_depths'][index]
    visible = volume['horizon_visible'][index]
    low = np.minimum(np.floor(depths).astype(int), config.height - 2)
    fraction = depths - low
    interpolated = (
        rgt[frame, low, lateral] * (1 - fraction)
        + rgt[frame, low + 1, lateral] * fraction
    )
    np.testing.assert_allclose(interpolated[visible], level, atol=2e-7)
  assert bool(volume['fault_mask'].any()) == (scenario in ('faulted', 'mixed'))
  assert bool(volume['unconformity_mask'].any()) == (
      scenario in ('unconformity', 'mixed')
  )


def test_erosion_removes_horizons_instead_of_interpolating_across_missing_ages():
  config = _config(num_horizons=40, scenarios=('unconformity',))
  volume = geology.generate_geological_volume(config, 19)
  rgt = volume['rgt']
  for frame, high, lateral in np.argwhere(volume['unconformity_mask']):
    levels_in_gap = (
        (volume['horizon_rgt'] > rgt[frame, high - 1, lateral])
        & (volume['horizon_rgt'] < rgt[frame, high, lateral])
    )
    assert not volume['horizon_visible'][levels_in_gap, frame, lateral].any()
  assert np.any(~volume['horizon_visible'])


@pytest.mark.parametrize('reverse_probability', (0.0, 1.0))
@pytest.mark.parametrize('stride', (1, 2, 4))
def test_tap_tracks_match_volume_after_reversal_and_stride(reverse_probability, stride):
  config = _config(frame_strides=(1, 2, 4), reverse_probability=reverse_probability)
  sample = geology.generate_geological_sample(
      config, 11, scenario='mixed', frame_stride=stride, include_volume=True
  )
  assert sample['video'].shape == (12, 96, 24, 3)
  np.testing.assert_array_equal(sample['video'][..., 0], sample['seismic'])
  np.testing.assert_array_equal(sample['frame_indices'], 24 + (np.arange(12) - 6) * stride)
  assert int(sample['scene_num_frames']) == 45
  assert bool(sample['sweep_reversed']) == bool(reverse_probability)
  assert sample['label_valid'].all()
  for index, (frame, depth, lateral) in enumerate(sample['query_points']):
    frame, lateral = int(frame), int(lateral)
    horizon = sample['trackgroup'][index]
    assert not sample['occluded'][index, frame]
    np.testing.assert_array_equal(
        sample['target_points'][index, :, 1],
        sample['horizon_depths'][horizon, :, lateral],
    )
    np.testing.assert_array_equal(
        ~sample['occluded'][index], sample['horizon_visible'][horizon, :, lateral]
    )
    np.testing.assert_array_equal(sample['target_points'][index, :, 0], lateral)
    np.testing.assert_array_equal(sample['target_points'][index, frame], [lateral, depth])


def test_dense_output_does_not_change_tracking_sample_or_rng_sequence():
  config = _config()
  first = geology.generate_geological_sample(config, 9)
  dense = geology.generate_geological_sample(config, 9, include_volume=True)
  second = geology.generate_geological_sample(config, 9)
  for key in first:
    np.testing.assert_array_equal(first[key], dense[key])
    np.testing.assert_array_equal(first[key], second[key])


def test_stride_views_share_the_same_center_and_geology():
  config = _config(frame_strides=(1, 2, 4))
  views = [geology.generate_geological_sample(
      config, 23, scenario='mixed', frame_stride=stride, include_volume=True
  ) for stride in config.frame_strides]
  assert all(view['frame_indices'][6] == 24 for view in views)
  for view in views[1:]:
    common = np.intersect1d(views[0]['frame_indices'], view['frame_indices'])
    first_indices = np.searchsorted(views[0]['frame_indices'], common)
    second_indices = np.searchsorted(view['frame_indices'], common)
    for key in ('video', 'rgt', 'fault_mask', 'unconformity_mask'):
      np.testing.assert_array_equal(
          views[0][key][first_indices], view[key][second_indices]
      )


def test_rendering_accepts_absent_horizons_at_last_depth_sample():
  config = synthetic.SyntheticSeismicConfig(
      num_frames=2, height=64, width=8, num_horizons=2, max_fault_throw=0,
  )
  depths = np.full((2, 2, 8), 63, dtype=np.float32)
  amplitudes = synthetic._render_amplitudes(
      depths, np.zeros_like(depths, dtype=bool), config, np.random.default_rng(0)
  )
  assert np.isfinite(amplitudes).all()


@pytest.mark.parametrize('override', (
    {'scenarios': ()}, {'scenarios': ('invalid',)},
    {'scenarios': ('layered', 'layered')}, {'noise_std': -1},
    {'noise_std': float('nan')}, {'max_fault_throw': float('inf')},
    {'max_fault_throw': 48}, {'max_faults': 0}, {'fault_damage_width': -1},
    {'fault_damage_width': 6},
    {'reverse_probability': 2}, {'frame_strides': (1, 1)},
    {'frame_strides': (1.5,)}, {'num_horizons': 2.5},
))
def test_invalid_configuration_is_rejected(override):
  with pytest.raises(ValueError):
    _config(**override).validate()


def test_invalid_requested_scenario_and_stride_are_rejected():
  with pytest.raises(ValueError, match='Scenario'):
    geology.generate_geological_volume(_config(scenarios=('layered',)), 1,
                                       scenario='mixed')
  with pytest.raises(ValueError, match='frame_stride'):
    geology.generate_geological_sample(_config(), 1, frame_stride=2)


def test_export_is_reproducible_split_isolated_and_pickle_free(tmp_path):
  config = _config(scenarios=('layered', 'mixed'), frame_strides=(1, 2))
  first = tmp_path / 'first'
  second = tmp_path / 'second'
  generate_suite.export_suite(first, config, train_samples=4,
                              validation_samples=1, test_samples=1, seed=2)
  generate_suite.export_suite(second, config, train_samples=5,
                              validation_samples=1, test_samples=1, seed=2)
  manifest = json.loads((first / 'manifest.json').read_text())
  assert manifest['split_counts'] == {'train': 4, 'validation': 1, 'test': 1}
  records = manifest['samples']
  assert len({r['seed'] for r in records}) == len(records)
  assert [(r['scenario'], r['frame_stride']) for r in records[:4]] == [
      ('layered', 1), ('mixed', 1), ('layered', 2), ('mixed', 2)
  ]
  for record in records:
    with np.load(first / record['path'], allow_pickle=False) as a, np.load(
        second / record['path'], allow_pickle=False
    ) as b:
      assert {'video', 'target_points', 'rgt', 'fault_mask', 'horizon_rgt'} <= set(a.files)
      for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
  with pytest.raises(FileExistsError):
    generate_suite.export_suite(first, config)


def test_invalid_export_leaves_no_output_directory(tmp_path):
  for overrides in ({'train_samples': -1}, {'seed': -1}):
    destination = tmp_path / 'invalid'
    with pytest.raises(ValueError):
      generate_suite.export_suite(destination, _config(), **overrides)
    assert not destination.exists()
