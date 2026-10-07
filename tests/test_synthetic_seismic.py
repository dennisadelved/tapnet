"""Tests for deterministic seismic generation and TAP coordinate semantics."""

import numpy as np
import pytest

from tapnet.seismic.synthetic import SyntheticSeismicConfig
from tapnet.seismic.synthetic import _make_horizons
from tapnet.seismic.synthetic import generate_synthetic_sample
from tapnet.seismic.synthetic import iter_synthetic_samples


def _small_config(**overrides):
  values = dict(
      num_frames=8,
      height=96,
      width=32,
      num_horizons=6,
      num_queries=12,
      wavelet_length=17,
  )
  values.update(overrides)
  return SyntheticSeismicConfig(**values)


def test_sample_shapes_ranges_and_dtypes():
  config = _small_config()
  sample = generate_synthetic_sample(config, 7)

  assert sample['video'].shape == (
      config.num_frames, config.height, config.width, 3
  )
  assert sample['query_points'].shape == (config.num_queries, 3)
  assert sample['target_points'].shape == (
      config.num_queries, config.num_frames, 2
  )
  assert sample['occluded'].shape == (
      config.num_queries, config.num_frames
  )
  assert sample['label_valid'].shape == sample['occluded'].shape
  assert sample['video'].dtype == np.float32
  assert sample['query_points'].dtype == np.float32
  assert sample['target_points'].dtype == np.float32
  assert sample['occluded'].dtype == np.bool_
  assert np.max(sample['video']) <= 1.0
  assert np.min(sample['video']) >= -1.0
  np.testing.assert_array_equal(sample['video'][..., 0], sample['video'][..., 1])
  np.testing.assert_array_equal(sample['video'][..., 1], sample['video'][..., 2])


def test_query_matches_target_and_lateral_coordinate_is_fixed():
  config = _small_config(termination_probability=1.0)
  sample = generate_synthetic_sample(config, 11)

  for query_index, query in enumerate(sample['query_points']):
    frame = int(query[0])
    target = sample['target_points'][query_index, frame]
    assert not sample['occluded'][query_index, frame]
    assert target[0] == pytest.approx(query[2])
    assert target[1] == pytest.approx(query[1])
    np.testing.assert_allclose(
        sample['target_points'][query_index, :, 0], query[2]
    )


def test_termination_produces_known_occlusion_not_missing_labels():
  config = _small_config(
      num_queries=64,
      termination_probability=1.0,
      fault_probability=0.0,
  )
  sample = generate_synthetic_sample(config, 19)

  assert np.any(sample['occluded'])
  assert np.any(~sample['occluded'])
  assert np.all(sample['label_valid'])


def test_horizons_do_not_cross():
  config = _small_config(fault_probability=1.0)
  surfaces, _, _ = _make_horizons(config, np.random.default_rng(29))
  assert np.all(np.diff(surfaces, axis=0) > 0.0)


def test_multiple_fault_config_generates_up_to_requested_count():
  config = _small_config(fault_probability=1.0, max_faults=3)
  counts = [
      _make_horizons(config, np.random.default_rng(seed))[2]
      for seed in range(20)
  ]

  assert all(1 <= count <= 3 for count in counts)
  assert any(count > 1 for count in counts)


def test_full_termination_probability_always_leaves_valid_query_points():
  config = _small_config(
      num_queries=32,
      termination_probability=1.0,
      reverse_probability=0.5,
  )
  for seed in range(50):
    sample = generate_synthetic_sample(config, seed)
    query_frames = sample['query_points'][:, 0].astype(np.int32)
    query_indices = np.arange(config.num_queries)
    assert np.all(~sample['occluded'][query_indices, query_frames])


def test_seeded_generation_is_reproducible_and_stream_advances():
  config = _small_config()
  first = generate_synthetic_sample(config, 23)
  repeated = generate_synthetic_sample(config, 23)
  for key in first:
    np.testing.assert_array_equal(first[key], repeated[key])

  stream = iter_synthetic_samples(config, seed=23)
  stream_first = next(stream)
  stream_second = next(stream)
  assert not np.array_equal(stream_first['video'], stream_second['video'])


@pytest.mark.parametrize(
    'override',
    [
        {'num_frames': 1},
        {'height': 32},
        {'num_queries': 0},
        {'max_faults': 0},
        {'wavelet_length': 16},
        {'fault_probability': 1.1},
    ],
)
def test_invalid_configuration_is_rejected(override):
  with pytest.raises(ValueError):
    _small_config(**override).validate()
