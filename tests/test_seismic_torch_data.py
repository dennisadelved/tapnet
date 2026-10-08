"""Tests for the NumPy-to-PyTorch seismic adapter."""

import pytest

torch = pytest.importorskip('torch')

from tapnet.seismic import synthetic
from tapnet.seismic import torch_data


def test_iterable_dataset_is_reproducible_and_tensor_valued():
  config = synthetic.SyntheticSeismicConfig(
      num_frames=2,
      height=64,
      width=16,
      num_horizons=4,
      num_queries=2,
      wavelet_length=17,
      max_fault_throw=2.0,
  )
  first = next(iter(torch_data.SyntheticSeismicIterableDataset(config, seed=9)))
  second = next(iter(torch_data.SyntheticSeismicIterableDataset(config, seed=9)))

  assert first['video'].shape == (2, 64, 16, 3)
  assert first['query_points'].shape == (2, 3)
  assert first['target_points'].shape == (2, 2, 2)
  assert first['video'].dtype == torch.float32
  assert first['occluded'].dtype == torch.bool
  for key in first:
    assert torch.equal(first[key], second[key])


def test_start_index_resumes_the_same_sample_sequence():
  config = synthetic.SyntheticSeismicConfig(
      num_frames=2,
      height=64,
      width=16,
      num_horizons=4,
      num_queries=2,
      wavelet_length=17,
      max_fault_throw=2.0,
  )
  stream = iter(torch_data.SyntheticSeismicIterableDataset(config, seed=11))
  next(stream)
  second = next(stream)
  resumed = next(
      iter(
          torch_data.SyntheticSeismicIterableDataset(
              config, seed=11, start_index=1
          )
      )
  )

  for key in second:
    assert torch.equal(second[key], resumed[key])


def test_multistride_dataset_is_balanced_and_resume_preserves_stride():
  config = synthetic.SyntheticSeismicConfig(
      num_frames=2,
      height=64,
      width=16,
      num_horizons=4,
      num_queries=2,
      wavelet_length=17,
      max_fault_throw=2.0,
      frame_strides=(1, 2, 4),
  )
  stream = iter(torch_data.SyntheticSeismicIterableDataset(config, seed=13))

  strides = [int(next(stream)['frame_stride']) for _ in range(4)]
  resumed = next(iter(
      torch_data.SyntheticSeismicIterableDataset(
          config, seed=13, start_index=4
      )
  ))

  assert strides == [1, 2, 4, 1]
  assert int(resumed['frame_stride']) == 2

