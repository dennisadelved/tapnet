"""Optional TensorFlow adapter tests."""

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

from tapnet.seismic.dataset import create_synthetic_seismic_dataset


def test_tensorflow_dataset_matches_training_contract():
  dataset = create_synthetic_seismic_dataset(
      seed=3,
      num_frames=4,
      height=96,
      width=16,
      num_horizons=4,
      num_queries=5,
      wavelet_length=17,
  )
  first = next(iter(dataset.take(1)))

  assert tuple(first['video'].shape) == (4, 96, 16, 3)
  assert tuple(first['query_points'].shape) == (5, 3)
  assert tuple(first['target_points'].shape) == (5, 4, 2)
  assert first['video'].dtype == tf.float32
  assert first['occluded'].dtype == tf.bool
  assert np.all(np.asarray(first['label_valid']))
