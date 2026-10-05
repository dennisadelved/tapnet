"""TensorFlow and evaluation adapters for synthetic seismic TAPIR data."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Mapping, Sequence

import numpy as np

from tapnet.seismic import synthetic


def _config_from_kwargs(**kwargs) -> synthetic.SyntheticSeismicConfig:
  field_names = {field.name for field in dataclasses.fields(
      synthetic.SyntheticSeismicConfig
  )}
  unknown = set(kwargs) - field_names
  if unknown:
    raise TypeError(f'Unknown synthetic seismic options: {sorted(unknown)}')
  return synthetic.SyntheticSeismicConfig(**kwargs)


def create_synthetic_seismic_dataset(
    batch_dims: Sequence[int] = (), seed: int = 0, **kwargs
):
  """Creates the unbatched infinite ``tf.data.Dataset`` used for training.

  ``batch_dims`` is accepted for compatibility with Kubric's constructor.  The
  shared experiment code performs device and per-device batching afterward.
  """
  del batch_dims
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(
        'TensorFlow is required for the training dataset adapter. Install the '
        'dependencies in requirements_seismic.txt.'
    ) from exc

  config = _config_from_kwargs(**kwargs)
  config.validate()
  output_signature = {
      'video': tf.TensorSpec(
          (config.num_frames, config.height, config.width, 3), tf.float32
      ),
      'query_points': tf.TensorSpec((config.num_queries, 3), tf.float32),
      'target_points': tf.TensorSpec(
          (config.num_queries, config.num_frames, 2), tf.float32
      ),
      'occluded': tf.TensorSpec(
          (config.num_queries, config.num_frames), tf.bool
      ),
      'label_valid': tf.TensorSpec(
          (config.num_queries, config.num_frames), tf.bool
      ),
      'trackgroup': tf.TensorSpec((config.num_queries,), tf.int32),
      'faulted': tf.TensorSpec((), tf.bool),
      'sweep_reversed': tf.TensorSpec((), tf.bool),
  }
  return tf.data.Dataset.from_generator(
      lambda: synthetic.iter_synthetic_samples(config, seed),
      output_signature=output_signature,
  )


def create_synthetic_seismic_eval_dataset(
    num_samples: int,
    seed: int = 1,
    dataset_key: str = 'seismic',
    **kwargs,
) -> Iterable[Mapping[str, Mapping[str, np.ndarray]]]:
  """Yields finite, batch-size-one examples for JAXline evaluation."""
  if num_samples < 1:
    raise ValueError('num_samples must be positive.')
  config = _config_from_kwargs(**kwargs)
  config.validate()
  stream = synthetic.iter_synthetic_samples(config, seed)
  for sample in itertools.islice(stream, num_samples):
    yield {
        dataset_key: {
            key: value[np.newaxis]
            for key, value in sample.items()
        }
    }
