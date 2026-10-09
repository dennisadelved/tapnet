"""PyTorch dataset adapter for procedural seismic tracking examples."""

from __future__ import annotations

from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils import data

from tapnet.seismic import synthetic
from tapnet.seismic import geology


class SyntheticSeismicIterableDataset(data.IterableDataset):
  """Reproducible infinite stream of tensor-valued seismic examples."""

  def __init__(
      self,
      config: synthetic.SyntheticSeismicConfig | geology.GeologicalSeismicConfig,
      seed: int = 0,
      start_index: int = 0,
  ) -> None:
    super().__init__()
    config.validate()
    if start_index < 0:
      raise ValueError('start_index must not be negative.')
    self._config = config
    self._seed = seed
    self._start_index = start_index

  def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
    worker = data.get_worker_info()
    worker_id = 0 if worker is None else worker.id
    worker_count = 1 if worker is None else worker.num_workers
    sample_index = self._start_index + worker_id
    while True:
      sample_seed = int(
          np.random.SeedSequence([self._seed, sample_index]).generate_state(
              1, dtype=np.uint64
          )[0]
      )
      if isinstance(self._config, geology.GeologicalSeismicConfig):
        scenario_count = len(self._config.scenarios)
        frame_stride = self._config.frame_strides[
            (sample_index // scenario_count) % len(self._config.frame_strides)
        ]
        sample = geology.generate_geological_sample(
            self._config, rng=sample_seed, frame_stride=frame_stride,
            scenario=self._config.scenarios[sample_index % scenario_count],
        )
      else:
        frame_stride = self._config.frame_strides[
            sample_index % len(self._config.frame_strides)
        ]
        sample = synthetic.generate_synthetic_sample(
            self._config, rng=sample_seed, frame_stride=frame_stride
        )
      yield {
          key: torch.from_numpy(np.asarray(value))
          for key, value in sample.items()
      }
      sample_index += worker_count

