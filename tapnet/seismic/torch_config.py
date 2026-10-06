"""Dependency-light configurations for native-Windows PyTorch training."""

from __future__ import annotations

import dataclasses

from tapnet.seismic.synthetic import SyntheticSeismicConfig


@dataclasses.dataclass(frozen=True)
class TorchTrainingConfig:
  """First-pass single-GPU training configuration."""

  synthetic: SyntheticSeismicConfig
  steps: int
  initial_resolution: tuple[int, int]
  pyramid_level: int
  query_chunk_size: int
  num_pips_iter: int = 4
  learning_rate: float = 1e-4
  end_learning_rate: float = 1e-6
  warmup_steps: int = 0
  weight_decay: float = 1e-2
  gradient_clip_norm: float = 1.0
  seed: int = 0
  fixed_batch: bool = False
  freeze_feature_encoder: bool = False

  def validate(self) -> None:
    self.synthetic.validate()
    if self.steps < 1:
      raise ValueError('steps must be positive.')
    if self.query_chunk_size < 1:
      raise ValueError('query_chunk_size must be positive.')
    if self.num_pips_iter < 1:
      raise ValueError('num_pips_iter must be positive.')
    if self.initial_resolution != (
        self.synthetic.height,
        self.synthetic.width,
    ):
      raise ValueError(
          'The first-pass config requires initial_resolution to match the '
          'synthetic image size.'
      )


def get_config(variant: str) -> TorchTrainingConfig:
  """Returns a documented smoke or constrained-VDI configuration."""
  if variant == 'smoke':
    config = TorchTrainingConfig(
        synthetic=SyntheticSeismicConfig(
            num_frames=2,
            height=64,
            width=64,
            num_horizons=4,
            num_queries=2,
            wavelet_length=17,
            max_fault_throw=2.0,
        ),
        steps=1,
        initial_resolution=(64, 64),
        pyramid_level=0,
        query_chunk_size=2,
        num_pips_iter=1,
    )
  elif variant == 'vdi-small':
    config = TorchTrainingConfig(
        synthetic=SyntheticSeismicConfig(
            num_frames=8,
            height=128,
            width=128,
            num_horizons=8,
            num_queries=16,
            wavelet_length=25,
            max_fault_throw=6.0,
        ),
        steps=2000,
        initial_resolution=(128, 128),
        pyramid_level=1,
        query_chunk_size=8,
        warmup_steps=100,
    )
  else:
    raise ValueError(
        f'Unknown PyTorch config {variant!r}; expected smoke or vdi-small.'
    )
  config.validate()
  return config

