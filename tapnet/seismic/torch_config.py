"""Dependency-light configurations for native-Windows PyTorch training."""

from __future__ import annotations

import dataclasses

from tapnet.seismic.synthetic import SyntheticSeismicConfig


CONFIG_VARIANTS = ('smoke', 'vdi-small', 'vdi-hard')


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
  encoder_learning_rate_multiplier: float = 1.0
  lateral_loss_weight: float = 0.25
  seed: int = 0
  fixed_batch: bool = False
  freeze_feature_encoder: bool = False
  amp_enabled: bool = True
  amp_dtype: str = 'bfloat16'

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
    if self.amp_dtype not in ('bfloat16', 'float16'):
      raise ValueError('amp_dtype must be bfloat16 or float16.')
    if not 0.0 < self.encoder_learning_rate_multiplier <= 1.0:
      raise ValueError(
          'encoder_learning_rate_multiplier must be greater than 0 and at '
          'most 1.'
      )
    if self.lateral_loss_weight <= 0.0:
      raise ValueError('lateral_loss_weight must be positive.')


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
  elif variant == 'vdi-hard':
    config = TorchTrainingConfig(
        synthetic=SyntheticSeismicConfig(
            num_frames=16,
            height=128,
            width=128,
            num_horizons=10,
            num_queries=24,
            wavelet_length=25,
            noise_std=0.18,
            max_fault_throw=12.0,
            fault_probability=1.0,
            max_faults=3,
            termination_probability=0.35,
        ),
        steps=5000,
        initial_resolution=(128, 128),
        pyramid_level=1,
        query_chunk_size=8,
        warmup_steps=250,
        learning_rate=5e-5,
        lateral_loss_weight=1.0,
    )
  else:
    raise ValueError(
        f'Unknown PyTorch config {variant!r}; expected one of '
        f'{CONFIG_VARIANTS!r}.'
    )
  config.validate()
  return config

