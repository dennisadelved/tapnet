"""Configuration tests for the PyTorch seismic training path."""

import dataclasses

import pytest

from tapnet.seismic import torch_config


@pytest.mark.parametrize('variant', torch_config.CONFIG_VARIANTS)
def test_config_variants_are_valid(variant):
  config = torch_config.get_config(variant)
  config.validate()


def test_vdi_config_is_explicitly_smaller_than_jax_default():
  config = torch_config.get_config('vdi-small')

  assert config.synthetic.num_frames == 8
  assert config.synthetic.height == 128
  assert config.synthetic.width == 128
  assert config.synthetic.num_queries == 16
  assert config.pyramid_level == 1


def test_hard_config_extends_temporal_and_fault_curriculum():
  config = torch_config.get_config('vdi-hard')

  assert config.synthetic.num_frames == 16
  assert config.synthetic.fault_probability == 1.0
  assert config.synthetic.max_faults == 3
  assert config.synthetic.max_fault_throw == 12.0
  assert config.synthetic.noise_std == 0.18
  assert config.synthetic.termination_probability == 0.35
  assert config.lateral_loss_weight == 1.0
  assert config.steps == 5000


def test_multistride_config_uses_aligned_32_frame_views():
  config = torch_config.get_config('vdi-multistride')

  assert config.synthetic.num_frames == 32
  assert config.synthetic.frame_strides == (1, 2, 4)
  assert config.synthetic.max_faults == 3
  assert config.lateral_loss_weight == 1.0
  assert config.steps == 3000


def test_fault_robust_config_expands_depth_and_fault_curriculum():
  config = torch_config.get_config('vdi-fault-robust')

  assert config.synthetic.num_frames == 32
  assert config.synthetic.frame_strides == (1, 2, 4)
  assert config.synthetic.height == 256
  assert config.synthetic.width == 128
  assert config.synthetic.min_fault_throw == 4.0
  assert config.synthetic.max_fault_throw == 80.0
  assert config.synthetic.max_fault_offset == 0.15
  assert not config.synthetic.divide_fault_throw_by_count
  assert config.synthetic.min_fault_damage_width == 1
  assert config.synthetic.max_fault_damage_width == 8
  assert config.synthetic.fault_query_probability == 0.75
  assert config.synthetic.center_aligned_views
  assert config.initial_resolution == (256, 128)
  assert config.query_chunk_size == 4
  assert config.steps == 5000


def test_unknown_config_is_rejected():
  with pytest.raises(ValueError, match='Unknown PyTorch config'):
    torch_config.get_config('undocumented')


def test_geology_training_config_uses_all_scenarios_and_multiple_strides():
  from tapnet.seismic import geology

  config = torch_config.get_config('vdi-geology')
  assert isinstance(config.synthetic, geology.GeologicalSeismicConfig)
  assert config.synthetic.scenarios == geology.SCENARIOS
  assert config.synthetic.frame_strides == (1, 2, 4)
  assert config.initial_resolution == (128, 128)
  assert config.steps == 5000


def test_cuda_precision_defaults_to_bfloat16():
  config = torch_config.get_config('vdi-small')

  assert config.amp_enabled
  assert config.amp_dtype == 'bfloat16'


def test_encoder_learning_rate_multiplier_is_bounded():
  config = torch_config.get_config('vdi-small')

  with pytest.raises(ValueError, match='encoder_learning_rate_multiplier'):
    dataclasses.replace(
        config, encoder_learning_rate_multiplier=0.0
    ).validate()

