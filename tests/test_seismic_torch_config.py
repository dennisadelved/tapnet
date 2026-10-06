"""Configuration tests for the PyTorch seismic training path."""

import pytest

from tapnet.seismic import torch_config


@pytest.mark.parametrize('variant', ['smoke', 'vdi-small'])
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


def test_unknown_config_is_rejected():
  with pytest.raises(ValueError, match='Unknown PyTorch config'):
    torch_config.get_config('undocumented')

