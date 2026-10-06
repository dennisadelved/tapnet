"""End-to-end autograd smoke test for the official PyTorch TAPIR model."""

import pytest

torch = pytest.importorskip('torch')

from tapnet.seismic import synthetic
from tapnet.seismic import torch_config
from tapnet.seismic import torch_losses
from tapnet.torch import tapir_model


def test_tapir_smoke_forward_and_backward():
  config = torch_config.get_config('smoke')
  sample = synthetic.generate_synthetic_sample(config.synthetic, rng=3)
  batch = {
      key: torch.from_numpy(value).unsqueeze(0)
      for key, value in sample.items()
  }
  model = tapir_model.TAPIR(
      num_pips_iter=config.num_pips_iter,
      pyramid_level=config.pyramid_level,
      initial_resolution=config.initial_resolution,
      extra_convs=True,
  )

  outputs = model(
      batch['video'],
      batch['query_points'],
      is_training=True,
      query_chunk_size=config.query_chunk_size,
  )
  loss, _ = torch_losses.seismic_supervised_loss(outputs, batch)
  loss.backward()

  assert torch.isfinite(loss)
  assert any(
      parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
      for parameter in model.parameters()
  )
