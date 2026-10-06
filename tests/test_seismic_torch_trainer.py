"""Focused safety tests for the PyTorch training loop."""

import pytest

torch = pytest.importorskip('torch')

from tapnet.seismic import train_torch


def test_nonfinite_gradient_is_rejected_before_optimizer_step():
  model = torch.nn.Linear(1, 1)
  for parameter in model.parameters():
    parameter.grad = torch.full_like(parameter, float('nan'))

  with pytest.raises(RuntimeError, match='non-finite'):
    train_torch._clip_gradients(model, max_norm=1.0)


def test_intermediate_scalar_format_includes_each_loss_term():
  scalars = {
      'position_loss_0': torch.tensor(1.0),
      'occlusion_loss_0': torch.tensor(2.0),
      'probability_loss_0': torch.tensor(3.0),
      'loss_0': torch.tensor(6.0),
  }

  text = train_torch._format_intermediate_scalars(scalars)

  assert text == (
      'stage_0_position=1.000000 stage_0_occlusion=2.000000 '
      'stage_0_probability=3.000000 stage_0_total=6.000000'
  )

