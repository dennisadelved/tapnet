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

