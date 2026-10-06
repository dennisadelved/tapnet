"""Numerical and gradient tests for the PyTorch seismic objective."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from tapnet.seismic import torch_losses


def _base_inputs():
  targets = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
  occluded = torch.zeros((1, 1, 3), dtype=torch.bool)
  logits = torch.full((1, 1, 3), -5.0, dtype=torch.float32)
  valid = torch.ones((1, 1, 3), dtype=torch.bool)
  return targets, occluded, logits, valid


def test_depth_and_lateral_errors_have_separate_weights():
  targets, occluded, logits, valid = _base_inputs()
  depth_points = targets.clone()
  lateral_points = targets.clone()
  depth_points[..., 1] = 2.0
  lateral_points[..., 0] = 2.0
  config = torch_losses.SeismicLossConfig(position_loss_weight=1.0)

  depth_loss, _, _ = torch_losses.seismic_tapir_loss(
      depth_points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      config=config,
  )
  lateral_loss, _, _ = torch_losses.seismic_tapir_loss(
      lateral_points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      config=config,
  )

  assert float(depth_loss) == pytest.approx(4.0 * float(lateral_loss))


def test_unknown_and_occluded_positions_do_not_affect_position_loss():
  targets, occluded, logits, valid = _base_inputs()
  points = targets.clone()
  points[0, 0, 1, 1] = 100.0
  points[0, 0, 2, 1] = 100.0
  valid[0, 0, 1] = False
  occluded[0, 0, 2] = True

  position_loss, _, _ = torch_losses.seismic_tapir_loss(
      points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      config=torch_losses.SeismicLossConfig(position_loss_weight=1.0),
  )

  assert float(position_loss) == pytest.approx(0.0)


def test_loss_gradient_is_finite():
  targets, occluded, logits, valid = _base_inputs()
  points = torch.ones_like(targets, requires_grad=True)
  position, occurrence, probability = torch_losses.seismic_tapir_loss(
      points,
      logits,
      targets,
      occluded,
      expected_dist=logits,
      label_valid=valid,
  )

  (position + occurrence + probability).backward()

  assert np.all(np.isfinite(points.grad.detach().numpy()))


def test_intermediate_predictions_are_deeply_supervised():
  targets, occluded, logits, valid = _base_inputs()
  points = torch.ones_like(targets, requires_grad=True)
  outputs = {
      'tracks': points,
      'occlusion': logits,
      'expected_dist': logits,
      'unrefined_tracks': [points],
      'unrefined_occlusion': [logits],
      'unrefined_expected_dist': [logits],
  }
  batch = {
      'target_points': targets,
      'occluded': occluded,
      'label_valid': valid,
  }

  total, scalars = torch_losses.seismic_supervised_loss(outputs, batch)
  final = (
      scalars['position_loss']
      + scalars['occlusion_loss']
      + scalars['probability_loss']
  )

  assert float(total.detach()) == pytest.approx(2.0 * float(final.detach()))
  assert float(scalars['loss_0'].detach()) == pytest.approx(
      float(final.detach())
  )
  assert float(scalars['intermediate_loss'].detach()) == pytest.approx(
      float(final.detach())
  )

