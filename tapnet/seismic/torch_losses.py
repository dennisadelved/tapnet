"""PyTorch losses for fixed-lateral seismic TAPIR supervision."""

from __future__ import annotations

import dataclasses
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class SeismicLossConfig:
  """Weights matching the first-pass JAX seismic objective."""

  position_loss_weight: float = 0.05
  depth_loss_weight: float = 1.0
  lateral_loss_weight: float = 0.25
  expected_dist_thresh: float = 2.0
  huber_loss_delta: float = 2.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  mask = mask.to(values.dtype)
  return torch.sum(values * mask) / torch.clamp(torch.sum(mask), min=1.0)


def _scalar_huber(error: torch.Tensor, delta: float) -> torch.Tensor:
  absolute_error = torch.abs(error)
  return torch.where(
      absolute_error <= delta,
      0.5 * torch.square(error),
      delta * (absolute_error - 0.5 * delta),
  )


def seismic_tapir_loss(
    points: torch.Tensor,
    occlusion: torch.Tensor,
    target_points: torch.Tensor,
    target_occ: torch.Tensor,
    *,
    expected_dist: torch.Tensor | None = None,
    label_valid: torch.Tensor | None = None,
    config: SeismicLossConfig = SeismicLossConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Computes the PyTorch equivalent of ``seismic_tapnet_loss``."""
  if label_valid is None:
    label_valid = torch.ones_like(target_occ, dtype=torch.bool)
  valid = label_valid.to(points.dtype)
  visible_valid = valid * (1.0 - target_occ.to(points.dtype))

  lateral_error = points[..., 0] - target_points[..., 0]
  depth_error = points[..., 1] - target_points[..., 1]
  depth_loss = _masked_mean(
      _scalar_huber(depth_error, config.huber_loss_delta), visible_valid
  )
  lateral_loss = _masked_mean(
      _scalar_huber(lateral_error, config.huber_loss_delta), visible_valid
  )
  position_loss = config.position_loss_weight * (
      config.depth_loss_weight * depth_loss
      + config.lateral_loss_weight * lateral_loss
  )

  occurrence_targets = target_occ.to(occlusion.dtype)
  occurrence_loss = F.binary_cross_entropy_with_logits(
      occlusion, occurrence_targets, reduction='none'
  )
  occurrence_loss = _masked_mean(occurrence_loss, valid)

  if expected_dist is None:
    probability_loss = points.new_zeros(())
  else:
    excessive_depth_error = (
        torch.abs(depth_error.detach()) > config.expected_dist_thresh
    ).to(expected_dist.dtype)
    probability_loss = F.binary_cross_entropy_with_logits(
        expected_dist, excessive_depth_error, reduction='none'
    )
    probability_loss = _masked_mean(probability_loss, visible_valid)

  return position_loss, occurrence_loss, probability_loss


def seismic_supervised_loss(
    outputs: Mapping[str, object],
    batch: Mapping[str, torch.Tensor],
    config: SeismicLossConfig = SeismicLossConfig(),
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
  """Sums final and intermediate TAPIR losses as in the JAX trainer."""

  def compute(
      tracks: torch.Tensor,
      occlusion: torch.Tensor,
      expected_dist: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return seismic_tapir_loss(
        tracks,
        occlusion,
        batch['target_points'],
        batch['occluded'],
        expected_dist=expected_dist,
        label_valid=batch.get('label_valid'),
        config=config,
    )

  position, occurrence, probability = compute(
      outputs['tracks'], outputs['occlusion'], outputs.get('expected_dist')
  )
  total = position + occurrence + probability
  scalars = {
      'position_loss': position,
      'occlusion_loss': occurrence,
      'probability_loss': probability,
  }

  unrefined_tracks = outputs.get('unrefined_tracks', ())
  unrefined_occlusion = outputs.get('unrefined_occlusion', ())
  unrefined_expected = outputs.get('unrefined_expected_dist', ())
  for index, (tracks, occlusion) in enumerate(
      zip(unrefined_tracks, unrefined_occlusion)
  ):
    expected = (
        unrefined_expected[index]
        if index < len(unrefined_expected)
        else None
    )
    step_position, step_occurrence, step_probability = compute(
        tracks, occlusion, expected
    )
    total = total + step_position + step_occurrence + step_probability
    scalars[f'position_loss_{index}'] = step_position
    scalars[f'occlusion_loss_{index}'] = step_occurrence
    scalars[f'probability_loss_{index}'] = step_probability

  scalars['loss'] = total
  return total, scalars

