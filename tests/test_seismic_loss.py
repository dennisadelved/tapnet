"""Numerical tests for the JAX seismic loss."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('optax')
pytest.importorskip('chex')

from tapnet.utils import model_utils


def _base_inputs():
  targets = jnp.zeros((1, 1, 3, 2), dtype=jnp.float32)
  occluded = jnp.zeros((1, 1, 3), dtype=bool)
  logits = jnp.full((1, 1, 3), -5.0, dtype=jnp.float32)
  valid = jnp.ones((1, 1, 3), dtype=bool)
  return targets, occluded, logits, valid


def test_depth_and_lateral_errors_have_separate_weights():
  targets, occluded, logits, valid = _base_inputs()
  depth_points = targets.at[..., 1].set(2.0)
  lateral_points = targets.at[..., 0].set(2.0)

  depth_loss, _, _ = model_utils.seismic_tapnet_loss(
      depth_points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      position_loss_weight=1.0,
      depth_loss_weight=1.0,
      lateral_loss_weight=0.25,
  )
  lateral_loss, _, _ = model_utils.seismic_tapnet_loss(
      lateral_points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      position_loss_weight=1.0,
      depth_loss_weight=1.0,
      lateral_loss_weight=0.25,
  )

  assert float(depth_loss) == pytest.approx(4.0 * float(lateral_loss))


def test_unknown_and_occluded_positions_do_not_affect_position_loss():
  targets, occluded, logits, valid = _base_inputs()
  points = targets.at[0, 0, 1, 1].set(100.0)
  points = points.at[0, 0, 2, 1].set(100.0)
  valid = valid.at[0, 0, 1].set(False)
  occluded = occluded.at[0, 0, 2].set(True)

  position_loss, _, _ = model_utils.seismic_tapnet_loss(
      points,
      logits,
      targets,
      occluded,
      label_valid=valid,
      position_loss_weight=1.0,
  )

  assert float(position_loss) == pytest.approx(0.0)


def test_loss_gradient_is_finite():
  targets, occluded, logits, valid = _base_inputs()

  def loss_fn(points):
    position, trackability, uncertainty = model_utils.seismic_tapnet_loss(
        points,
        logits,
        targets,
        occluded,
        expected_dist=logits,
        label_valid=valid,
    )
    return position + trackability + uncertainty

  gradient = jax.grad(loss_fn)(jnp.ones_like(targets))
  assert np.all(np.isfinite(np.asarray(gradient)))
