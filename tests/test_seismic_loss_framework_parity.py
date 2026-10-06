"""Cross-framework value parity for the seismic loss port."""

import numpy as np
import pytest

jnp = pytest.importorskip('jax.numpy')
torch = pytest.importorskip('torch')
pytest.importorskip('optax')
pytest.importorskip('chex')

from tapnet.seismic import torch_losses
from tapnet.utils import model_utils


def test_pytorch_loss_matches_jax_reference_on_fixed_tensors():
  rng = np.random.default_rng(21)
  points = rng.normal(size=(2, 3, 4, 2)).astype(np.float32)
  targets = rng.normal(size=(2, 3, 4, 2)).astype(np.float32)
  occurrence_logits = rng.normal(size=(2, 3, 4)).astype(np.float32)
  probability_logits = rng.normal(size=(2, 3, 4)).astype(np.float32)
  target_occ = rng.random(size=(2, 3, 4)) < 0.25
  label_valid = rng.random(size=(2, 3, 4)) < 0.8

  jax_losses = model_utils.seismic_tapnet_loss(
      jnp.asarray(points),
      jnp.asarray(occurrence_logits),
      jnp.asarray(targets),
      jnp.asarray(target_occ),
      expected_dist=jnp.asarray(probability_logits),
      label_valid=jnp.asarray(label_valid),
  )
  torch_result = torch_losses.seismic_tapir_loss(
      torch.from_numpy(points),
      torch.from_numpy(occurrence_logits),
      torch.from_numpy(targets),
      torch.from_numpy(target_occ),
      expected_dist=torch.from_numpy(probability_logits),
      label_valid=torch.from_numpy(label_valid),
  )

  np.testing.assert_allclose(
      np.asarray([float(value) for value in jax_losses]),
      np.asarray([float(value) for value in torch_result]),
      rtol=1e-6,
      atol=1e-6,
  )

