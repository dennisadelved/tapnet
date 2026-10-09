"""Physical invariants of Wu's folding and curved fault transformations."""

import dataclasses

import numpy as np
import pytest

from tapnet.seismic import structure


@pytest.mark.parametrize('slip_sign', (-1, 1))
def test_curved_fault_moves_both_blocks_without_opening_a_gap(slip_sign):
  fault = structure.sample_faults(128, 1, 20, np.random.default_rng(9))[0]
  fault = dataclasses.replace(fault, max_slip=slip_sign * abs(fault.max_slip))
  np.testing.assert_allclose(fault.rotation @ fault.rotation.T, np.eye(3), atol=1e-15)
  controls = fault.surface_controls
  np.testing.assert_allclose(
      fault.surface(controls[:, 0] * fault.strike_radius,
                    controls[:, 1] * fault.dip_radius), controls[:, 2], atol=1e-12,
  )
  x = np.linspace(-0.6, 0.6, 31) * fault.strike_radius
  y = np.linspace(-0.6, 0.6, 31) * fault.dip_radius
  # Samples arbitrarily close to either side stay close to the curved sheet.
  for distance in (-1e-6, 1e-6, -5.0, 5.0):
    original = fault._global((x, y, fault.surface(x, y) + distance))
    deformed = fault.forward(original)
    local = fault._local(deformed)
    np.testing.assert_allclose(local[2] - fault.surface(*local[:2]), distance, atol=1e-12)
    assert np.max(np.abs(local[1] - y)) > 1
    restored = fault.restore(deformed)
    np.testing.assert_allclose(restored, original, atol=1e-8)


def test_finite_slip_vanishes_at_tips_and_outside_the_drag_zone():
  fault = structure.WuFault((0, 0, 64), 30, 65, 12, 40, 50, 20)
  slip, derivative = fault.slip_profile(np.array([0, 40, 80]), np.zeros(3))
  np.testing.assert_array_equal(slip, [12, 0, 0])
  np.testing.assert_array_equal(derivative, 0)
  local = (np.array([0, 40, 80]), np.zeros(3), np.array([21, 0, 0]))
  original = fault._global(local)
  np.testing.assert_allclose(fault.forward(original), original, atol=1e-13)


def test_sequential_faults_restore_in_reverse_order():
  rng = np.random.default_rng(13)
  faults = structure.sample_faults(128, 3, 40, rng)
  assert len(faults) > 1
  original = tuple(rng.uniform(-50, 50, 1000) for _ in range(2)) + (
      rng.uniform(0, 127, 1000),
  )
  deformed = original
  for fault in faults:
    deformed = fault.forward(deformed)
  for fault in reversed(faults):
    deformed = fault.restore(deformed)
  np.testing.assert_allclose(deformed, original, atol=1e-8)


def test_depth_scaled_fold_inverse_recovers_flat_stratigraphy():
  dip, gaussians = structure.sample_fold(128, 'folded', np.random.default_rng(8))
  x, y = np.meshgrid(np.linspace(-60, 60, 21), np.linspace(-60, 60, 23))
  gaussian = sum(amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
                 for cx, cy, width, amplitude in gaussians)
  for age in (0, 32, 64, 127):
    depth = age + dip[0] * x + dip[1] * y + 1.5 * age / 127 * gaussian
    np.testing.assert_allclose(
        structure.restore_fold((x, y, depth), 128, dip, gaussians), age, atol=1e-13,
    )
