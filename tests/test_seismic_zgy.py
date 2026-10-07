"""Tests for bounded ZGY reads and survey-coordinate conversion."""

import numpy as np
import pytest

from tapnet.seismic import zgy


class _FakeReader:
  size = (10, 12, 14)
  zstart = 1000.0
  zinc = 4.0
  annotstart = (200.0, 400.0)
  annotinc = (2.0, 5.0)
  corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
  meta = {'zunitname': 'ms', 'hunitname': 'm'}

  def __init__(self):
    self.reads = []

  def read(self, start, destination):
    self.reads.append((tuple(start), tuple(destination.shape)))
    grid = np.indices(destination.shape)
    destination[...] = grid[0] * 100 + grid[1] * 10 + grid[2]


def test_bounded_zgy_read_uses_inline_crossline_sample_order():
  reader = _FakeReader()

  values = zgy.read_window(reader, (2, 3, 4), (3, 4, 5))

  assert values.shape == (3, 4, 5)
  assert values.dtype == np.float32
  assert reader.reads == [((2, 3, 4), (3, 4, 5))]


def test_bounded_zgy_read_rejects_out_of_bounds_window():
  with pytest.raises(ValueError, match='exceeds'):
    zgy.read_window(_FakeReader(), (9, 0, 0), (2, 1, 1))


def test_zgy_annotation_round_trip_preserves_fractional_index():
  geometry = zgy.ZgyGeometry.from_reader(_FakeReader())
  expected = (2.5, 3.25, 4.75)

  annotation = zgy.index_to_annotation(geometry, *expected)
  actual = zgy.annotation_to_index(geometry, *annotation)

  np.testing.assert_allclose(actual, expected)


def test_zgy_world_conversion_interpolates_ordered_corners():
  geometry = zgy.ZgyGeometry.from_reader(_FakeReader())

  world_x, world_y = zgy.index_to_world(
      geometry, np.array([0.0, 9.0, 4.5]), np.array([0.0, 11.0, 5.5])
  )

  np.testing.assert_allclose(world_x, [0.0, 1.0, 0.5])
  np.testing.assert_allclose(world_y, [0.0, 1.0, 0.5])


def test_centered_window_is_clamped_at_each_boundary():
  assert zgy.centered_window_start(2.0, 8, 20) == 0
  assert zgy.centered_window_start(10.0, 8, 20) == 6
  assert zgy.centered_window_start(19.0, 8, 20) == 12
