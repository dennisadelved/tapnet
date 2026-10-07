"""Tests for multi-seed peak picking and ZGY window assignment."""

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('matplotlib')

from tapnet.seismic import infer_zgy_peaks_torch
from tapnet.seismic import torch_config
from tapnet.seismic import zgy


class _ArrayReader:

  def __init__(self, values):
    self.values = values
    self.size = values.shape

  def read(self, start, destination):
    slices = tuple(
        slice(origin, origin + length)
        for origin, length in zip(start, destination.shape)
    )
    destination[...] = self.values[slices]


def _geometry(size):
  return zgy.ZgyGeometry(
      size=size,
      zstart=1000.0,
      zinc=4.0,
      annotstart=(200.0, 400.0),
      annotinc=(2.0, 5.0),
      corners=((0.0, 0.0), (69.0, 0.0), (0.0, 79.0), (69.0, 79.0)),
      zunitname='ms',
      hunitname='m',
  )


def test_peak_picker_supports_each_seismic_polarity():
  trace = np.array([0.0, 3.0, 0.0, -4.0, 0.0, 2.0, 0.0])

  positive, _ = infer_zgy_peaks_torch._pick_trace_peaks(
      trace,
      polarity='positive',
      relative_threshold=0.0,
      min_distance=1,
      max_peaks=None,
  )
  negative, _ = infer_zgy_peaks_torch._pick_trace_peaks(
      trace,
      polarity='negative',
      relative_threshold=0.0,
      min_distance=1,
      max_peaks=None,
  )
  both, _ = infer_zgy_peaks_torch._pick_trace_peaks(
      trace,
      polarity='both',
      relative_threshold=0.0,
      min_distance=1,
      max_peaks=None,
  )

  np.testing.assert_array_equal(positive, [1, 5])
  np.testing.assert_array_equal(negative, [3])
  np.testing.assert_array_equal(both, [1, 3, 5])


def test_peak_picker_keeps_stronger_event_within_minimum_distance():
  trace = np.array([0.0, 3.0, 0.0, -4.0, 0.0, 2.0, 0.0])

  peaks, _ = infer_zgy_peaks_torch._pick_trace_peaks(
      trace,
      polarity='both',
      relative_threshold=0.0,
      min_distance=3,
      max_peaks=None,
  )

  np.testing.assert_array_equal(peaks, [3])


def test_peak_windows_cover_full_trace_and_maximize_edge_margin():
  starts = infer_zgy_peaks_torch._depth_window_starts(195, 128)
  assignments = infer_zgy_peaks_torch._assign_peaks_to_windows(
      np.array([1, 100, 194]), starts, 128
  )

  np.testing.assert_array_equal(starts, [0, 64, 67])
  np.testing.assert_array_equal(assignments, [0, 1, 2])


def test_cycle_diagnostics_measure_return_and_ignore_invalid_endpoint():
  forward_tracks = np.array(
      [[[[4.0, 3.0], [5.0, 4.0], [6.0, 5.0]]]], dtype=np.float32
  ).reshape(1, 3, 2)
  cycle_tracks = np.full((1, 2, 3, 2), np.nan, dtype=np.float32)
  cycle_tracks[0, 0] = forward_tracks[0]
  cycle_tracks[0, 0, 1] = [5.5, 4.25]
  cycle_trackability = np.full((1, 2, 3), np.nan, dtype=np.float32)
  cycle_trackability[0, 0] = [0.8, 0.9, 0.7]

  result = infer_zgy_peaks_torch._cycle_diagnostics(
      forward_tracks=forward_tracks,
      forward_trackability=np.array([[0.8, 0.9, 0.7]], dtype=np.float32),
      source_queries=np.array([[1.0, 4.0, 5.0]], dtype=np.float32),
      endpoint_frames=np.array([0, 2], dtype=np.int32),
      endpoint_in_bounds=np.array([[True, False]]),
      cycle_tracks=cycle_tracks,
      cycle_trackability=cycle_trackability,
      visibility_threshold=0.5,
  )

  assert result['roundtrip_lateral_error'][0, 0] == pytest.approx(0.5)
  assert result['roundtrip_depth_error'][0, 0] == pytest.approx(0.25)
  assert result['roundtrip_euclidean_error'][0, 0] == pytest.approx(
      np.hypot(0.5, 0.25)
  )
  assert result['confidence_valid'].tolist() == [[True, False]]
  assert np.isnan(result['roundtrip_euclidean_error'][0, 1])

  summary = infer_zgy_peaks_torch._summarize_cycle_diagnostics(
      result, np.array([[True, False]])
  )

  assert summary['attempted_cycle_count'] == 2
  assert summary['in_bounds_cycle_count'] == 1
  assert summary['confidence_valid_cycle_count'] == 1
  assert summary['confidence_valid'][
      'roundtrip_depth_error_samples'
  ]['mean'] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ('sweep', 'expected_shape', 'expected_start', 'expected_query'),
    [
        ('inline', (2, 90, 64), (34, 8, 0), (1, 32)),
        ('crossline', (2, 90, 64), (3, 39, 0), (1, 32)),
    ],
)
def test_full_depth_sweep_block_preserves_model_axis_order(
    sweep, expected_shape, expected_start, expected_query
):
  config = torch_config.get_config('smoke')
  values = np.arange(70 * 80 * 90, dtype=np.float32).reshape(70, 80, 90)

  block, start, query_frame, query_lateral, snapped = (
      infer_zgy_peaks_torch._read_sweep_block(
          _ArrayReader(values),
          _geometry(values.shape),
          35.0,
          40.0,
          config,
          sweep,
      )
  )

  assert block.shape == expected_shape
  assert start == expected_start
  assert (query_frame, query_lateral) == expected_query
  assert snapped == (35, 40)
