"""Tests for real-volume TAPIR patch and coordinate mapping."""

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('matplotlib')

from tapnet.seismic import infer_zgy_torch
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


def _geometry(size=(70, 80, 90)):
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


def test_inline_sweep_transposes_zgy_axes_to_frame_depth_lateral():
  config = torch_config.get_config('smoke')
  values = np.arange(70 * 80 * 90, dtype=np.float32).reshape(70, 80, 90)
  query_index = (35.0, 40.0, 45.0)

  video, start, query = infer_zgy_torch._window_and_video(
      _ArrayReader(values), _geometry(), query_index, config, 'inline'
  )

  assert video.shape == (2, 64, 64)
  np.testing.assert_array_equal(
      video,
      np.transpose(values[34:36, 8:72, 13:77], (0, 2, 1)),
  )
  assert start == (34, 8, 13)
  np.testing.assert_allclose(query, [1.0, 32.0, 32.0])


def test_crossline_sweep_transposes_zgy_axes_to_frame_depth_lateral():
  config = torch_config.get_config('smoke')
  values = np.arange(70 * 80 * 90, dtype=np.float32).reshape(70, 80, 90)
  query_index = (35.0, 40.0, 45.0)

  video, start, query = infer_zgy_torch._window_and_video(
      _ArrayReader(values), _geometry(), query_index, config, 'crossline'
  )

  assert video.shape == (2, 64, 64)
  np.testing.assert_array_equal(
      video,
      np.transpose(values[3:67, 39:41, 13:77], (1, 2, 0)),
  )
  assert start == (3, 39, 13)
  np.testing.assert_allclose(query, [1.0, 32.0, 32.0])


def test_model_tracks_map_back_to_survey_coordinates():
  tracks = np.array([[3.0, 5.0], [4.0, 6.0]], dtype=np.float32)

  result = infer_zgy_torch._survey_tracks(
      tracks, (10, 20, 30), _geometry(), 'inline'
  )

  np.testing.assert_allclose(result['inline_index'], [10.0, 11.0])
  np.testing.assert_allclose(result['crossline_index'], [23.0, 24.0])
  np.testing.assert_allclose(result['z_index'], [35.0, 36.0])
  np.testing.assert_allclose(result['inline_annotation'], [220.0, 222.0])
  np.testing.assert_allclose(result['crossline_annotation'], [515.0, 520.0])
  np.testing.assert_allclose(result['z_coordinate'], [1140.0, 1144.0])
  np.testing.assert_allclose(result['world_x'], [10.0, 11.0])
  np.testing.assert_allclose(result['world_y'], [23.0, 24.0])


def test_patch_normalization_records_and_replaces_nonfinite_values():
  amplitude = np.array([0.0, 2.0, -4.0, np.nan], dtype=np.float32)

  normalized, scale, nonfinite_count = infer_zgy_torch._normalize_amplitude(
      amplitude, 100.0
  )

  np.testing.assert_allclose(normalized, [0.0, 0.5, -1.0, 0.0])
  assert scale == pytest.approx(4.0)
  assert nonfinite_count == 1


@pytest.mark.parametrize('sweep', ['inline', 'crossline'])
def test_real_track_curtain_renders_for_each_sweep(tmp_path, sweep):
  geometry = _geometry()
  tracks = np.array([[32.0, 31.0], [32.0, 32.0]], dtype=np.float32)
  start = (10, 20, 30)
  survey = infer_zgy_torch._survey_tracks(
      tracks, start, geometry, sweep
  )
  output = tmp_path / f'{sweep}.png'

  infer_zgy_torch._write_curtain(
      output,
      amplitude=np.zeros((2, 64, 64), dtype=np.float32),
      query_point=np.array([1.0, 32.0, 32.0], dtype=np.float32),
      survey_tracks=survey,
      trackability=np.array([0.25, 0.75], dtype=np.float32),
      visibility_threshold=0.5,
      geometry=geometry,
      start=start,
      sweep=sweep,
  )

  assert output.is_file()
  assert output.stat().st_size > 0
