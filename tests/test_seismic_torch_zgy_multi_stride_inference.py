"""Tests for aligned multi-stride inference on real ZGY data."""

import contextlib
import csv
import json
import sys

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('matplotlib')

from tapnet.seismic import infer_zgy_peaks_torch_multi_stride as multi
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


def test_view_indices_contain_center_and_clamp_at_boundaries():
  centered, centered_query = multi._view_indices(20, 4, 3, 50)
  left, left_query = multi._view_indices(0, 4, 3, 50)
  right, right_query = multi._view_indices(49, 4, 3, 50)

  np.testing.assert_array_equal(centered, [14, 17, 20, 23])
  assert centered_query == 2
  np.testing.assert_array_equal(left, [0, 3, 6, 9])
  assert left_query == 0
  np.testing.assert_array_equal(right, [40, 43, 46, 49])
  assert right_query == 3


def test_view_indices_reject_span_larger_than_survey_axis():
  with pytest.raises(ValueError, match='require 13 lines'):
    multi._view_indices(5, 4, 4, 10)


@pytest.mark.parametrize(
    ('sweep', 'expected_start'),
    [('inline', (31, 8, 0)), ('crossline', (3, 36, 0))],
)
def test_multistride_read_is_one_dense_block_with_aligned_source(
    sweep, expected_start
):
  values = np.arange(70 * 80 * 90, dtype=np.float32).reshape(70, 80, 90)

  block, start, query_lateral, snapped, views, query_frames = (
      multi._read_multistride_block(
          _ArrayReader(values),
          _geometry(),
          35.0,
          40.0,
          frame_count=2,
          frame_strides=(1, 2, 4),
          width=64,
          sweep=sweep,
      )
  )

  assert block.shape == (5, 90, 64)
  assert start == expected_start
  assert query_lateral == 32
  assert snapped == (35, 40)
  assert query_frames == {1: 1, 2: 1, 4: 1}
  for stride, indices in views.items():
    assert indices[query_frames[stride]] == 4


def test_strided_model_tracks_map_to_physical_survey_indices():
  tracks = np.array([[3.0, 5.0], [4.0, 6.0]], dtype=np.float32)

  result = multi._survey_tracks_at_indices(
      tracks,
      np.array([11, 15]),
      (10, 20, 0),
      30,
      _geometry(),
      'inline',
  )

  np.testing.assert_allclose(result['inline_index'], [11.0, 15.0])
  np.testing.assert_allclose(result['crossline_index'], [23.0, 24.0])
  np.testing.assert_allclose(result['z_index'], [35.0, 36.0])
  np.testing.assert_allclose(result['inline_annotation'], [222.0, 230.0])


def test_cross_scale_disagreement_uses_only_shared_physical_lines():
  geometry = _geometry()
  results = {
      1: {
          'geometry': geometry,
          'sweep_indices': np.array([10, 11, 12, 13]),
          'model_tracks': np.array(
              [[[3.0, 5.0], [3.0, 6.0], [3.0, 7.0], [3.0, 8.0]]]
          ),
          'trackability': np.array([[0.9, 0.9, 0.9, 0.9]]),
      },
      2: {
          'geometry': geometry,
          'sweep_indices': np.array([10, 12, 14, 16]),
          'model_tracks': np.array(
              [[[4.0, 6.0], [4.0, 9.0], [4.0, 10.0], [4.0, 11.0]]]
          ),
          'trackability': np.array([[0.8, 0.8, 0.8, 0.8]]),
      },
  }

  rows, summary = multi._cross_scale_disagreement(
      results,
      np.array([5]),
      np.array([1.0]),
      sweep='inline',
      visibility_threshold=0.5,
  )

  assert [row['sweep_index'] for row in rows] == [10, 12]
  assert [row['absolute_depth_disagreement_samples'] for row in rows] == [
      1.0, 2.0
  ]
  assert summary['1_vs_2']['shared_physical_line_count'] == 2
  assert summary['1_vs_2']['all'][
      'absolute_lateral_disagreement_traces'
  ]['mean'] == pytest.approx(1.0)


def test_cli_writes_per_stride_and_disagreement_artifacts(
    tmp_path, monkeypatch
):
  values = np.zeros((70, 80, 90), dtype=np.float32)
  values[:, :, 10] = 3.0
  values[:, :, 20] = -4.0

  class Reader(_ArrayReader):

    def __init__(self, array):
      super().__init__(array)
      geometry = _geometry(array.shape)
      self.zstart = geometry.zstart
      self.zinc = geometry.zinc
      self.annotstart = geometry.annotstart
      self.annotinc = geometry.annotinc
      self.corners = geometry.corners
      self.meta = {
          'zunitname': geometry.zunitname,
          'hunitname': geometry.hunitname,
      }

  @contextlib.contextmanager
  def open_reader(_path):
    yield Reader(values)

  class Model:

    def load_state_dict(self, _state):
      return None

    def to(self, _device):
      return self

    def eval(self):
      return self

  def run_model(_model, video, queries, **_kwargs):
    query_count = len(queries)
    frame_count = len(video)
    tracks = np.empty((query_count, frame_count, 2), dtype=np.float32)
    tracks[..., 0] = queries[:, None, 2]
    tracks[..., 1] = queries[:, None, 1] + np.arange(frame_count)
    logits = np.full((query_count, frame_count), -4.0, dtype=np.float32)
    trackability = np.full(
        (query_count, frame_count), 0.9, dtype=np.float32
    )
    return tracks, logits, logits, trackability

  output = tmp_path / 'output'
  monkeypatch.setattr(multi.zgy, 'open_zgy_reader', open_reader)
  monkeypatch.setattr(
      multi.infer_torch,
      '_load_model_checkpoint',
      lambda *_args: ({}, {'step': 7}),
  )
  monkeypatch.setattr(
      multi.infer_torch, '_validate_checkpoint_config', lambda *_args: None
  )
  monkeypatch.setattr(
      multi.train_torch, '_select_device', lambda _requested: multi.torch.device('cpu')
  )
  monkeypatch.setattr(multi.train_torch, '_build_model', lambda _config: Model())
  monkeypatch.setattr(multi, '_run_model', run_model)
  monkeypatch.setattr(
      sys,
      'argv',
      [
          'multi',
          '--input', str(tmp_path / 'cube.zgy'),
          '--checkpoint', str(tmp_path / 'checkpoint.pt'),
          '--output-dir', str(output),
          '--config', 'smoke',
          '--frames-per-view', '2',
          '--frame-strides', '1', '2',
          '--sweep', 'crossline',
          '--coordinates', 'index',
          '--query-inline', '35',
          '--query-crossline', '40',
          '--max-peaks', '2',
          '--cycle-consistency',
          '--device', 'cpu',
      ],
  )

  multi.main()

  expected = {
      'tracks_stride1.csv',
      'tracks_stride2.csv',
      'cross_scale_disagreement.csv',
      'cycle_consistency_stride1.csv',
      'cycle_consistency_stride2.csv',
      'predictions.npz',
      'track_curtain_multistride.png',
      'summary.json',
  }
  assert expected <= {path.name for path in output.iterdir()}
  with (output / 'cross_scale_disagreement.csv').open(
      newline='', encoding='utf-8'
  ) as handle:
    rows = list(csv.DictReader(handle))
  assert rows
  assert {row['sweep_index'] for row in rows} == {'40'}
  summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
  assert summary['model_config']['frame_strides'] == [1, 2]
  assert summary['fusion']['performed'] is False
  assert summary['cycle_consistency']['enabled'] is True
