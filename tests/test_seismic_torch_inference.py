"""Tests for held-out PyTorch seismic inference artifacts and metrics."""

import csv

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('matplotlib')

from tapnet.seismic import infer_torch
from tapnet.seismic import torch_config
from tapnet.seismic import train_torch


def test_trackability_probability_matches_tapir_visibility_rule():
  occlusion = np.array([0.0, 100.0], dtype=np.float32)
  expected_distance = np.array([0.0, 0.0], dtype=np.float32)

  probability = infer_torch._trackability_probability(
      occlusion, expected_distance
  )

  np.testing.assert_allclose(probability, [0.25, 0.0], atol=1e-6)


def test_metrics_mask_occluded_and_invalid_positions():
  target = np.zeros((1, 1, 4, 2), dtype=np.float32)
  predicted = target.copy()
  predicted[0, 0, :, 1] = [1.0, 100.0, 100.0, 3.0]
  occluded = np.array([[[False, True, False, False]]])
  valid = np.array([[[True, True, False, True]]])
  trackability = np.array([[[0.9, 0.1, 0.9, 0.9]]], dtype=np.float32)

  metrics = infer_torch._compute_metrics(
      predicted, target, occluded, valid, trackability, 0.5
  )

  assert metrics['valid_position_count'] == 2
  assert metrics['depth_mae_samples'] == pytest.approx(2.0)
  assert metrics['depth_within_1_sample'] == pytest.approx(0.5)
  assert metrics['visibility_accuracy'] == pytest.approx(1.0)


def test_tracks_csv_blanks_position_error_when_target_is_hidden(tmp_path):
  query = np.array([[[0.0, 10.0, 2.0]]], dtype=np.float32)
  target = np.zeros((1, 1, 2, 2), dtype=np.float32)
  predicted = np.ones_like(target)
  occluded = np.array([[[False, True]]])
  valid = np.ones_like(occluded)
  confidence = np.array([[[0.9, 0.1]]], dtype=np.float32)
  path = tmp_path / 'tracks.csv'

  infer_torch._write_tracks_csv(
      path,
      example_seeds=[123],
      query_points=query,
      predicted_tracks=predicted,
      target_tracks=target,
      target_occluded=occluded,
      label_valid=valid,
      trackability=confidence,
      trackgroup=np.array([[7]], dtype=np.int32),
      faulted=np.array([True]),
      sweep_reversed=np.array([False]),
      frame_strides=np.array([2], dtype=np.int32),
      frame_indices=np.array([[0, 2]], dtype=np.int32),
      visibility_threshold=0.5,
  )

  with path.open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
  assert rows[0]['absolute_depth_error'] == '1.0'
  assert rows[0]['query_trackgroup'] == '7'
  assert rows[0]['example_faulted'] == 'True'
  assert rows[0]['temporal_stride'] == '2'
  assert rows[1]['scene_frame_index'] == '2'
  assert rows[1]['absolute_depth_error'] == ''
  assert rows[1]['target_visible'] == 'False'


def test_inference_loads_model_shard_without_optimizer_state(tmp_path):
  config = torch_config.get_config('smoke')
  model = torch.nn.Linear(1, 1)
  optimizer = torch.optim.AdamW(model.parameters())
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
  scaler = torch.amp.GradScaler('cuda', enabled=False)
  latest = train_torch._save_checkpoint(
      tmp_path,
      step=3,
      config=config,
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      scaler=scaler,
  )

  model_state, metadata = infer_torch._load_model_checkpoint(
      latest, torch.device('cpu')
  )

  assert model_state.keys() == model.state_dict().keys()
  assert metadata['step'] == 3
  assert 'optimizer' not in metadata
