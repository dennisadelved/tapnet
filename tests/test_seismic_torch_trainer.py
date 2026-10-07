"""Focused safety tests for the PyTorch training loop."""

import csv
import sys

import pytest

torch = pytest.importorskip('torch')

from tapnet.seismic import train_torch
from tapnet.seismic import torch_config


def test_training_cli_accepts_temporal_length_override(monkeypatch):
  monkeypatch.setattr(
      sys,
      'argv',
      ['train_torch', '--config', 'vdi-hard', '--num-frames', '64'],
  )

  args = train_torch._parse_args()

  assert args.num_frames == 64


def test_nonfinite_gradient_is_rejected_before_optimizer_step():
  model = torch.nn.Linear(1, 1)
  for parameter in model.parameters():
    parameter.grad = torch.full_like(parameter, float('nan'))

  with pytest.raises(RuntimeError, match='non-finite'):
    train_torch._clip_gradients(model, max_norm=1.0)


def test_intermediate_scalar_format_includes_each_loss_term():
  scalars = {
      'position_loss_0': torch.tensor(1.0),
      'occlusion_loss_0': torch.tensor(2.0),
      'probability_loss_0': torch.tensor(3.0),
      'loss_0': torch.tensor(6.0),
  }

  text = train_torch._format_intermediate_scalars(scalars)

  assert text == (
      'stage_0_position=1.000000 stage_0_occlusion=2.000000 '
      'stage_0_probability=3.000000 stage_0_total=6.000000'
  )


def test_metrics_csv_is_self_describing_and_appendable(tmp_path):
  path = tmp_path / 'metrics.csv'
  fieldnames = train_torch._metrics_fieldnames(num_stages=2)
  row = {
      'run_id': 'run-a',
      'step': 1,
      'loss': 1.25,
      'stage_0_total': 0.75,
      'stage_1_total': 0.5,
  }

  train_torch._prepare_metrics_csv(path, fieldnames)
  train_torch._append_metrics_csv(path, fieldnames, row)
  train_torch._prepare_metrics_csv(path, fieldnames)
  train_torch._append_metrics_csv(path, fieldnames, {**row, 'step': 2})

  with path.open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
  assert [row['step'] for row in rows] == ['1', '2']
  assert rows[0]['stage_0_total'] == '0.75'
  assert 'source_checkpoint' in rows[0]
  assert 'training_config_json' in rows[0]
  assert 'loss_config_json' in rows[0]
  assert 'checkpoint_mode' in rows[0]
  assert 'encoder_learning_rate' in rows[0]
  assert 'peak_cuda_memory_gib' in rows[0]


def test_optimizer_uses_lower_learning_rate_for_encoder_parameters():
  class ToyModel(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.resnet_torch = torch.nn.Linear(2, 2)
      self.extra_convs = torch.nn.Linear(2, 2)
      self.head = torch.nn.Linear(2, 2)

  groups = train_torch._optimizer_parameter_groups(
      ToyModel(),
      weight_decay=0.01,
      learning_rate=1e-4,
      encoder_learning_rate_multiplier=0.1,
  )

  group_settings = {
      (group['lr'], group['weight_decay']) for group in groups
  }
  assert group_settings == {
      (1e-4, 0.01),
      (1e-4, 0.0),
      (1e-5, 0.01),
      (1e-5, 0.0),
  }


def test_failed_checkpoint_removes_partial_and_preserves_latest(
    tmp_path, monkeypatch
):
  config = torch_config.get_config('smoke')
  model = torch.nn.Linear(1, 1)
  optimizer = torch.optim.AdamW(model.parameters())
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
  scaler = torch.amp.GradScaler('cuda', enabled=False)
  latest = tmp_path / 'latest.pt'
  latest.write_bytes(b'valid previous checkpoint')

  def fail_save(*_args, **_kwargs):
    raise RuntimeError('simulated write failure')

  monkeypatch.setattr(torch, 'save', fail_save)

  with pytest.raises(RuntimeError, match='previous latest.pt was preserved'):
    train_torch._save_checkpoint(
        tmp_path,
        step=2,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

  assert latest.read_bytes() == b'valid previous checkpoint'
  assert not list(tmp_path.glob('checkpoint_step*.tmp'))


def test_sharded_checkpoint_round_trip_and_replaces_previous_shards(tmp_path):
  config = torch_config.get_config('smoke')
  model = torch.nn.Linear(1, 1)
  optimizer = torch.optim.AdamW(model.parameters())
  model(torch.ones((1, 1))).sum().backward()
  optimizer.step()
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
  scaler = torch.amp.GradScaler('cuda', enabled=False)

  latest = train_torch._save_checkpoint(
      tmp_path,
      step=1,
      config=config,
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      scaler=scaler,
  )
  first_manifest = torch.load(latest, weights_only=True)
  restored = train_torch._load_torch_file(latest, torch.device('cpu'))

  assert first_manifest['checkpoint_format'] == 'sharded_v1'
  assert restored['step'] == 1
  assert restored['model'].keys() == model.state_dict().keys()
  assert restored['optimizer']['state']

  train_torch._save_checkpoint(
      tmp_path,
      step=2,
      config=config,
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      scaler=scaler,
  )
  second_manifest = torch.load(latest, weights_only=True)

  assert second_manifest['step'] == 2
  assert second_manifest['shards'] != first_manifest['shards']
  for filename in first_manifest['shards'].values():
    assert not (tmp_path / filename).exists()


def test_model_only_checkpoint_omits_optimizer_and_loads_parameters(tmp_path):
  config = torch_config.get_config('smoke')
  model = torch.nn.Linear(1, 1)
  optimizer = torch.optim.AdamW(model.parameters())
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
  scaler = torch.amp.GradScaler('cuda', enabled=False)

  latest = train_torch._save_checkpoint(
      tmp_path,
      step=1,
      config=config,
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      scaler=scaler,
      checkpoint_mode='model-only',
  )
  manifest = torch.load(latest, weights_only=True)
  restored = train_torch._load_torch_file(latest, torch.device('cpu'))

  assert manifest['checkpoint_mode'] == 'model-only'
  assert set(manifest['shards']) == {'model', 'training'}
  assert 'optimizer' not in restored
  assert restored['model'].keys() == model.state_dict().keys()

