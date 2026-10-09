"""Single-device PyTorch training entry point for synthetic seismic TAPIR."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Mapping
import uuid

import numpy as np
import torch
from torch.utils import data

from tapnet.seismic import torch_config
from tapnet.seismic import torch_data
from tapnet.seismic import torch_losses
from tapnet.torch import tapir_model


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--config', choices=torch_config.CONFIG_VARIANTS, default='smoke'
  )
  parser.add_argument('--steps', type=int, default=None)
  parser.add_argument('--frame-stride', type=int, default=None,
                      help='Train with one survey-line stride instead of the configured mix.')
  parser.add_argument(
      '--num-frames',
      type=int,
      default=None,
      help='Override the synthetic training sequence length.',
  )
  parser.add_argument('--output-dir', type=Path, default=None)
  parser.add_argument('--pretrained-checkpoint', type=Path)
  parser.add_argument('--resume', type=Path)
  parser.add_argument(
      '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
  )
  parser.add_argument('--checkpoint-every', type=int, default=100)
  parser.add_argument(
      '--checkpoint-mode',
      choices=('full', 'model-only'),
      default='full',
      help=(
          'full saves resumable optimizer state; model-only is smaller and '
          'can only be used for parameter initialization.'
      ),
  )
  parser.add_argument('--disable-amp', action='store_true')
  parser.add_argument(
      '--amp-dtype', choices=('bfloat16', 'float16'), default=None
  )
  parser.add_argument(
      '--overfit-one-batch',
      action='store_true',
      help='Reuse one deterministic batch for the learning sanity check.',
  )
  parser.add_argument(
      '--train-feature-encoder',
      action='store_true',
      help='Also update the ResNet/extra-convolution feature encoder.',
  )
  parser.add_argument(
      '--encoder-lr-multiplier',
      type=float,
      default=None,
      help='Feature-encoder learning rate relative to the head learning rate.',
  )
  args = parser.parse_args()
  if args.pretrained_checkpoint and args.resume:
    parser.error('--pretrained-checkpoint and --resume are mutually exclusive.')
  if args.steps is not None and args.steps < 1:
    parser.error('--steps must be positive.')
  if args.num_frames is not None and args.num_frames < 2:
    parser.error('--num-frames must be at least 2 when supplied.')
  if args.frame_stride is not None and args.frame_stride < 1:
    parser.error('--frame-stride must be positive.')
  if args.checkpoint_every < 1:
    parser.error('--checkpoint-every must be positive.')
  if args.encoder_lr_multiplier is not None and not (
      0.0 < args.encoder_lr_multiplier <= 1.0
  ):
    parser.error('--encoder-lr-multiplier must be greater than 0 and at most 1.')
  return args


def _select_device(requested: str) -> torch.device:
  if requested == 'auto':
    requested = 'cuda' if torch.cuda.is_available() else 'cpu'
  if requested == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError(
        'CUDA was requested but torch.cuda.is_available() is False. Install '
        'a CUDA-enabled Windows PyTorch wheel and verify VDI GPU access.'
    )
  return torch.device(requested)


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_torch_file(path: Path, device: torch.device):
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = torch.load(path, map_location=device, weights_only=True)
  if not isinstance(payload, dict) or payload.get('checkpoint_format') != (
      'sharded_v1'
  ):
    return payload
  shards = payload['shards']
  training = torch.load(
      _checkpoint_shard_path(path.parent, shards['training']),
      map_location=device,
      weights_only=True,
  )
  training['model'] = torch.load(
      _checkpoint_shard_path(path.parent, shards['model']),
      map_location=device,
      weights_only=True,
  )
  if 'optimizer' in shards:
    training['optimizer'] = torch.load(
        _checkpoint_shard_path(path.parent, shards['optimizer']),
        map_location=device,
        weights_only=True,
    )
  return training


def _build_model(
    config: torch_config.TorchTrainingConfig,
) -> tapir_model.TAPIR:
  return tapir_model.TAPIR(
      bilinear_interp_with_depthwise_conv=False,
      num_pips_iter=config.num_pips_iter,
      pyramid_level=config.pyramid_level,
      initial_resolution=config.initial_resolution,
      extra_convs=True,
      use_casual_conv=False,
  )


def _freeze_feature_encoder(model: tapir_model.TAPIR) -> None:
  modules = [model.resnet_torch]
  if model.extra_convs is not None:
    modules.append(model.extra_convs)
  for module in modules:
    module.requires_grad_(False)


def _optimizer_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    encoder_learning_rate_multiplier: float,
) -> list[dict[str, object]]:
  grouped_parameters = {
      ('head', 'decay'): [],
      ('head', 'no_decay'): [],
      ('encoder', 'decay'): [],
      ('encoder', 'no_decay'): [],
  }
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
      continue
    component = (
        'encoder'
        if name.startswith(('resnet_torch.', 'extra_convs.'))
        else 'head'
    )
    decay = 'no_decay' if name.endswith('.bias') else 'decay'
    grouped_parameters[(component, decay)].append(parameter)
  if not any(grouped_parameters.values()):
    raise ValueError('No trainable model parameters remain.')
  parameter_groups = []
  for (component, decay), parameters in grouped_parameters.items():
    if not parameters:
      continue
    rate_multiplier = (
        encoder_learning_rate_multiplier if component == 'encoder' else 1.0
    )
    parameter_groups.append(
        {
            'params': parameters,
            'weight_decay': weight_decay if decay == 'decay' else 0.0,
            'lr': learning_rate * rate_multiplier,
        }
    )
  return parameter_groups


def _lr_multiplier(
    step: int, config: torch_config.TorchTrainingConfig
) -> float:
  if config.warmup_steps and step < config.warmup_steps:
    return float(step + 1) / config.warmup_steps
  decay_steps = max(config.steps - config.warmup_steps, 1)
  progress = min(max(step - config.warmup_steps, 0) / decay_steps, 1.0)
  end_ratio = config.end_learning_rate / config.learning_rate
  return end_ratio + 0.5 * (1.0 - end_ratio) * (
      1.0 + math.cos(math.pi * progress)
  )


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
  return {
      key: value.to(device, non_blocking=device.type == 'cuda')
      for key, value in batch.items()
  }


def _clip_gradients(
    model: torch.nn.Module, max_norm: float
) -> torch.Tensor:
  """Clips finite gradients and refuses an invalid optimizer update."""
  return torch.nn.utils.clip_grad_norm_(
      model.parameters(), max_norm, error_if_nonfinite=True
  )


def _format_intermediate_scalars(
    scalars: Mapping[str, torch.Tensor],
) -> str:
  """Formats every deeply supervised unrefined output for diagnostics."""
  stage_indices = sorted(
      int(name.removeprefix('position_loss_'))
      for name in scalars
      if name.startswith('position_loss_')
  )
  fields = []
  for index in stage_indices:
    for label, scalar_name in (
        ('position', f'position_loss_{index}'),
        ('occlusion', f'occlusion_loss_{index}'),
        ('probability', f'probability_loss_{index}'),
        ('total', f'loss_{index}'),
    ):
      value = float(scalars[scalar_name].detach())
      fields.append(f'stage_{index}_{label}={value:.6f}')
  return ' '.join(fields)


def _metrics_fieldnames(
    num_stages: int, *, include_temporal_stride: bool = False
) -> list[str]:
  """Returns the stable, self-contained training CSV schema."""
  fields = [
      'run_id',
      'step',
      'total_steps',
      'config',
      'fixed_batch',
      'seed',
      'device',
      'gpu',
      'torch_version',
      'cuda_version',
      'precision',
      'feature_encoder',
      'source_checkpoint',
      'output_dir',
      'checkpoint_every',
      'checkpoint_mode',
      'training_config_json',
      'loss_config_json',
      'elapsed_seconds',
      'step_seconds',
      'learning_rate',
      'encoder_learning_rate',
      'gradient_norm',
      'peak_cuda_memory_gib',
      'loss',
      'position_loss',
      'occlusion_loss',
      'probability_loss',
      'intermediate_loss',
  ]
  if include_temporal_stride:
    insertion = fields.index('elapsed_seconds')
    fields[insertion:insertion] = ['temporal_stride', 'scene_num_frames']
  for index in range(num_stages):
    fields.extend(
        (
            f'stage_{index}_position',
            f'stage_{index}_occlusion',
            f'stage_{index}_probability',
            f'stage_{index}_total',
        )
    )
  return fields


def _prepare_metrics_csv(path: Path, fieldnames: list[str]) -> None:
  """Creates a CSV header or validates an existing append target."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.is_file() and path.stat().st_size:
    with path.open('r', newline='', encoding='utf-8') as handle:
      existing_header = next(csv.reader(handle), None)
    if existing_header != fieldnames:
      raise ValueError(
          f'Existing metrics schema does not match this trainer: {path}'
      )
    return
  with path.open('w', newline='', encoding='utf-8') as handle:
    csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _append_metrics_csv(
    path: Path, fieldnames: list[str], row: Mapping[str, object]
) -> None:
  """Appends and flushes one training step by closing the file immediately."""
  with path.open('a', newline='', encoding='utf-8') as handle:
    csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def _config_signature(config: torch_config.TorchTrainingConfig) -> dict:
  serialized = dataclasses.asdict(config)
  return {
      key: serialized[key]
      for key in (
          'synthetic',
          'initial_resolution',
          'pyramid_level',
          'query_chunk_size',
          'num_pips_iter',
          'fixed_batch',
          'freeze_feature_encoder',
          'amp_enabled',
          'amp_dtype',
          'encoder_learning_rate_multiplier',
      )
  }


def _checkpoint_shard_path(directory: Path, filename: str) -> Path:
  """Resolves a generated checkpoint shard without allowing path traversal."""
  relative = Path(filename)
  if relative.name != filename or not filename.startswith('checkpoint_step'):
    raise ValueError(f'Invalid checkpoint shard name: {filename!r}')
  return directory / relative


def _read_sharded_manifest(path: Path) -> dict | None:
  """Reads a small local manifest without loading a legacy checkpoint."""
  if not path.is_file() or path.stat().st_size > 1024**2:
    return None
  try:
    payload = torch.load(path, map_location='cpu', weights_only=True)
  except Exception:
    return None
  if isinstance(payload, dict) and payload.get('checkpoint_format') == (
      'sharded_v1'
  ):
    return payload
  return None


def _save_torch_payload(payload: object, path: Path) -> None:
  """Owns the file handle so Windows releases it after serialization errors."""
  with path.open('wb') as handle:
    torch.save(payload, handle)


def _save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    config: torch_config.TorchTrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    checkpoint_mode: str = 'full',
) -> Path:
  if checkpoint_mode not in ('full', 'model-only'):
    raise ValueError(f'Unsupported checkpoint mode: {checkpoint_mode!r}')
  output_dir.mkdir(parents=True, exist_ok=True)
  destination = output_dir / 'latest.pt'
  previous_manifest = _read_sharded_manifest(destination)
  token = uuid.uuid4().hex
  prefix = f'checkpoint_step{step:08d}_{token}'
  shard_names = {
      'model': f'{prefix}.model.pt',
      'training': f'{prefix}.training.pt',
  }
  if checkpoint_mode == 'full':
    shard_names['optimizer'] = f'{prefix}.optimizer.pt'
  shard_payloads = {
      'model': model.state_dict(),
      'training': {
          'step': step,
          'config': dataclasses.asdict(config),
          'config_signature': _config_signature(config),
          'scheduler': scheduler.state_dict(),
          'scaler': scaler.state_dict(),
          'checkpoint_mode': checkpoint_mode,
      },
  }
  if checkpoint_mode == 'full':
    shard_payloads['optimizer'] = optimizer.state_dict()
  final_shards = {
      key: _checkpoint_shard_path(output_dir, name)
      for key, name in shard_names.items()
  }
  temporary_shards = {
      key: path.with_suffix(path.suffix + '.tmp')
      for key, path in final_shards.items()
  }
  manifest_temporary = output_dir / f'latest_{token}.pt.tmp'
  created_paths = []
  active_shard = 'none'
  try:
    for key in shard_names:
      active_shard = key
      _save_torch_payload(shard_payloads[key], temporary_shards[key])
      temporary_shards[key].replace(final_shards[key])
      created_paths.append(final_shards[key])
    _save_torch_payload(
        {
            'checkpoint_format': 'sharded_v1',
            'checkpoint_mode': checkpoint_mode,
            'step': step,
            'shards': shard_names,
        },
        manifest_temporary,
    )
    manifest_temporary.replace(destination)
  except Exception as error:
    cleanup_failures = []
    cleanup_targets = (
        list(temporary_shards.values())
        + created_paths
        + [manifest_temporary]
    )
    for path in cleanup_targets:
      try:
        path.unlink(missing_ok=True)
      except OSError as cleanup_error:
        cleanup_failures.append(f'{path.name}: {cleanup_error}')
    cleanup_status = (
        'partial new checkpoint files removed'
        if not cleanup_failures
        else 'cleanup failures: ' + '; '.join(cleanup_failures)
    )
    free_gib = shutil.disk_usage(output_dir).free / 1024**3
    raise RuntimeError(
        f'Sharded checkpoint save failed at step {step} while writing '
        f'{active_shard!r}; {free_gib:.3f} GiB volume-level free space in '
        f'{output_dir} (this may not reflect a user or directory quota); '
        f'{cleanup_status}. Any previous latest.pt was preserved.'
    ) from error

  if previous_manifest is not None:
    for filename in previous_manifest['shards'].values():
      previous_path = _checkpoint_shard_path(output_dir, filename)
      if previous_path not in final_shards.values():
        try:
          previous_path.unlink(missing_ok=True)
        except OSError:
          pass
  return destination


def main() -> None:
  args = _parse_args()
  config = torch_config.get_config(args.config)
  if args.steps is not None:
    config = dataclasses.replace(config, steps=args.steps)
  if args.frame_stride is not None:
    config = dataclasses.replace(
        config, synthetic=dataclasses.replace(
            config.synthetic, frame_strides=(args.frame_stride,)
        ),
    )
  if args.num_frames is not None:
    config = dataclasses.replace(
        config,
        synthetic=dataclasses.replace(
            config.synthetic, num_frames=args.num_frames
        ),
    )
  if args.overfit_one_batch:
    config = dataclasses.replace(config, fixed_batch=True)
  if args.disable_amp:
    config = dataclasses.replace(config, amp_enabled=False)
  if args.amp_dtype is not None:
    config = dataclasses.replace(config, amp_dtype=args.amp_dtype)
  if args.encoder_lr_multiplier is not None:
    config = dataclasses.replace(
        config,
        encoder_learning_rate_multiplier=args.encoder_lr_multiplier,
    )
  output_dir = args.output_dir or Path(
      f'checkpoints/seismic_tapir_torch_{args.config}'
  )
  device = _select_device(args.device)
  resume_state = (
      _load_torch_file(args.resume, device) if args.resume else None
  )
  if resume_state is not None:
    if 'optimizer' not in resume_state:
      raise ValueError(
          'The requested --resume checkpoint is model-only and cannot restore '
          'optimizer/scheduler state. Pass it with --pretrained-checkpoint to '
          'start a new optimizer instead.'
      )
    saved_config = resume_state.get('config', {})
    saved_freeze = bool(saved_config.get('freeze_feature_encoder', False))
    saved_encoder_lr_multiplier = float(
        saved_config.get('encoder_learning_rate_multiplier', 1.0)
    )
    if saved_freeze and args.train_feature_encoder:
      raise ValueError(
          'A frozen-encoder checkpoint cannot be resumed with '
          '--train-feature-encoder because its optimizer state excludes those '
          'parameters. Start a new fine-tuning run from model weights instead.'
      )
    if (
        args.encoder_lr_multiplier is not None
        and args.encoder_lr_multiplier != saved_encoder_lr_multiplier
    ):
      raise ValueError(
          'Encoder learning-rate multiplier cannot change during exact resume. '
          'Start a new run from model parameters instead.'
      )
    config = dataclasses.replace(
        config,
        freeze_feature_encoder=saved_freeze,
        encoder_learning_rate_multiplier=saved_encoder_lr_multiplier,
    )
  elif args.pretrained_checkpoint:
    config = dataclasses.replace(
        config, freeze_feature_encoder=not args.train_feature_encoder
    )
  config.validate()
  _set_seed(config.seed)

  print(f'device={device}')
  gpu_name = ''
  if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(device)
    print(f'gpu={gpu_name}')
    torch.cuda.reset_peak_memory_stats(device)

  model = _build_model(config)
  start_step = 0
  if resume_state is not None:
    saved_signature = dict(resume_state.get('config_signature', {}))
    saved_signature.setdefault('encoder_learning_rate_multiplier', 1.0)
    if saved_signature != _config_signature(config):
      raise ValueError(
          'Resume checkpoint model/data configuration does not match the '
          'requested configuration.'
      )
    model.load_state_dict(resume_state['model'])
    start_step = int(resume_state['step'])
  elif args.pretrained_checkpoint:
    pretrained = _load_torch_file(args.pretrained_checkpoint, device)
    model.load_state_dict(pretrained.get('model', pretrained))

  if config.freeze_feature_encoder:
    _freeze_feature_encoder(model)
    feature_encoder_status = 'frozen'
  else:
    feature_encoder_status = 'trainable'
  print(f'feature_encoder={feature_encoder_status}')
  model.to(device)
  model.train()

  parameter_groups = _optimizer_parameter_groups(
      model,
      config.weight_decay,
      config.learning_rate,
      config.encoder_learning_rate_multiplier,
  )
  optimizer = torch.optim.AdamW(
      parameter_groups,
      lr=config.learning_rate,
      betas=(0.9, 0.95),
      eps=1e-8,
  )
  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer, lambda step: _lr_multiplier(step, config)
  )
  amp_enabled = device.type == 'cuda' and config.amp_enabled
  amp_dtype = (
      torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
  )
  scaler_enabled = amp_enabled and amp_dtype == torch.float16
  scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
  precision = config.amp_dtype if amp_enabled else 'float32'
  print(f'precision={precision}')
  if resume_state is not None:
    optimizer.load_state_dict(resume_state['optimizer'])
    scheduler.load_state_dict(resume_state['scheduler'])
    scaler.load_state_dict(resume_state['scaler'])

  dataset = torch_data.SyntheticSeismicIterableDataset(
      config.synthetic,
      seed=config.seed,
      start_index=0 if config.fixed_batch else start_step,
  )
  loader = data.DataLoader(
      dataset,
      batch_size=1,
      num_workers=0,
      pin_memory=device.type == 'cuda',
  )
  stream = iter(loader)
  fixed_batch = (
      _move_batch(next(stream), device) if config.fixed_batch else None
  )
  loss_config = torch_losses.SeismicLossConfig(
      lateral_loss_weight=config.lateral_loss_weight
  )
  metrics_path = output_dir / 'metrics.csv'
  include_temporal_stride = len(config.synthetic.frame_strides) > 1
  metric_fieldnames = _metrics_fieldnames(
      config.num_pips_iter,
      include_temporal_stride=include_temporal_stride,
  )
  _prepare_metrics_csv(metrics_path, metric_fieldnames)
  print(f'metrics_csv={metrics_path}')
  run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
      '%Y%m%dT%H%M%S.%fZ'
  )
  source_checkpoint = str(args.resume or args.pretrained_checkpoint or '')
  training_started_at = time.perf_counter()

  last_checkpoint_step = start_step
  for step in range(start_step, config.steps):
    step_started_at = time.perf_counter()
    batch = (
        fixed_batch
        if fixed_batch is not None
        else _move_batch(next(stream), device)
    )
    optimizer.zero_grad(set_to_none=True)
    learning_rate = float(optimizer.param_groups[0]['lr'])
    autocast = (
        torch.amp.autocast('cuda', dtype=amp_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with autocast:
      outputs = model(
          batch['video'],
          batch['query_points'],
          is_training=True,
          query_chunk_size=config.query_chunk_size,
      )
      loss, scalars = torch_losses.seismic_supervised_loss(
          outputs, batch, loss_config
      )
    if not torch.isfinite(loss):
      raise FloatingPointError(f'Non-finite loss at step {step + 1}: {loss}')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = _clip_gradients(model, config.gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    completed_step = step + 1
    summary_names = (
        'loss',
        'position_loss',
        'occlusion_loss',
        'probability_loss',
        'intermediate_loss',
    )
    scalar_text = ' '.join(
        f'{name}={float(scalars[name].detach()):.6f}'
        for name in summary_names
    )
    step_finished_at = time.perf_counter()
    peak_cuda_memory_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == 'cuda'
        else ''
    )
    metric_row = {
        'run_id': run_id,
        'step': completed_step,
        'total_steps': config.steps,
        'config': args.config,
        'fixed_batch': config.fixed_batch,
        'seed': config.seed,
        'device': str(device),
        'gpu': gpu_name,
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda or '',
        'precision': precision,
        'feature_encoder': feature_encoder_status,
        'source_checkpoint': source_checkpoint,
        'output_dir': str(output_dir),
        'checkpoint_every': args.checkpoint_every,
        'checkpoint_mode': args.checkpoint_mode,
        'training_config_json': json.dumps(
            dataclasses.asdict(config), sort_keys=True, separators=(',', ':')
        ),
        'loss_config_json': json.dumps(
            dataclasses.asdict(loss_config),
            sort_keys=True,
            separators=(',', ':'),
        ),
        'elapsed_seconds': step_finished_at - training_started_at,
        'step_seconds': step_finished_at - step_started_at,
        'learning_rate': learning_rate,
        'encoder_learning_rate': (
            learning_rate * config.encoder_learning_rate_multiplier
            if feature_encoder_status == 'trainable'
            else ''
        ),
        'gradient_norm': float(gradient_norm),
        'peak_cuda_memory_gib': peak_cuda_memory_gib,
        **{
            name: float(scalars[name].detach()) for name in summary_names
        },
    }
    if include_temporal_stride:
      metric_row.update({
          'temporal_stride': int(batch['frame_stride'].item()),
          'scene_num_frames': int(batch['scene_num_frames'].item()),
      })
    for index in range(config.num_pips_iter):
      metric_row.update(
          {
              f'stage_{index}_position': float(
                  scalars[f'position_loss_{index}'].detach()
              ),
              f'stage_{index}_occlusion': float(
                  scalars[f'occlusion_loss_{index}'].detach()
              ),
              f'stage_{index}_probability': float(
                  scalars[f'probability_loss_{index}'].detach()
              ),
              f'stage_{index}_total': float(
                  scalars[f'loss_{index}'].detach()
              ),
          }
      )
    _append_metrics_csv(metrics_path, metric_fieldnames, metric_row)
    print(
        f'step={completed_step}/{config.steps} {scalar_text} '
        f'gradient_norm={float(gradient_norm):.6f} '
        f'lr={learning_rate:.8f}'
    )
    intermediate_text = _format_intermediate_scalars(scalars)
    if intermediate_text:
      print(f'intermediate_step={completed_step} {intermediate_text}')

    if completed_step % args.checkpoint_every == 0:
      checkpoint = _save_checkpoint(
          output_dir,
          step=completed_step,
          config=config,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          scaler=scaler,
          checkpoint_mode=args.checkpoint_mode,
      )
      print(f'checkpoint={checkpoint}')
      last_checkpoint_step = completed_step

  if last_checkpoint_step != config.steps:
    checkpoint = _save_checkpoint(
        output_dir,
        step=config.steps,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        checkpoint_mode=args.checkpoint_mode,
    )
    print(f'checkpoint={checkpoint}')
  if device.type == 'cuda':
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f'peak_cuda_memory_gib={peak_gib:.3f}')


if __name__ == '__main__':
  main()

