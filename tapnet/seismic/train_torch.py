"""Single-device PyTorch training entry point for synthetic seismic TAPIR."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
from pathlib import Path
import random
from typing import Mapping

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
      '--config', choices=('smoke', 'vdi-small'), default='smoke'
  )
  parser.add_argument('--steps', type=int, default=None)
  parser.add_argument('--output-dir', type=Path, default=None)
  parser.add_argument('--pretrained-checkpoint', type=Path)
  parser.add_argument('--resume', type=Path)
  parser.add_argument(
      '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
  )
  parser.add_argument('--checkpoint-every', type=int, default=100)
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
  args = parser.parse_args()
  if args.pretrained_checkpoint and args.resume:
    parser.error('--pretrained-checkpoint and --resume are mutually exclusive.')
  if args.steps is not None and args.steps < 1:
    parser.error('--steps must be positive.')
  if args.checkpoint_every < 1:
    parser.error('--checkpoint-every must be positive.')
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
  return torch.load(path, map_location=device, weights_only=True)


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
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, object]]:
  decay = []
  no_decay = []
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
      continue
    if name.endswith('.bias'):
      no_decay.append(parameter)
    else:
      decay.append(parameter)
  if not decay and not no_decay:
    raise ValueError('No trainable model parameters remain.')
  return [
      {'params': decay, 'weight_decay': weight_decay},
      {'params': no_decay, 'weight_decay': 0.0},
  ]


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
      )
  }


def _save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    config: torch_config.TorchTrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  destination = output_dir / 'latest.pt'
  temporary = output_dir / 'latest.pt.tmp'
  torch.save(
      {
          'step': step,
          'config': dataclasses.asdict(config),
          'config_signature': _config_signature(config),
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'scaler': scaler.state_dict(),
      },
      temporary,
  )
  temporary.replace(destination)
  return destination


def main() -> None:
  args = _parse_args()
  config = torch_config.get_config(args.config)
  if args.steps is not None:
    config = dataclasses.replace(config, steps=args.steps)
  if args.overfit_one_batch:
    config = dataclasses.replace(config, fixed_batch=True)
  if args.disable_amp:
    config = dataclasses.replace(config, amp_enabled=False)
  if args.amp_dtype is not None:
    config = dataclasses.replace(config, amp_dtype=args.amp_dtype)
  output_dir = args.output_dir or Path(
      f'checkpoints/seismic_tapir_torch_{args.config}'
  )
  device = _select_device(args.device)
  resume_state = (
      _load_torch_file(args.resume, device) if args.resume else None
  )
  if resume_state is not None:
    saved_config = resume_state.get('config', {})
    saved_freeze = bool(saved_config.get('freeze_feature_encoder', False))
    if saved_freeze and args.train_feature_encoder:
      raise ValueError(
          'A frozen-encoder checkpoint cannot be resumed with '
          '--train-feature-encoder because its optimizer state excludes those '
          'parameters. Start a new fine-tuning run from model weights instead.'
      )
    config = dataclasses.replace(
        config, freeze_feature_encoder=saved_freeze
    )
  elif args.pretrained_checkpoint:
    config = dataclasses.replace(
        config, freeze_feature_encoder=not args.train_feature_encoder
    )
  config.validate()
  _set_seed(config.seed)

  print(f'device={device}')
  if device.type == 'cuda':
    print(f'gpu={torch.cuda.get_device_name(device)}')
    torch.cuda.reset_peak_memory_stats(device)

  model = _build_model(config)
  start_step = 0
  if resume_state is not None:
    if resume_state.get('config_signature') != _config_signature(config):
      raise ValueError(
          'Resume checkpoint model/data configuration does not match the '
          'requested configuration.'
      )
    model.load_state_dict(resume_state['model'])
    start_step = int(resume_state['step'])
  elif args.pretrained_checkpoint:
    pretrained = _load_torch_file(args.pretrained_checkpoint, device)
    model.load_state_dict(pretrained)

  if config.freeze_feature_encoder:
    _freeze_feature_encoder(model)
    print('feature_encoder=frozen')
  else:
    print('feature_encoder=trainable')
  model.to(device)
  model.train()

  parameter_groups = _optimizer_parameter_groups(model, config.weight_decay)
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
  loss_config = torch_losses.SeismicLossConfig()

  last_checkpoint_step = start_step
  for step in range(start_step, config.steps):
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
    )
    print(f'checkpoint={checkpoint}')
  if device.type == 'cuda':
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f'peak_cuda_memory_gib={peak_gib:.3f}')


if __name__ == '__main__':
  main()

