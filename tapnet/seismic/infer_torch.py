"""Held-out synthetic inference for the native PyTorch seismic TAPIR model."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tapnet.seismic import synthetic
from tapnet.seismic import torch_config
from tapnet.seismic import train_torch


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--checkpoint', type=Path, required=True, help='latest.pt or legacy .pt.'
  )
  parser.add_argument(
      '--config', choices=torch_config.CONFIG_VARIANTS, default='vdi-small'
  )
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument(
      '--num-frames',
      type=int,
      default=None,
      help='Override the synthetic evaluation sequence length.',
  )
  parser.add_argument(
      '--frame-stride',
      type=int,
      default=None,
      help='Evaluate one explicit survey-line stride instead of the config mix.',
  )
  parser.add_argument('--num-examples', type=int, default=8)
  parser.add_argument(
      '--seed',
      type=int,
      default=1_000_000,
      help='Held-out stream seed; keep it distinct from training seed 0.',
  )
  parser.add_argument(
      '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
  )
  parser.add_argument('--disable-amp', action='store_true')
  parser.add_argument(
      '--amp-dtype', choices=('bfloat16', 'float16'), default='bfloat16'
  )
  parser.add_argument('--visibility-threshold', type=float, default=0.5)
  parser.add_argument('--queries-per-image', type=int, default=4)
  args = parser.parse_args()
  if args.num_examples < 1:
    parser.error('--num-examples must be positive.')
  if args.num_frames is not None and args.num_frames < 2:
    parser.error('--num-frames must be at least 2 when supplied.')
  if args.frame_stride is not None and args.frame_stride < 1:
    parser.error('--frame-stride must be positive when supplied.')
  if not 0.0 < args.visibility_threshold < 1.0:
    parser.error('--visibility-threshold must be between 0 and 1.')
  if args.queries_per_image < 1:
    parser.error('--queries-per-image must be positive.')
  return args


def _load_model_checkpoint(
    path: Path, device: torch.device
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
  """Loads model weights and small metadata without loading optimizer state."""
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = torch.load(path, map_location=device, weights_only=True)
  if isinstance(payload, dict) and payload.get('checkpoint_format') == (
      'sharded_v1'
  ):
    shards = payload['shards']
    metadata = torch.load(
        train_torch._checkpoint_shard_path(path.parent, shards['training']),
        map_location='cpu',
        weights_only=True,
    )
    model_state = torch.load(
        train_torch._checkpoint_shard_path(path.parent, shards['model']),
        map_location=device,
        weights_only=True,
    )
    return model_state, metadata
  if not isinstance(payload, dict):
    raise ValueError(f'Unsupported checkpoint payload in {path}.')
  model_state = payload.get('model', payload)
  return model_state, payload


def _validate_checkpoint_config(
    metadata: Mapping[str, object], config: torch_config.TorchTrainingConfig
) -> None:
  saved = metadata.get('config')
  if not isinstance(saved, dict):
    return
  expected = dataclasses.asdict(config)
  for key in ('initial_resolution', 'pyramid_level', 'num_pips_iter'):
    if saved.get(key) != expected[key]:
      raise ValueError(
          f'Checkpoint {key}={saved.get(key)!r} does not match '
          f'--config {key}={expected[key]!r}.'
      )


def _trackability_probability(
    occlusion_logits: np.ndarray, expected_dist_logits: np.ndarray
) -> np.ndarray:
  """Matches TAPIR's published visible/trackable post-processing score."""
  visible_probability = 1.0 / (
      1.0 + np.exp(np.clip(occlusion_logits, -60.0, 60.0))
  )
  accurate_probability = 1.0 / (
      1.0 + np.exp(np.clip(expected_dist_logits, -60.0, 60.0))
  )
  return visible_probability * accurate_probability


def _safe_ratio(numerator: int, denominator: int) -> float:
  return float(numerator / denominator) if denominator else 0.0


def _compute_metrics(
    predicted_tracks: np.ndarray,
    target_tracks: np.ndarray,
    target_occluded: np.ndarray,
    label_valid: np.ndarray,
    trackability: np.ndarray,
    visibility_threshold: float,
) -> dict[str, float | int]:
  valid = label_valid.astype(bool)
  target_visible = valid & ~target_occluded.astype(bool)
  predicted_visible = trackability > visibility_threshold
  depth_error = np.abs(predicted_tracks[..., 1] - target_tracks[..., 1])
  lateral_error = np.abs(predicted_tracks[..., 0] - target_tracks[..., 0])
  visible_depth_error = depth_error[target_visible]
  visible_lateral_error = lateral_error[target_visible]
  if not visible_depth_error.size:
    raise ValueError('Evaluation contains no visible, valid target positions.')

  true_positive = int(np.sum(predicted_visible & target_visible))
  false_positive = int(np.sum(predicted_visible & valid & ~target_visible))
  false_negative = int(np.sum(~predicted_visible & target_visible))
  correct_visibility = int(
      np.sum((predicted_visible == target_visible) & valid)
  )
  precision = _safe_ratio(true_positive, true_positive + false_positive)
  recall = _safe_ratio(true_positive, true_positive + false_negative)
  return {
      'valid_position_count': int(np.sum(target_visible)),
      'valid_visibility_count': int(np.sum(valid)),
      'depth_mae_samples': float(np.mean(visible_depth_error)),
      'depth_rmse_samples': float(
          np.sqrt(np.mean(np.square(visible_depth_error)))
      ),
      'lateral_mae_traces': float(np.mean(visible_lateral_error)),
      'depth_within_1_sample': float(np.mean(visible_depth_error <= 1.0)),
      'depth_within_2_samples': float(np.mean(visible_depth_error <= 2.0)),
      'depth_within_4_samples': float(np.mean(visible_depth_error <= 4.0)),
      'gross_depth_error_gt_8_samples': float(
          np.mean(visible_depth_error > 8.0)
      ),
      'visibility_accuracy': _safe_ratio(
          correct_visibility, int(np.sum(valid))
      ),
      'visibility_precision': precision,
      'visibility_recall': recall,
      'visibility_f1': _safe_ratio(
          2.0 * precision * recall, precision + recall
      ),
      'predicted_trackable_fraction': float(
          np.mean(predicted_visible[valid])
      ),
      'mean_trackability_probability': float(np.mean(trackability[valid])),
  }


def _write_tracks_csv(
    path: Path,
    *,
    example_seeds: list[int],
    query_points: np.ndarray,
    predicted_tracks: np.ndarray,
    target_tracks: np.ndarray,
    target_occluded: np.ndarray,
    label_valid: np.ndarray,
    trackability: np.ndarray,
    trackgroup: np.ndarray,
    faulted: np.ndarray,
    sweep_reversed: np.ndarray,
    frame_strides: np.ndarray,
    frame_indices: np.ndarray,
    visibility_threshold: float,
) -> None:
  fields = [
      'example_id',
      'example_seed',
      'example_faulted',
      'sweep_reversed',
      'temporal_stride',
      'query_id',
      'query_trackgroup',
      'frame',
      'scene_frame_index',
      'query_frame',
      'query_depth',
      'query_lateral',
      'predicted_lateral',
      'predicted_depth',
      'target_lateral',
      'target_depth',
      'label_valid',
      'target_visible',
      'trackability_probability',
      'predicted_visible',
      'absolute_lateral_error',
      'absolute_depth_error',
  ]
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for example_id, example_seed in enumerate(example_seeds):
      for query_id, query in enumerate(query_points[example_id]):
        for frame in range(predicted_tracks.shape[2]):
          predicted = predicted_tracks[example_id, query_id, frame]
          target = target_tracks[example_id, query_id, frame]
          valid = bool(label_valid[example_id, query_id, frame])
          target_visible = valid and not bool(
              target_occluded[example_id, query_id, frame]
          )
          confidence = float(trackability[example_id, query_id, frame])
          writer.writerow({
              'example_id': example_id,
              'example_seed': example_seed,
              'example_faulted': bool(faulted[example_id]),
              'sweep_reversed': bool(sweep_reversed[example_id]),
              'temporal_stride': int(frame_strides[example_id]),
              'query_id': query_id,
              'query_trackgroup': int(trackgroup[example_id, query_id]),
              'frame': frame,
              'scene_frame_index': int(frame_indices[example_id, frame]),
              'query_frame': float(query[0]),
              'query_depth': float(query[1]),
              'query_lateral': float(query[2]),
              'predicted_lateral': float(predicted[0]),
              'predicted_depth': float(predicted[1]),
              'target_lateral': float(target[0]),
              'target_depth': float(target[1]),
              'label_valid': valid,
              'target_visible': target_visible,
              'trackability_probability': confidence,
              'predicted_visible': confidence > visibility_threshold,
              'absolute_lateral_error': (
                  float(abs(predicted[0] - target[0]))
                  if target_visible
                  else ''
              ),
              'absolute_depth_error': (
                  float(abs(predicted[1] - target[1]))
                  if target_visible
                  else ''
              ),
          })


def _write_track_curtain(
    path: Path,
    *,
    video_amplitude: np.ndarray,
    query_points: np.ndarray,
    predicted_tracks: np.ndarray,
    target_tracks: np.ndarray,
    target_occluded: np.ndarray,
    trackability: np.ndarray,
    frame_indices: np.ndarray,
    visibility_threshold: float,
    max_queries: int,
) -> None:
  query_count = min(max_queries, len(query_points))
  column_count = min(2, query_count)
  row_count = (query_count + column_count - 1) // column_count
  figure, axes = plt.subplots(
      row_count, column_count, figsize=(7 * column_count, 3 * row_count),
      squeeze=False, constrained_layout=True
  )
  frames = frame_indices
  for query_id, axis in enumerate(axes.flat):
    if query_id >= query_count:
      axis.axis('off')
      continue
    query = query_points[query_id]
    lateral = int(np.clip(round(float(query[2])), 0, video_amplitude.shape[2] - 1))
    curtain = video_amplitude[:, :, lateral].T
    scale = max(float(np.percentile(np.abs(curtain), 99.0)), 1e-6)
    axis.imshow(
        curtain,
        cmap='gray',
        vmin=-scale,
        vmax=scale,
        origin='upper',
        aspect='auto',
        extent=(
            frames[0] - 0.5,
            frames[-1] + 0.5,
            curtain.shape[0] - 0.5,
            -0.5,
        ),
    )
    target_depth = target_tracks[query_id, :, 1].copy()
    target_depth[target_occluded[query_id]] = np.nan
    predicted_depth = predicted_tracks[query_id, :, 1]
    predicted_visible = trackability[query_id] > visibility_threshold
    axis.plot(frames, target_depth, color='#00e5ff', linewidth=2, label='target')
    axis.plot(
        frames, predicted_depth, color='#ff2d95', linewidth=1.5,
        label='prediction'
    )
    axis.scatter(
        frames[predicted_visible], predicted_depth[predicted_visible],
        color='#ff2d95', s=12
    )
    axis.scatter(
        [frames[int(query[0])]], [query[1]], marker='*', color='#ffe600',
        edgecolor='black',
        s=90, zorder=4, label='query'
    )
    axis.set_title(f'query {query_id}, lateral trace {lateral}')
    axis.set_xlabel('scene frame (inline/crossline offset)')
    axis.set_ylabel('depth sample')
    axis.legend(loc='upper right', fontsize='small')
  figure.savefig(path, dpi=140)
  plt.close(figure)


def main() -> None:
  args = _parse_args()
  config = torch_config.get_config(args.config)
  if args.num_frames is not None:
    config = dataclasses.replace(
        config,
        synthetic=dataclasses.replace(
            config.synthetic, num_frames=args.num_frames
        ),
    )
  config.validate()
  if (
      args.frame_stride is not None
      and args.frame_stride not in config.synthetic.frame_strides
  ):
    raise ValueError(
        f'--frame-stride {args.frame_stride} is not configured for '
        f'{args.config}: {config.synthetic.frame_strides!r}.'
    )
  device = train_torch._select_device(args.device)
  model_state, checkpoint_metadata = _load_model_checkpoint(
      args.checkpoint, device
  )
  _validate_checkpoint_config(checkpoint_metadata, config)
  model = train_torch._build_model(config)
  model.load_state_dict(model_state)
  model.to(device)
  model.eval()

  amp_enabled = device.type == 'cuda' and not args.disable_amp
  amp_dtype = (
      torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  print(f'device={device}')
  if device.type == 'cuda':
    print(f'gpu={torch.cuda.get_device_name(device)}')
  print(f'precision={args.amp_dtype if amp_enabled else "float32"}')

  collected: dict[str, list[np.ndarray]] = {
      'video_amplitude': [],
      'query_points': [],
      'predicted_tracks': [],
      'target_tracks': [],
      'occlusion_logits': [],
      'expected_dist_logits': [],
      'target_occluded': [],
      'label_valid': [],
      'trackgroup': [],
      'faulted': [],
      'sweep_reversed': [],
      'frame_stride': [],
      'scene_num_frames': [],
      'frame_indices': [],
  }
  example_seeds = []
  for example_id in range(args.num_examples):
    example_seed = int(
        np.random.SeedSequence([args.seed, example_id]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    example_seeds.append(example_seed)
    sample = synthetic.generate_synthetic_sample(
        config.synthetic, rng=example_seed, frame_stride=args.frame_stride
    )
    video = torch.from_numpy(sample['video']).unsqueeze(0).to(device)
    queries = torch.from_numpy(sample['query_points']).unsqueeze(0).to(device)
    autocast = (
        torch.amp.autocast('cuda', dtype=amp_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
      outputs = model(
          video,
          queries,
          is_training=False,
          query_chunk_size=config.query_chunk_size,
      )
    for key, output_key in (
        ('predicted_tracks', 'tracks'),
        ('occlusion_logits', 'occlusion'),
        ('expected_dist_logits', 'expected_dist'),
    ):
      collected[key].append(
          outputs[output_key][0].detach().float().cpu().numpy()
      )
    collected['video_amplitude'].append(sample['video'][..., 0])
    collected['query_points'].append(sample['query_points'])
    collected['target_tracks'].append(sample['target_points'])
    for key in (
        'target_occluded',
        'label_valid',
        'trackgroup',
        'faulted',
        'sweep_reversed',
        'frame_stride',
        'scene_num_frames',
        'frame_indices',
    ):
      sample_key = 'occluded' if key == 'target_occluded' else key
      collected[key].append(np.asarray(sample[sample_key]))
    print(f'example={example_id + 1}/{args.num_examples} seed={example_seed}')

  arrays = {key: np.stack(values) for key, values in collected.items()}
  trackability = _trackability_probability(
      arrays['occlusion_logits'], arrays['expected_dist_logits']
  ).astype(np.float32)
  arrays['trackability_probability'] = trackability
  arrays['example_seed'] = np.asarray(example_seeds, dtype=np.uint64)
  metrics = _compute_metrics(
      arrays['predicted_tracks'],
      arrays['target_tracks'],
      arrays['target_occluded'],
      arrays['label_valid'],
      trackability,
      args.visibility_threshold,
  )

  np.savez_compressed(args.output_dir / 'predictions.npz', **arrays)
  _write_tracks_csv(
      args.output_dir / 'tracks.csv',
      example_seeds=example_seeds,
      query_points=arrays['query_points'],
      predicted_tracks=arrays['predicted_tracks'],
      target_tracks=arrays['target_tracks'],
      target_occluded=arrays['target_occluded'],
      label_valid=arrays['label_valid'],
      trackability=trackability,
      trackgroup=arrays['trackgroup'],
      faulted=arrays['faulted'],
      sweep_reversed=arrays['sweep_reversed'],
      frame_strides=arrays['frame_stride'],
      frame_indices=arrays['frame_indices'],
      visibility_threshold=args.visibility_threshold,
  )
  for example_id in range(args.num_examples):
    _write_track_curtain(
        args.output_dir / f'example_{example_id:03d}_tracks.png',
        video_amplitude=arrays['video_amplitude'][example_id],
        query_points=arrays['query_points'][example_id],
        predicted_tracks=arrays['predicted_tracks'][example_id],
        target_tracks=arrays['target_tracks'][example_id],
        target_occluded=arrays['target_occluded'][example_id],
        trackability=trackability[example_id],
        frame_indices=arrays['frame_indices'][example_id],
        visibility_threshold=args.visibility_threshold,
        max_queries=args.queries_per_image,
    )
  summary = {
      'created_at_utc': datetime.datetime.now(
          datetime.timezone.utc
      ).isoformat(),
      'checkpoint': str(args.checkpoint.resolve()),
      'checkpoint_step': checkpoint_metadata.get('step'),
      'config': args.config,
      'synthetic_config': dataclasses.asdict(config.synthetic),
      'evaluation_seed': args.seed,
      'example_seeds': [str(seed) for seed in example_seeds],
      'num_examples': args.num_examples,
      'visibility_threshold': args.visibility_threshold,
      'visibility_rule': (
          '(1-sigmoid(occlusion_logit)) * '
          '(1-sigmoid(expected_distance_logit)) > threshold'
      ),
      'device': str(device),
      'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
      'torch_version': str(torch.__version__),
      'precision': args.amp_dtype if amp_enabled else 'float32',
      'metrics': metrics,
      'artifacts': {
          'predictions': 'predictions.npz',
          'tracks': 'tracks.csv',
          'visualizations': 'example_NNN_tracks.png',
      },
  }
  with (args.output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write('\n')
  print(json.dumps(metrics, sort_keys=True))
  print(f'inference_output={args.output_dir}')


if __name__ == '__main__':
  main()
