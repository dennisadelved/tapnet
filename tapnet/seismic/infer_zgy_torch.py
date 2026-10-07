"""Run one seeded TAPIR track through a bounded patch of a real ZGY cube."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tapnet.seismic import infer_torch
from tapnet.seismic import torch_config
from tapnet.seismic import train_torch
from tapnet.seismic import zgy


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument(
      '--config', choices=('smoke', 'vdi-small'), default='vdi-small'
  )
  parser.add_argument('--sweep', choices=('inline', 'crossline'), default='inline')
  parser.add_argument(
      '--coordinates', choices=('annotation', 'index'), default='annotation'
  )
  parser.add_argument('--query-inline', type=float, required=True)
  parser.add_argument('--query-crossline', type=float, required=True)
  parser.add_argument('--query-z', type=float, required=True)
  parser.add_argument(
      '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
  )
  parser.add_argument('--disable-amp', action='store_true')
  parser.add_argument(
      '--amp-dtype', choices=('bfloat16', 'float16'), default='bfloat16'
  )
  parser.add_argument('--visibility-threshold', type=float, default=0.5)
  parser.add_argument('--normalization-percentile', type=float, default=99.5)
  args = parser.parse_args()
  if args.input.suffix.lower() != '.zgy':
    parser.error('--input must be a .zgy file.')
  if not 0.0 < args.visibility_threshold < 1.0:
    parser.error('--visibility-threshold must be between 0 and 1.')
  if not 0.0 < args.normalization_percentile <= 100.0:
    parser.error('--normalization-percentile must be in (0, 100].')
  return args


def _window_and_video(
    reader,
    geometry: zgy.ZgyGeometry,
    query_index: tuple[float, float, float],
    config: torch_config.TorchTrainingConfig,
    sweep: str,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
  """Reads one model patch and returns video plus model-space query point."""
  frame_count = config.synthetic.num_frames
  height, width = config.initial_resolution
  inline_index, crossline_index, z_index = query_index
  depth_start = zgy.centered_window_start(z_index, height, geometry.size[2])
  if sweep == 'inline':
    frame_start = zgy.centered_window_start(
        inline_index, frame_count, geometry.size[0]
    )
    lateral_start = zgy.centered_window_start(
        crossline_index, width, geometry.size[1]
    )
    start = (frame_start, lateral_start, depth_start)
    raw = zgy.read_window(reader, start, (frame_count, width, height))
    video = np.transpose(raw, (0, 2, 1))
    query = np.asarray(
        [
            inline_index - frame_start,
            z_index - depth_start,
            crossline_index - lateral_start,
        ],
        dtype=np.float32,
    )
  elif sweep == 'crossline':
    frame_start = zgy.centered_window_start(
        crossline_index, frame_count, geometry.size[1]
    )
    lateral_start = zgy.centered_window_start(
        inline_index, width, geometry.size[0]
    )
    start = (lateral_start, frame_start, depth_start)
    raw = zgy.read_window(reader, start, (width, frame_count, height))
    video = np.transpose(raw, (1, 2, 0))
    query = np.asarray(
        [
            crossline_index - frame_start,
            z_index - depth_start,
            inline_index - lateral_start,
        ],
        dtype=np.float32,
    )
  else:
    raise ValueError(f'Unsupported sweep: {sweep!r}')
  return video, start, query


def _normalize_amplitude(
    amplitude: np.ndarray, percentile: float
) -> tuple[np.ndarray, float, int]:
  """Applies the synthetic renderer's absolute-percentile scaling to a patch."""
  finite = np.isfinite(amplitude)
  finite_count = int(np.sum(finite))
  if not finite_count:
    raise ValueError('The selected ZGY patch contains no finite amplitudes.')
  scale = float(np.percentile(np.abs(amplitude[finite]), percentile))
  if scale <= np.finfo(np.float32).eps:
    raise ValueError('The selected ZGY patch has zero amplitude scale.')
  clean = np.where(finite, amplitude, 0.0)
  normalized = np.clip(clean / scale, -1.0, 1.0).astype(np.float32)
  return normalized, scale, int(amplitude.size - finite_count)


def _survey_tracks(
    tracks: np.ndarray,
    start: tuple[int, int, int],
    geometry: zgy.ZgyGeometry,
    sweep: str,
) -> dict[str, np.ndarray]:
  """Maps model [lateral, depth] tracks back to survey indices/annotations."""
  frame = np.arange(tracks.shape[0], dtype=np.float32)
  if sweep == 'inline':
    inline_index = start[0] + frame
    crossline_index = start[1] + tracks[:, 0]
  else:
    inline_index = start[0] + tracks[:, 0]
    crossline_index = start[1] + frame
  z_index = start[2] + tracks[:, 1]
  inline = geometry.annotstart[0] + inline_index * geometry.annotinc[0]
  crossline = (
      geometry.annotstart[1] + crossline_index * geometry.annotinc[1]
  )
  z_coordinate = geometry.zstart + z_index * geometry.zinc
  world_x, world_y = zgy.index_to_world(
      geometry, inline_index, crossline_index
  )
  return {
      'inline_index': inline_index,
      'crossline_index': crossline_index,
      'z_index': z_index,
      'inline_annotation': inline,
      'crossline_annotation': crossline,
      'z_coordinate': z_coordinate,
      'world_x': world_x,
      'world_y': world_y,
  }


def _write_tracks_csv(
    path: Path,
    *,
    survey_tracks: dict[str, np.ndarray],
    model_tracks: np.ndarray,
    trackability: np.ndarray,
    visibility_threshold: float,
    sweep: str,
) -> None:
  fields = [
      'frame',
      'sweep',
      'inline_index',
      'crossline_index',
      'z_index',
      'inline_annotation',
      'crossline_annotation',
      'z_coordinate',
      'world_x',
      'world_y',
      'model_lateral',
      'model_depth',
      'trackability_probability',
      'predicted_visible',
  ]
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for frame in range(len(model_tracks)):
      writer.writerow({
          'frame': frame,
          'sweep': sweep,
          **{
              key: float(values[frame])
              for key, values in survey_tracks.items()
          },
          'model_lateral': float(model_tracks[frame, 0]),
          'model_depth': float(model_tracks[frame, 1]),
          'trackability_probability': float(trackability[frame]),
          'predicted_visible': (
              float(trackability[frame]) > visibility_threshold
          ),
      })


def _write_curtain(
    path: Path,
    *,
    amplitude: np.ndarray,
    query_point: np.ndarray,
    survey_tracks: dict[str, np.ndarray],
    trackability: np.ndarray,
    visibility_threshold: float,
    geometry: zgy.ZgyGeometry,
    start: tuple[int, int, int],
    sweep: str,
) -> None:
  lateral = int(
      np.clip(round(float(query_point[2])), 0, amplitude.shape[2] - 1)
  )
  curtain = amplitude[:, :, lateral].T
  if sweep == 'inline':
    horizontal = survey_tracks['inline_annotation']
    horizontal_label = 'inline annotation'
    fixed_label = 'crossline'
    fixed_value = geometry.annotstart[1] + (
        start[1] + query_point[2]
    ) * geometry.annotinc[1]
  else:
    horizontal = survey_tracks['crossline_annotation']
    horizontal_label = 'crossline annotation'
    fixed_label = 'inline'
    fixed_value = geometry.annotstart[0] + (
        start[0] + query_point[2]
    ) * geometry.annotinc[0]
  h_step = float(horizontal[1] - horizontal[0])
  vertical = survey_tracks['z_coordinate']
  query_horizontal = float(horizontal[0] + query_point[0] * h_step)
  query_z = geometry.zstart + (start[2] + query_point[1]) * geometry.zinc
  z_top = geometry.zstart + start[2] * geometry.zinc
  z_bottom = z_top + (curtain.shape[0] - 1) * geometry.zinc
  scale = max(float(np.percentile(np.abs(curtain), 99.0)), 1e-6)
  figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
  axis.imshow(
      curtain,
      cmap='gray',
      vmin=-scale,
      vmax=scale,
      origin='upper',
      aspect='auto',
      extent=(
          horizontal[0] - h_step / 2,
          horizontal[-1] + h_step / 2,
          z_bottom + geometry.zinc / 2,
          z_top - geometry.zinc / 2,
      ),
  )
  visible = trackability > visibility_threshold
  axis.plot(horizontal, vertical, color='#ff2d95', linewidth=2, label='prediction')
  axis.scatter(
      horizontal[visible], vertical[visible], color='#ff2d95', s=25,
      label='predicted visible'
  )
  axis.scatter(
      [query_horizontal], [query_z], marker='*', color='#ffe600',
      edgecolor='black', s=130, zorder=4, label='query'
  )
  axis.set_title(f'{sweep} sweep at {fixed_label} {fixed_value:g}')
  axis.set_xlabel(horizontal_label)
  axis.set_ylabel(f'Z ({geometry.zunitname or "header units"})')
  axis.legend(loc='best')
  figure.savefig(path, dpi=160)
  plt.close(figure)


def main() -> None:
  args = _parse_args()
  config = torch_config.get_config(args.config)
  device = train_torch._select_device(args.device)
  model_state, checkpoint_metadata = infer_torch._load_model_checkpoint(
      args.checkpoint, device
  )
  infer_torch._validate_checkpoint_config(checkpoint_metadata, config)

  with zgy.open_zgy_reader(args.input) as reader:
    geometry = zgy.ZgyGeometry.from_reader(reader)
    requested = (args.query_inline, args.query_crossline, args.query_z)
    query_index = (
        zgy.annotation_to_index(geometry, *requested)
        if args.coordinates == 'annotation'
        else requested
    )
    query_index = zgy.validate_index(query_index, geometry)
    raw_amplitude, start, query_point = _window_and_video(
        reader, geometry, query_index, config, args.sweep
    )
  amplitude, scale, nonfinite_count = _normalize_amplitude(
      raw_amplitude, args.normalization_percentile
  )
  video = np.repeat(amplitude[..., None], 3, axis=-1)

  model = train_torch._build_model(config)
  model.load_state_dict(model_state)
  model.to(device)
  model.eval()
  amp_enabled = device.type == 'cuda' and not args.disable_amp
  amp_dtype = (
      torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
  )
  autocast = (
      torch.amp.autocast('cuda', dtype=amp_dtype)
      if amp_enabled
      else contextlib.nullcontext()
  )
  with torch.inference_mode(), autocast:
    outputs = model(
        torch.from_numpy(video).unsqueeze(0).to(device),
        torch.from_numpy(query_point).reshape(1, 1, 3).to(device),
        is_training=False,
        query_chunk_size=config.query_chunk_size,
    )
  model_tracks = outputs['tracks'][0, 0].detach().float().cpu().numpy()
  occlusion_logits = (
      outputs['occlusion'][0, 0].detach().float().cpu().numpy()
  )
  expected_dist_logits = (
      outputs['expected_dist'][0, 0].detach().float().cpu().numpy()
  )
  trackability = infer_torch._trackability_probability(
      occlusion_logits, expected_dist_logits
  ).astype(np.float32)
  survey_tracks = _survey_tracks(model_tracks, start, geometry, args.sweep)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  _write_tracks_csv(
      args.output_dir / 'tracks.csv',
      survey_tracks=survey_tracks,
      model_tracks=model_tracks,
      trackability=trackability,
      visibility_threshold=args.visibility_threshold,
      sweep=args.sweep,
  )
  np.savez_compressed(
      args.output_dir / 'predictions.npz',
      raw_amplitude=raw_amplitude,
      normalized_amplitude=amplitude,
      query_index=np.asarray(query_index, dtype=np.float64),
      query_point_model=query_point,
      model_tracks=model_tracks,
      occlusion_logits=occlusion_logits,
      expected_dist_logits=expected_dist_logits,
      trackability_probability=trackability,
      **survey_tracks,
  )
  _write_curtain(
      args.output_dir / 'track_curtain.png',
      amplitude=amplitude,
      query_point=query_point,
      survey_tracks=survey_tracks,
      trackability=trackability,
      visibility_threshold=args.visibility_threshold,
      geometry=geometry,
      start=start,
      sweep=args.sweep,
  )
  query_annotation = zgy.index_to_annotation(geometry, *query_index)
  query_frame = int(round(float(query_point[0])))
  diagnostics = {
      'trackability_min': float(np.min(trackability)),
      'trackability_mean': float(np.mean(trackability)),
      'trackability_max': float(np.max(trackability)),
      'max_lateral_drift_traces': float(
          np.max(np.abs(model_tracks[:, 0] - query_point[2]))
      ),
      'max_depth_deviation_samples': float(
          np.max(np.abs(model_tracks[:, 1] - query_point[1]))
      ),
      'query_frame_lateral_residual_traces': float(
          model_tracks[query_frame, 0] - query_point[2]
      ),
      'query_frame_depth_residual_samples': float(
          model_tracks[query_frame, 1] - query_point[1]
      ),
  }
  summary = {
      'created_at_utc': datetime.datetime.now(
          datetime.timezone.utc
      ).isoformat(),
      'input_zgy': str(args.input.resolve()),
      'checkpoint': str(args.checkpoint.resolve()),
      'checkpoint_step': checkpoint_metadata.get('step'),
      'config': args.config,
      'model_config': {
          'initial_resolution': list(config.initial_resolution),
          'num_frames': config.synthetic.num_frames,
          'query_chunk_size': config.query_chunk_size,
          'num_pips_iter': config.num_pips_iter,
          'pyramid_level': config.pyramid_level,
      },
      'sweep': args.sweep,
      'device': str(device),
      'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
      'precision': args.amp_dtype if amp_enabled else 'float32',
      'geometry': dataclasses.asdict(geometry),
      'query': {
          'requested_coordinate_system': args.coordinates,
          'requested': list(requested),
          'index': list(query_index),
          'annotation': list(query_annotation),
          'model': query_point.tolist(),
      },
      'window': {
          'start_inline_crossline_sample': list(start),
          'raw_shape_inline_crossline_sample': list(
              (config.synthetic.num_frames, config.synthetic.width,
               config.synthetic.height)
              if args.sweep == 'inline'
              else (config.synthetic.width, config.synthetic.num_frames,
                    config.synthetic.height)
          ),
          'model_video_shape_frame_depth_lateral': list(amplitude.shape),
      },
      'normalization': {
          'method': 'clip(amplitude / abs_percentile, -1, 1)',
          'percentile': args.normalization_percentile,
          'scale': scale,
          'nonfinite_replaced_with_zero': nonfinite_count,
      },
      'diagnostics_not_accuracy_metrics': diagnostics,
      'visibility_threshold': args.visibility_threshold,
      'trackability_rule': (
          '(1-sigmoid(occlusion_logit)) * '
          '(1-sigmoid(expected_distance_logit))'
      ),
      'artifacts': {
          'tracks': 'tracks.csv',
          'predictions': 'predictions.npz',
          'visualization': 'track_curtain.png',
      },
  }
  with (args.output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write('\n')
  print(f'device={device}')
  print(f'zgy_size={geometry.size}')
  print(f'query_index={query_index}')
  print(f'window_start={start}')
  print(f'normalization_scale={scale:.8g}')
  print(f'inference_output={args.output_dir}')


if __name__ == '__main__':
  main()
