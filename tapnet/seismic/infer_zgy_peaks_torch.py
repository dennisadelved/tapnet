"""Seed TAPIR from multiple extrema on one trace of a real ZGY cube."""

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
from tapnet.seismic import infer_zgy_torch
from tapnet.seismic import torch_config
from tapnet.seismic import train_torch
from tapnet.seismic import zgy


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument(
      '--config', choices=torch_config.CONFIG_VARIANTS, default='vdi-small'
  )
  parser.add_argument(
      '--num-frames',
      type=int,
      default=None,
      help='Number of inline/crossline survey lines in the inference sweep.',
  )
  parser.add_argument('--sweep', choices=('inline', 'crossline'), default='inline')
  parser.add_argument(
      '--coordinates', choices=('annotation', 'index'), default='annotation'
  )
  parser.add_argument('--query-inline', type=float, required=True)
  parser.add_argument('--query-crossline', type=float, required=True)
  parser.add_argument(
      '--peak-polarity',
      choices=('positive', 'negative', 'both'),
      default='both',
  )
  parser.add_argument(
      '--peak-relative-threshold',
      type=float,
      default=0.1,
      help='Minimum absolute amplitude relative to the trace p99.5 scale.',
  )
  parser.add_argument('--peak-min-distance', type=int, default=4)
  parser.add_argument('--max-peaks', type=int, default=None)
  parser.add_argument(
      '--cycle-consistency',
      action='store_true',
      help=(
          'Re-query from each predicted first/last-frame endpoint and measure '
          'return error at the original source trace.'
      ),
  )
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
  if args.num_frames is not None and args.num_frames < 2:
    parser.error('--num-frames must be at least 2 when supplied.')
  if not 0.0 <= args.peak_relative_threshold <= 1.0:
    parser.error('--peak-relative-threshold must be in [0, 1].')
  if args.peak_min_distance < 1:
    parser.error('--peak-min-distance must be positive.')
  if args.max_peaks is not None and args.max_peaks < 1:
    parser.error('--max-peaks must be positive when supplied.')
  if not 0.0 < args.visibility_threshold < 1.0:
    parser.error('--visibility-threshold must be between 0 and 1.')
  if not 0.0 < args.normalization_percentile <= 100.0:
    parser.error('--normalization-percentile must be in (0, 100].')
  return args


def _pick_trace_peaks(
    trace: np.ndarray,
    *,
    polarity: str,
    relative_threshold: float,
    min_distance: int,
    max_peaks: int | None,
) -> tuple[np.ndarray, float]:
  """Finds thresholded local extrema and suppresses nearby weaker extrema."""
  trace = np.asarray(trace, dtype=np.float32)
  finite = np.isfinite(trace)
  if not np.any(finite):
    raise ValueError('The selected trace contains no finite amplitudes.')
  clean = np.where(finite, trace, 0.0)
  scale = float(np.percentile(np.abs(clean[finite]), 99.5))
  if scale <= np.finfo(np.float32).eps:
    raise ValueError('The selected trace has zero amplitude scale.')
  center = clean[1:-1]
  positive = (center > clean[:-2]) & (center >= clean[2:])
  negative = (center < clean[:-2]) & (center <= clean[2:])
  if polarity == 'positive':
    candidate = positive
  elif polarity == 'negative':
    candidate = negative
  elif polarity == 'both':
    candidate = positive | negative
  else:
    raise ValueError(f'Unsupported peak polarity: {polarity!r}')
  candidate &= np.abs(center) >= relative_threshold * scale
  indices = np.flatnonzero(candidate) + 1
  strongest_first = indices[np.argsort(-np.abs(clean[indices]))]
  selected = []
  for index in strongest_first:
    if all(abs(int(index) - kept) >= min_distance for kept in selected):
      selected.append(int(index))
      if max_peaks is not None and len(selected) >= max_peaks:
        break
  if not selected:
    raise ValueError(
        'No trace peaks passed the requested polarity, threshold, and spacing.'
    )
  return np.asarray(sorted(selected), dtype=np.int32), scale


def _depth_window_starts(depth_size: int, height: int) -> np.ndarray:
  """Returns half-overlapping windows, including one aligned to the end."""
  if height > depth_size:
    raise ValueError(
        f'Model depth {height} exceeds cube sample count {depth_size}.'
    )
  if height == depth_size:
    return np.asarray([0], dtype=np.int32)
  stride = max(height // 2, 1)
  starts = list(range(0, depth_size - height + 1, stride))
  final_start = depth_size - height
  if starts[-1] != final_start:
    starts.append(final_start)
  return np.asarray(starts, dtype=np.int32)


def _assign_peaks_to_windows(
    peaks: np.ndarray, starts: np.ndarray, height: int
) -> np.ndarray:
  """Assigns each peak to the containing window with the largest edge margin."""
  assignments = []
  for peak in peaks:
    margins = np.minimum(peak - starts, starts + height - 1 - peak)
    margins[(peak < starts) | (peak >= starts + height)] = -1
    best = int(np.argmax(margins))
    if margins[best] < 0:
      raise RuntimeError(f'No model window contains peak sample {peak}.')
    assignments.append(best)
  return np.asarray(assignments, dtype=np.int32)


def _read_sweep_block(
    reader,
    geometry: zgy.ZgyGeometry,
    inline_index: float,
    crossline_index: float,
    config: torch_config.TorchTrainingConfig,
    sweep: str,
) -> tuple[np.ndarray, tuple[int, int, int], int, int, tuple[int, int]]:
  """Reads all samples for one bounded sweep around a snapped source trace."""
  frame_count = config.synthetic.num_frames
  width = config.synthetic.width
  snapped_inline = int(round(inline_index))
  snapped_crossline = int(round(crossline_index))
  if sweep == 'inline':
    frame_start = zgy.centered_window_start(
        snapped_inline, frame_count, geometry.size[0]
    )
    lateral_start = zgy.centered_window_start(
        snapped_crossline, width, geometry.size[1]
    )
    start = (frame_start, lateral_start, 0)
    raw = zgy.read_window(
        reader, start, (frame_count, width, geometry.size[2])
    )
    block = np.transpose(raw, (0, 2, 1))
    query_frame = snapped_inline - frame_start
    query_lateral = snapped_crossline - lateral_start
  elif sweep == 'crossline':
    frame_start = zgy.centered_window_start(
        snapped_crossline, frame_count, geometry.size[1]
    )
    lateral_start = zgy.centered_window_start(
        snapped_inline, width, geometry.size[0]
    )
    start = (lateral_start, frame_start, 0)
    raw = zgy.read_window(
        reader, start, (width, frame_count, geometry.size[2])
    )
    block = np.transpose(raw, (1, 2, 0))
    query_frame = snapped_crossline - frame_start
    query_lateral = snapped_inline - lateral_start
  else:
    raise ValueError(f'Unsupported sweep: {sweep!r}')
  return (
      block,
      start,
      query_frame,
      query_lateral,
      (snapped_inline, snapped_crossline),
  )


def _write_tracks_csv(
    path: Path,
    *,
    peaks: np.ndarray,
    seed_amplitude: np.ndarray,
    model_tracks: np.ndarray,
    survey_tracks: dict[str, np.ndarray],
    trackability: np.ndarray,
    visibility_threshold: float,
    sweep: str,
) -> None:
  fields = [
      'seed_id',
      'seed_sample_index',
      'seed_amplitude',
      'seed_polarity',
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
    for seed_id, peak in enumerate(peaks):
      polarity = 'positive' if seed_amplitude[seed_id] > 0 else 'negative'
      for frame in range(model_tracks.shape[1]):
        writer.writerow({
            'seed_id': seed_id,
            'seed_sample_index': int(peak),
            'seed_amplitude': float(seed_amplitude[seed_id]),
            'seed_polarity': polarity,
            'frame': frame,
            'sweep': sweep,
            **{
                key: float(values[seed_id, frame])
                for key, values in survey_tracks.items()
            },
            'model_lateral': float(model_tracks[seed_id, frame, 0]),
            'model_depth': float(model_tracks[seed_id, frame, 1]),
            'trackability_probability': float(trackability[seed_id, frame]),
            'predicted_visible': (
                float(trackability[seed_id, frame]) > visibility_threshold
            ),
        })


def _cycle_diagnostics(
    *,
    forward_tracks: np.ndarray,
    forward_trackability: np.ndarray,
    source_queries: np.ndarray,
    endpoint_frames: np.ndarray,
    endpoint_in_bounds: np.ndarray,
    cycle_tracks: np.ndarray,
    cycle_trackability: np.ndarray,
    visibility_threshold: float,
) -> dict[str, np.ndarray]:
  """Measures endpoint-reseeded tracks against the original forward tracks."""
  seed_count, endpoint_count = endpoint_in_bounds.shape
  source_frames = np.rint(source_queries[:, 0]).astype(np.int32)
  source_xy = source_queries[:, [2, 1]]
  forward_endpoint_trackability = np.stack(
      [forward_trackability[:, frame] for frame in endpoint_frames], axis=1
  )
  returned_source_xy = np.full(
      (seed_count, endpoint_count, 2), np.nan, dtype=np.float32
  )
  returned_source_trackability = np.full(
      (seed_count, endpoint_count), np.nan, dtype=np.float32
  )
  path_mean_error = np.full(
      (seed_count, endpoint_count), np.nan, dtype=np.float32
  )
  path_max_error = np.full(
      (seed_count, endpoint_count), np.nan, dtype=np.float32
  )
  for seed_id in range(seed_count):
    for endpoint_id in range(endpoint_count):
      if not endpoint_in_bounds[seed_id, endpoint_id]:
        continue
      source_frame = source_frames[seed_id]
      returned_source_xy[seed_id, endpoint_id] = cycle_tracks[
          seed_id, endpoint_id, source_frame
      ]
      returned_source_trackability[seed_id, endpoint_id] = cycle_trackability[
          seed_id, endpoint_id, source_frame
      ]
      path_error = np.linalg.norm(
          cycle_tracks[seed_id, endpoint_id] - forward_tracks[seed_id],
          axis=-1,
      )
      path_mean_error[seed_id, endpoint_id] = float(np.mean(path_error))
      path_max_error[seed_id, endpoint_id] = float(np.max(path_error))
  return_delta = returned_source_xy - source_xy[:, None, :]
  lateral_error = np.abs(return_delta[..., 0])
  depth_error = np.abs(return_delta[..., 1])
  euclidean_error = np.linalg.norm(return_delta, axis=-1)
  confidence_valid = (
      endpoint_in_bounds
      & (forward_endpoint_trackability > visibility_threshold)
      & (returned_source_trackability > visibility_threshold)
  )
  return {
      'source_frame': source_frames,
      'source_xy': source_xy.astype(np.float32),
      'forward_endpoint_trackability': forward_endpoint_trackability.astype(
          np.float32
      ),
      'returned_source_xy': returned_source_xy,
      'returned_source_trackability': returned_source_trackability,
      'roundtrip_lateral_error': lateral_error.astype(np.float32),
      'roundtrip_depth_error': depth_error.astype(np.float32),
      'roundtrip_euclidean_error': euclidean_error.astype(np.float32),
      'path_mean_euclidean_error': path_mean_error,
      'path_max_euclidean_error': path_max_error,
      'confidence_valid': confidence_valid,
  }


def _metric_distribution(
    values: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
  selected = np.asarray(values, dtype=np.float64)[mask]
  selected = selected[np.isfinite(selected)]
  if not len(selected):
    return {
        'count': 0,
        'mean': None,
        'median': None,
        'p95': None,
        'max': None,
    }
  return {
      'count': int(len(selected)),
      'mean': float(np.mean(selected)),
      'median': float(np.median(selected)),
      'p95': float(np.percentile(selected, 95.0)),
      'max': float(np.max(selected)),
  }


def _summarize_cycle_diagnostics(
    diagnostics: dict[str, np.ndarray], endpoint_in_bounds: np.ndarray
) -> dict[str, object]:
  confident = diagnostics['confidence_valid']
  return {
      'attempted_cycle_count': int(endpoint_in_bounds.size),
      'in_bounds_cycle_count': int(np.sum(endpoint_in_bounds)),
      'confidence_valid_cycle_count': int(np.sum(confident)),
      'in_bounds': {
          'roundtrip_lateral_error_traces': _metric_distribution(
              diagnostics['roundtrip_lateral_error'], endpoint_in_bounds
          ),
          'roundtrip_depth_error_samples': _metric_distribution(
              diagnostics['roundtrip_depth_error'], endpoint_in_bounds
          ),
          'roundtrip_euclidean_error_model_pixels': _metric_distribution(
              diagnostics['roundtrip_euclidean_error'], endpoint_in_bounds
          ),
          'path_mean_euclidean_error_model_pixels': _metric_distribution(
              diagnostics['path_mean_euclidean_error'], endpoint_in_bounds
          ),
      },
      'confidence_valid': {
          'roundtrip_lateral_error_traces': _metric_distribution(
              diagnostics['roundtrip_lateral_error'], confident
          ),
          'roundtrip_depth_error_samples': _metric_distribution(
              diagnostics['roundtrip_depth_error'], confident
          ),
          'roundtrip_euclidean_error_model_pixels': _metric_distribution(
              diagnostics['roundtrip_euclidean_error'], confident
          ),
          'path_mean_euclidean_error_model_pixels': _metric_distribution(
              diagnostics['path_mean_euclidean_error'], confident
          ),
      },
  }


def _write_cycle_csv(
    path: Path,
    *,
    peaks: np.ndarray,
    seed_amplitude: np.ndarray,
    endpoint_frames: np.ndarray,
    endpoint_queries: np.ndarray,
    endpoint_in_bounds: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> None:
  fields = [
      'seed_id',
      'seed_sample_index',
      'seed_amplitude',
      'seed_polarity',
      'endpoint',
      'endpoint_frame',
      'endpoint_query_lateral',
      'endpoint_query_depth',
      'endpoint_in_bounds',
      'forward_endpoint_trackability',
      'returned_source_lateral',
      'returned_source_depth',
      'returned_source_trackability',
      'roundtrip_lateral_error_traces',
      'roundtrip_depth_error_samples',
      'roundtrip_euclidean_error_model_pixels',
      'path_mean_euclidean_error_model_pixels',
      'path_max_euclidean_error_model_pixels',
      'confidence_valid',
  ]
  endpoint_names = ('start', 'end')
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for seed_id, peak in enumerate(peaks):
      polarity = 'positive' if seed_amplitude[seed_id] > 0 else 'negative'
      for endpoint_id, endpoint_name in enumerate(endpoint_names):
        returned = diagnostics['returned_source_xy'][seed_id, endpoint_id]
        writer.writerow({
            'seed_id': seed_id,
            'seed_sample_index': int(peak),
            'seed_amplitude': float(seed_amplitude[seed_id]),
            'seed_polarity': polarity,
            'endpoint': endpoint_name,
            'endpoint_frame': int(endpoint_frames[endpoint_id]),
            'endpoint_query_lateral': float(
                endpoint_queries[seed_id, endpoint_id, 2]
            ),
            'endpoint_query_depth': float(
                endpoint_queries[seed_id, endpoint_id, 1]
            ),
            'endpoint_in_bounds': bool(
                endpoint_in_bounds[seed_id, endpoint_id]
            ),
            'forward_endpoint_trackability': float(
                diagnostics['forward_endpoint_trackability'][
                    seed_id, endpoint_id
                ]
            ),
            'returned_source_lateral': float(returned[0]),
            'returned_source_depth': float(returned[1]),
            'returned_source_trackability': float(
                diagnostics['returned_source_trackability'][
                    seed_id, endpoint_id
                ]
            ),
            'roundtrip_lateral_error_traces': float(
                diagnostics['roundtrip_lateral_error'][seed_id, endpoint_id]
            ),
            'roundtrip_depth_error_samples': float(
                diagnostics['roundtrip_depth_error'][seed_id, endpoint_id]
            ),
            'roundtrip_euclidean_error_model_pixels': float(
                diagnostics['roundtrip_euclidean_error'][seed_id, endpoint_id]
            ),
            'path_mean_euclidean_error_model_pixels': float(
                diagnostics['path_mean_euclidean_error'][seed_id, endpoint_id]
            ),
            'path_max_euclidean_error_model_pixels': float(
                diagnostics['path_max_euclidean_error'][seed_id, endpoint_id]
            ),
            'confidence_valid': bool(
                diagnostics['confidence_valid'][seed_id, endpoint_id]
            ),
        })


def _write_curtain(
    path: Path,
    *,
    block: np.ndarray,
    query_frame: int,
    query_lateral: int,
    peaks: np.ndarray,
    survey_tracks: dict[str, np.ndarray],
    trackability: np.ndarray,
    visibility_threshold: float,
    geometry: zgy.ZgyGeometry,
    start: tuple[int, int, int],
    sweep: str,
) -> None:
  curtain = block[:, :, query_lateral].T
  frames = np.arange(block.shape[0], dtype=np.float32)
  if sweep == 'inline':
    horizontal = geometry.annotstart[0] + (
        start[0] + frames
    ) * geometry.annotinc[0]
    horizontal_label = 'inline annotation'
    fixed_label = 'crossline'
    fixed_value = geometry.annotstart[1] + (
        start[1] + query_lateral
    ) * geometry.annotinc[1]
  else:
    horizontal = geometry.annotstart[1] + (
        start[1] + frames
    ) * geometry.annotinc[1]
    horizontal_label = 'crossline annotation'
    fixed_label = 'inline'
    fixed_value = geometry.annotstart[0] + (
        start[0] + query_lateral
    ) * geometry.annotinc[0]
  h_step = float(horizontal[1] - horizontal[0])
  vertical = geometry.zstart + np.arange(
      block.shape[1], dtype=np.float32
  ) * geometry.zinc
  scale = max(float(np.percentile(np.abs(curtain), 99.0)), 1e-6)
  figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
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
          vertical[-1] + geometry.zinc / 2,
          vertical[0] - geometry.zinc / 2,
      ),
  )
  color_map = plt.get_cmap('turbo')
  minimum_peak = float(peaks[0])
  maximum_peak = float(peaks[-1])
  if maximum_peak == minimum_peak:
    maximum_peak += 1.0
  normalizer = matplotlib.colors.Normalize(
      vmin=minimum_peak, vmax=maximum_peak
  )
  query_horizontal = float(horizontal[query_frame])
  for seed_id, peak in enumerate(peaks):
    color = color_map(normalizer(float(peak)))
    predicted_z = survey_tracks['z_coordinate'][seed_id]
    visible = trackability[seed_id] > visibility_threshold
    axis.plot(horizontal, predicted_z, color=color, linewidth=1.4, alpha=0.9)
    axis.scatter(
        horizontal[visible], predicted_z[visible], color=color, s=10
    )
    axis.scatter(
        [query_horizontal], [vertical[peak]], marker='*', color=color,
        edgecolor='black', linewidth=0.4, s=45, zorder=4
    )
  mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
  figure.colorbar(mappable, ax=axis, label='seed sample index')
  axis.set_title(
      f'{len(peaks)} peak-seeded tracks; {sweep} sweep at '
      f'{fixed_label} {fixed_value:g}'
  )
  axis.set_xlabel(horizontal_label)
  axis.set_ylabel(f'Z ({geometry.zunitname or "header units"})')
  figure.savefig(path, dpi=160)
  plt.close(figure)


def main() -> None:
  args = _parse_args()
  config = infer_zgy_torch._real_volume_config(
      torch_config.get_config(args.config), args.num_frames
  )
  device = train_torch._select_device(args.device)
  model_state, checkpoint_metadata = infer_torch._load_model_checkpoint(
      args.checkpoint, device
  )
  infer_torch._validate_checkpoint_config(checkpoint_metadata, config)

  with zgy.open_zgy_reader(args.input) as reader:
    geometry = zgy.ZgyGeometry.from_reader(reader)
    if args.coordinates == 'annotation':
      inline_index, crossline_index, _ = zgy.annotation_to_index(
          geometry,
          args.query_inline,
          args.query_crossline,
          geometry.zstart,
      )
    else:
      inline_index, crossline_index = (
          args.query_inline,
          args.query_crossline,
      )
    zgy.validate_index((inline_index, crossline_index, 0.0), geometry)
    block, base_start, query_frame, query_lateral, snapped = (
        _read_sweep_block(
            reader,
            geometry,
            inline_index,
            crossline_index,
            config,
            args.sweep,
        )
    )
  trace = block[query_frame, :, query_lateral]
  peaks, trace_scale = _pick_trace_peaks(
      trace,
      polarity=args.peak_polarity,
      relative_threshold=args.peak_relative_threshold,
      min_distance=args.peak_min_distance,
      max_peaks=args.max_peaks,
  )
  window_starts = _depth_window_starts(
      geometry.size[2], config.synthetic.height
  )
  assignments = _assign_peaks_to_windows(
      peaks, window_starts, config.synthetic.height
  )

  model = train_torch._build_model(config)
  model.load_state_dict(model_state)
  model.to(device)
  model.eval()
  amp_enabled = device.type == 'cuda' and not args.disable_amp
  amp_dtype = (
      torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
  )

  collected = {
      'seed_sample_index': [],
      'seed_amplitude': [],
      'query_window_index': [],
      'query_point_model': [],
      'model_tracks': [],
      'occlusion_logits': [],
      'expected_dist_logits': [],
      'trackability_probability': [],
  }
  cycle_collected = {
      'endpoint_queries': [],
      'endpoint_in_bounds': [],
      'tracks': [],
      'trackability_probability': [],
  }
  endpoint_frames = np.asarray(
      [0, config.synthetic.num_frames - 1], dtype=np.int32
  )
  survey_collected: dict[str, list[np.ndarray]] = {}
  raw_windows = []
  normalized_windows = []
  used_window_starts = []
  window_scales = []
  window_nonfinite = []
  for source_window_index in np.unique(assignments):
    depth_start = int(window_starts[source_window_index])
    selected = np.flatnonzero(assignments == source_window_index)
    selected_peaks = peaks[selected]
    raw_patch = block[
        :, depth_start : depth_start + config.synthetic.height, :
    ]
    normalized, scale, nonfinite_count = (
        infer_zgy_torch._normalize_amplitude(
            raw_patch, args.normalization_percentile
        )
    )
    queries = np.stack(
        [
            np.full(len(selected_peaks), query_frame, dtype=np.float32),
            selected_peaks.astype(np.float32) - depth_start,
            np.full(len(selected_peaks), query_lateral, dtype=np.float32),
        ],
        axis=-1,
    )
    video = np.repeat(normalized[..., None], 3, axis=-1)
    autocast = (
        torch.amp.autocast('cuda', dtype=amp_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
      outputs = model(
          torch.from_numpy(video).unsqueeze(0).to(device),
          torch.from_numpy(queries).unsqueeze(0).to(device),
          is_training=False,
          query_chunk_size=config.query_chunk_size,
      )
    tracks = outputs['tracks'][0].detach().float().cpu().numpy()
    occlusion = outputs['occlusion'][0].detach().float().cpu().numpy()
    expected = outputs['expected_dist'][0].detach().float().cpu().numpy()
    trackability = infer_torch._trackability_probability(
        occlusion, expected
    ).astype(np.float32)
    if args.cycle_consistency:
      endpoint_xy = tracks[:, endpoint_frames, :]
      endpoint_queries = np.empty(
          (len(selected_peaks), len(endpoint_frames), 3), dtype=np.float32
      )
      endpoint_queries[..., 0] = endpoint_frames[None, :]
      endpoint_queries[..., 1] = endpoint_xy[..., 1]
      endpoint_queries[..., 2] = endpoint_xy[..., 0]
      endpoint_in_bounds = (
          np.all(np.isfinite(endpoint_xy), axis=-1)
          & (endpoint_xy[..., 0] >= 0.0)
          & (endpoint_xy[..., 0] <= config.synthetic.width - 1)
          & (endpoint_xy[..., 1] >= 0.0)
          & (endpoint_xy[..., 1] <= config.synthetic.height - 1)
      )
      cycle_tracks = np.full(
          (
              len(selected_peaks),
              len(endpoint_frames),
              config.synthetic.num_frames,
              2,
          ),
          np.nan,
          dtype=np.float32,
      )
      cycle_trackability = np.full(
          (
              len(selected_peaks),
              len(endpoint_frames),
              config.synthetic.num_frames,
          ),
          np.nan,
          dtype=np.float32,
      )
      if np.any(endpoint_in_bounds):
        valid_queries = endpoint_queries.reshape(-1, 3)[
            endpoint_in_bounds.reshape(-1)
        ]
        cycle_autocast = (
            torch.amp.autocast('cuda', dtype=amp_dtype)
            if amp_enabled
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), cycle_autocast:
          cycle_outputs = model(
              torch.from_numpy(video).unsqueeze(0).to(device),
              torch.from_numpy(valid_queries).unsqueeze(0).to(device),
              is_training=False,
              query_chunk_size=config.query_chunk_size,
          )
        valid_tracks = (
            cycle_outputs['tracks'][0].detach().float().cpu().numpy()
        )
        valid_occlusion = (
            cycle_outputs['occlusion'][0].detach().float().cpu().numpy()
        )
        valid_expected = (
            cycle_outputs['expected_dist'][0].detach().float().cpu().numpy()
        )
        valid_trackability = infer_torch._trackability_probability(
            valid_occlusion, valid_expected
        ).astype(np.float32)
        cycle_tracks.reshape(-1, config.synthetic.num_frames, 2)[
            endpoint_in_bounds.reshape(-1)
        ] = valid_tracks
        cycle_trackability.reshape(-1, config.synthetic.num_frames)[
            endpoint_in_bounds.reshape(-1)
        ] = valid_trackability
    window_output_index = len(raw_windows)
    raw_windows.append(raw_patch)
    normalized_windows.append(normalized)
    used_window_starts.append(depth_start)
    window_scales.append(scale)
    window_nonfinite.append(nonfinite_count)
    for local_index, global_index in enumerate(selected):
      peak = int(peaks[global_index])
      survey = infer_zgy_torch._survey_tracks(
          tracks[local_index],
          (base_start[0], base_start[1], depth_start),
          geometry,
          args.sweep,
      )
      collected['seed_sample_index'].append(peak)
      collected['seed_amplitude'].append(float(trace[peak]))
      collected['query_window_index'].append(window_output_index)
      collected['query_point_model'].append(queries[local_index])
      collected['model_tracks'].append(tracks[local_index])
      collected['occlusion_logits'].append(occlusion[local_index])
      collected['expected_dist_logits'].append(expected[local_index])
      collected['trackability_probability'].append(
          trackability[local_index]
      )
      if args.cycle_consistency:
        cycle_collected['endpoint_queries'].append(
            endpoint_queries[local_index]
        )
        cycle_collected['endpoint_in_bounds'].append(
            endpoint_in_bounds[local_index]
        )
        cycle_collected['tracks'].append(cycle_tracks[local_index])
        cycle_collected['trackability_probability'].append(
            cycle_trackability[local_index]
        )
      for key, values in survey.items():
        survey_collected.setdefault(key, []).append(values)

  arrays = {key: np.asarray(values) for key, values in collected.items()}
  survey_arrays = {
      key: np.stack(values) for key, values in survey_collected.items()
  }
  order = np.argsort(arrays['seed_sample_index'])
  arrays = {key: values[order] for key, values in arrays.items()}
  survey_arrays = {key: values[order] for key, values in survey_arrays.items()}
  cycle_arrays = None
  cycle_diagnostics = None
  cycle_summary = None
  if args.cycle_consistency:
    cycle_arrays = {
        key: np.asarray(values)[order]
        for key, values in cycle_collected.items()
    }
    cycle_diagnostics = _cycle_diagnostics(
        forward_tracks=arrays['model_tracks'],
        forward_trackability=arrays['trackability_probability'],
        source_queries=arrays['query_point_model'],
        endpoint_frames=endpoint_frames,
        endpoint_in_bounds=cycle_arrays['endpoint_in_bounds'],
        cycle_tracks=cycle_arrays['tracks'],
        cycle_trackability=cycle_arrays['trackability_probability'],
        visibility_threshold=args.visibility_threshold,
    )
    cycle_summary = _summarize_cycle_diagnostics(
        cycle_diagnostics, cycle_arrays['endpoint_in_bounds']
    )
  peaks = arrays['seed_sample_index'].astype(np.int32)
  seed_amplitude = arrays['seed_amplitude'].astype(np.float32)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  _write_tracks_csv(
      args.output_dir / 'tracks.csv',
      peaks=peaks,
      seed_amplitude=seed_amplitude,
      model_tracks=arrays['model_tracks'],
      survey_tracks=survey_arrays,
      trackability=arrays['trackability_probability'],
      visibility_threshold=args.visibility_threshold,
      sweep=args.sweep,
  )
  if args.cycle_consistency:
    _write_cycle_csv(
        args.output_dir / 'cycle_consistency.csv',
        peaks=peaks,
        seed_amplitude=seed_amplitude,
        endpoint_frames=endpoint_frames,
        endpoint_queries=cycle_arrays['endpoint_queries'],
        endpoint_in_bounds=cycle_arrays['endpoint_in_bounds'],
        diagnostics=cycle_diagnostics,
    )
  cycle_npz = {}
  if args.cycle_consistency:
    cycle_npz = {
        'cycle_endpoint_frames': endpoint_frames,
        'cycle_endpoint_queries': cycle_arrays['endpoint_queries'],
        'cycle_endpoint_in_bounds': cycle_arrays['endpoint_in_bounds'],
        'cycle_model_tracks': cycle_arrays['tracks'],
        'cycle_trackability_probability': cycle_arrays[
            'trackability_probability'
        ],
        **{
            f'cycle_{key}': value
            for key, value in cycle_diagnostics.items()
        },
    }
  np.savez_compressed(
      args.output_dir / 'predictions.npz',
      source_trace=trace,
      raw_windows=np.stack(raw_windows),
      normalized_windows=np.stack(normalized_windows),
      window_depth_start=np.asarray(used_window_starts, dtype=np.int32),
      window_normalization_scale=np.asarray(window_scales, dtype=np.float32),
      **arrays,
      **survey_arrays,
      **cycle_npz,
  )
  _write_curtain(
      args.output_dir / 'track_curtain.png',
      block=block,
      query_frame=query_frame,
      query_lateral=query_lateral,
      peaks=peaks,
      survey_tracks=survey_arrays,
      trackability=arrays['trackability_probability'],
      visibility_threshold=args.visibility_threshold,
      geometry=geometry,
      start=base_start,
      sweep=args.sweep,
  )
  mean_trackability = float(
      np.mean(arrays['trackability_probability'])
  )
  max_lateral_drift = float(
      np.max(
          np.abs(arrays['model_tracks'][..., 0] - query_lateral)
      )
  )
  snapped_annotation = zgy.index_to_annotation(
      geometry, snapped[0], snapped[1], 0.0
  )[:2]
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
      'source_trace': {
          'requested_coordinate_system': args.coordinates,
          'requested_inline_crossline': [
              args.query_inline,
              args.query_crossline,
          ],
          'fractional_index': [inline_index, crossline_index],
          'snapped_index': list(snapped),
          'snapped_annotation': list(snapped_annotation),
          'query_frame_model': query_frame,
          'query_lateral_model': query_lateral,
      },
      'peak_picker': {
          'polarity': args.peak_polarity,
          'relative_threshold': args.peak_relative_threshold,
          'threshold_reference': '99.5 percentile of absolute source trace',
          'trace_scale': trace_scale,
          'minimum_distance_samples': args.peak_min_distance,
          'max_peaks': args.max_peaks,
          'selected_peak_count': int(len(peaks)),
          'selected_sample_indices': peaks.tolist(),
      },
      'model_windows': {
          'overlap': '50 percent except end alignment',
          'used_window_count': len(raw_windows),
          'depth_starts': used_window_starts,
          'normalization_percentile': args.normalization_percentile,
          'normalization_scales': window_scales,
          'nonfinite_replaced_with_zero': window_nonfinite,
      },
      'diagnostics_not_accuracy_metrics': {
          'mean_trackability_probability': mean_trackability,
          'max_lateral_drift_traces': max_lateral_drift,
      },
      'cycle_consistency': {
          'enabled': args.cycle_consistency,
          'endpoint_definition': 'first and last frame of the forward track',
          'confidence_rule': (
              'forward endpoint and returned source trackability both exceed '
              f'{args.visibility_threshold}'
          ),
          'metrics': cycle_summary,
      },
      'visibility_threshold': args.visibility_threshold,
      'artifacts': {
          'tracks': 'tracks.csv',
          'predictions': 'predictions.npz',
          'visualization': 'track_curtain.png',
          **(
              {'cycle_consistency': 'cycle_consistency.csv'}
              if args.cycle_consistency
              else {}
          ),
      },
  }
  with (args.output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write('\n')
  print(f'device={device}')
  print(f'zgy_size={geometry.size}')
  print(f'source_trace_index={snapped}')
  print(f'selected_peaks={len(peaks)}')
  print(f'model_windows={len(raw_windows)}')
  if args.cycle_consistency:
    print(
        'confidence_valid_cycles='
        f"{cycle_summary['confidence_valid_cycle_count']}/"
        f"{cycle_summary['attempted_cycle_count']}"
    )
  print(f'inference_output={args.output_dir}')


if __name__ == '__main__':
  main()
