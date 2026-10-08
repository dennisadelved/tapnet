"""Run aligned multi-stride peak-seeded TAPIR inference on a real ZGY cube."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tapnet.seismic import infer_torch
from tapnet.seismic import infer_zgy_peaks_torch as peaks_infer
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
      '--config', choices=torch_config.CONFIG_VARIANTS,
      default='vdi-multistride'
  )
  parser.add_argument(
      '--frames-per-view', type=int, default=None,
      help='Model frames in each uniformly sampled view.',
  )
  parser.add_argument(
      '--frame-strides', type=int, nargs='+', default=None,
      help='Positive survey-line strides. Defaults to the config values.',
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
  parser.add_argument('--peak-relative-threshold', type=float, default=0.1)
  parser.add_argument('--peak-min-distance', type=int, default=4)
  parser.add_argument('--max-peaks', type=int, default=None)
  parser.add_argument('--cycle-consistency', action='store_true')
  parser.add_argument(
      '--agreement-depth-tolerance', type=float, default=2.0,
      help='Maximum depth spread in samples for a multi-stride agreement flag.',
  )
  parser.add_argument(
      '--agreement-lateral-tolerance', type=float, default=2.0,
      help='Maximum lateral spread in traces for a multi-stride agreement flag.',
  )
  parser.add_argument(
      '--agreement-min-trackability', type=float, default=None,
      help='Minimum probability at every stride; defaults to visibility threshold.',
  )
  parser.add_argument(
      '--rebase-agreed', action='store_true',
      help='Run one new multi-stride view from each outer continuous-agreement anchor.',
  )
  parser.add_argument(
      '--rebase-min-distance', type=int, default=1,
      help='Minimum survey-index distance from the original source for rebasing.',
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
  if args.frames_per_view is not None and args.frames_per_view < 2:
    parser.error('--frames-per-view must be at least 2 when supplied.')
  if args.frame_strides is not None:
    if len(args.frame_strides) < 2:
      parser.error('--frame-strides requires at least two strides.')
    if any(stride < 1 for stride in args.frame_strides):
      parser.error('--frame-strides must contain positive integers.')
    if len(set(args.frame_strides)) != len(args.frame_strides):
      parser.error('--frame-strides must not contain duplicates.')
  if not 0.0 <= args.peak_relative_threshold <= 1.0:
    parser.error('--peak-relative-threshold must be in [0, 1].')
  if args.peak_min_distance < 1:
    parser.error('--peak-min-distance must be positive.')
  if args.max_peaks is not None and args.max_peaks < 1:
    parser.error('--max-peaks must be positive when supplied.')
  if not 0.0 < args.visibility_threshold < 1.0:
    parser.error('--visibility-threshold must be between 0 and 1.')
  if args.agreement_depth_tolerance < 0.0:
    parser.error('--agreement-depth-tolerance must be non-negative.')
  if args.agreement_lateral_tolerance < 0.0:
    parser.error('--agreement-lateral-tolerance must be non-negative.')
  if (
      args.agreement_min_trackability is not None
      and not 0.0 < args.agreement_min_trackability < 1.0
  ):
    parser.error('--agreement-min-trackability must be between 0 and 1.')
  if args.rebase_min_distance < 1:
    parser.error('--rebase-min-distance must be positive.')
  if not 0.0 < args.normalization_percentile <= 100.0:
    parser.error('--normalization-percentile must be in (0, 100].')
  return args


def _view_indices(
    center: int, frame_count: int, stride: int, limit: int
) -> tuple[np.ndarray, int]:
  """Returns an in-bounds uniform view containing ``center`` exactly."""
  if frame_count < 2 or stride < 1:
    raise ValueError('frame_count must be >= 2 and stride must be positive.')
  if center < 0 or center >= limit:
    raise ValueError(f'Center {center} is outside axis length {limit}.')
  required_span = (frame_count - 1) * stride + 1
  if required_span > limit:
    raise ValueError(
        f'{frame_count} frames at stride {stride} require {required_span} '
        f'lines, exceeding the sweep-axis length {limit}.'
    )
  minimum_query_frame = max(
      0, frame_count - 1 - (limit - 1 - center) // stride
  )
  maximum_query_frame = min(frame_count - 1, center // stride)
  query_frame = min(
      max(frame_count // 2, minimum_query_frame), maximum_query_frame
  )
  indices = center + (
      np.arange(frame_count, dtype=np.int32) - query_frame
  ) * stride
  if indices[query_frame] != center:
    raise RuntimeError('Multi-stride source-trace alignment failed.')
  return indices, int(query_frame)


def _read_multistride_block(
    reader,
    geometry: zgy.ZgyGeometry,
    inline_index: float,
    crossline_index: float,
    *,
    frame_count: int,
    frame_strides: tuple[int, ...],
    width: int,
    sweep: str,
) -> tuple[
    np.ndarray,
    tuple[int, int, int],
    int,
    tuple[int, int],
    dict[int, np.ndarray],
    dict[int, int],
]:
  """Reads the minimal dense block containing every aligned stride view."""
  snapped_inline = int(round(inline_index))
  snapped_crossline = int(round(crossline_index))
  if sweep == 'inline':
    center = snapped_inline
    limit = geometry.size[0]
    lateral_center = snapped_crossline
    lateral_limit = geometry.size[1]
  elif sweep == 'crossline':
    center = snapped_crossline
    limit = geometry.size[1]
    lateral_center = snapped_inline
    lateral_limit = geometry.size[0]
  else:
    raise ValueError(f'Unsupported sweep: {sweep!r}')

  views = {}
  query_frames = {}
  for stride in frame_strides:
    views[stride], query_frames[stride] = _view_indices(
        center, frame_count, stride, limit
    )
  dense_start = min(int(indices[0]) for indices in views.values())
  dense_stop = max(int(indices[-1]) for indices in views.values()) + 1
  lateral_start = zgy.centered_window_start(
      lateral_center, width, lateral_limit
  )
  query_lateral = lateral_center - lateral_start
  if sweep == 'inline':
    start = (dense_start, lateral_start, 0)
    raw = zgy.read_window(
        reader,
        start,
        (dense_stop - dense_start, width, geometry.size[2]),
    )
    block = np.transpose(raw, (0, 2, 1))
  else:
    start = (lateral_start, dense_start, 0)
    raw = zgy.read_window(
        reader,
        start,
        (width, dense_stop - dense_start, geometry.size[2]),
    )
    block = np.transpose(raw, (1, 2, 0))
  local_views = {
      stride: indices - dense_start for stride, indices in views.items()
  }
  return (
      block,
      start,
      query_lateral,
      (snapped_inline, snapped_crossline),
      local_views,
      query_frames,
  )


def _physical_sweep_indices(
    local_indices: np.ndarray, start: tuple[int, int, int], sweep: str
) -> np.ndarray:
  axis_start = start[0] if sweep == 'inline' else start[1]
  return axis_start + local_indices


def _survey_tracks_at_indices(
    tracks: np.ndarray,
    sweep_indices: np.ndarray,
    start: tuple[int, int, int],
    depth_start: int,
    geometry: zgy.ZgyGeometry,
    sweep: str,
) -> dict[str, np.ndarray]:
  """Maps strided model tracks to survey indices and coordinates."""
  sweep_indices = np.asarray(sweep_indices, dtype=np.float32)
  if sweep == 'inline':
    inline_index = sweep_indices
    crossline_index = start[1] + tracks[:, 0]
  elif sweep == 'crossline':
    inline_index = start[0] + tracks[:, 0]
    crossline_index = sweep_indices
  else:
    raise ValueError(f'Unsupported sweep: {sweep!r}')
  z_index = depth_start + tracks[:, 1]
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


def _run_model(
    model: torch.nn.Module,
    video: np.ndarray,
    queries: np.ndarray,
    *,
    device: torch.device,
    query_chunk_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        query_chunk_size=query_chunk_size,
    )
  tracks = outputs['tracks'][0].detach().float().cpu().numpy()
  occlusion = outputs['occlusion'][0].detach().float().cpu().numpy()
  expected = outputs['expected_dist'][0].detach().float().cpu().numpy()
  trackability = infer_torch._trackability_probability(
      occlusion, expected
  ).astype(np.float32)
  return tracks, occlusion, expected, trackability


def _write_stride_tracks_csv(
    path: Path,
    *,
    stride: int,
    sweep_indices: np.ndarray,
    peaks: np.ndarray,
    seed_amplitude: np.ndarray,
    model_tracks: np.ndarray,
    survey_tracks: dict[str, np.ndarray],
    trackability: np.ndarray,
    visibility_threshold: float,
    sweep: str,
) -> None:
  fields = [
      'temporal_stride', 'seed_id', 'seed_sample_index', 'seed_amplitude',
      'seed_polarity', 'model_frame', 'sweep_index', 'sweep', 'inline_index',
      'crossline_index', 'z_index', 'inline_annotation',
      'crossline_annotation', 'z_coordinate', 'world_x', 'world_y',
      'model_lateral', 'model_depth', 'trackability_probability',
      'predicted_visible',
  ]
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for seed_id, peak in enumerate(peaks):
      polarity = 'positive' if seed_amplitude[seed_id] > 0 else 'negative'
      for frame, sweep_index in enumerate(sweep_indices):
        writer.writerow({
            'temporal_stride': stride,
            'seed_id': seed_id,
            'seed_sample_index': int(peak),
            'seed_amplitude': float(seed_amplitude[seed_id]),
            'seed_polarity': polarity,
            'model_frame': frame,
            'sweep_index': int(sweep_index),
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


def _cross_scale_disagreement(
    results: dict[int, dict[str, object]],
    peaks: np.ndarray,
    seed_amplitude: np.ndarray,
    *,
    sweep: str,
    visibility_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
  """Compares predictions only at physical lines shared by two views."""
  rows = []
  summary = {}
  for stride_a, stride_b in itertools.combinations(sorted(results), 2):
    first = results[stride_a]
    second = results[stride_b]
    common, first_frames, second_frames = np.intersect1d(
        first['sweep_indices'],
        second['sweep_indices'],
        assume_unique=True,
        return_indices=True,
    )
    pair_depth = []
    pair_lateral = []
    pair_confidence = []
    pair_both_visible = []
    for seed_id, peak in enumerate(peaks):
      polarity = 'positive' if seed_amplitude[seed_id] > 0 else 'negative'
      for sweep_index, frame_a, frame_b in zip(
          common, first_frames, second_frames
      ):
        track_a = first['model_tracks'][seed_id, frame_a]
        track_b = second['model_tracks'][seed_id, frame_b]
        confidence_a = float(first['trackability'][seed_id, frame_a])
        confidence_b = float(second['trackability'][seed_id, frame_b])
        depth_disagreement = float(abs(track_a[1] - track_b[1]))
        lateral_disagreement = float(abs(track_a[0] - track_b[0]))
        confidence_disagreement = float(abs(confidence_a - confidence_b))
        both_visible = (
            confidence_a > visibility_threshold
            and confidence_b > visibility_threshold
        )
        if sweep == 'inline':
          annotation = (
              first['geometry'].annotstart[0]
              + sweep_index * first['geometry'].annotinc[0]
          )
        else:
          annotation = (
              first['geometry'].annotstart[1]
              + sweep_index * first['geometry'].annotinc[1]
          )
        rows.append({
            'stride_a': stride_a,
            'stride_b': stride_b,
            'seed_id': seed_id,
            'seed_sample_index': int(peak),
            'seed_amplitude': float(seed_amplitude[seed_id]),
            'seed_polarity': polarity,
            'sweep_index': int(sweep_index),
            'sweep_annotation': float(annotation),
            'depth_a': float(track_a[1]),
            'depth_b': float(track_b[1]),
            'absolute_depth_disagreement_samples': depth_disagreement,
            'lateral_a': float(track_a[0]),
            'lateral_b': float(track_b[0]),
            'absolute_lateral_disagreement_traces': lateral_disagreement,
            'trackability_a': confidence_a,
            'trackability_b': confidence_b,
            'absolute_trackability_disagreement': confidence_disagreement,
            'both_predicted_visible': both_visible,
        })
        pair_depth.append(depth_disagreement)
        pair_lateral.append(lateral_disagreement)
        pair_confidence.append(confidence_disagreement)
        pair_both_visible.append(both_visible)
    mask = np.ones(len(pair_depth), dtype=bool)
    visible_mask = np.asarray(pair_both_visible, dtype=bool)
    key = f'{stride_a}_vs_{stride_b}'
    summary[key] = {
        'shared_physical_line_count': int(len(common)),
        'comparison_count': int(len(pair_depth)),
        'both_predicted_visible_count': int(np.sum(visible_mask)),
        'all': {
            'absolute_depth_disagreement_samples': (
                peaks_infer._metric_distribution(np.asarray(pair_depth), mask)
            ),
            'absolute_lateral_disagreement_traces': (
                peaks_infer._metric_distribution(np.asarray(pair_lateral), mask)
            ),
            'absolute_trackability_disagreement': (
                peaks_infer._metric_distribution(
                    np.asarray(pair_confidence), mask
                )
            ),
        },
        'both_predicted_visible': {
            'absolute_depth_disagreement_samples': (
                peaks_infer._metric_distribution(
                    np.asarray(pair_depth), visible_mask
                )
            ),
            'absolute_lateral_disagreement_traces': (
                peaks_infer._metric_distribution(
                    np.asarray(pair_lateral), visible_mask
                )
            ),
        },
    }
  return rows, summary


def _write_disagreement_csv(path: Path, rows: list[dict[str, object]]) -> None:
  if not rows:
    raise ValueError('No shared physical lines were available for comparison.')
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _multi_stride_agreement(
    results: dict[int, dict[str, object]],
    peaks: np.ndarray,
    seed_amplitude: np.ndarray,
    *,
    source_sweep_index: int,
    sweep: str,
    depth_tolerance: float,
    lateral_tolerance: float,
    minimum_trackability: float,
    minimum_rebase_distance: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
  """Flags all-stride agreement and selects continuous outer anchors."""
  strides = sorted(results)
  common = np.asarray(results[strides[0]]['sweep_indices'])
  for stride in strides[1:]:
    common = np.intersect1d(common, results[stride]['sweep_indices'])
  frame_by_stride = {
      stride: {
          int(sweep_index): frame
          for frame, sweep_index in enumerate(results[stride]['sweep_indices'])
      }
      for stride in strides
  }
  rows = []
  rows_by_seed = {seed_id: [] for seed_id in range(len(peaks))}
  for seed_id, peak in enumerate(peaks):
    polarity = 'positive' if seed_amplitude[seed_id] > 0 else 'negative'
    for sweep_index_value in common:
      sweep_index = int(sweep_index_value)
      frames = {
          stride: frame_by_stride[stride][sweep_index]
          for stride in strides
      }
      depths = np.asarray([
          results[stride]['survey_tracks']['z_index'][
              seed_id, frames[stride]
          ]
          for stride in strides
      ], dtype=np.float32)
      lateral_key = 'crossline_index' if sweep == 'inline' else 'inline_index'
      laterals = np.asarray([
          results[stride]['survey_tracks'][lateral_key][
              seed_id, frames[stride]
          ]
          for stride in strides
      ], dtype=np.float32)
      trackability = np.asarray([
          results[stride]['trackability'][seed_id, frames[stride]]
          for stride in strides
      ], dtype=np.float32)
      finite = bool(
          np.all(np.isfinite(depths))
          and np.all(np.isfinite(laterals))
          and np.all(np.isfinite(trackability))
      )
      depth_spread = float(np.ptp(depths)) if finite else float('nan')
      lateral_spread = float(np.ptp(laterals)) if finite else float('nan')
      min_trackability = float(np.min(trackability)) if finite else float('nan')
      agrees = bool(
          finite
          and depth_spread <= depth_tolerance
          and lateral_spread <= lateral_tolerance
          and min_trackability >= minimum_trackability
      )
      geometry = results[strides[0]]['geometry']
      annotation_axis = 0 if sweep == 'inline' else 1
      row = {
          'seed_id': seed_id,
          'seed_sample_index': int(peak),
          'seed_amplitude': float(seed_amplitude[seed_id]),
          'seed_polarity': polarity,
          'sweep_index': sweep_index,
          'sweep_annotation': float(
              geometry.annotstart[annotation_axis]
              + sweep_index * geometry.annotinc[annotation_axis]
          ),
          'distance_from_source_indices': abs(
              sweep_index - source_sweep_index
          ),
          'consensus_z_index': float(np.median(depths)),
          'consensus_lateral_index': float(np.median(laterals)),
          'depth_spread_samples': depth_spread,
          'lateral_spread_traces': lateral_spread,
          'minimum_trackability_probability': min_trackability,
          'multi_resolution_agree': agrees,
          'continuous_agreement_from_source': False,
          'selected_rebase_anchor': False,
          'rebase_direction': '',
      }
      rows.append(row)
      rows_by_seed[seed_id].append(row)

  anchors = []
  for seed_rows in rows_by_seed.values():
    source_rows = [
        row for row in seed_rows if row['sweep_index'] == source_sweep_index
    ]
    if not source_rows or not source_rows[0]['multi_resolution_agree']:
      continue
    source_rows[0]['continuous_agreement_from_source'] = True
    for direction, label in ((-1, 'lower_index'), (1, 'higher_index')):
      candidates = sorted(
          (
              row for row in seed_rows
              if direction * (row['sweep_index'] - source_sweep_index) > 0
          ),
          key=lambda row: row['distance_from_source_indices'],
      )
      outermost = None
      for row in candidates:
        if not row['multi_resolution_agree']:
          break
        row['continuous_agreement_from_source'] = True
        if row['distance_from_source_indices'] >= minimum_rebase_distance:
          outermost = row
      if outermost is not None:
        outermost['selected_rebase_anchor'] = True
        outermost['rebase_direction'] = label
        anchors.append(dict(outermost))

  agreed = sum(bool(row['multi_resolution_agree']) for row in rows)
  continuous = sum(
      bool(row['continuous_agreement_from_source']) for row in rows
  )
  summary = {
      'shared_physical_line_count': int(len(common)),
      'comparison_count': len(rows),
      'agreement_count': agreed,
      'continuous_agreement_count': continuous,
      'selected_anchor_count': len(anchors),
      'seed_count_with_anchor': len({row['seed_id'] for row in anchors}),
      'depth_tolerance_samples': depth_tolerance,
      'lateral_tolerance_traces': lateral_tolerance,
      'minimum_trackability_probability': minimum_trackability,
      'minimum_rebase_distance_indices': minimum_rebase_distance,
  }
  return rows, anchors, summary


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
  if not rows:
    return
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _run_rebased_views(
    reader,
    anchors: list[dict[str, object]],
    *,
    geometry: zgy.ZgyGeometry,
    frame_count: int,
    frame_strides: tuple[int, ...],
    width: int,
    height: int,
    sweep: str,
    model: torch.nn.Module,
    device: torch.device,
    query_chunk_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    normalization_percentile: float,
    visibility_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
  """Runs one centered multi-stride hop from continuous-agreement anchors."""
  if not anchors:
    return [], {'anchor_count': 0, 'view_count': 0, 'track_row_count': 0}
  depth_starts = peaks_infer._depth_window_starts(geometry.size[2], height)
  rows = []
  view_summaries = []
  anchors_by_line = {}
  for anchor in anchors:
    anchors_by_line.setdefault(int(anchor['sweep_index']), []).append(anchor)

  for anchor_sweep_index, line_anchors in sorted(anchors_by_line.items()):
    lateral_center = float(np.median([
        anchor['consensus_lateral_index'] for anchor in line_anchors
    ]))
    if sweep == 'inline':
      inline_index = anchor_sweep_index
      crossline_index = lateral_center
    else:
      inline_index = lateral_center
      crossline_index = anchor_sweep_index
    (
        block,
        base_start,
        _query_lateral,
        _snapped,
        local_views,
        query_frames,
    ) = _read_multistride_block(
        reader,
        geometry,
        inline_index,
        crossline_index,
        frame_count=frame_count,
        frame_strides=frame_strides,
        width=width,
        sweep=sweep,
    )
    lateral_start = base_start[1] if sweep == 'inline' else base_start[0]
    anchor_depths = np.rint([
        anchor['consensus_z_index'] for anchor in line_anchors
    ]).astype(np.int32)
    assignments = peaks_infer._assign_peaks_to_windows(
        anchor_depths, depth_starts, height
    )
    sweep_indices = {
        stride: _physical_sweep_indices(indices, base_start, sweep)
        for stride, indices in local_views.items()
    }
    view_summaries.append({
        'anchor_sweep_index': anchor_sweep_index,
        'anchor_count': len(line_anchors),
        'lateral_center_index': lateral_center,
        'sweep_index_range_by_stride': {
            str(stride): [
                int(sweep_indices[stride][0]),
                int(sweep_indices[stride][-1]),
            ]
            for stride in frame_strides
        },
    })

    for window_index in np.unique(assignments):
      selected = np.flatnonzero(assignments == window_index)
      depth_start = int(depth_starts[window_index])
      dense_patch = block[:, depth_start : depth_start + height, :]
      dense_normalized, _scale, _nonfinite = (
          infer_zgy_torch._normalize_amplitude(
              dense_patch, normalization_percentile
          )
      )
      for stride in frame_strides:
        selected_anchors = [line_anchors[index] for index in selected]
        queries = np.asarray([
            [
                query_frames[stride],
                float(anchor['consensus_z_index']) - depth_start,
                float(anchor['consensus_lateral_index']) - lateral_start,
            ]
            for anchor in selected_anchors
        ], dtype=np.float32)
        if np.any(queries[:, 2] < 0.0) or np.any(queries[:, 2] > width - 1):
          raise ValueError(
              'Agreed anchors at one sweep line exceed the lateral model '
              'window; reduce drift before rebasing.'
          )
        video = np.repeat(
            dense_normalized[local_views[stride]][..., None], 3, axis=-1
        )
        tracks, _occlusion, _expected, trackability = _run_model(
            model,
            video,
            queries,
            device=device,
            query_chunk_size=query_chunk_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for local_index, anchor in enumerate(selected_anchors):
          survey_tracks = _survey_tracks_at_indices(
              tracks[local_index],
              sweep_indices[stride],
              base_start,
              depth_start,
              geometry,
              sweep,
          )
          for frame, sweep_index in enumerate(sweep_indices[stride]):
            rows.append({
                'seed_id': int(anchor['seed_id']),
                'seed_sample_index': int(anchor['seed_sample_index']),
                'rebase_direction': anchor['rebase_direction'],
                'anchor_sweep_index': anchor_sweep_index,
                'anchor_consensus_z_index': float(
                    anchor['consensus_z_index']
                ),
                'anchor_consensus_lateral_index': float(
                    anchor['consensus_lateral_index']
                ),
                'temporal_stride': stride,
                'model_frame': frame,
                'sweep_index': int(sweep_index),
                'inline_index': float(
                    survey_tracks['inline_index'][frame]
                ),
                'crossline_index': float(
                    survey_tracks['crossline_index'][frame]
                ),
                'z_index': float(survey_tracks['z_index'][frame]),
                'inline_annotation': float(
                    survey_tracks['inline_annotation'][frame]
                ),
                'crossline_annotation': float(
                    survey_tracks['crossline_annotation'][frame]
                ),
                'z_coordinate': float(
                    survey_tracks['z_coordinate'][frame]
                ),
                'world_x': float(survey_tracks['world_x'][frame]),
                'world_y': float(survey_tracks['world_y'][frame]),
                'model_lateral': float(tracks[local_index, frame, 0]),
                'model_depth': float(tracks[local_index, frame, 1]),
                'trackability_probability': float(
                    trackability[local_index, frame]
                ),
                'predicted_visible': bool(
                    trackability[local_index, frame]
                    > visibility_threshold
                ),
            })
  summary = {
      'anchor_count': len(anchors),
      'view_count': len(view_summaries),
      'track_row_count': len(rows),
      'views': view_summaries,
  }
  return rows, summary


def _write_multistride_curtain(
    path: Path,
    *,
    block: np.ndarray,
    start: tuple[int, int, int],
    query_lateral: int,
    peaks: np.ndarray,
    results: dict[int, dict[str, object]],
    geometry: zgy.ZgyGeometry,
    sweep: str,
    agreement_rows: list[dict[str, object]] | None = None,
) -> None:
  curtain = block[:, :, query_lateral].T
  dense_indices = _physical_sweep_indices(
      np.arange(block.shape[0], dtype=np.int32), start, sweep
  )
  if sweep == 'inline':
    horizontal = (
        geometry.annotstart[0] + dense_indices * geometry.annotinc[0]
    )
    horizontal_label = 'inline annotation'
  else:
    horizontal = (
        geometry.annotstart[1] + dense_indices * geometry.annotinc[1]
    )
    horizontal_label = 'crossline annotation'
  vertical = geometry.zstart + np.arange(
      block.shape[1], dtype=np.float32
  ) * geometry.zinc
  h_step = float(horizontal[1] - horizontal[0])
  scale = max(float(np.percentile(np.abs(curtain), 99.0)), 1e-6)
  figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
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
  normalizer = matplotlib.colors.Normalize(
      vmin=float(peaks[0]), vmax=float(max(peaks[-1], peaks[0] + 1))
  )
  line_styles = ('-', '--', ':', '-.')
  for style_index, stride in enumerate(sorted(results)):
    result = results[stride]
    sweep_key = 'inline_annotation' if sweep == 'inline' else (
        'crossline_annotation'
    )
    x_values = result['survey_tracks'][sweep_key]
    z_values = result['survey_tracks']['z_coordinate']
    for seed_id, peak in enumerate(peaks):
      axis.plot(
          x_values[seed_id],
          z_values[seed_id],
          color=color_map(normalizer(float(peak))),
          linestyle=line_styles[style_index % len(line_styles)],
          linewidth=1.1,
          alpha=0.8,
      )
  legend_handles = [
      plt.Line2D(
          [0], [0], color='black',
          linestyle=line_styles[index % len(line_styles)],
          label=f'stride {stride}',
      )
      for index, stride in enumerate(sorted(results))
  ]
  agreement_rows = agreement_rows or []
  agreed = [row for row in agreement_rows if row['multi_resolution_agree']]
  anchors = [row for row in agreement_rows if row['selected_rebase_anchor']]
  if agreed:
    axis.scatter(
        [row['sweep_annotation'] for row in agreed],
        [
            geometry.zstart + row['consensus_z_index'] * geometry.zinc
            for row in agreed
        ],
        s=8,
        c='#39ff88',
        marker='o',
        linewidths=0,
        alpha=0.8,
        zorder=4,
    )
    legend_handles.append(plt.Line2D(
        [0], [0], color='#39ff88', marker='o', linestyle='none',
        label='multi-stride agreement',
    ))
  if anchors:
    axis.scatter(
        [row['sweep_annotation'] for row in anchors],
        [
            geometry.zstart + row['consensus_z_index'] * geometry.zinc
            for row in anchors
        ],
        s=42,
        c='#ffd43b',
        edgecolors='black',
        marker='*',
        linewidths=0.5,
        zorder=5,
    )
    legend_handles.append(plt.Line2D(
        [0], [0], color='#ffd43b', marker='*', markeredgecolor='black',
        linestyle='none', label='selected rebase anchor',
    ))
  axis.legend(handles=legend_handles, loc='upper right')
  axis.set_title(f'{len(peaks)} peak-seeded tracks; aligned multi-stride views')
  axis.set_xlabel(horizontal_label)
  axis.set_ylabel(f'Z ({geometry.zunitname or "header units"})')
  figure.savefig(path, dpi=160)
  plt.close(figure)


def main() -> None:
  args = _parse_args()
  config = torch_config.get_config(args.config)
  frame_count = args.frames_per_view or config.synthetic.num_frames
  frame_strides = tuple(
      args.frame_strides or config.synthetic.frame_strides
  )
  if len(frame_strides) < 2:
    raise ValueError(
        'Multi-stride inference requires at least two configured strides.'
    )
  if args.frames_per_view is not None:
    config = dataclasses.replace(
        config,
        synthetic=dataclasses.replace(
            config.synthetic, num_frames=args.frames_per_view
        ),
    )
  config.validate()
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
          args.query_inline, args.query_crossline
      )
    zgy.validate_index((inline_index, crossline_index, 0.0), geometry)
    (
        block,
        base_start,
        query_lateral,
        snapped,
        local_views,
        query_frames,
    ) = _read_multistride_block(
        reader,
        geometry,
        inline_index,
        crossline_index,
        frame_count=frame_count,
        frame_strides=frame_strides,
        width=config.synthetic.width,
        sweep=args.sweep,
    )

  source_local_index = (
      snapped[0] - base_start[0]
      if args.sweep == 'inline'
      else snapped[1] - base_start[1]
  )
  trace = block[source_local_index, :, query_lateral]
  peaks, trace_scale = peaks_infer._pick_trace_peaks(
      trace,
      polarity=args.peak_polarity,
      relative_threshold=args.peak_relative_threshold,
      min_distance=args.peak_min_distance,
      max_peaks=args.max_peaks,
  )
  window_starts = peaks_infer._depth_window_starts(
      geometry.size[2], config.synthetic.height
  )
  assignments = peaks_infer._assign_peaks_to_windows(
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
  sweep_indices = {
      stride: _physical_sweep_indices(indices, base_start, args.sweep)
      for stride, indices in local_views.items()
  }
  collected = {
      stride: {
          'seed_sample_index': [],
          'seed_amplitude': [],
          'query_window_index': [],
          'query_point_model': [],
          'model_tracks': [],
          'occlusion_logits': [],
          'expected_dist_logits': [],
          'trackability_probability': [],
      }
      for stride in frame_strides
  }
  survey_collected = {stride: {} for stride in frame_strides}
  cycle_collected = {
      stride: {
          'endpoint_queries': [],
          'endpoint_in_bounds': [],
          'tracks': [],
          'trackability_probability': [],
      }
      for stride in frame_strides
  }
  endpoint_frames = np.asarray([0, frame_count - 1], dtype=np.int32)
  raw_windows = []
  normalized_windows = []
  used_window_starts = []
  window_scales = []
  window_nonfinite = []

  for source_window_index in np.unique(assignments):
    depth_start = int(window_starts[source_window_index])
    selected = np.flatnonzero(assignments == source_window_index)
    selected_peaks = peaks[selected]
    dense_raw_patch = block[
        :, depth_start : depth_start + config.synthetic.height, :
    ]
    dense_normalized, scale, nonfinite_count = (
        infer_zgy_torch._normalize_amplitude(
            dense_raw_patch, args.normalization_percentile
        )
    )
    window_output_index = len(raw_windows)
    raw_windows.append(dense_raw_patch)
    normalized_windows.append(dense_normalized)
    used_window_starts.append(depth_start)
    window_scales.append(scale)
    window_nonfinite.append(nonfinite_count)

    for stride in frame_strides:
      normalized = dense_normalized[local_views[stride]]
      queries = np.stack(
          [
              np.full(
                  len(selected_peaks), query_frames[stride], dtype=np.float32
              ),
              selected_peaks.astype(np.float32) - depth_start,
              np.full(
                  len(selected_peaks), query_lateral, dtype=np.float32
              ),
          ],
          axis=-1,
      )
      video = np.repeat(normalized[..., None], 3, axis=-1)
      tracks, occlusion, expected, trackability = _run_model(
          model,
          video,
          queries,
          device=device,
          query_chunk_size=config.query_chunk_size,
          amp_enabled=amp_enabled,
          amp_dtype=amp_dtype,
      )
      endpoint_queries = None
      endpoint_in_bounds = None
      cycle_tracks = None
      cycle_trackability = None
      if args.cycle_consistency:
        endpoint_xy = tracks[:, endpoint_frames]
        endpoint_queries = np.empty(
            (len(selected_peaks), 2, 3), dtype=np.float32
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
            (len(selected_peaks), 2, frame_count, 2),
            np.nan,
            dtype=np.float32,
        )
        cycle_trackability = np.full(
            (len(selected_peaks), 2, frame_count),
            np.nan,
            dtype=np.float32,
        )
        if np.any(endpoint_in_bounds):
          valid_queries = endpoint_queries.reshape(-1, 3)[
              endpoint_in_bounds.reshape(-1)
          ]
          (
              valid_tracks,
              valid_occlusion,
              valid_expected,
              valid_trackability,
          ) = _run_model(
              model,
              video,
              valid_queries,
              device=device,
              query_chunk_size=config.query_chunk_size,
              amp_enabled=amp_enabled,
              amp_dtype=amp_dtype,
          )
          del valid_occlusion, valid_expected
          cycle_tracks.reshape(-1, frame_count, 2)[
              endpoint_in_bounds.reshape(-1)
          ] = valid_tracks
          cycle_trackability.reshape(-1, frame_count)[
              endpoint_in_bounds.reshape(-1)
          ] = valid_trackability

      for local_index, global_index in enumerate(selected):
        peak = int(peaks[global_index])
        survey = _survey_tracks_at_indices(
            tracks[local_index],
            sweep_indices[stride],
            base_start,
            depth_start,
            geometry,
            args.sweep,
        )
        values = collected[stride]
        values['seed_sample_index'].append(peak)
        values['seed_amplitude'].append(float(trace[peak]))
        values['query_window_index'].append(window_output_index)
        values['query_point_model'].append(queries[local_index])
        values['model_tracks'].append(tracks[local_index])
        values['occlusion_logits'].append(occlusion[local_index])
        values['expected_dist_logits'].append(expected[local_index])
        values['trackability_probability'].append(trackability[local_index])
        for key, survey_values in survey.items():
          survey_collected[stride].setdefault(key, []).append(survey_values)
        if args.cycle_consistency:
          cycle_values = cycle_collected[stride]
          cycle_values['endpoint_queries'].append(
              endpoint_queries[local_index]
          )
          cycle_values['endpoint_in_bounds'].append(
              endpoint_in_bounds[local_index]
          )
          cycle_values['tracks'].append(cycle_tracks[local_index])
          cycle_values['trackability_probability'].append(
              cycle_trackability[local_index]
          )

  results = {}
  cycle_summaries = {}
  for stride in frame_strides:
    arrays = {
        key: np.asarray(values) for key, values in collected[stride].items()
    }
    survey_arrays = {
        key: np.stack(values)
        for key, values in survey_collected[stride].items()
    }
    order = np.argsort(arrays['seed_sample_index'])
    arrays = {key: values[order] for key, values in arrays.items()}
    survey_arrays = {
        key: values[order] for key, values in survey_arrays.items()
    }
    result = {
        'geometry': geometry,
        'sweep_indices': sweep_indices[stride],
        'model_tracks': arrays['model_tracks'],
        'trackability': arrays['trackability_probability'],
        'arrays': arrays,
        'survey_tracks': survey_arrays,
    }
    if args.cycle_consistency:
      cycle_arrays = {
          key: np.asarray(values)[order]
          for key, values in cycle_collected[stride].items()
      }
      diagnostics = peaks_infer._cycle_diagnostics(
          forward_tracks=arrays['model_tracks'],
          forward_trackability=arrays['trackability_probability'],
          source_queries=arrays['query_point_model'],
          endpoint_frames=endpoint_frames,
          endpoint_in_bounds=cycle_arrays['endpoint_in_bounds'],
          cycle_tracks=cycle_arrays['tracks'],
          cycle_trackability=cycle_arrays['trackability_probability'],
          visibility_threshold=args.visibility_threshold,
      )
      cycle_summaries[str(stride)] = (
          peaks_infer._summarize_cycle_diagnostics(
              diagnostics, cycle_arrays['endpoint_in_bounds']
          )
      )
      result['cycle_arrays'] = cycle_arrays
      result['cycle_diagnostics'] = diagnostics
    results[stride] = result

  peaks = results[frame_strides[0]]['arrays'][
      'seed_sample_index'
  ].astype(np.int32)
  seed_amplitude = results[frame_strides[0]]['arrays'][
      'seed_amplitude'
  ].astype(np.float32)
  disagreement_rows, disagreement_summary = _cross_scale_disagreement(
      results,
      peaks,
      seed_amplitude,
      sweep=args.sweep,
      visibility_threshold=args.visibility_threshold,
  )
  source_sweep_index = snapped[0] if args.sweep == 'inline' else snapped[1]
  agreement_min_trackability = (
      args.agreement_min_trackability
      if args.agreement_min_trackability is not None
      else args.visibility_threshold
  )
  agreement_rows, rebase_anchors, agreement_summary = (
      _multi_stride_agreement(
          results,
          peaks,
          seed_amplitude,
          source_sweep_index=source_sweep_index,
          sweep=args.sweep,
          depth_tolerance=args.agreement_depth_tolerance,
          lateral_tolerance=args.agreement_lateral_tolerance,
          minimum_trackability=agreement_min_trackability,
          minimum_rebase_distance=args.rebase_min_distance,
      )
  )
  rebase_rows = []
  rebase_summary = {
      'enabled': args.rebase_agreed,
      'anchor_count': len(rebase_anchors),
      'view_count': 0,
      'track_row_count': 0,
  }
  if args.rebase_agreed and rebase_anchors:
    with zgy.open_zgy_reader(args.input) as reader:
      rebase_rows, rebase_run_summary = _run_rebased_views(
          reader,
          rebase_anchors,
          geometry=geometry,
          frame_count=frame_count,
          frame_strides=frame_strides,
          width=config.synthetic.width,
          height=config.synthetic.height,
          sweep=args.sweep,
          model=model,
          device=device,
          query_chunk_size=config.query_chunk_size,
          amp_enabled=amp_enabled,
          amp_dtype=amp_dtype,
          normalization_percentile=args.normalization_percentile,
          visibility_threshold=args.visibility_threshold,
      )
    rebase_summary.update(rebase_run_summary)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  npz_values = {
      'source_trace': trace,
      'dense_raw_windows': np.stack(raw_windows),
      'dense_normalized_windows': np.stack(normalized_windows),
      'window_depth_start': np.asarray(used_window_starts, dtype=np.int32),
      'window_normalization_scale': np.asarray(
          window_scales, dtype=np.float32
      ),
      'window_nonfinite_count': np.asarray(window_nonfinite, dtype=np.int64),
  }
  stride_diagnostics = {}
  for stride, result in results.items():
    arrays = result['arrays']
    survey_arrays = result['survey_tracks']
    _write_stride_tracks_csv(
        args.output_dir / f'tracks_stride{stride}.csv',
        stride=stride,
        sweep_indices=result['sweep_indices'],
        peaks=peaks,
        seed_amplitude=seed_amplitude,
        model_tracks=arrays['model_tracks'],
        survey_tracks=survey_arrays,
        trackability=arrays['trackability_probability'],
        visibility_threshold=args.visibility_threshold,
        sweep=args.sweep,
    )
    for key, values in arrays.items():
      npz_values[f'stride{stride}_{key}'] = values
    for key, values in survey_arrays.items():
      npz_values[f'stride{stride}_{key}'] = values
    npz_values[f'stride{stride}_sweep_indices'] = result['sweep_indices']
    npz_values[f'stride{stride}_query_frame'] = np.asarray(
        query_frames[stride], dtype=np.int32
    )
    if args.cycle_consistency:
      cycle_arrays = result['cycle_arrays']
      diagnostics = result['cycle_diagnostics']
      peaks_infer._write_cycle_csv(
          args.output_dir / f'cycle_consistency_stride{stride}.csv',
          peaks=peaks,
          seed_amplitude=seed_amplitude,
          endpoint_frames=endpoint_frames,
          endpoint_queries=cycle_arrays['endpoint_queries'],
          endpoint_in_bounds=cycle_arrays['endpoint_in_bounds'],
          diagnostics=diagnostics,
      )
      for key, values in cycle_arrays.items():
        npz_values[f'stride{stride}_cycle_{key}'] = values
      for key, values in diagnostics.items():
        npz_values[f'stride{stride}_cycle_diagnostic_{key}'] = values
    stride_diagnostics[str(stride)] = {
        'mean_trackability_probability': float(
            np.mean(arrays['trackability_probability'])
        ),
        'max_lateral_drift_traces': float(
            np.max(
                np.abs(arrays['model_tracks'][..., 0] - query_lateral)
            )
        ),
    }
  np.savez_compressed(args.output_dir / 'predictions.npz', **npz_values)
  _write_disagreement_csv(
      args.output_dir / 'cross_scale_disagreement.csv', disagreement_rows
  )
  _write_rows_csv(
      args.output_dir / 'multi_resolution_agreement.csv', agreement_rows
  )
  _write_rows_csv(args.output_dir / 'rebase_anchors.csv', rebase_anchors)
  _write_rows_csv(args.output_dir / 'rebased_tracks.csv', rebase_rows)
  _write_multistride_curtain(
      args.output_dir / 'track_curtain_multistride.png',
      block=block,
      start=base_start,
      query_lateral=query_lateral,
      peaks=peaks,
      results=results,
      geometry=geometry,
      sweep=args.sweep,
      agreement_rows=agreement_rows,
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
          'frames_per_view': frame_count,
          'frame_strides': list(frame_strides),
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
              args.query_inline, args.query_crossline
          ],
          'fractional_index': [inline_index, crossline_index],
          'snapped_index': list(snapped),
          'snapped_annotation': list(snapped_annotation),
          'query_lateral_model': query_lateral,
          'query_frames_by_stride': {
              str(key): value for key, value in query_frames.items()
          },
      },
      'views': {
          str(stride): {
              'query_frame_model': query_frames[stride],
              'sweep_indices': sweep_indices[stride].tolist(),
              'sweep_annotations': (
                  geometry.annotstart[0]
                  + sweep_indices[stride] * geometry.annotinc[0]
                  if args.sweep == 'inline'
                  else geometry.annotstart[1]
                  + sweep_indices[stride] * geometry.annotinc[1]
              ).tolist(),
          }
          for stride in frame_strides
      },
      'peak_picker': {
          'polarity': args.peak_polarity,
          'relative_threshold': args.peak_relative_threshold,
          'trace_scale': trace_scale,
          'minimum_distance_samples': args.peak_min_distance,
          'max_peaks': args.max_peaks,
          'selected_peak_count': int(len(peaks)),
          'selected_sample_indices': peaks.tolist(),
      },
      'model_windows': {
          'used_window_count': len(raw_windows),
          'depth_starts': used_window_starts,
          'shared_normalization_across_strides': True,
          'normalization_percentile': args.normalization_percentile,
          'normalization_scales': window_scales,
          'nonfinite_replaced_with_zero': window_nonfinite,
      },
      'diagnostics_not_accuracy_metrics': {
          'by_stride': stride_diagnostics,
          'cross_scale_disagreement': disagreement_summary,
          'multi_resolution_agreement': agreement_summary,
      },
      'rebasing': rebase_summary,
      'cycle_consistency': {
          'enabled': args.cycle_consistency,
          'by_stride': cycle_summaries,
      },
      'visibility_threshold': args.visibility_threshold,
      'fusion': {
          'performed': False,
          'reason': (
              'First pass exports raw aligned views and disagreement without '
              'averaging potentially different reflectors across faults.'
          ),
      },
      'artifacts': {
          'tracks_by_stride': {
              str(stride): f'tracks_stride{stride}.csv'
              for stride in frame_strides
          },
          'cross_scale_disagreement': 'cross_scale_disagreement.csv',
          'multi_resolution_agreement': 'multi_resolution_agreement.csv',
          **(
              {
                  'rebase_anchors': 'rebase_anchors.csv',
                  'rebased_tracks': 'rebased_tracks.csv',
              }
              if args.rebase_agreed and rebase_anchors
              else {}
          ),
          'predictions': 'predictions.npz',
          'visualization': 'track_curtain_multistride.png',
          **(
              {
                  'cycle_consistency_by_stride': {
                      str(stride): f'cycle_consistency_stride{stride}.csv'
                      for stride in frame_strides
                  }
              }
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
  print(f'frame_strides={frame_strides}')
  print(f'selected_peaks={len(peaks)}')
  print(f'model_windows={len(raw_windows)}')
  print(f'multi_resolution_agreements={agreement_summary["agreement_count"]}')
  print(f'rebase_anchors={len(rebase_anchors)}')
  print(f'inference_output={args.output_dir}')


if __name__ == '__main__':
  main()
