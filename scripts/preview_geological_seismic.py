"""Compare seismic, RGT, and structural labels for every geological scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tapnet.seismic import geology


def _fault_examples(output):
  """Shows clean normal/reverse faults and their age correlation side by side."""
  config = geology.GeologicalSeismicConfig(
      num_frames=256, height=256, width=256, num_horizons=20,
      max_faults=1, max_fault_throw=80, noise_std=0,
  )
  figure, axes = plt.subplots(2, 3, figsize=(13, 7))
  for row, seed in enumerate((3, 0)):
    volume = geology.generate_geological_volume(config, seed, scenario='faulted')
    lateral = config.width // 2
    seismic = volume['seismic'][:, :, lateral].T
    rgt = volume['rgt'][:, :, lateral].T
    mask = volume['fault_mask'][:, :, lateral].T
    label = 'Normal fault' if volume['fault_parameters'][0, 5] > 0 else 'Reverse fault'
    axes[row, 0].imshow(seismic, cmap='gray', vmin=-1, vmax=1, aspect='auto')
    axes[row, 1].imshow(rgt, cmap='turbo', vmin=-1, vmax=1, aspect='auto')
    axes[row, 2].imshow(seismic, cmap='gray', vmin=-1, vmax=1, aspect='auto')
    axes[row, 2].contour(mask, levels=[0.5], colors=['red'], linewidths=0.7)
    # Contours display every age intersection, including repeated branches.
    axes[row, 2].contour(np.ma.array(rgt, mask=mask),
                         levels=volume['horizon_rgt'], cmap='turbo', linewidths=0.7)
    axes[row, 0].set_ylabel(f'{label} (seed {seed})\ndepth sample')
    for axis in axes[row]:
      axis.set_xlabel('sweep index')
  for axis, title in zip(axes[0], (
      'Deformed reflectivity + Ricker (no noise)', 'Shared material age (RGT)',
      'Age contours / fault label (red)',
  )):
    axis.set_title(title)
  figure.tight_layout()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--fault-examples', action='store_true',
                      help='Show fixed clean normal/reverse examples instead of all scenarios.')
  args = parser.parse_args()
  if args.fault_examples:
    _fault_examples(args.output)
    return
  config = geology.GeologicalSeismicConfig(
      num_frames=256, height=256, width=256, num_queries=48,
      reverse_probability=0.0,
  )
  figure, axes = plt.subplots(len(geology.SCENARIOS), 3, figsize=(12, 17))
  for row, scenario in enumerate(geology.SCENARIOS):
    sample = geology.generate_geological_sample(
        config, args.seed + row, scenario=scenario, include_volume=True
    )
    # Sweep sections expose fault crossings; lateral sections expose clinoforms.
    if scenario == 'clinoform':
      frame = config.num_frames // 2
      section = sample['seismic'][frame]
      rgt = sample['rgt'][frame]
      depths = sample['horizon_depths'][:, frame]
      visible = sample['horizon_visible'][:, frame] & sample['horizon_valid'][:, frame]
      mask = sample['fault_mask'][frame] | sample['unconformity_mask'][frame]
      xlabel = 'lateral trace'
    else:
      lateral = config.width // 2
      section = sample['seismic'][:, :, lateral].T
      rgt = sample['rgt'][:, :, lateral].T
      depths = sample['horizon_depths'][:, :, lateral]
      visible = sample['horizon_visible'][:, :, lateral] & sample['horizon_valid'][:, :, lateral]
      mask = (
          sample['fault_mask'][:, :, lateral]
          | sample['unconformity_mask'][:, :, lateral]
      ).T
      xlabel = 'sweep index'
    axes[row, 0].imshow(section, cmap='gray', vmin=-1, vmax=1, aspect='auto')
    axes[row, 1].imshow(rgt, cmap='turbo', vmin=-1, vmax=1, aspect='auto')
    axes[row, 2].imshow(section, cmap='gray', vmin=-1, vmax=1, aspect='auto')
    for horizon in range(config.num_horizons):
      axes[row, 2].plot(
          np.where(visible[horizon], depths[horizon], np.nan), linewidth=0.7
      )
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (1, 0, 0, 0.4)
    axes[row, 2].imshow(overlay, aspect='auto')
    axes[row, 0].set_ylabel(f'{scenario}\ndepth sample')
    for axis in axes[row]:
      axis.set_xlabel(xlabel)
  for axis, title in zip(axes[0], (
      'Synthetic seismic', 'Ground-truth RGT [-1, 1]',
      'Visible horizons / boundaries (red)',
  )):
    axis.set_title(title)
  figure.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=130)
  plt.close(figure)


if __name__ == '__main__':
  main()
