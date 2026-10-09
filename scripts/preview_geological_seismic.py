"""Compare seismic, RGT, and structural labels for every geological scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tapnet.seismic import geology


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=7)
  args = parser.parse_args()
  config = geology.GeologicalSeismicConfig(
      num_frames=64, height=128, width=96, num_queries=48,
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
      visible = sample['horizon_visible'][:, frame]
      mask = sample['fault_mask'][frame] | sample['unconformity_mask'][frame]
      xlabel = 'lateral trace'
    else:
      lateral = config.width // 2
      section = sample['seismic'][:, :, lateral].T
      rgt = sample['rgt'][:, :, lateral].T
      depths = sample['horizon_depths'][:, :, lateral]
      visible = sample['horizon_visible'][:, :, lateral]
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
