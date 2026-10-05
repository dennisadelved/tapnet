"""Render a labeled synthetic seismic sample for visual quality control."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tapnet.seismic.synthetic import SyntheticSeismicConfig
from tapnet.seismic.synthetic import generate_synthetic_sample


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--frames', type=int, default=24)
  parser.add_argument('--height', type=int, default=256)
  parser.add_argument('--width', type=int, default=256)
  args = parser.parse_args()

  config = SyntheticSeismicConfig(
      num_frames=args.frames,
      height=args.height,
      width=args.width,
  )
  sample = generate_synthetic_sample(config, args.seed)
  frame_ids = (0, config.num_frames // 2, config.num_frames - 1)
  figure, axes = plt.subplots(1, len(frame_ids), figsize=(15, 5), sharey=True)
  for axis, frame_id in zip(axes, frame_ids):
    axis.imshow(
        sample['video'][frame_id, ..., 0],
        cmap='gray',
        vmin=-1,
        vmax=1,
        aspect='auto',
    )
    targets = sample['target_points'][:, frame_id]
    visible = ~sample['occluded'][:, frame_id]
    axis.scatter(
        targets[visible, 0],
        targets[visible, 1],
        s=8,
        c=sample['trackgroup'][visible],
        cmap='turbo',
    )
    axis.scatter(
        targets[~visible, 0],
        targets[~visible, 1],
        s=8,
        facecolors='none',
        edgecolors='red',
    )
    axis.set_title(f'frame {frame_id}')
    axis.set_xlabel('lateral trace')
  axes[0].set_ylabel('depth sample')
  figure.suptitle(
      f'seed={args.seed}, faulted={bool(sample["faulted"])}, '
      f'reversed={bool(sample["sweep_reversed"])}'
  )
  figure.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=150)
  plt.close(figure)


if __name__ == '__main__':
  main()
