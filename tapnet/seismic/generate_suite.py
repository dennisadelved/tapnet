"""Export reproducible geological seismic training/evaluation splits to NPZ."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from tapnet.seismic import geology


def export_suite(
    output_dir: Path,
    config: geology.GeologicalSeismicConfig,
    *,
    train_samples: int = 100,
    validation_samples: int = 14,
    test_samples: int = 14,
    seed: int = 0,
) -> Path:
  """Writes paired volumes and TAP labels, keeping each scene in one split.

  Split seeds are separate SeedSequence namespaces; changing a split's size
  never changes another split. Each NPZ opens with ``allow_pickle=False``.
  An existing destination is refused to avoid mixing old and new datasets.
  """
  config.validate()
  counts = {
      'train': train_samples, 'validation': validation_samples, 'test': test_samples
  }
  if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
         for n in counts.values()) or not any(counts.values()):
    raise ValueError('Split counts must be non-negative integers with a total > 0.')
  if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
    raise ValueError('seed must be a non-negative integer.')
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=False)
  records = []
  for split_id, (split, count) in enumerate(counts.items()):
    split_dir = output_dir / split
    split_dir.mkdir()
    for index in range(count):
      sample_seed = int(np.random.SeedSequence(
          [seed, split_id, index, 0x53454953]
      ).generate_state(1, dtype=np.uint64)[0])
      scenario = config.scenarios[index % len(config.scenarios)]
      stride = config.frame_strides[
          (index // len(config.scenarios)) % len(config.frame_strides)
      ]
      sample = geology.generate_geological_sample(
          config, sample_seed, scenario=scenario,
          frame_stride=stride, include_volume=True,
      )
      relative_path = f'{split}/{index:06d}.npz'
      np.savez_compressed(output_dir / relative_path, **sample)
      records.append({
          'path': relative_path, 'split': split, 'index': index,
          'seed': sample_seed, 'scenario': scenario, 'frame_stride': stride,
      })
  manifest = {
      'format_version': 1,
      'generator': 'tapnet.seismic.geology',
      'numpy_version': np.__version__,
      'reference': 'https://arxiv.org/abs/2605.01273',
      'method': 'paper-inspired stratigraphic deformation and Ricker convolution',
      'axis_order': ['frame', 'depth', 'lateral'],
      'query_order': ['frame', 'depth', 'lateral'],
      'target_order': ['lateral', 'depth'],
      'rgt_range': [-1, 1],
      'scenario_ids': dict(enumerate(geology.SCENARIOS)),
      'config': dataclasses.asdict(config),
      'seed': seed,
      'split_counts': counts,
      'samples': records,
  }
  manifest_path = output_dir / 'manifest.json'
  manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
  return manifest_path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--train-samples', type=int, default=100)
  parser.add_argument('--validation-samples', type=int, default=14)
  parser.add_argument('--test-samples', type=int, default=14)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--frames', type=int, default=32)
  parser.add_argument('--height', type=int, default=128)
  parser.add_argument('--width', type=int, default=128)
  parser.add_argument('--horizons', type=int, default=20)
  parser.add_argument('--queries', type=int, default=24)
  parser.add_argument('--noise-std', type=float, default=0.12)
  parser.add_argument('--max-fault-throw', type=float, default=20.0)
  parser.add_argument('--max-faults', type=int, default=3)
  parser.add_argument('--frame-strides', type=int, nargs='+', default=[1])
  parser.add_argument('--scenarios', choices=geology.SCENARIOS, nargs='+',
                      default=list(geology.SCENARIOS))
  args = parser.parse_args()
  config = geology.GeologicalSeismicConfig(
      num_frames=args.frames, height=args.height, width=args.width,
      num_horizons=args.horizons, num_queries=args.queries,
      noise_std=args.noise_std, max_fault_throw=args.max_fault_throw,
      max_faults=args.max_faults, frame_strides=tuple(args.frame_strides),
      scenarios=tuple(args.scenarios),
  )
  try:
    manifest = export_suite(
        args.output_dir, config, train_samples=args.train_samples,
        validation_samples=args.validation_samples, test_samples=args.test_samples,
        seed=args.seed,
    )
  except (ValueError, FileExistsError) as exc:
    parser.error(str(exc))
  print(f'Wrote {manifest}')


if __name__ == '__main__':
  main()
