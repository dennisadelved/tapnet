# Geological synthetic seismic suite

This suite generates paired 3D seismic amplitudes and relative geologic time
(RGT), structural masks, and TAPIR horizon tracks. It is inspired by Section 2.6
of Dou et al., *Learning Stratigraphically Consistent Relative Geologic Time
from 3D Seismic Data via Sinusoidal Mapping*,
[arXiv:2605.01273v3](https://arxiv.org/abs/2605.01273v3).
The paper describes 2,000 paired 512-cubed volumes and structural coverage,
but does not give a complete generator specification. This implementation is
an independent, smaller procedural approximation, not a reproduction of their
dataset or RGT-Est neural network.

## Generate a saved dataset

Run from the repository root using the seismic environment with NumPy installed:

```powershell
python -m tapnet.seismic.generate_suite --output-dir datasets/geological-seismic --train-samples 140 --validation-samples 21 --test-samples 21 --seed 42
```

The destination must be new. It contains `manifest.json` and separate `train`,
`validation`, and `test` directories of compressed NPZ files. Each scene stays
in one split. The manifest records config, axis conventions, scenario IDs, and
each sample's seed, scenario, stride, and path. Separate seed namespaces keep
the splits independent of each other and of the online training stream.
Changing a split's size does not change existing samples in another split.

Defaults are 32 frames, 128 depth samples, 128 lateral traces, 20 horizon ages,
24 queries, and stride 1. Dimensions, counts, noise, fault throw/count, scenarios,
and frame strides have CLI options. For example:

```powershell
python -m tapnet.seismic.generate_suite --output-dir datasets/geological-multistride --train-samples 210 --validation-samples 21 --test-samples 21 --frame-strides 1 2 4 --seed 42
```

Generation cycles through all seven scenarios before advancing the stride,
covering the full scenario/stride product every 21 samples in this example.
All stride views are centered on the same scene index, so narrower views retain
the central structures and shared indices refer to identical geology.
Whole-scene seeds differ between splits; slices of the same scene are never
randomly divided between training and evaluation.

## Train and evaluate TAPIR

The existing native PyTorch trainer consumes the suite online, without requiring
precomputed files or a new model architecture:

```powershell
python -m tapnet.seismic.train_torch --config geology-smoke --device cpu --output-dir checkpoints/geology-smoke --checkpoint-mode model-only
python -m tapnet.seismic.train_torch --config vdi-geology --device cuda --output-dir checkpoints/geology-training
```

`geology-smoke` is one 2-frame, 64-by-64 update. `vdi-geology` uses 32-frame
128-by-128 views, 24 queries, all scenarios, strides 1/2/4, and 5,000 updates.
It is a starting configuration; memory/throughput on the target GPU and useful
model quality have not been measured. Use the existing
`--pretrained-checkpoint` option to initialize from compatible TAPIR weights.
Legacy configs continue to use the original generator.

The iterable dataset derives samples from `(seed, sample_index)` and partitions
indices between DataLoader workers. Training resume preserves both scene seeds
and the scenario/stride schedule. Dense volumes are computed for geology but
are omitted from the training batch because TAPIR uses point supervision.

Held-out inference supports both new configs:

```powershell
python -m tapnet.seismic.infer_torch --config geology-smoke --checkpoint checkpoints/geology-smoke/latest.pt --output-dir checkpoints/geology-eval --device cpu --num-examples 7
```

For a `vdi-geology` checkpoint, select `--config vdi-geology` and use at least
21 examples to cover every scenario/stride pair. Keep the evaluation seed
distinct from the training seed, as the existing inference CLI requires.
Saved NPZ volumes can separately serve dense RGT/fault training; this change
does not add an RGT model or a trainer that reads these files.

## Geological and label conventions

| Scenario | Geometry |
| --- | --- |
| `layered` | Horizontal, irregularly spaced beds |
| `dipping` | Strong inline and crossline dip |
| `folded` | Local Gaussian folds with changing layer thickness |
| `faulted` | Folded strata cut by multiple signed vertical offsets |
| `unconformity` | Eroded folded package overlain by younger cover |
| `clinoform` | Laterally migrating sigmoid depositional fronts |
| `mixed` | Folding, faulting, clinoforms, and erosion together |

RGT is normalized once over the full scene to `[-1, 1]`, before slicing. It
increases with depth on each trace. Horizon age values are shared across all
traces and fault blocks. Erosion inserts an age hiatus; an age inside that
hiatus has no visible reflector. RGT normalization is never performed per trace.
Seismic is rendered from the same horizon geometry, plus an unconformity
contact reflector where present, using randomized polarity/amplitude,
zero-phase Ricker convolution, correlated noise, depth gain, and global robust
amplitude scaling. Missing horizons contribute no reflectivity.

| NPZ field | Shape / convention |
| --- | --- |
| `seismic`, `rgt` | float32 `[T, H, W]`, range `[-1, 1]` |
| `fault_mask`, `unconformity_mask` | bool `[T, H, W]` |
| `horizon_depths`, `horizon_visible` | `[K, T, W]`, float32 depths / bool |
| `horizon_rgt` | float32 `[K]`, ordered age values |
| `video` | float32 `[T, H, W, 3]`, repeated scalar seismic |
| `query_points` | float32 `[N, 3]`, `[frame, depth, lateral]` |
| `target_points` | float32 `[N, T, 2]`, `[lateral, depth]` |
| `occluded`, `label_valid` | bool `[N, T]` |
| `trackgroup` | int32 `[N]`, index into the horizon arrays |
| `frame_indices` | int32 `[T]`, indices in the oriented scene |
| `scenario_id`, `fault_count`, `frame_stride`, `scene_num_frames` | int32 scalars |
| `faulted`, `sweep_reversed` | bool scalars |

`fault_mask` includes the rasterized fault sheet and optional damage width;
reflectors in that mask are hidden. `unconformity_mask` marks the first sample
below the erosional contact. Known disappearance through erosion, fault cores,
or leaving the sampled depth window sets `occluded=True`, while
`label_valid=True` remains set. Positions for absent horizons are bounded
surrogates and must be masked out of position losses. Queries always start at
a visible target. Targets keep lateral trace fixed and follow one age surface.
Reversal and frame striding apply identically to seismic, RGT, masks, and labels.

## Python API and preview

```python
import numpy as np
from tapnet.seismic.geology import GeologicalSeismicConfig
from tapnet.seismic.geology import generate_geological_sample
from tapnet.seismic.geology import generate_geological_volume

config = GeologicalSeismicConfig()
volume = generate_geological_volume(config, 42, scenario='mixed')
sample = generate_geological_sample(config, 42, include_volume=True)
with np.load('datasets/geological-seismic/train/000000.npz', allow_pickle=False) as saved:
    amplitudes, rgt = saved['seismic'], saved['rgt']
```

```powershell
python -m scripts.preview_geological_seismic --output docs/geological_seismic_preview.png
python -m pytest tests/test_seismic_geology.py tests/test_seismic_torch_data.py tests/test_seismic_torch_config.py
```

![Seismic, RGT, and horizons for all scenarios](geological_seismic_preview.png)

## Limits and verification

Faults are vertical planes with signed offsets; negative throw does not model
reverse-fault repetition, overturned strata, dipping/listric faults, or fault
drag. Folds and clinoforms are analytic approximations. The renderer does not
simulate impedance/velocity physics, wave propagation, illumination, multiples,
diffractions, migration, salt, or survey acquisition. Horizons can be thin enough
for wavelet interference, so an age surface is not necessarily a seismic peak.
There is no field-data calibration or demonstrated synthetic-to-field accuracy.

Automated tests check RGT ordering, age-surface interpolation, erosion hiatuses,
track/volume alignment after reversal and striding, deterministic generation,
all scenario/stride combinations, worker partitioning, stream resume, NPZ
round-trips, and split isolation. A CPU smoke update verifies the data/model/loss
path and checkpoint writing; it is not evidence of model quality.

Validation on 2026-10-09: the repository test suite passed with **133 tests**.
One CPU update completed with `geology-smoke`, saved a model checkpoint, and
held-out inference ran successfully on all seven scenarios. An additional 210
small scenes (30 seeds per scenario) generated without empty-query failures.
