# Synthetic seismic TAPIR training prototype

## Status and provenance

This is a first-pass research implementation for training TAPIR to follow a
seismic reflector while stepping through inline or crossline slices. It is
built on Google DeepMind TAPNet commit
`730cda1c730877cfedbe01bf87fb1cadb78a565d`.

The prototype is executable end to end, but it is not yet a trained or
validated seismic model. Smoke checkpoints are generated locally under
`checkpoints/`, are ignored by Git, and must not be treated as useful models.

This file is the implementation ledger. Any future shortcut, omitted physical
effect, failed test, or unverified assumption should be added here when the code
changes.

## Scientific scope

The current task is deliberately narrower than "track any seismic voxel."

- A video frame is a vertical seismic slice.
- Video time is progression through inline or crossline index.
- Image `y` is depth/time sample.
- Image `x` is the lateral trace coordinate within the slice.
- A query denotes a known synthetic horizon at one frame and lateral trace.
- Its target keeps the lateral coordinate fixed and follows that horizon's
  depth through all frames.
- `occluded=True` means the synthetic reflector is intentionally terminated.
- `label_valid=False` is reserved for unknown labels and is distinct from a
  geological termination. Synthetic examples currently have complete labels.

The exact TAP tensors are:

```text
video          float32 [T, H, W, 3]       range [-1, 1]
query_points   float32 [N, 3]             [frame, depth, lateral]
target_points  float32 [N, T, 2]          [lateral, depth]
occluded       bool    [N, T]             known termination
label_valid    bool    [N, T]             supervision is known
trackgroup     int32   [N]                synthetic horizon identity
```

## Implemented

### Synthetic generator

`tapnet/seismic/synthetic.py` generates reproducible examples using only NumPy.
It currently includes:

- Smooth dip, cross-dip, curvature, and sinusoidal folding.
- A probabilistic planar fault with a shared vertical throw.
- Non-crossing horizon families with sub-sample depth coordinates.
- Probabilistic horizon termination and explicit occlusion labels.
- Random reflector polarity and laterally varying amplitude.
- Random-frequency zero-phase Ricker convolution.
- Correlated Gaussian noise and a simple depth-dependent gain.
- Robust amplitude clipping/scaling to `[-1, 1]`.
- Random reversal of the slice sequence.
- Scalar amplitude repeated into three channels to preserve the upstream TAPIR
  architecture and checkpoint shape.
- Deterministic seeds and an infinite training stream.

`tapnet/seismic/dataset.py` wraps the generator as an unbatched
`tf.data.Dataset`. Existing experiment code still performs per-device and
device batching. The Kubric import was made lazy so Kubric is not required for
seismic-only training. Natural-video colour augmentation is disabled for the
seismic dataset.

`tapnet/seismic/torch_data.py` provides a separate
`torch.utils.data.IterableDataset`. It does not import TensorFlow or JAX. Each
sample is derived from `(base_seed, sample_index)`, so restarting at a saved
training step resumes the same deterministic sample sequence without replaying
all preceding synthetic examples.

### Seismic loss

`tapnet.utils.model_utils.seismic_tapnet_loss` adds:

- Native trace/sample coordinates instead of automatic 256-by-256 rescaling.
- Separate depth and lateral Huber terms.
- A lower-weight lateral-lock loss because target tracks must remain at their
  seed lateral coordinate.
- Trackability binary cross-entropy.
- Expected-distance supervision based on vertical sample error.
- A `label_valid` mask so an absent annotation is not trained as an occlusion.

The default first-pass weights are:

```text
position_loss_weight = 0.05
depth_loss_weight = 1.0
lateral_loss_weight = 0.25
huber_loss_delta = 2 samples/traces
expected_dist_thresh = 2 depth samples
```

These are starting values, not tuned values.

`tapnet/seismic/torch_losses.py` ports the same objective to PyTorch, including
valid-label masking and losses on every unrefined TAPIR prediction. Fixed-tensor
tests cover weighting, masks, intermediate supervision, and finite gradients.

### Native-Windows PyTorch trainer

`tapnet/seismic/train_torch.py` is a single-device trainer using Google
DeepMind's official `tapnet.torch.tapir_model.TAPIR` implementation. It includes:

- native PyTorch data loading with batch size one;
- CUDA automatic mixed precision, disabled automatically on CPU;
- AdamW with the current learning rate, beta, gradient clipping, weight decay,
  warmup, and cosine-decay intent;
- optional discriminative feature-encoder learning rate through
  `--encoder-lr-multiplier`, while retaining the established head schedule;
- bias parameters excluded from weight decay to approximate the JAX optimizer;
- optional official PyTorch checkpoint initialization;
- default feature-encoder freezing when a pretrained checkpoint is supplied;
- atomic sharded checkpoints: a small `latest.pt` manifest is replaced only
  after separate model, optimizer, and training-state files are complete;
- optional `--checkpoint-mode model-only` output for constrained storage;
- exact synthetic-stream continuation on resume;
- an explicit `--overfit-one-batch` learning sanity-check mode;
- aggregate intermediate loss plus per-stage position, occlusion, probability,
  and total loss logging for every deeply supervised unrefined output; and
- a continuously flushed `metrics.csv` in every output directory, containing
  metrics plus enough configuration and environment context for handoff; and
- CUDA peak allocated-memory reporting at process exit.

For `vdi-small`, `stage_0` is the initial cost-volume prediction and
`stage_1` through `stage_3` are the unrefined mixer predictions. The ordinary
`position_loss`, `occlusion_loss`, and `probability_loss` fields refer to the
final model output. `intermediate_loss` is the sum of all four stage totals;
the optimization objective remains `loss = final losses + intermediate_loss`.
This is a logging-only change and does not alter checkpoint compatibility or
loss weighting.

Each CSV row contains a UTC run ID, step and target step count, configuration,
fixed-batch flag, seed, device and GPU, PyTorch and CUDA versions, precision,
encoder trainability, source checkpoint, elapsed and step times, learning rate,
pre-clipping gradient norm, cumulative peak allocated CUDA memory, all final
losses, aggregate intermediate loss, and all per-stage losses. The file is
closed after every row, so completed rows survive a later training or checkpoint
failure. Resumed processes append with a new run ID. Attach `metrics.csv` in
future updates instead of copying terminal output; the session and checkpoint
columns preserve the needed context.

Full checkpoints remain the default and are exact-resume artifacts. Model-only
checkpoints omit AdamW state and are substantially smaller. They cannot be used
with `--resume`; pass them with `--pretrained-checkpoint` to load parameters and
start a new optimizer. The selected mode is recorded in both the manifest and
CSV. This is an explicit storage tradeoff, not an equivalent resume mechanism.

The upstream PyTorch inference model used an in-place residual addition in
`tapnet/torch/nets.py`. It was changed to an equivalent out-of-place addition
because the in-place form invalidated tensors required by autograd. A full-model
forward/backward test protects this requirement.

### Native-Windows PyTorch held-out inference

`tapnet/seismic/infer_torch.py` loads legacy, model-only, or full sharded
PyTorch checkpoints without loading the optimizer shard. It runs deterministic
synthetic examples from a seed that is separate from the training stream and
uses `model.eval()` plus `torch.inference_mode()`. CUDA BF16 is the default on
the VDI; `--disable-amp` selects FP32.

Every inference output directory contains:

- `summary.json`: checkpoint identity, step, evaluation configuration, exact
  example seeds, precision/device context, visibility rule, and aggregate
  metrics;
- `tracks.csv`: one row per example/query/frame with fault/reversal/horizon
  context, query, prediction, target, validity, target visibility,
  trackability probability, predicted visibility, and errors. Position errors
  are blank for terminated or invalid targets;
- `predictions.npz`: compressed lossless NumPy arrays for the single-channel
  input amplitude, queries, predictions, labels, logits, trackability,
  horizon groups, fault/reversal flags, and example seeds; and
- `example_NNN_tracks.png`: interpreter-facing fixed-lateral seismic curtains.
  Cyan is the visible target, magenta is the prediction, magenta points pass
  the trackability threshold, and the yellow star is the query.

The trackability probability follows upstream TAPIR post-processing:

```text
(1 - sigmoid(occlusion_logit))
  * (1 - sigmoid(expected_distance_logit))
```

The default visibility decision threshold is `0.5` and is recorded in every
output. Aggregate position metrics use only known, visible targets. Visibility
metrics use every known target, including true terminations. Current metrics
are depth MAE/RMSE, lateral MAE, depth accuracy within 1/2/4 samples, gross
depth-error rate above 8 samples, visibility accuracy/precision/recall/F1,
predicted-trackable fraction, and mean trackability probability.

This evaluator is synthetic; the separate bounded real-volume path is described
below. It does not emit SEG-Y/ZGY horizons, dense surfaces, confidence
calibration curves, or metrics stratified by faults, terminations, and sweep
direction. The PNG background is sampled at the query's fixed lateral trace;
the CSV/NPZ must be used to inspect any predicted lateral drift.

### Real ZGY seeded-track smoke path

`tapnet/seismic/zgy.py` and `tapnet/seismic/infer_zgy_torch.py` implement the
first real-volume path. The design follows the bounded-reader pattern inspected
in the local `fault-prob-sfm/src/data/streaming.py` and
`fault-prob-sfm/src/inference/streaming_io.py` implementations rather than
loading a whole cube:

- OpenZGY array order is treated as `[inline, crossline, sample]`;
- one caller-owned `float32` window is read with `ZgyReader.read`;
- the reader is held in a context manager and closed on every exit path;
- `size`, `zstart`, `zinc`, `annotstart`, `annotinc`, ordered world corners,
  and unit names are preserved; and
- model predictions are converted back to fractional cube indices, line
  annotations, Z coordinates, and interpolated world X/Y.

By default, an inline sweep reads `[8 frames, 128 crosslines, 128 samples]` and
transposes it to model order `[frame, depth, lateral]`. A crossline sweep reads
`[128 inlines, 8 frames, 128 samples]` and performs the corresponding
transpose. `--num-frames N` overrides the number of survey lines for either
real-ZGY inference command without changing the 128-trace lateral or
128-sample depth window. The line window is centered on the seed and clamped at
cube boundaries. Cubes smaller than the requested window are rejected rather
than silently padded or resampled. Larger values increase inference memory and
runtime approximately linearly and have not yet been accuracy- or
memory-benchmarked on the L40-12Q.

The current CLI accepts one seed in either index coordinates or
`[inline annotation, crossline annotation, Z header coordinate]`. It produces:

- `tracks.csv` with model coordinates, cube indices, annotations, Z, world X/Y,
  trackability, and the visibility decision for every frame;
- `predictions.npz` with the exact bounded input, normalized input, logits,
  tracks, coordinates, and confidence;
- `track_curtain.png` on the seed's fixed lateral trace; and
- `summary.json` with input/checkpoint provenance, complete ZGY geometry,
  window/axis mapping, normalization, query mapping, and stability diagnostics.

Real amplitudes are normalized per inference patch using the same functional
form as the synthetic renderer: divide by the `99.5` percentile of absolute
finite amplitude and clip to `[-1, 1]`. Non-finite values are replaced by zero
and counted in `summary.json`; an all-nonfinite or zero-scale patch is rejected.
This patch-local scaling is an explicit first-pass domain-adaptation shortcut.
It is not yet a survey-level normalization policy and may make confidence
incomparable across patches.

The single-seed CLI does not ingest a real interpreted horizon, calculate
real-data accuracy, fuse tracks into a surface, enforce inline/crossline
agreement, or write a horizon grid/ZGY. The reported drift and trackability
fields are diagnostics, not accuracy metrics. A confident track can still
follow the wrong reflector. The separate multi-seed CLI is described below.

### Multiple seeds from one real trace

`tapnet/seismic/infer_zgy_peaks_torch.py` automates multiple queries from one
inline/crossline trace. It reads the full sample axis only for the bounded
frame-by-128-trace sweep block (8 frames by default, or `--num-frames N`),
picks local extrema on the source trace, and batches all compatible queries in
overlapping model windows.

The default peak policy is explicit and configurable:

```text
polarity                  both positive peaks and negative troughs
relative amplitude        >= 0.1 * p99.5(abs(source trace))
minimum seed spacing      4 samples
maximum peak count        unlimited unless --max-peaks is supplied
```

Literal positive peaks use `--peak-polarity positive`; negative events use
`negative`. The detector uses immediate-neighbor extrema, so the first and last
samples cannot be selected. Candidates are processed strongest-first and a
weaker candidate inside `--peak-min-distance` of an accepted event is removed.
This is deterministic amplitude picking, not a prominence, phase, or geological
event detector.

Depth is covered with half-overlapping 128-sample windows plus an end-aligned
window. Each seed is assigned to the containing window with the largest edge
margin. All seeds assigned to a window are sent to TAPIR together; the existing
query chunk size limits model memory. Fractional line requests are snapped to
the nearest actual source trace and both requested and snapped coordinates are
recorded.

The multi-seed output has the same four artifact types as single-seed inference.
`tracks.csv` adds seed ID, seed sample, amplitude and polarity. The NPZ also
contains the exact source trace, every raw/normalized model window, window
assignment, logits, and all survey coordinates. The PNG shows the full source
trace curtain with tracks colored by seed sample.

This first pass intentionally omits wavelet-aware peak consolidation, automatic
polarity selection, peak picking across missing samples, amplitude/phase
attributes, multi-trace seed voting, horizon ordering, track crossing checks,
and surface fusion. It also reads the complete sample axis for the bounded
horizontal block; very deep cubes may require depth-streamed peak detection.

### Forward/backward cycle consistency

Pass `--cycle-consistency` to the multi-seed real-ZGY command to run the first
label-free consistency test. For every forward peak-seeded track, the test:

1. takes the predicted `[lateral, depth]` point at the first frame and at the
   last frame;
2. re-queries TAPIR on the same normalized model window from each endpoint;
3. measures the returned position at the original source frame against the
   original seed; and
4. measures mean and maximum full-path disagreement between the original and
   endpoint-seeded tracks.

Endpoints outside the 128-by-128 model window are recorded as out of bounds
and are not silently clipped. A cycle is `confidence_valid` only when the
forward endpoint and returned source probabilities both exceed
`--visibility-threshold`. Metrics are reported separately for every in-bounds
cycle and for this confidence-qualified subset. No geometric pass threshold is
hard-coded in the first pass; the CSV retains lateral, depth, Euclidean, and
full-path errors so thresholds can be selected from observed distributions.

Example using the 500-step adapted checkpoint and the same 400-crossline test:

```powershell
$cube = 'D:\data\survey.zgy'
$checkpoint = 'checkpoints\seismic_tapir_torch_vdi_pretrained_frozen500\latest.pt'
$output = 'checkpoints\real_zgy_crossline_peaks400_cycle'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_zgy_peaks_torch `
  --input $cube `
  --checkpoint $checkpoint `
  --output-dir $output `
  --config vdi-small `
  --num-frames 400 `
  --sweep crossline `
  --coordinates annotation `
  --query-inline 22434 `
  --query-crossline 282 `
  --peak-polarity both `
  --peak-relative-threshold 0.1 `
  --peak-min-distance 4 `
  --max-peaks 20 `
  --cycle-consistency `
  --device cuda
```

In addition to the normal inference artifacts, this writes
`cycle_consistency.csv`, adds lossless endpoint queries, in-bounds masks,
reseeded tracks, probabilities, and diagnostic arrays to `predictions.npz`, and
adds aggregate distributions to `summary.json`. The terminal reports the
number of confidence-valid cycles out of 40 attempted cycles for 20 seeds.

Cycle consistency is not accuracy: both passes can consistently follow the
same wrong reflector. It is also affected by endpoint confidence, paths leaving
the local lateral/depth window, and the very long 400-frame extrapolation from
8-frame synthetic training. The first implementation does not yet render a
cycle plot, compare inline and crossline cycles, sample amplitudes along the
predicted 3D paths, or calculate thresholds from interpreted validation data.
The dependency-independent cycle metric tests pass locally; an end-to-end local
ZGY smoke was omitted because the current workspace virtual environment lacks
`pyzgy`. Subsequent VDI baseline and trained runs completed end to end, as
recorded in the validation section.

`tapnet/seismic/torch_config.py` contains four explicit variants:

- `smoke`: 64-by-64, 2 frames, 2 queries, one refinement iteration, one step,
  random initialization, and plumbing validation only.
- `vdi-small`: 128-by-128, 8 frames, 16 queries, the checkpoint-compatible
  pyramid, and 2,000 planned steps. This is the initial L40-12Q experiment, not
  a tuned final configuration.
- `vdi-hard`: 128-by-128, 16 frames, 10 horizons, 24 queries, one to three
  planar faults in every example, a cumulative 12-sample fault-throw budget,
  noise standard deviation 0.18, termination probability 0.35, and 5,000
  planned steps. It uses a `5e-5` peak learning rate with 250 warmup steps and
  raises fixed-lateral supervision from 0.25 to 1.0 relative to depth. This is
  the second-stage curriculum for the Windows VDI, not a final geological
  simulator.
- `vdi-multistride`: 128-by-128, 32-frame views sampled at balanced strides
  1, 2, and 4 from one rendered 125-line scene, otherwise retaining the hard
  fault/noise/query settings. It plans 3,000 new-optimizer steps.

### Evaluation

`eval_seismic_synthetic` uses a fixed seed disjoint from the training stream and
reports:

- Mean absolute depth error in samples.
- Mean absolute lateral error in traces.
- Fractions within 1, 2, and 4 depth samples.
- Fraction with a gross error greater than 8 samples.
- Trackability accuracy.
- Visible precision and recall.
- Seismic-native position and trackability losses.

The evaluation runner is configured as one-off evaluation so it exits after the
latest checkpoint instead of waiting for new checkpoints.

### Configuration and quality checks

- `configs/seismic_tapir_config.py` is the default 256-by-256, 24-frame
  synthetic configuration.
- `configs/seismic_tapir_config.py:smoke` is a 64-by-64, 2-frame, 2-query,
  one-step integration configuration. It tests plumbing only.
- `scripts/preview_synthetic_seismic.py` renders labeled examples.
- `docs/synthetic_seismic_preview.png` is the inspected seed-5 example. Solid
  dots are visible targets and hollow red dots are known terminations.
- Unit tests cover shapes, data types, amplitude range, coordinate ordering,
  fixed-lateral targets, termination semantics, reproducibility, configuration
  validation, TensorFlow and PyTorch wrapping, JAX/PyTorch loss
  masking/weighting, finite gradients, deterministic resume samples, and a full
  PyTorch TAPIR backward pass.

## Setup

There are now two framework paths. JAX/JAXline remains the reference and needs
Linux or WSL2 for NVIDIA GPU training. PyTorch is the recommended target for the
locked-down Windows VDI because official PyTorch CUDA wheels support native
Windows. Do not mix their checkpoints or environment instructions.

### Target Windows VDI: no-admin feasibility gate

Reported target VDI on 2026-10-05:

- vGPU profile: NVIDIA L40-12Q;
- dedicated GPU memory visible to the VDI: 11.0 GB;
- dedicated memory already in use when inspected in Task Manager: 4.6 GB;
- shared GPU memory: 192 GB; and
- Windows device driver version: 32.0.15.8253, dated 2026-04-15.

Environment probe on 2026-10-06:

```text
torch.__version__          2.14.1+cpu
torch.cuda.is_available()  False
torch.version.cuda         None
device                     GPU unavailable
```

This result proves that the CPU-only PyTorch wheel was installed; it does not
yet prove that CUDA compute is blocked by the VDI. The next test is to replace
that wheel with the official Windows CUDA 12.6 build:

```powershell
.venv-torch\Scripts\python.exe -m pip uninstall -y torch
.venv-torch\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

CUDA 12.6 is selected conservatively for the first VDI test. Do not install the
CPU requirements again between this command and the CUDA verification, because
an incorrectly configured package source could replace the CUDA wheel.

Correction recorded on 2026-10-06: the first version of
`requirements_seismic_torch.txt` included an unqualified `torch` entry. Following
the documented setup could therefore install the default CPU wheel. That was a
project setup defect, not a user error. The entry has been removed; PyTorch must
now be installed explicitly from the selected CUDA wheel index before installing
the remaining requirements.

Second clean-environment correction recorded on 2026-10-06: the first
PyTorch-only requirements list omitted `dm-tree`. The official TAPIR PyTorch
model imports it as `tree`, while the development environment already had it as
a transitive JAX-side dependency and therefore hid the omission. The target VDI
correctly failed with `ModuleNotFoundError: No module named 'tree'`. `dm-tree`
has been added explicitly. The audited direct third-party imports for the
PyTorch seismic path are now PyTorch itself, NumPy, `einshape`, and `dm-tree`.

The shared-memory figure is system RAM and must not be counted as CUDA device
memory for JAX/XLA capacity planning. The effective training ceiling is the
11 GB vGPU framebuffer, less display and other-process usage. The `-12Q` profile
is a partition of an L40; it does not expose the physical L40's full memory.

No-admin installation of Python packages in a virtual environment is possible,
but that does not bypass the operating-system requirement: the official JAX
CUDA wheels run on Linux, not native Windows. A Conda or `venv` environment on
native Windows therefore cannot make this training code use the vGPU.

Before requesting any VDI change, run these non-administrative PowerShell checks:

```powershell
nvidia-smi
where.exe python
py -0p
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

Interpretation:

1. If PyTorch reports CUDA and the L40-12Q, use the native-Windows PyTorch path.
2. If PyTorch is absent, install an approved CUDA wheel into a user-owned
   virtual environment; administrator access is normally unnecessary.
3. If a CUDA-enabled PyTorch wheel reports no GPU, the vGPU driver, license, or
   VDI policy is blocking compute access and must be handled by the provider.
4. WSL2 or a Linux container is needed only if retaining JAX GPU training.

At this stage of setup the VDI CUDA path was unverified. The later validation
records below supersede that initial status: CUDA forward/backward now works on
the L40-12Q. The 11 GB profile remains the real device-memory limit; do not
infer usable VRAM from the 203 GB combined Task Manager figure.

VDI update on 2026-10-06: the CUDA-enabled PyTorch environment successfully
reported `device=cuda` and `gpu=NVIDIA L40-12Q`. This proves that native-Windows
PyTorch can access the assigned vGPU. The first training attempt then stopped
before model construction with `FileNotFoundError` for
`checkpoints/pretrained/bootstapir_checkpoint_v2.pt`. This is expected when the
repository is cloned because `checkpoints/` is Git-ignored. It is a missing
artifact, not a CUDA or training failure. CUDA forward/backward, peak VRAM, and
throughput remain unverified until the checkpoint is copied or downloaded and
the command is rerun.

Second VDI run on 2026-10-06 loaded the checkpoint and completed forward and
backward execution with the feature encoder frozen, but it did **not** complete
a valid optimizer update. FP16 produced finite loss `13.255537` but a non-finite
pre-clipping gradient norm (`nan`). `GradScaler` therefore skipped
`optimizer.step()`, after which the old trainer incorrectly advanced the learning
rate scheduler, emitted a scheduler-order warning, and saved a checkpoint. That
checkpoint is invalid and must not be resumed. Peak allocated CUDA memory was
only 0.677 GiB, so this failure indicates numerical overflow rather than an
out-of-memory condition.

Corrective changes:

- CUDA automatic mixed precision now defaults to BF16, which the L40 supports
  and which has a substantially larger numerical range than FP16;
- FP16 remains an explicit `--amp-dtype float16` option rather than the default;
- `--disable-amp` provides the FP32 diagnostic path;
- non-finite gradient norms now raise before the optimizer, scheduler, or
  checkpoint save; and
- precision settings are included in checkpoint compatibility checks.

The required next gate at that point was a one-step FP32 run followed by BF16
with the updated trainer. Both later completed with finite gradient norms and
without a scheduler warning, as recorded below.

FP32 VDI gate completed successfully on 2026-10-06:

```text
device                    cuda
gpu                       NVIDIA L40-12Q
feature encoder           frozen
precision                 float32
loss                      13.134300
position loss             0.933581
occlusion loss            1.552396
probability loss          0.475143
pre-clipping gradient     1178.522827
peak allocated CUDA       0.814 GiB
checkpoint                saved
```

The gradient norm was finite and clipped to the configured maximum of 1.0; no
scheduler warning occurred. The large pre-clipping norm remains a stability
warning for longer training. The displayed learning rate of `0.00000200` in
this run was the rate prepared for the next step, not the rate used by the
optimizer. Because `vdi-small` has a 100-step warmup, the first update used
`0.00000100`. The trainer now captures and reports the rate actually used by
the update.

The FP32 result proves that the native-Windows CUDA training path and current
constrained geometry fit comfortably in the 11 GB vGPU partition. The remaining
precision gate is one BF16 step with the updated default configuration.

BF16 VDI gate completed successfully on 2026-10-06:

```text
device                    cuda
gpu                       NVIDIA L40-12Q
feature encoder           frozen
precision                 bfloat16
loss                      12.852102
position loss             1.043687
occlusion loss            1.469949
probability loss          0.502973
pre-clipping gradient     2185.319092
learning rate used        0.00000100
peak allocated CUDA       0.815 GiB
checkpoint                saved
```

The BF16 update had finite gradients, no scheduler warning, and valid checkpoint
output. BF16 is therefore the preferred VDI precision. The pre-clipping gradient
norm is nearly twice the FP32 smoke value and was clipped to 1.0. This does not
invalidate the execution gate, but the one-step result alone was not evidence
of stable optimization. The fixed-batch run below subsequently supplied that
evidence; held-out evaluation and intermediate-loss inspection still block the
full 2,000-step experiment.

The 100-step fixed-batch BF16 VDI run completed all optimizer steps on
2026-10-06 without a non-finite loss or gradient:

```text
                              step 1          step 100
total deeply supervised loss  12.852102       0.775462
final position loss             1.043687       0.005962
final occlusion loss            1.469949       0.000000
final probability loss          0.502973       0.000001
pre-clipping gradient norm   2185.319092       8.531075
learning rate used              0.000001       0.000100
```

This is a 94.0% decrease in total loss and a 99.4% decrease in the final-head
position loss. It validates sustained BF16 optimization on the L40-12Q and
shows that the final prediction head can memorize the fixed synthetic batch.
The reported total is higher than the three displayed components because it
also sums supervision for four unrefined TAPIR predictions; those intermediate
components are currently calculated but not printed. Their residual loss means
that the complete deep-supervision overfit gate is only partially passed.

The initial console capture ended immediately after the step-100 line, but the
complete output subsequently confirmed normal completion:

```text
checkpoint=checkpoints\seismic_tapir_torch_vdi_bf16_overfit100\latest.pt
peak_cuda_memory_gib=0.874
```

The 100-step run therefore saved its intended checkpoint and remained far below
the 11 GB vGPU allocation in peak memory reported by PyTorch.

The same fixed batch was then resumed from step 100 through step 300. This was
run before intermediate-loss logging was added, so it provides an optimization
result but does not isolate the residual loss:

```text
                              step 100        step 300
total deeply supervised loss   0.775462        0.480608
final position loss            0.005962        0.000027
final occlusion loss            0.000000        0.000000
final probability loss          0.000001        0.000000
pre-clipping gradient norm      8.531075        0.587641
learning rate used              0.000100        0.00000101
checkpoint                    checkpoints/seismic_tapir_torch_vdi_bf16_overfit300/latest.pt
peak allocated CUDA           0.876 GiB
```

The continuation was numerically stable and reduced the residual total loss by
38.0%, while the final output converged essentially to zero loss. At step 300,
approximately `0.480581` of the `0.480608` total comes from unprinted unrefined
outputs. The learning-rate schedule had also reached its approximately `1e-6`
floor. Continuing this same schedule without first identifying the responsible
stage and loss term is not a useful diagnostic.

The frozen feature encoder uses `InstanceNorm2d` with
`track_running_stats=False`, not running-stat BatchNorm. Calling `model.train()`
therefore does not silently change normalization statistics in the frozen
encoder and does not explain the intermediate-loss plateau.

The per-stage step-301 diagnostic isolated the residual:

```text
final output total       0.000024
intermediate total       0.480867
stage 0 position         0.110461
stage 0 occlusion        0.000005
stage 0 probability      0.370307
stage 0 total            0.480773
stage 1 total            0.000049
stage 2 total            0.000023
stage 3 total            0.000023
overall loss             0.480891
pre-clipping gradient    0.622427
peak allocated CUDA      0.876 GiB
```

Stage 0 is the initial cost-volume prediction and accounts for 99.98% of the
intermediate loss. Its expected-distance loss accounts for 77.0% of the stage-0
residual, position for 23.0%, and occlusion is effectively solved. All three
mixer refinements and the final output have overfit the fixed batch.

The reference JAX trainer also applies position, occlusion, and probability
supervision equally to every item in `unrefined_tracks`, including stage 0.
The residual is therefore not caused by accidental extra supervision in the
PyTorch port. Do not remove or down-weight stage 0 without an explicit ablation.
The next controlled test is a fresh fixed-batch run from the official pretrained
checkpoint with the feature encoder trainable, preceded by a one-step CUDA
memory and gradient gate.

### PyTorch alternative for a locked-down Windows VDI

PyTorch is the preferred alternative if the VDI cannot provide WSL2 or a Linux
container. Official PyTorch CUDA wheels support Windows, and installing a wheel
inside a user-owned virtual environment normally does not require administrator
rights. GPU access still depends on the VDI driver, vGPU license/policy, and the
ability to download or obtain the approved wheel.

This repository already contains Google DeepMind's PyTorch TAPIR architecture in
`tapnet/torch/` and publishes matching PyTorch checkpoints. This avoids porting
the neural network itself. It does **not** provide an official PyTorch TAPIR
training framework: the upstream training loop, optimizer integration, losses,
evaluation, and checkpoint management are JAX/JAXline code. Google DeepMind
links to a third-party PyTorch training project but explicitly states that it is
not affiliated and its accuracy has not been verified.

Implemented scope:

1. The framework-independent NumPy generator is reused unchanged.
2. The PyTorch data adapter does not route through TensorFlow.
3. The complete first-pass seismic loss and unrefined losses are ported.
4. The trainer uses the official PyTorch TAPIR module and loads the published
   BootsTAPIR v2 checkpoint with strict key matching.
5. Optimizer, schedule, gradient clipping, checkpoint/resume, mixed precision,
   and peak-memory reporting are implemented.
6. Deterministic CPU tests and full-model backward tests are implemented.
7. The `vdi-small` configuration uses batch size one, reduced geometry, and a
   frozen feature encoder by default when pretrained weights are supplied.

Known omissions and risks:

- held-out PyTorch inference now exists, but it has not yet evaluated a model
  trained on varying synthetic batches;
- the 300-step BF16 fixed-batch run drives the final-head losses near zero, but
  approximately 0.480581 aggregate intermediate-refinement loss remains; the
  logger attributes nearly all of it to stage 0;
- exact parity with the JAX checkpoint/training trajectory is not expected;
- the current JAX/JAXline configuration cannot be reused directly;
- peak allocated VRAM was approximately 0.815 GiB for the one-step BF16 run and
  0.874 GiB for the 100-step fixed-batch BF16 run; and
- corporate package-index and checkpoint-download policies may require an
  offline wheel/checkpoint transfer.

Do not delete the JAX implementation. Keep it as the tested reference until the
PyTorch CUDA smoke, fixed-batch overfit, evaluation, and held-out tests have all
passed.

#### Native Windows PyTorch setup

Create the environment without administrator privileges:

```powershell
python -m venv .venv-torch
.venv-torch\Scripts\python.exe -m pip install --upgrade pip
```

Use the official PyTorch selector to install a Windows/Pip/CUDA wheel compatible
with the VDI. Then install the remaining local requirements and verify CUDA:

```powershell
.venv-torch\Scripts\python.exe -m pip install -r requirements_seismic_torch_dev.txt
.venv-torch\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

Run commands from the repository root. Do not use `pip install -e .` for this
environment yet: the upstream `pyproject.toml` lists JAX/JAXline as unconditional
dependencies. `requirements_seismic_torch.txt` intentionally supplies only the
PyTorch-path runtime packages so the target environment need not install JAX or
TensorFlow. Matplotlib is included only for the inference PNG artifacts, and
`pyzgy==0.1.1` supplies the `openzgy.api` reader used for local ZGY files.
`sdglue` is not required for local files. ZFP-compressed ZGY input may require
the optional `zfpy` package and has not been tested on the Windows VDI.

Download or copy the official checkpoint to an ignored local directory:

```powershell
New-Item -ItemType Directory -Force checkpoints\pretrained
curl.exe -L --fail --output checkpoints\pretrained\bootstapir_checkpoint_v2.pt https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
```

If corporate policy blocks the URL, transfer the approved file through the
organization's permitted mechanism. The locally validated file size was
218,886,140 bytes; no upstream cryptographic checksum was found, so size alone
is not an authenticity guarantee.

### Windows with an NVIDIA GPU: WSL2 setup

Observed development host on 2026-10-05:

- GPU: NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU;
- dedicated VRAM reported by `nvidia-smi`: 6113 MiB;
- NVIDIA driver: 596.58;
- maximum CUDA version reported by the driver: 13.2; and
- WSL was not installed at the time of inspection.

The driver is new enough for the CUDA 13 JAX wheel. The approximately 6 GB of
VRAM is expected to be the main constraint. The smoke configuration should be
tested first; the default 24-frame, 256-by-256 configuration is not assumed to
fit until measured. If it does not fit, create and document a reduced GPU
configuration instead of silently changing the default experiment.

Assumptions for this path:

- the machine has a supported NVIDIA GPU;
- Windows 11, or a Windows 10 release that supports WSL2, is installed;
- virtualization is enabled in the firmware; and
- the NVIDIA **Windows** driver supports CUDA in WSL.

In an Administrator PowerShell terminal, list the available distributions and
install Ubuntu 22.04 so the WSL environment uses the Python 3.10 version already
validated by this prototype:

```powershell
wsl --update
wsl --list --online
wsl --install -d Ubuntu-22.04
```

Restart Windows if requested. Install or update the NVIDIA Windows driver before
continuing. Do not install an NVIDIA Linux display driver inside WSL; WSL exposes
the Windows host driver to Linux. In the Ubuntu terminal, first confirm that the
GPU is visible:

```bash
nvidia-smi
```

Keep the training checkout in the WSL Linux filesystem (for example,
`~/tap-testing`) rather than under `/mnt/c` to avoid slower filesystem access.
The seismic changes must first be committed and pushed to an accessible branch,
or the complete working tree must be copied, because cloning upstream TAPNet
alone does not contain this prototype.

Inside the WSL checkout, create the environment and install the CUDA-enabled
JAX wheel before the project requirements:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "jax[cuda13]"
python -m pip install -r requirements_seismic_dev.txt
```

If the installed GPU or Windows driver cannot support the CUDA 13 JAX wheel,
use `jax[cuda12]` instead. Do not install both variants. Verify the environment:

```bash
python --version
python -c "import jax; print(jax.__version__); print(jax.devices())"
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pytest tests -q
```

`jax.devices()` must include a GPU before attempting the default experiment.
This WSL2 GPU path has not yet been executed for this prototype. GPU memory use,
throughput, suspend/resume behavior, and checkpoint restart remain explicit
future tests.

### Native Linux or managed Linux GPU container

Create an isolated environment and install the correct accelerator-specific JAX
build first. One native Linux/NVIDIA example is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda13]"
python -m pip install -r requirements_seismic_dev.txt
```

Check the active backend before starting a long run:

```bash
python -c "import jax; print(jax.devices())"
```

### Native Windows: CPU smoke testing only

On PowerShell, CPU-only smoke setup is:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements_seismic_dev.txt
```

## Commands

Run all tests:

```bash
python -m pytest tests -q
```

Run the native-Windows PyTorch plumbing smoke test. Use `--device cpu` only for
development; the VDI gate must use `--device cuda`:

```powershell
python -m tapnet.seismic.train_torch --config smoke --device cuda --checkpoint-every 1
```

Run exactly one constrained pretrained VDI step before scheduling training:

```powershell
python -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 1 `
  --device cuda `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_cuda_smoke `
  --checkpoint-every 1
```

Record the reported `peak_cuda_memory_gib`, elapsed time, losses, gradient norm,
PyTorch version, CUDA runtime, driver, and GPU name. Do not start the 2,000-step
run until this command completes with memory headroom.

Run the fixed-batch learning sanity check separately from normal training:

```powershell
python -m tapnet.seismic.train_torch `
  --config smoke `
  --steps 20 `
  --device cuda `
  --overfit-one-batch `
  --output-dir checkpoints\seismic_tapir_torch_overfit `
  --checkpoint-every 20
```

The validated pretrained VDI version of this gate is:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 100 `
  --device cuda `
  --overfit-one-batch `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_bf16_overfit100 `
  --checkpoint-every 100
```

After updating to the per-stage logger, inspect the step-300 checkpoint with
one additional update:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 301 `
  --device cuda `
  --overfit-one-batch `
  --resume checkpoints\seismic_tapir_torch_vdi_bf16_overfit300\latest.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_bf16_diagnostic301 `
  --checkpoint-every 1
```

The first output line reports the final and aggregate intermediate losses. The
following `intermediate_step=301` line reports position, occlusion,
probability, and total loss for stages 0 through 3. This diagnostic uses the
existing optimizer state and approximately `1e-6` learning-rate floor; it is
not a new training experiment.

The diagnostic showed that only stage 0 remains material. Before a longer
trainable-encoder overfit run, execute a one-step memory and gradient gate:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 1 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_trainable_encoder_smoke1 `
  --checkpoint-every 1
```

Only if that step has finite loss and gradients and fits in device memory,
continue the same optimizer state to step 300:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 300 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --resume checkpoints\seismic_tapir_torch_vdi_trainable_encoder_smoke1\latest.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_trainable_encoder_overfit300 `
  --checkpoint-every 50
```

This must be a new trainable-encoder run. The frozen-encoder step-301 optimizer
has no state for encoder parameters and is intentionally rejected if resumed
with `--train-feature-encoder`.

The one-step trainable-encoder gate completed successfully on the L40-12Q:

```text
loss                       12.852104
final position              1.043687
final occlusion             1.469949
final probability           0.502973
intermediate loss           9.835494
pre-clipping gradient    5298.271973
learning rate used          0.000001
peak allocated CUDA         1.252 GiB
checkpoint                  checkpoints/seismic_tapir_torch_vdi_trainable_encoder_smoke1/latest.pt
```

The update and checkpoint were valid and all reported values were finite. Peak
allocated memory increased by 0.376 GiB relative to the 0.876 GiB frozen-encoder
diagnostic and remains far below the 11 GB allocation. The pre-clipping gradient
is large and was clipped to the configured norm of 1.0. The 300-step continuation
above is now cleared to run, with gradient finiteness and stage-0 losses as the
primary monitoring signals.

The trainable-encoder continuation first reached step 100 but failed while
writing the monolithic step-100 checkpoint:

```text
RuntimeError: ios_base::badbit set: iostream stream error
RuntimeError: unexpected pos 476526592 vs 476526480
```

The retry with enhanced error reporting failed at the same byte offset while
writing step 50, despite reporting 1858.483 GiB free. This rules out ordinary
volume free-space exhaustion but not a user/profile quota, filesystem filter,
antivirus/security product, or a failure specific to PyTorch's monolithic ZIP
writer. It was not a CUDA out-of-memory or non-finite-gradient failure.

A subsequent sharded save also failed with `OSError: [Errno 28] No space left
on device`, while the volume API still reported 1858.479 GiB free. The sharded
cleanup succeeded. This proves that a user/profile/directory quota or storage
filter, rather than monolithic file size alone, is blocking writes. Sharding
improves atomicity but cannot bypass the effective quota.

The model-only gate under `%LOCALAPPDATA%` failed while writing the model shard
at approximately 4.1 MB, again with `Errno 28` and 1858.478 GiB volume-level
free space. Only the 2,163-byte `metrics.csv` was written. The effective limit
therefore applies across the user's profile rather than only to `Documents`,
and neither sharding nor omitting optimizer state can work around it. No useful
model checkpoint can be preserved until existing user files are removed or the
VDI quota is increased.

The user-level checkpoint inventory identified approximately 3.65 GiB of
generated `.pt` artifacts plus the 208.75 MiB official pretrained checkpoint,
consistent with an approximately 4 GiB effective profile quota. The generated
artifacts consist of superseded precision smokes, fixed-batch memorization
checkpoints, unstable trainable-encoder checkpoints, and one invalid 454.45 MiB
temporary file. None is a useful generalizing model. They are approved cleanup
targets once their exact paths are previewed. Retain
`checkpoints/pretrained/bootstapir_checkpoint_v2.pt` and the small CSV logs.
Deleting the generated `.pt` files is not reversible, but their validation
results are recorded here and the experiments are reproducible.

After those generated `.pt` artifacts were removed, the full sharded
trainable-encoder gate successfully saved step 1 on the VDI. It reported finite
loss `12.852104`, finite pre-clipping gradient norm `5300.103516`, and 1.252 GiB
peak allocated CUDA memory. The output contains a `latest.pt` manifest plus
model, optimizer, and training-state shards. Sharded saving is therefore
VDI-validated; loading and exact optimizer continuation remain gated on the
step-2 resume test.

The step-2 resume gate then loaded the sharded manifest and all three state
shards, restored the optimizer and scheduler, appended to the same CSV, and
saved a replacement checkpoint. Loss decreased from 12.852104 at step 1 to
12.688643 at step 2, the actual head learning rate advanced from `1e-6` to
`2e-6`, gradients remained finite, and peak allocated CUDA memory was 1.319 GiB.
Full sharded save and exact resume are therefore VDI-validated.

The shared `1e-4` peak rate destabilized the earlier trainable-encoder run even
though that rate was stable with the encoder frozen. The trainer now supports a
separate encoder multiplier without changing the default. The initial ablation
uses `--encoder-lr-multiplier 0.1`, giving a `1e-5` encoder peak while retaining
the `1e-4` head peak. Both rates are recorded in `metrics.csv`; the multiplier
is checkpointed and cannot change during exact resume.

The older trainer left an invalid `latest.pt.tmp`. Because those saves wrote the
temporary file before replacing `latest.pt`, a previously completed step-50
`latest.pt` should remain intact; neither failed attempt persisted its current
state. Confirm rather than assume the saved step:

```powershell
Get-PSDrive -Name C | Select-Object Used, Free
Get-ChildItem checkpoints\seismic_tapir_torch_vdi_trainable_encoder_overfit300 `
  -Force | Select-Object Name, Length, LastWriteTime
.venv-torch\Scripts\python.exe -c "import torch; p=r'checkpoints\seismic_tapir_torch_vdi_trainable_encoder_overfit300\latest.pt'; print(torch.load(p, map_location='cpu', weights_only=True)['step'])"
```

If `latest.pt` loads and `latest.pt.tmp` exists, the latter is the incomplete
monolithic artifact and can be removed using its exact path after the Python
process exits. The trainer now avoids this large monolithic write: it writes
model, optimizer, and training metadata to separate, uniquely named shards,
then atomically replaces the small `latest.pt` manifest. Existing monolithic
checkpoints remain readable. If any shard fails, new partial files are removed
and the previous manifest/checkpoint is preserved. After a successful save,
shards belonging to the preceding manifest are removed to retain latest-only
semantics. Full and model-only sharded formats are unit-tested, but neither has
completed a VDI save while the effective quota is exhausted.

After synchronizing the sharded-checkpoint revision, validate it in a new output
directory:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 1 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_sharded_smoke1 `
  --checkpoint-every 1

.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 2 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --resume checkpoints\seismic_tapir_torch_vdi_sharded_smoke1\latest.pt `
  --output-dir checkpoints\seismic_tapir_torch_vdi_sharded_resume2 `
  --checkpoint-every 1
```

Both directories must contain `metrics.csv`, a small `latest.pt`, and one each
of `.model.pt`, `.optimizer.pt`, and `.training.pt`. The second command must
start at step 2. Attach the second run's `metrics.csv` for review.

After the save/resume gate passes, run the controlled discriminative-rate
ablation from the official checkpoint:

```powershell
$tapnetOutput = 'checkpoints\seismic_tapir_torch_vdi_encoder_lr01_overfit300'
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 300 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --encoder-lr-multiplier 0.1 `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir $tapnetOutput `
  --checkpoint-every 50
```

This is a fresh run, not a resume of the unstable shared-rate experiment. The
head schedule still peaks at `1e-4`; the encoder schedule peaks at `1e-5`.
Attach only `$tapnetOutput\metrics.csv` after completion or failure.

The first non-overfit training/inference experiment uses varying synthetic
batches, the stable frozen pretrained encoder, and one fixed held-out seed for
a before/after comparison. Refresh the non-PyTorch runtime packages first so
the new PNG writer dependency is present:

```powershell
.venv-torch\Scripts\python.exe -m pip install -r requirements_seismic_torch.txt
```

Evaluate the unadapted official checkpoint on exactly the held-out set that
will be reused after training:

```powershell
$baselineOutput = 'checkpoints\seismic_tapir_torch_vdi_baseline_eval_seed1000000'
.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_torch `
  --checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --config vdi-small `
  --output-dir $baselineOutput `
  --num-examples 32 `
  --seed 1000000 `
  --device cuda
```

Then run a 500-step pilot on 500 different deterministic synthetic examples.
Do not add `--overfit-one-batch`: this run measures adaptation rather than
memorization. The encoder remains frozen because full-encoder stability has not
yet passed the discriminative-rate ablation.

```powershell
$trainingOutput = 'checkpoints\seismic_tapir_torch_vdi_synthetic_pilot500'
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 500 `
  --device cuda `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir $trainingOutput `
  --checkpoint-every 100
```

Run inference on the identical held-out examples from the saved model:

```powershell
$inferenceOutput = 'checkpoints\seismic_tapir_torch_vdi_pilot500_eval_seed1000000'
.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_torch `
  --checkpoint "$trainingOutput\latest.pt" `
  --config vdi-small `
  --output-dir $inferenceOutput `
  --num-examples 32 `
  --seed 1000000 `
  --device cuda
```

Compare `$baselineOutput\summary.json` with
`$inferenceOutput\summary.json`, inspect every generated PNG, and retain both
`tracks.csv` files for query-level failure analysis. Improvement from one
500-step run is not guaranteed and is not a promotion result. The main gate is
lower held-out depth/lateral error without collapsing visibility recall. Do not
select or tune on this seed repeatedly; a second untouched synthetic test seed
must be introduced after the experiment design stabilizes.

### Harder second-stage synthetic curriculum

The 500-step frozen-encoder model improved 400-frame real forward continuity
but retained large lateral drift and was less cycle-consistent than the
baseline on the accidentally executed 8-frame cycle test. The next experiment
therefore changes difficulty and temporal coverage rather than simply extending
the original stream. `vdi-hard` adds multiple fault planes, increases the
cumulative throw budget from 6 to 12 samples, doubles training frames from 8
to 16, raises noise and termination frequency, increases queries from 16 to
24, and gives fixed-lateral error the same loss weight as depth error.

Start a new optimizer and scheduler from the successful 500-step model weights.
This is intentionally `--pretrained-checkpoint`, not `--resume`, because the
data configuration, loss, and schedule changed. The feature encoder remains
frozen; do not pass `--train-feature-encoder`.

First evaluate the existing 500-step checkpoint on the untouched hard stream;
this is the before measurement for the new curriculum:

```powershell
$hardBaseline = 'checkpoints\seismic_tapir_torch_vdi_frozen500_hard_eval_seed2000000'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_torch `
  --checkpoint checkpoints\seismic_tapir_torch_vdi_pretrained_frozen500\latest.pt `
  --config vdi-hard `
  --output-dir $hardBaseline `
  --num-examples 32 `
  --seed 2000000 `
  --device cuda
```

```powershell
$hardOutput = 'checkpoints\seismic_tapir_torch_vdi_hard5000'

.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-hard `
  --steps 5000 `
  --device cuda `
  --pretrained-checkpoint checkpoints\seismic_tapir_torch_vdi_pretrained_frozen500\latest.pt `
  --output-dir $hardOutput `
  --checkpoint-every 500
```

Expected startup includes `feature_encoder=frozen`, `precision=bfloat16`, and a
`source_checkpoint` pointing at the 500-step model in `metrics.csv`. Full
sharded checkpoints remain resumable and replace the previous generated shard
set at each save. Resume an interrupted run without changing its configuration:

```powershell
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-hard `
  --steps 5000 `
  --device cuda `
  --resume "$hardOutput\latest.pt" `
  --output-dir $hardOutput `
  --checkpoint-every 500
```

Evaluate on a hard held-out stream before real-data comparison:

```powershell
$hardEval = 'checkpoints\seismic_tapir_torch_vdi_hard5000_eval_seed2000000'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_torch `
  --checkpoint "$hardOutput\latest.pt" `
  --config vdi-hard `
  --output-dir $hardEval `
  --num-examples 32 `
  --seed 2000000 `
  --device cuda
```

Do not compare raw training loss directly with the easier run: the sample
distribution and lateral loss weight changed. Promotion requires comparing the
`$hardBaseline\summary.json` and `$hardEval\summary.json`, then comparing the
500-step and hard checkpoints on the original held-out `vdi-small` set and
repeating the identical 400-frame forward and cycle commands. The hard run
should reduce lateral drift and cycle error without degrading depth continuity
or visibility.

The harder generator still omits non-planar and listric faults, fault shadows,
unconformities, stratigraphic pinch-outs beyond simple termination masks, salt,
multiples, acquisition footprints, phase rotations, survey-specific wavelets,
and real amplitude statistics. Multiple synthetic faults share a total throw
budget so the 128-sample window remains valid. Adding every omitted effect at
once would make failures uninterpretable; subsequent additions should be
driven by held-out and real-data error modes.

### 64-frame temporal curriculum

The hard 16-frame model used only 1.3869 GiB of PyTorch CUDA allocation on the
L40-12Q, while the real application asks for 400 frames. Training and synthetic
evaluation therefore accept `--num-frames N` as an explicit override. The next
curriculum uses 64 frames: four times the hard training horizon and one sixth of
the real sweep. Jumping directly to 400 is intentionally avoided until memory,
runtime, optimization, and held-out behavior are measured at 64.

Run one complete forward/backward/checkpoint step first. This is a memory and
plumbing gate, not training evidence:

```powershell
$hardOutput = 'S:\Seismic\User\dadel\tapnet\checkpoints\seismic_tapir_torch_vdi_hard5000'
$longSmoke = 'S:\Seismic\User\dadel\tapnet\runs\seismic_tapir_torch_vdi_long64_smoke1'

.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-hard `
  --num-frames 64 `
  --steps 1 `
  --device cuda `
  --pretrained-checkpoint "$hardOutput\latest.pt" `
  --output-dir $longSmoke `
  --checkpoint-every 1 `
  --checkpoint-mode model-only
```

Proceed only if the step is finite and `peak_cuda_memory_gib` leaves safe room
below the VDI's 11 GiB dedicated limit. Approximately 8 GiB or less is the
initial operational gate; this is deliberately conservative because other VDI
processes and non-PyTorch CUDA allocations are not included in PyTorch's peak.
If it exceeds that gate or fails with CUDA OOM, repeat at 32 frames and record
the failure rather than reducing other dimensions silently.

Before training, evaluate the hard-5,000 checkpoint on an untouched 64-frame
stream so the new stage has an exact baseline:

```powershell
$longBaseline = 'S:\Seismic\User\dadel\tapnet\runs\seismic_tapir_torch_vdi_hard5000_long64_eval_seed3000000'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_torch `
  --checkpoint "$hardOutput\latest.pt" `
  --config vdi-hard `
  --num-frames 64 `
  --output-dir $longBaseline `
  --num-examples 32 `
  --seed 3000000 `
  --device cuda
```

If both gates pass, start a fresh 3,000-step optimizer/scheduler from the hard
weights. This is not an exact resume because sequence length changes:

```powershell
$longOutput = 'S:\Seismic\User\dadel\tapnet\checkpoints\seismic_tapir_torch_vdi_long64_3000'

.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-hard `
  --num-frames 64 `
  --steps 3000 `
  --device cuda `
  --pretrained-checkpoint "$hardOutput\latest.pt" `
  --output-dir $longOutput `
  --checkpoint-every 500
```

The feature encoder remains frozen and the hard fault/noise/lateral-loss
settings are retained. An interrupted long run must include both
`--config vdi-hard` and `--num-frames 64` when resumed, or its checkpoint
signature will correctly reject the mismatch. After training, evaluate the
same seed with the long checkpoint and then repeat the identical real 400-frame
forward/cycle test. Promotion requires better 64-frame held-out results and
lower real cycle/lateral error without degrading forward depth continuity.

### Temporal multi-stride first pass

An alternative to feeding 64 or more consecutive traces is to form three
aligned views of the same underlying seismic neighborhood. Each view contains
32 frames, sampled at survey-line strides 1, 2, and 4. Their respective spans
are 32, 63, and 125 survey lines when both endpoints are counted. This is a
good fit for the existing TAPIR implementation if the three views are run as
separate, uniformly sampled sequences. It is not safe to concatenate them into
one 96-frame sequence: the refinement network applies one-dimensional
convolutions along the frame axis and is given frame order, but no physical
survey-line coordinate or frame-spacing value. It would therefore treat a
one-line step, a two-line step, and a four-line step as identical adjacent
steps. Duplicate and discontinuously ordered lines would add a second
ambiguity.

There is room for the separate-view approach without changing pretrained model
weights. The feature encoder operates on frames independently, the initial
cost volume matches the query feature against every frame, and the model accepts
a variable number of frames. The existing `pyramid_level` setting is an image
height/width feature pyramid and must not be confused with temporal or
survey-line stride. Temporal coupling happens later in the PIPs mixer, so each
individual view must retain monotonic, constant survey-line spacing.

The recommended first prototype is deliberately outside the TAPIR core:

1. Generate one high-resolution synthetic scene covering at least 125 lines.
2. Derive the stride-1, stride-2, and stride-4 views from that same rendered
   scene, including targets, visibility, query-frame remapping, and a shared
   amplitude normalization scale.
3. During training, select one of the three strides per example with a balanced
   distribution. This exposes the unchanged model to all scales while keeping
   memory close to one 32-frame run. Do not hold three backward graphs in memory.
4. During inference, run the three views separately and map every prediction
   back to its actual survey-line index. At lines shared by multiple views,
   measure scale disagreement before any fusion.
5. Use a coarse prediction as an anchor for a consecutive-trace local window,
   then obtain the final dense track from the local window. Confidence-weighted
   averaging alone is not sufficient because the 400-frame experiment showed
   that high apparent trackability can coexist with poor lateral cycle return.

A minimal fusion rule should prefer the consecutive-trace result where it
exists, use the sparse view to extend coverage, and reject or flag locations
where depth, lateral position, polarity, or forward/backward cycle checks
disagree. A robust median or confidence-weighted depth estimate can be tested
only at overlapping physical lines; lateral identity should not be averaged
across clearly different events. Near faults, stride 4 can skip the transition
entirely, so cross-scale disagreement is useful evidence rather than noise to
smooth away.

This design does not by itself cover a 400-line sweep. A 32-frame stride-4 view
spans only 125 lines. Full coverage still requires overlapping coarse windows,
a larger stride, or hierarchical reseeding. The first experiment should stop at
125 lines so the effect of scale can be isolated before adding window stitching.

The first training-side implementation is `--config vdi-multistride`. The
generator renders one 125-line scene and then extracts one 32-frame view at
stride 1, 2, or 4. Strides cycle exactly in that order, remain deterministic
across resume, and are recorded as `temporal_stride` in every metrics row.
`scene_num_frames` and the exact physical `frame_indices` are retained in the
generated sample. All views from a forced common seed share the same geology
and amplitude normalization.

The first pass deliberately does not change the TAPIR core, run all three
views in one training step, fuse predictions, or add a consistency loss. Real
multi-stride inference is implemented separately in
`tapnet.seismic.infer_zgy_peaks_torch_multi_stride`: it reads one bounded dense
block, constructs aligned stride views with identical source peaks and shared
normalization, and exports disagreement rather than averaging across faults.

Use the hard 5,000-step checkpoint as the preferred initialization when it is
available. Otherwise the official BootsTAPIR checkpoint is a valid plumbing
fallback, but it does not isolate the effect of multi-stride training from the
earlier seismic curriculum.

Run a one-step CUDA gate first:

```powershell
$python = 'C:\Users\adelv\Documents\venvs\seismic-pytorch\Scripts\python.exe'
$sourceCheckpoint = 'checkpoints\seismic_tapir_torch_vdi_hard5000\latest.pt'
$smokeOutput = 'checkpoints\seismic_tapir_torch_vdi_multistride_smoke1'

& $python -m tapnet.seismic.train_torch `
  --config vdi-multistride `
  --steps 1 `
  --device cuda `
  --pretrained-checkpoint $sourceCheckpoint `
  --output-dir $smokeOutput `
  --checkpoint-every 1 `
  --checkpoint-mode model-only
```

Confirm a finite loss, `feature_encoder=frozen`, `precision=bfloat16`, and safe
peak CUDA memory. Then start the 3,000-step run with a new optimizer:

```powershell
$multiOutput = 'checkpoints\seismic_tapir_torch_vdi_multistride3000'

& $python -m tapnet.seismic.train_torch `
  --config vdi-multistride `
  --steps 3000 `
  --device cuda `
  --pretrained-checkpoint $sourceCheckpoint `
  --output-dir $multiOutput `
  --checkpoint-every 500
```

Resume without changing the config or total-step target:

```powershell
& $python -m tapnet.seismic.train_torch `
  --config vdi-multistride `
  --steps 3000 `
  --device cuda `
  --resume "$multiOutput\latest.pt" `
  --output-dir $multiOutput `
  --checkpoint-every 500
```

Evaluate each stride separately on the same untouched seed:

```powershell
foreach ($stride in 1, 2, 4) {
  & $python -m tapnet.seismic.infer_torch `
    --checkpoint "$multiOutput\latest.pt" `
    --config vdi-multistride `
    --frame-stride $stride `
    --output-dir "${multiOutput}_eval_stride${stride}_seed4000000" `
    --num-examples 32 `
    --seed 4000000 `
    --device cuda
}
```

### Large-fault multi-stride fine-tuning

`vdi-fault-robust` keeps 32-frame stride-1/2/4 sampling but expands the model
depth crop from 128 to 256 samples. The three views are centered on the same
physical source line. Each faulted scene contains one major 4--80-sample throw
without dividing the throw by fault count, a 1--8-line reflector-free damage
core labelled occluded, and a visible displaced continuation. Seventy-five
percent of queries are drawn from horizon/lateral pairs that cross that core
when such pairs are available. Twenty percent of scenes remain fault-free.

Start a new optimizer from the encoder-fine-tuned multi-stride weights; do not
use `--resume` because the data and input resolution changed:

```powershell
$python = '.\.venv-torch\Scripts\python.exe'
$sourceCheckpoint = 'S:\Seismic\User\dadel\tapnet\checkpoints\seismic_tapir_torch_vdi_multistride_encoderft3000\latest.pt'
$faultOutput = 'S:\Seismic\User\dadel\tapnet\checkpoints\seismic_tapir_torch_vdi_faultrobust5000'

& $python -m tapnet.seismic.train_torch `
  --config vdi-fault-robust `
  --steps 5000 `
  --pretrained-checkpoint $sourceCheckpoint `
  --output-dir $faultOutput `
  --train-feature-encoder `
  --encoder-lr-multiplier 0.1 `
  --checkpoint-every 250 `
  --device cuda
```

The 256-by-128 full-encoder CUDA smoke used 2.664 GiB peak allocated memory.
Real-data inference from this checkpoint must use `--config vdi-fault-robust`
and should initially keep `--frames-per-view 32`; increasing inference to 128
frames changes horizontal coverage but not the trained vertical displacement
distribution.

Run aligned stride-1/2/4 inference on one real source trace:

```powershell
$cube = 'D:\data\survey.zgy'
$multiOutput = 'S:\Seismic\User\dadel\tapnet\checkpoints\seismic_tapir_torch_vdi_multistride3000'
$realMulti = 'S:\Seismic\User\dadel\tapnet\runs\real_zgy_multistride_inline2391'

& $python -m tapnet.seismic.infer_zgy_peaks_torch_multi_stride `
  --input $cube `
  --checkpoint "$multiOutput\latest.pt" `
  --output-dir $realMulti `
  --config vdi-multistride `
  --frames-per-view 32 `
  --frame-strides 1 2 4 `
  --sweep crossline `
  --coordinates annotation `
  --query-inline 2391 `
  --query-crossline 880 `
  --peak-polarity both `
  --peak-relative-threshold 0.1 `
  --peak-min-distance 4 `
  --agreement-depth-tolerance 2 `
  --agreement-lateral-tolerance 2 `
  --agreement-min-trackability 0.5 `
  --rebase-agreed `
  --rebase-min-distance 4 `
  --cycle-consistency `
  --device cuda
```

The output contains `tracks_stride1.csv`, `tracks_stride2.csv`,
`tracks_stride4.csv`, `cross_scale_disagreement.csv`,
`multi_resolution_agreement.csv`, `predictions.npz`,
`track_curtain_multistride.png`, and `summary.json`. Agreement requires every
stride at a shared physical line to pass the depth, lateral, and trackability
thresholds. Green plot markers show agreement. A rebase anchor is selected only
at the outer edge of an unbroken agreement path from the original source; an
isolated agreement beyond a failure is not eligible. With `--rebase-agreed`,
the command centers one new multi-stride view on each selected anchor and writes
`rebase_anchors.csv` plus `rebased_tracks.csv`. Gold stars show those anchors.
This is one rebase hop, not recursive propagation. Cycle CSVs are written per
stride when requested. `summary.json` explicitly records `fusion.performed` as
false; cross-scale agreement is a gate for reseeding, not yet an accepted
horizon or accuracy measurement.

Before promotion, compare three controlled variants on identical held-out scene
seeds: the existing consecutive-frame model, a model trained with balanced
random stride, and three-view inference using that model. Report results
separately for strides 1, 2, and 4 in actual survey-line units, split depth and
lateral errors by distance to a fault, record cross-scale disagreement, and
repeat the real forward/backward cycle test. The multi-stride prototype succeeds
only if it reduces long-range lateral cycle error without degrading dense
stride-1 depth continuity. GPU headroom is not the deciding constraint for this
prototype; identity preservation across scales is.

Run a real seeded-track smoke after choosing a visible reflector in a seismic
viewer. Annotation mode expects the actual inline number, crossline number, and
Z header coordinate (for example TWT if that is how the cube is authored):

```powershell
$cube = 'D:\data\survey.zgy'
$checkpoint = 'checkpoints\seismic_tapir_torch_vdi_synthetic_pilot500\latest.pt'
$realOutput = 'checkpoints\real_zgy_inline_seed001'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_zgy_torch `
  --input $cube `
  --checkpoint $checkpoint `
  --output-dir $realOutput `
  --config vdi-small `
  --sweep inline `
  --coordinates annotation `
  --query-inline 43332 `
  --query-crossline 39734 `
  --query-z 884 `
  --device cuda
```

Replace the example seed values with a pick from the target cube. If only
zero-based voxel indices are known, pass `--coordinates index` and provide
inline index, crossline index, and sample index through the same three query
arguments. Repeat with `--sweep crossline` into a different output directory.
The two tracks should meet at the seed and should be reviewed together; the CLI
does not yet calculate their disagreement automatically.

To select and track all configured extrema on one trace, omit `--query-z` and
use the multi-seed command:

```powershell
$peakOutput = 'checkpoints\real_zgy_inline_trace_peaks001'

.venv-torch\Scripts\python.exe -m tapnet.seismic.infer_zgy_peaks_torch `
  --input $cube `
  --checkpoint $checkpoint `
  --output-dir $peakOutput `
  --config vdi-small `
  --sweep inline `
  --coordinates annotation `
  --query-inline 43332 `
  --query-crossline 39734 `
  --peak-polarity both `
  --peak-relative-threshold 0.1 `
  --peak-min-distance 4 `
  --device cuda
```

Run the same trace with `--sweep crossline` into a separate directory. Use
`--max-peaks` only when a deliberate compute cap is needed; if used, the
strongest accepted extrema are retained and the cap is recorded. Raising the
relative threshold or minimum distance changes the scientific seed-selection
policy and must be treated as an experiment parameter, not an invisible
performance shortcut.

If the full gate remains blocked after removing obsolete artifacts, test whether
the quota is specific to `Documents` using a model-only output under local app
data:

```powershell
$tapnetOutput = Join-Path $env:LOCALAPPDATA 'tapnet-checkpoints\model-only-smoke1'
.venv-torch\Scripts\python.exe -m tapnet.seismic.train_torch `
  --config vdi-small `
  --steps 1 `
  --device cuda `
  --overfit-one-batch `
  --train-feature-encoder `
  --checkpoint-mode model-only `
  --pretrained-checkpoint checkpoints\pretrained\bootstapir_checkpoint_v2.pt `
  --output-dir $tapnetOutput `
  --checkpoint-every 1
```

This command intentionally includes the official pretrained checkpoint. The
failed command ending in `_new` omitted both `--pretrained-checkpoint` and
`--resume`, so it started from random weights and is not evidence for the
planned pretrained encoder ablation.

Optimization was also unstable as the shared learning rate reached `1e-4`:
total loss was 0.503898 at step 76 but rose to 5.181211 at step 100, with a
pre-clipping gradient norm of 1328.516235. Do not resume the trainable-encoder
run unchanged after fixing storage. A lower encoder learning rate or separate
encoder/head parameter-group rates must be tested first.

Create a visual sample:

```bash
python -m scripts.preview_synthetic_seismic \
  --output docs/synthetic_seismic_preview.png \
  --seed 5
```

Run the CPU integration smoke test:

```bash
python -m tapnet.training.experiment \
  --config=configs/seismic_tapir_config.py:smoke \
  --jaxline_mode=train

python -m tapnet.training.experiment \
  --config=configs/seismic_tapir_config.py:smoke \
  --jaxline_mode=eval_seismic_synthetic
```

Run the default synthetic experiment only after completing the GPU and
single-batch-overfit gates below:

```bash
python -m tapnet.training.experiment \
  --config=configs/seismic_tapir_config.py \
  --jaxline_mode=train
```

Checkpoints are written beneath `checkpoints/`, which is intentionally ignored
by Git because even a smoke checkpoint is approximately 373 MB.

## Validation record

Validated on 2026-10-05 for JAX, 2026-10-06 for PyTorch, and 2026-10-07 for
the real-ZGY comparison and cycle-consistency setup:

- Upstream revision: `730cda1c730877cfedbe01bf87fb1cadb78a565d`.
- Python 3.10 workspace-local virtual environment.
- JAX 0.6.2 CPU, JAXlib 0.6.2, TensorFlow 2.21.0.
- Final combined `python -m pytest -q` after adding configurable real-ZGY
  sweep length, cycle consistency, the hard curriculum, and synthetic temporal
  overrides: **67 passed**.
- Python bytecode compilation: passed.
- Import of seismic config and `tapnet.training.experiment`: passed without
  Kubric installed.
- Visual inspection of `docs/synthetic_seismic_preview.png`: horizon labels,
  a fault displacement, and terminated labels align with the rendered data.
- Smoke training: one complete forward/backward/update step, 31.07M parameters,
  finite loss and gradients, checkpoint saved.
- Smoke evaluation: two deterministic examples, seismic metrics returned, and
  one-off evaluator exited normally.
- PyTorch 2.14.1+cpu on Python 3.10.11: all data, loss, configuration, resume
  sequence, and full-model autograd tests passed.
- `vdi-hard` generation was sampled for 100 deterministic seeds: fault counts
  were 32 one-fault, 27 two-fault, and 41 three-fault examples; every tested
  horizon set remained ordered without crossings. A complete sample had shape
  `[16, 128, 128, 3]` with 24 queries. This validates generator/configuration
  plumbing, not a CUDA training step or geological realism.
- PyTorch/JAX seismic loss values match on fixed randomized tensors within
  `1e-6` relative and absolute tolerance.
- The upstream in-place residual addition was reproduced as an autograd failure,
  changed to an out-of-place equivalent, and verified by the full TAPIR backward
  test.
- PyTorch random-weight smoke: one 64-by-64 forward/backward/update step,
  checkpoint save, and resumed second step completed on CPU.
- Official `bootstapir_checkpoint_v2.pt`: 218,886,140 bytes, strict state load
  succeeded with all keys matching the 54,699,335-parameter PyTorch model.
- Native PyTorch inference completed end to end on CPU for one deterministic
  `vdi-small` example from the official checkpoint. It wrote `summary.json`,
  128 long-form CSV rows, compressed lossless arrays, and an inspected seismic
  curtain PNG. Depth MAE was 4.9537 samples and visibility recall was 0.1681;
  this one-example dependency/plumbing result is not a benchmark. Tests also
  verify masked metrics, CSV semantics, the TAPIR visibility rule, and loading
  the model shard without the optimizer shard.
- Real ZGY inference completed in both inline and crossline directions on
  `LowCretClino_ajax_2Deriv_phaserot_ELC_ajax.zgy`, a local
  845-by-559-by-195 cube with 2-by-2 line annotation increments, `zstart=496`,
  and `zinc=4`. The annotation seed `[43332, 39734, 884]` mapped exactly to
  index `[422, 279, 97]`. Each direction performed one bounded read, produced
  CSV/NPZ/JSON/PNG artifacts, and preserved cube indices, annotations, Z, and
  world X/Y. Inline and crossline trackability ranges were 0.9848--0.9899 and
  0.9978--0.9997; maximum predicted lateral drift from the seed was 1.395 and
  1.337 traces. These are stability diagnostics from the unadapted official
  checkpoint, not ground-truth accuracy or evidence of real-data readiness.
  The independent world-coordinate interpolation matched
  `ZgyReader.indexToWorld` exactly for a fractional predicted location.
- Multi-seed inference on the same real trace selected 29 positive/negative
  extrema using the documented 0.1 relative threshold and four-sample spacing.
  Three overlapping depth windows produced 232 track rows in each sweep. At
  the source trace, inline/crossline predicted-depth disagreement averaged
  0.0775 samples and reached 0.6079 samples; mean source-frame residual from
  the selected peak was 0.0947 samples inline and 0.0478 samples crossline.
  Mean trackability was 0.9966 inline and 0.9992 crossline, while maximum
  lateral drift reached 4.052 and 4.387 traces. This validates execution and
  exposes drift; without interpreted targets it does not validate correctness.
- User-provided VDI artifacts inspected on 2026-10-07 contained a separate
  500-step varying-synthetic training run and a 20-seed, 256-frame real-ZGY
  crossline sweep. The training run was finite throughout and reduced mean
  total loss from 17.2311 over its first 50 steps to 8.6947 over its last 50,
  but its CSV records an empty `source_checkpoint`, a trainable encoder, and
  only 8 training frames. It therefore trained all weights from random
  initialization rather than adapting BootsTAPIR. The real sweep contained
  5,120 predictions; mean trackability was 0.2737 and only 10.76% exceeded the
  0.5 visibility threshold. Per-seed maximum adjacent-frame depth jumps ranged
  from 2.75 to 63.37 samples. The plot draws every low-confidence trajectory as
  a solid line and marks threshold-passing locations with dots, so the solid
  curves must not be read as accepted horizons. The tracks CSV does not record
  checkpoint provenance, so linkage between these two supplied artifacts
  remains unverified. This run is rejected as an accuracy result because it
  combines random initialization with a 32-fold train/inference frame-count
  mismatch (8 versus 256) and has no interpreted real-data target. Future tests
  must compare the official pretrained checkpoint and the adapted checkpoint
  on identical 8-frame and then 16/32-frame windows before trying 256 frames.
- A follow-up real-ZGY crossline curtain and tracks CSV supplied on 2026-10-07
  were much more stable. The run used 20 seeds over 400 frames (8,000 rows,
  crossline annotations 82--481). Mean/median trackability were 0.8101/0.9396,
  85.66% of predictions exceeded 0.5, and the largest adjacent-frame depth
  jump was 5.94 samples, compared with 0.2737 mean trackability, 10.76%
  accepted, and a 63.37-sample jump in the rejected random-weight run. At the
  source frame, absolute depth residual averaged 0.2753 samples and reached
  0.7468. Most trajectories remained locally phase-consistent, although two
  deep tracks near 3,400--3,600 Z units make visible corrections at the left
  side and close tracks around 2,250--2,450 may switch or merge events.
  Predicted lateral displacement from the source trace remains material: the
  per-seed maximum averaged 30.25 traces and reached 55.64. The current PNG is
  consequently a projection of a 3D path onto the fixed-inline source curtain;
  it does not show the amplitude actually sampled at displaced lateral
  coordinates. The separately supplied 500-step training CSV correctly records
  `bootstapir_checkpoint_v2.pt` as its source, a frozen encoder, varying data,
  BF16, no non-finite metrics, and mean total loss decreasing from 5.1899 over
  the first 50 steps to 2.2022 over the last 50. The tracks CSV still lacks
  checkpoint provenance, so association with that training run cannot be
  proven without `summary.json`. There are still no interpreted targets; these
  are stability and confidence diagnostics, not real-data accuracy.
- A user-labeled official-checkpoint baseline tracks CSV was supplied for the
  same 20 seeds, but it used 256 frames over crossline annotations 154--409
  while the adapted run used 400 frames over 82--481. Across the common
  256-crossline interval, baseline versus adapted mean trackability was
  0.3803 versus 0.8479, median trackability was 0.3013 versus 0.9577, and the
  fraction above 0.5 was 34.36% versus 89.32%. Mean per-seed maximum adjacent
  depth jump fell from 9.55 to 1.26 samples and mean per-seed maximum lateral
  drift fell from 54.89 to 22.50 traces; global maxima fell from 44.39 to 2.57
  samples and 118.87 to 40.61 traces respectively. Predictions from the two
  checkpoints differed by 1.83 depth samples on average, 1.18 at the median,
  5.66 at the 95th percentile, and 43.49 at the worst point. That worst point
  was seed 15 at crossline annotation 234, where both checkpoints assigned low
  trackability. At the source crossline, the baseline adhered more exactly to
  the supplied seed (0.1007 mean absolute sample residual versus 0.2753 for the
  adapted run), though both were sub-sample on average. This comparison favors
  the adapted model for continuity and model-reported confidence but is not a
  controlled accuracy result: temporal window lengths differ, the baseline CSV
  contains no checkpoint provenance, and synthetic fine-tuning may recalibrate
  confidence without improving geology. A definitive A/B must use identical
  `--num-frames`, inputs, seeds, normalization, and interpreted targets.
- The user then supplied a 400-frame version of the user-labeled baseline,
  enabling a frame-for-frame comparison with the adapted run: both contain the
  same 20 seeds, 8,000 rows, and crossline annotations 82--481. Baseline versus
  adapted mean trackability was 0.2876 versus 0.8101, median trackability was
  0.1555 versus 0.9396, and the fraction above 0.5 was 24.96% versus 85.66%.
  Mean per-seed maximum adjacent depth jump fell from 20.67 to 2.08 samples
  and the global maximum fell from 109.38 to 5.94. Mean per-seed maximum
  lateral drift fell from 69.31 to 30.25 traces and the global maximum from
  124.97 to 55.64. The baseline retained tighter source-frame anchoring
  (0.0989 mean and 0.2112 maximum absolute depth residual versus 0.2753 and
  0.7468), although both remain sub-sample at the source. Across all rows, the
  checkpoints differed by 2.74 depth samples on average, 1.28 at the median,
  6.34 at the 95th percentile, and 137.70 at the maximum. Where both models
  exceeded the 0.5 threshold (1,940 rows), depth disagreement averaged only
  1.37 samples. The adapted model alone exceeded the threshold on 4,913 rows;
  this large gain could represent improved tracking, synthetic-domain
  confidence recalibration, or both. The largest disagreement was seed 19 at
  crossline 113, where both scores were low (baseline 0.00008, adapted 0.0722).
  This controlled comparison establishes improved numerical continuity and
  model-reported trackability, not geological accuracy. Interpreted targets or
  independent inline/crossline consistency are still required to distinguish
  correct tracking from smooth, confident tracking of the wrong reflector.
- The first VDI cycle-consistency outputs completed for both the user-labeled
  baseline and trained checkpoints, validating the end-to-end cycle artifact
  path. Both CSVs contain endpoint frames 0 and 7, so they are 8-frame tests,
  not the requested 400-frame test (which must contain endpoint frame 399).
  All 40 cycles per checkpoint were in bounds and confidence-valid. Baseline
  versus trained mean round-trip Euclidean error was 0.3737 versus 0.9859 model
  pixels; mean absolute lateral error was 0.2124 versus 0.6980 traces; mean
  absolute depth error was 0.2507 versus 0.5391 samples; and mean full-path
  disagreement was 0.3153 versus 0.7817 model pixels. The corresponding maxima
  were 0.8696 versus 2.3562 for round-trip error and 0.7508 versus 2.1608 for
  mean path disagreement. The trained model was less cycle-consistent on this
  local 8-frame test despite being substantially smoother and more confident
  on the earlier 400-frame forward run. This exposes a real tradeoff or
  calibration change that must be tested again over the intended 400 frames.
  A correct rerun must include `--num-frames 400`, and the resulting CSV must
  show `endpoint_frame=399` for every `endpoint=end` row.
- The `vdi-hard` 5,000-step VDI curriculum completed on 2026-10-07 from
  `seismic_tapir_torch_vdi_pretrained_frozen500/latest.pt` with the feature
  encoder frozen, varying examples, CUDA BF16, and all intended hard settings
  recorded in `metrics.csv`. All 5,000 steps were present and every logged loss
  and gradient was finite. Mean total loss decreased monotonically by broad
  blocks: 4.2364 (steps 1--500), 3.7914 (501--1,000), 3.4321
  (1,001--2,000), 3.1406 (2,001--3,000), 2.9403 (3,001--4,000), and 2.7721
  (4,001--5,000). First/last 100-step means were 4.5305 and 2.7501; final-step
  loss was 2.3332. Peak CUDA allocation was 1.3869 GiB and elapsed time was
  0.4701 hours. Logged gradient norms are pre-clipping: all exceeded the 1.0
  limit, 247 exceeded 100, and the maximum was 1,599.26 at step 4,458. The
  configured global-norm clipping was applied before every optimizer step, and
  the isolated spike remained finite, but it reinforces that held-out metrics
  rather than training loss must decide promotion. At that stage no held-out
  or real-data result had yet been supplied for the checkpoint.
- Held-out evaluation of the step-5,000 hard checkpoint then completed on 32
  examples from hard seed 2,000,000 and easy seed 1,000,000. On `vdi-hard`,
  depth MAE/RMSE were 0.4877/1.1474 samples, 91.74%/97.26%/98.57% of valid
  positions were within 1/2/4 samples, gross errors above 8 samples were 0.51%,
  lateral MAE was 0.3814 traces, and visibility accuracy/F1 were
  0.9782/0.9864. On `vdi-small`, depth MAE/RMSE improved to 0.3560/0.8869,
  96.39%/98.87%/99.42% were within 1/2/4 samples, gross errors were 0.41%,
  lateral MAE was 0.3587 traces, and visibility accuracy/F1 were
  0.9849/0.9915. Mean trackability was 0.7882 hard and 0.8819 easy. These are
  strong absolute held-out synthetic results with no obvious easy-distribution
  collapse. A strict improvement claim still requires the earlier 500-step
  checkpoint evaluated on these exact same seeds/configurations; the supplied
  summaries only evaluate the hard checkpoint. At that point, real 400-frame
  forward and cycle behavior remained unmeasured.
- The hard step-5,000 checkpoint was subsequently evaluated on the same real
  20-seed, 400-crossline sweep with cycle consistency. The summary confirms
  endpoint frames 0/399, checkpoint step 5,000, `vdi-hard`, seven depth
  windows, and 40 attempted/in-bounds cycles. Forward mean/median trackability
  were 0.8515/0.9931 and 86.80% of 8,000 predictions exceeded 0.5. Compared
  with the earlier 500-step adapted run, mean per-seed maximum lateral drift
  improved from 30.25 to 23.56 traces, global maximum drift from 55.64 to
  45.81, mean per-seed maximum adjacent depth jump from 2.08 to 1.44 samples,
  and source-frame mean absolute depth residual from 0.2753 to 0.1072 samples.
  The global depth-jump maximum worsened from 5.94 to 7.45 samples, so the
  improvement is not uniform. Hard and 500-step depths differed by 1.71
  samples on average, 0.90 at the median, and 4.59 at the 95th percentile.
- Long-range cycle closure exposed the remaining blocker. Only 26 of 40 cycles
  were confidence-valid. Across all in-bounds cycles, mean round-trip lateral,
  depth, and Euclidean errors were 14.32 traces, 2.54 samples, and 15.28 model
  pixels; median Euclidean error was 13.13 and no cycle returned within two
  model pixels. Among confidence-valid cycles, depth closure was substantially
  better than lateral closure: 61.5% returned within one depth sample, 69.2%
  within two, and 92.3% within four, but mean lateral error remained 13.98
  traces. Large confidence-qualified failures reached 45.42 model pixels,
  demonstrating that visibility confidence does not establish point identity.
  The curtain remains visually coherent and forward metrics improved, but the
  single 400-frame call does not preserve lateral identity under endpoint
  reseeding. The next implementation should evaluate overlapping temporal
  windows near the 16-frame training horizon and propagate/stitch tracks in
  both directions before adding more synthetic training steps. This is a
  stability proposal, not yet implemented; it must be compared with the same
  one-shot output and cycle metrics.
- Pretrained constrained smoke: one 128-by-128, 8-frame, 16-query update with
  the feature encoder frozen completed on CPU and saved a checkpoint.
- That pretrained step reported a pre-clipping gradient norm of 1110.08; the
  configured global-norm limit of 1.0 was applied. This is a warning to monitor
  loss scaling and stability on CUDA, not evidence that the chosen learning rate
  is safe.
- Checkpoint state restoration and deterministic continuation at the next sample
  index completed. Exact uninterrupted-versus-resumed numerical parity has not
  yet been measured.
- A 20-step fixed-batch CPU sanity run reduced total deeply supervised loss from
  3.061385 to 1.707801. This demonstrates short-horizon learning but is not the
  near-zero overfit result required for model promotion.
- Five generated CPU validation checkpoints totaling 2,821,188,215 bytes were
  deleted after their save/resume behavior was verified. They are reproducible
  with the commands above. The 218,886,140-byte official pretrained checkpoint
  remains under the Git-ignored `checkpoints/pretrained/` directory.
- A pretrained frozen-encoder checkpoint resumed for a second update with the
  encoder still frozen and the compatible optimizer state restored. Changing
  encoder trainability during resume is rejected; it requires a new run.
- The precision-safety revision passed a fresh CPU trainer step with finite
  gradient norm and no scheduler warning. A regression test injects NaN
  gradients and verifies that checkpoint-producing training is rejected before
  the optimizer step.
- A 100-step pretrained, frozen-encoder, fixed-batch BF16 run completed on the
  Windows L40-12Q without non-finite values. Total deeply supervised loss fell
  from 12.852102 to 0.775462; final position loss fell from 1.043687 to
  0.005962; and final occlusion/probability losses fell to approximately zero.
  The remaining total is attributed to the four supervised unrefined outputs,
  which are not yet included in console logging. The checkpoint was saved to
  `checkpoints/seismic_tapir_torch_vdi_bf16_overfit100/latest.pt`, and PyTorch
  reported 0.874 GiB peak allocated CUDA memory.
- Resuming the same fixed batch through step 300 reduced total loss from
  0.775462 at step 100 to 0.480608, while final position loss reached 0.000027
  and both final classification losses rounded to zero. All gradients remained
  finite, `latest.pt` was saved, and peak allocated CUDA memory was 0.876 GiB.
  The remaining approximately 0.480581 loss belongs to unprinted intermediate
  outputs, and the learning rate had reached approximately `1e-6`.
- Intermediate diagnostics now report the aggregate unrefined loss and the
  position, occlusion, probability, and total loss for each unrefined stage.
  Tests verify exact loss accounting and the console format. This logging-only
  revision passed the full 33-test suite and retains checkpoint compatibility.
- The step-301 diagnostic attributed 0.480773 of 0.480867 intermediate loss to
  stage 0 (99.98%). Stage-0 probability loss was 0.370307, position loss was
  0.110461, and occlusion loss was 0.000005. Stages 1 through 3 and the final
  output were all effectively zero. The checkpoint saved successfully and peak
  allocated CUDA memory remained 0.876 GiB.
- A fresh trainable-encoder BF16 update from the official checkpoint completed
  with finite loss and a finite pre-clipping gradient norm of 5298.271973. The
  norm was clipped to 1.0, the checkpoint saved successfully, and peak allocated
  CUDA memory was 1.252 GiB. This passes the execution/memory gate but is not
  evidence of stable full-encoder optimization beyond one step.
- The trainable-encoder continuation remained finite through step 100, but loss
  rose to 5.181211 as the learning rate reached `1e-4`, and the step-100
  checkpoint failed with an iostream write error. The run is not a successful
  overfit result. A retry failed at the same byte offset at step 50 despite
  1858.483 GiB reported free. CSV append/flush and atomic sharded checkpoint
  save/load/replacement/failure behavior are now regression-tested; the full
  suite contains 36 passing tests. The sharded format is not yet VDI-validated.
- A sharded VDI retry failed with explicit `Errno 28` despite 1858.479 GiB
  volume-level free space. Partial new shards were removed and the preceding
  checkpoint was preserved. Full/model-only manifest behavior, parameter-only
  loading, and rejection of model-only exact resume are covered by the final
  37-test suite. The quota/storage-filter cause remains external and unresolved.
- A model-only save under `%LOCALAPPDATA%` also failed after approximately
  4.1 MB, while its 2,163-byte CSV succeeded. This establishes an account/profile
  storage blocker. Training computation can run, but durable model training is
  blocked until quota is reclaimed or increased.
- After reclaiming approximately 3.65 GiB of obsolete generated checkpoints, a
  full sharded step-1 checkpoint saved successfully. Loss and gradients were
  finite and peak allocated CUDA memory was 1.252 GiB. VDI shard writing is
  validated; the step-2 load/resume result is still required.
- The step-2 sharded resume restored model, optimizer, scheduler, and fixed-batch
  state, appended CSV metrics, and saved a replacement checkpoint. Loss was
  12.688643, the pre-clipping gradient norm was finite at 7571.959961, and peak
  allocated CUDA memory was 1.319 GiB. Full VDI save/resume is validated.
- Discriminative encoder/head learning-rate groups and multiplier validation are
  covered by tests. The final suite contains 39 passing tests; the `0.1` encoder
  multiplier has not yet been run on the VDI.

The smoke evaluation followed a single random-weight update. Its numerical
accuracy is intentionally not recorded as a benchmark because it provides no
evidence of learning or generalization.

## Intentionally omitted or not yet validated

These omissions are material and must not be inferred as implemented:

1. **No useful model has been trained.** The default 20,000-step configuration
   has not been run on a GPU, tuned, or shown to converge.
2. **The complete one-batch overfit gate is partially passed.** On CUDA BF16,
   the pretrained frozen-encoder model drives the final-head losses near zero,
   and stages 1 through 3 also reach near-zero loss. Stage 0 alone retains
   `0.480773`, primarily expected-distance loss. A trainable-encoder overfit
   ablation is required before changing supervision or loss weights.
3. **Pretrained CUDA optimization is validated, not generalization.** The
   official PyTorch BootsTAPIR checkpoint loads strictly and completed 100
   frozen-encoder BF16 updates. The held-out evaluator is implemented, but no
   model trained on varying synthetic batches has been evaluated on the VDI.
4. **The real-volume path is only a seeded ZGY smoke.** Bounded local ZGY reads,
   survey-coordinate restoration, inline/crossline inference, and multi-seed
   batching for extrema on one trace exist. SEG-Y, NumPy cube and
   interpreted-horizon ingestion, external seed tables, spatial multi-trace
   seeding, survey-level normalization, surface fusion, and real-data split
   logic do not exist yet. ZFP-compressed and cloud-hosted ZGY input are not
   validated.
5. **No full seismic forward model.** The procedural renderer does not model
   velocity, illumination, multiples, diffractions, migration artifacts,
   acquisition footprint, nonstationary wavelets, arbitrary phase rotation,
   anisotropic sampling, or elastic effects.
6. **Simplified faults and terminations.** Fault throw is shared by the horizon
   family and reflector termination is a simple planar mask. Fault drag,
   damage zones, growth faults, unconformity erosion, pinch-outs, and ambiguous
   cross-fault correlation are absent.
7. **No hard lateral constraint.** Lateral drift is penalized but TAPIR can still
   predict it. Masking the cost volume to a vertical corridor or replacing the
   2D head with a 1D depth head has not been tested.
8. **No surface-level consistency.** Tracks are trained independently. There
   is no inline/crossline intersection loss, forward/reverse consistency,
   horizon-ordering loss, dense surface fusion, or topology constraint.
9. **No baseline comparison.** Constant-depth, adjacent-trace correlation,
   dynamic programming, and dip-guided propagation baselines are not yet
   implemented.
10. **No self-supervised adaptation.** Teacher/student consistency and
    high-confidence pseudo-label training are future work.
11. **No physical-unit accuracy metrics.** Real outputs include header Z and
    world X/Y, but no interpreted real target exists from which to calculate
    errors in milliseconds or metres.
12. **GPU validation is limited to one VDI.** Single-device CUDA BF16/FP32
    updates work on the L40-12Q. Throughput, multi-GPU reduction, long-run memory
    behavior, and exact numerical restart parity have not been tested.
13. **No prestack support.** The current input represents one scalar post-stack
    amplitude repeated into three channels.
14. **No data-volume caching/profile.** Synthetic generation is online and has
    not been profiled against GPU consumption at the default settings.

## Required next tests

### P0: establish that the prototype can learn

1. Start a fresh pretrained fixed-batch run with head rate `1e-4` and encoder
   multiplier `0.1`. Compare its stage-0 curve and stability against the failed
   shared-rate run. Do not resume the unstable shared-rate checkpoint or alter
   stage-0 loss weights.
2. Run the default tensor shapes for several steps on the target GPU and record
   peak memory, compile time, and examples/second.
3. Train on a small fixed synthetic train/validation split and plot every loss
   and seismic metric versus step.
4. Visually inspect predicted tracks, including false-confidence examples,
   faults, sequence reversal, and terminations.
5. Implement constant-depth and local cross-correlation baselines. Learned
   tracking must beat both on held-out synthetic seeds.
6. Verify checkpoint restart reproduces the next-step loss within numerical
   tolerance.

### P1: improve synthetic realism and test design choices

1. Add phase rotation, wavelet variation by depth, structured acquisition
   noise, missing traces, multiples, diffractions, unconformities, and fault
   damage zones one effect at a time.
2. Maintain clean and stress-test validation suites rather than increasing all
   difficulty simultaneously.
3. Implement parameter-only loading from an upstream TAPIR checkpoint and
   compare random initialization, frozen-backbone warm-up, and full fine-tuning.
4. Compare repeated amplitude channels with a one-channel stem and selected
   seismic-attribute channels.
5. Ablate lateral loss weight and test a hard vertical search corridor.
6. Add inline/crossline intersection and reverse-direction consistency metrics
   before adding them as losses.
7. Calibrate predicted confidence against depth error and report
   accuracy-versus-coverage curves.

### P2: real-data transition

1. Retain the documented ZGY coordinate contract and define interpreted-horizon
   grid ingestion, SEG-Y/NumPy cube support, and their coordinate transforms.
2. Build survey-level train/validation/test splits; never randomly split
   overlapping neighboring patches.
3. Decide and document the geological policy at faults, unconformities, and
   horizon terminations before producing labels.
4. Preserve three states in real labels: trackable, genuinely untrackable, and
   unknown/unlabeled.
5. Fit normalization only on each training survey and audit phase, polarity,
   bandwidth, sample interval, and trace-spacing differences.
6. Evaluate inline and crossline sweeps independently and at their intersections.
7. Only after supervised real-data evaluation is stable, add conservative
   teacher/student pseudo-label adaptation.

## Promotion gates

Do not call the prototype successful until all of these are true:

- It overfits a fixed batch.
- It beats constant-depth and cross-correlation baselines on unseen synthetic
  seeds.
- Confidence is positively calibrated with actual depth error.
- Both sweep directions agree at intersections within a declared tolerance.
- It improves over classical propagation on a spatially or survey-held-out real
  test set.
- Failure cases around faults and terminations have been reviewed by a seismic
  interpreter under a written label policy.
