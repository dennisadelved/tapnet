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
  validation, TensorFlow wrapping, JAX loss masking/weighting, and finite loss
  gradients.

## Setup

Use Linux or a Linux GPU container for actual training. Native Windows JAX is
CPU-only; it is sufficient for the smoke test but not the default training run.

Create an isolated environment and install the correct accelerator-specific JAX
build first. One Linux/NVIDIA example is:

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

Validated on 2026-10-05:

- Upstream revision: `730cda1c730877cfedbe01bf87fb1cadb78a565d`.
- Python 3.10 workspace-local virtual environment.
- JAX 0.6.2 CPU, JAXlib 0.6.2, TensorFlow 2.21.0.
- `python -m pytest tests -q`: **17 passed**.
- Python bytecode compilation: passed.
- Import of seismic config and `tapnet.training.experiment`: passed without
  Kubric installed.
- Visual inspection of `docs/synthetic_seismic_preview.png`: horizon labels,
  a fault displacement, and terminated labels align with the rendered data.
- Smoke training: one complete forward/backward/update step, 31.07M parameters,
  finite loss and gradients, checkpoint saved.
- Smoke evaluation: two deterministic examples, seismic metrics returned, and
  one-off evaluator exited normally.

The smoke evaluation followed a single random-weight update. Its numerical
accuracy is intentionally not recorded as a benchmark because it provides no
evidence of learning or generalization.

## Intentionally omitted or not yet validated

These omissions are material and must not be inferred as implemented:

1. **No useful model has been trained.** The default 20,000-step configuration
   has not been run on a GPU, tuned, or shown to converge.
2. **No one-batch overfit test yet.** The current one-step smoke test only proves
   that execution and gradients work.
3. **No pretrained checkpoint initialization.** The current config initializes
   TAPIR randomly. Upstream released checkpoints and JAXline resume checkpoints
   have different practical roles; compatible parameter-only initialization
   must be implemented and verified before claiming transfer learning.
4. **No real-volume reader.** SEG-Y, ZGY, NumPy cube, horizon-grid ingestion,
   survey normalization, inline/crossline metadata, and real-data split logic do
   not exist yet.
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
11. **No physical-unit metrics.** Errors are in samples and traces, not
    milliseconds, metres, or survey coordinates.
12. **No distributed training validation.** Only a single CPU device smoke run
    was performed. GPU memory use, throughput, mixed precision, multi-GPU
    reduction, and checkpoint restart have not been tested.
13. **No prestack support.** The current input represents one scalar post-stack
    amplitude repeated into three channels.
14. **No data-volume caching/profile.** Synthetic generation is online and has
    not been profiled against GPU consumption at the default settings.

## Required next tests

### P0: establish that the prototype can learn

1. Add a finite, fixed synthetic training set and overfit one batch until depth
   error is near zero. Failure blocks all larger experiments.
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

1. Define the accepted cube and horizon-grid formats and coordinate transforms.
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
