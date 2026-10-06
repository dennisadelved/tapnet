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
- bias parameters excluded from weight decay to approximate the JAX optimizer;
- optional official PyTorch checkpoint initialization;
- default feature-encoder freezing when a pretrained checkpoint is supplied;
- atomic `latest.pt` checkpoints containing model, optimizer, scheduler,
  gradient-scaler, step, and configuration state;
- exact synthetic-stream continuation on resume;
- an explicit `--overfit-one-batch` learning sanity-check mode; and
- CUDA peak allocated-memory reporting at process exit.

The upstream PyTorch inference model used an in-place residual addition in
`tapnet/torch/nets.py`. It was changed to an equivalent out-of-place addition
because the in-place form invalidated tensors required by autograd. A full-model
forward/backward test protects this requirement.

`tapnet/seismic/torch_config.py` contains two explicit variants:

- `smoke`: 64-by-64, 2 frames, 2 queries, one refinement iteration, one step,
  random initialization, and plumbing validation only.
- `vdi-small`: 128-by-128, 8 frames, 16 queries, the checkpoint-compatible
  pyramid, and 2,000 planned steps. This is the initial L40-12Q experiment, not
  a tuned final configuration.

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

The supplied console capture ends immediately after the step-100 line. It does
not contain the expected `checkpoint=...`, `peak_cuda_memory_gib=...`, or a
returned PowerShell prompt. Checkpoint persistence and peak memory for this
specific run therefore remain unverified even though all 100 training updates
completed.

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

- no PyTorch evaluation command or seismic-metric report exists yet;
- the BF16 fixed-batch run drives the final-head losses near zero, but the
  aggregate intermediate-refinement loss remains non-zero and is not logged;
- exact parity with the JAX checkpoint/training trajectory is not expected;
- the current JAX/JAXline configuration cannot be reused directly;
- peak allocated VRAM for one-step runs is approximately 0.815 GiB, but the
  peak for the 100-step run was not present in the supplied console capture;
  and
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
TensorFlow.

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

Validated on 2026-10-05 for JAX and 2026-10-06 for PyTorch:

- Upstream revision: `730cda1c730877cfedbe01bf87fb1cadb78a565d`.
- Python 3.10 workspace-local virtual environment.
- JAX 0.6.2 CPU, JAXlib 0.6.2, TensorFlow 2.21.0.
- Final combined `python -m pytest tests -q`: **32 passed**.
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
- PyTorch/JAX seismic loss values match on fixed randomized tensors within
  `1e-6` relative and absolute tolerance.
- The upstream in-place residual addition was reproduced as an autograd failure,
  changed to an out-of-place equivalent, and verified by the full TAPIR backward
  test.
- PyTorch random-weight smoke: one 64-by-64 forward/backward/update step,
  checkpoint save, and resumed second step completed on CPU.
- Official `bootstapir_checkpoint_v2.pt`: 218,886,140 bytes, strict state load
  succeeded with all keys matching the 54,699,335-parameter PyTorch model.
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
  which are not yet included in console logging. The supplied capture did not
  show the post-step checkpoint or peak-memory messages.

The smoke evaluation followed a single random-weight update. Its numerical
accuracy is intentionally not recorded as a benchmark because it provides no
evidence of learning or generalization.

## Intentionally omitted or not yet validated

These omissions are material and must not be inferred as implemented:

1. **No useful model has been trained.** The default 20,000-step configuration
   has not been run on a GPU, tuned, or shown to converge.
2. **The complete one-batch overfit gate is partially passed.** On CUDA BF16,
   the pretrained frozen-encoder model drives the final-head losses near zero,
   but the sum over intermediate refinement outputs remains `0.775462` at step
   100. Per-refinement logging and a longer run are still required.
3. **Pretrained CUDA optimization is validated, not generalization.** The
   official PyTorch BootsTAPIR checkpoint loads strictly and completed 100
   frozen-encoder BF16 updates. No held-out PyTorch evaluation has been run.
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
12. **GPU validation is limited to one VDI.** Single-device CUDA BF16/FP32
    updates work on the L40-12Q. Throughput, multi-GPU reduction, long-run memory
    behavior, and exact numerical restart parity have not been tested.
13. **No prestack support.** The current input represents one scalar post-stack
    amplitude repeated into three channels.
14. **No data-volume caching/profile.** Synthetic generation is online and has
    not been profiled against GPU consumption at the default settings.

## Required next tests

### P0: establish that the prototype can learn

1. Print aggregate or per-refinement losses during fixed-batch training, then
   extend the BF16 run to determine whether every deeply supervised refinement
   output can overfit. The final output already reaches near-zero error; failure
   of the intermediate outputs to improve blocks larger experiments.
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
