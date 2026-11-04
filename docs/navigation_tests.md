# Navigation benchmark operations

This benchmark evaluates navigation independently of aircraft control. Navigation
outputs do not drive the aircraft. The default trajectory is an **analytical
kinematic landing generator**, not a recorded flight or a flight-dynamics
simulation. Kinematic generation, recorded-input replay and recorded-trajectory
rerendering require no MATLAB installation, Engine, toolbox or license. These
commands never install/start MATLAB, edit algorithm parameters, or shut down the
machine. Do not use the closed-loop launcher for this experiment.

## Environment and preflight

Production defaults are Ubuntu 22.04 x86_64, Python 3.12, PyTorch 2.8.0 with CUDA
12.8, `--device cuda --renderer blender --render-device OPTIX`. The NVIDIA driver
and GPU/container access must already work; the checker does not install drivers.
Inference and Cycles have separate device choices. A requested GPU must work;
neither check silently substitutes CPU.

```bash
# Audit only: no package, Python, renderer or tmux installation.
bash shell/run_navigation_tests.sh --check-only
# Equivalent explicit action:
bash shell/run_navigation_tests.sh check --check-only

# Install/repair missing or incompatible pinned dependencies, then probe them.
bash shell/run_navigation_tests.sh check
```

Omitted device/render options follow the selected YAML config; explicit CLI
options take precedence. The launcher reuses a compatible Python or bootstraps Python 3.12 into
`.venv-navigation`. `PYTHON_BIN` selects an existing interpreter explicitly;
Linux requires Python 3.12. Bootstrap itself needs Python >=3.10, supplied by
Ubuntu 22.04. A local macOS CPU check can reuse Python 3.11/3.12 and the separately
pinned Torch 2.5.1 profile. It is not the production CUDA environment.

`requirements-navigation-test.txt` includes the shared numerical/image pins
directly from `requirements-data.txt`. It does not depend on the replay or
closed-loop requirements chain, which may be absent in a deployment, nor on the
old Torch/cu124 lock files. Required package versions and their dependency
closure are audited before installing with `only-if-needed`. The CUDA check
also distinguishes a correct cu128 wheel from a CPU/cu126 build having the same
Torch version. Native imports, a CUDA tensor operation, checkpoint loading and
an actual `HDetector` forward pass with finite outputs are checked. A blank probe
image need not produce a detection; this check establishes execution, not accuracy.

The checkpoint comes from the benchmark's `base_config`, currently
`main/navigation_observer.yaml`, which selects
`visual_training/checkpoints/four_weather_v1/best.pt`. The checker records its
size and SHA256, rejects missing files/LFS pointers, and never downloads or
substitutes weights. The shipped checkpoint SHA256 is
`1605b4a258414d931f08c630dc1bd83c3946f1a91bfa595614ee0e54df186ab3`.

For Blender, the checker optionally prepares `.venv-render-server` with Python
3.11 and the existing `requirements-render-server.txt` (bpy 4.5.3). Set
`RENDER_PYTHON=/absolute/venv/bin/python` or `--render-python PATH` to use an
existing isolated Python 3.11 environment. bpy is never installed in the
coordinator. The existing library-isolation helper is used for both the bpy
probe and an actual small Cycles render on the requested backend. `--renderer
cpu` and fixed-input `--source` replay skip all bpy/runtime/library installation.
Analysis skips checkpoint, GPU and renderer probes. `--check-only` still runs
probes and may write logs/caches; it means no dependency installation.

Downloads reuse `shell/download_mirrors.sh`. For example:

```bash
LANDING_DOWNLOAD_PROFILE=china bash shell/run_navigation_tests.sh check
```

This selects the existing TUNA PyPI/Ubuntu/Python-source and NJU Python-binary
routes. Existing `LANDING_PYPI_INDEX`, `LANDING_PYTHON_BINARY_MIRROR`,
`LANDING_PYTHON_SOURCE_MIRROR` and `LANDING_APT_MIRROR` overrides remain available.
Torch uses its separate cu128 index, configurable with `LANDING_TORCH_INDEX`.
Read-only failures require fixing the named dependency and rerunning; do not
replace the CUDA profile with a generic historical requirements file.

## Launch, resume and analyze

All relative paths are interpreted from the repository root. Options support
both `--option value` and `--option=value`; quote paths containing spaces.
Dependencies are checked before every smoke/run, including background runs.

```bash
# Two seconds of simulated time; always foreground, not a two-second wall timeout.
bash shell/run_navigation_tests.sh smoke --output outputs/navigation_tests/smoke_check

# Local smoke using synthetic CPU rendering; exclude from scientific results.
PYTHON_BIN=.venv/bin/python bash shell/run_navigation_tests.sh smoke \
  --device cpu --renderer cpu --render-device CPU \
  --output outputs/navigation_tests/local_cpu_smoke

# Full suite, using the frozen configuration's seed count.
bash shell/run_navigation_tests.sh run --suite all --output outputs/navigation_tests/study_01

# Primary matrix only; background execution is optional and run-only.
bash shell/run_navigation_tests.sh run --suite main \
  --output outputs/navigation_tests/primary_01 \
  --background --session navigation_primary

# Resume with exactly the original config, suite, sources and other overrides.
bash shell/run_navigation_tests.sh run --suite main --resume \
  --output outputs/navigation_tests/primary_01

# Rebuild analysis from an existing experiment without starting a simulation.
bash shell/run_navigation_tests.sh analyze --output outputs/navigation_tests/primary_01
```

The main entry point is `python -m main.navigation_tests {smoke,run,analyze}`.
Its config defaults to `configs/navigation_tests.yaml`. Omitting `--output`
autogenerates a timestamped directory below `outputs/navigation_tests/` (`run_…`
for runs, `smoke_…` for smoke). Analysis and resume require an existing explicit
output. A new run must use a new directory. `--resume` is run-only; the main module
checks frozen fingerprints and sensor integrity before reusing completed work.
Changed code/configuration/weights/input require a new experiment directory.

`--suite main|gimbal|stress|all` is optional; the full config defaults to `all`.
Smoke defaults to one clear/medium primary case and two seconds.
`--duration SECONDS` is **smoke-only**, positive and a multiple of 0.02 s. Formal
runs use complete trajectories. `--seeds-per-cell N` overrides the full-run
matrix for pilots; reduced matrices and smoke are not the 240-sortie study.
Smoke fixes its own one-seed pilot regardless of the full-run seed setting.

The launcher prints `outputs/logs/navigation_tests/<timestamp>_<action>_<pid>/`.
It contains `console.log`, `environment.json`, selected interpreter paths, the
quoted `worker.sh`, and `exit.code` after completion. Preflight and foreground
failures propagate through `tee`. With `--background`, successful launch means
the tmux session started; inspect `exit.code` and the benchmark manifest for
completion. tmux >=3.2 (Ubuntu 22.04 default) is required only for background
execution. Attach with `tmux attach -t navigation_primary`; detach with Ctrl-B D.
The worker preserves argument boundaries and selected runtime/GPU variables
even when an existing tmux server has an older environment.

## Trajectory and observation provenance

| Mode | Inputs and permitted interpretation |
| --- | --- |
| No source option | Analytical prescribed kinematic trajectories with independent test seeds from the existing motion family. Synthetic navigation evaluation; no flight dynamics or actual flight record. |
| `--source DIR` | One complete actual closed-loop recorded run containing `run.json`, chronological `steps.jsonl`, synchronized encoder history and saved `sensors/*.npz`. Replay those fixed observations once across methods. Main suite only; no rerendering or MATLAB. |
| `--trajectory DIR` | A recorded run supplying body/deck truth in `run.json`/`steps.jsonl`. Rerender weather, prescribed motion perturbations and gimbal conditions with new sensor records. These are derivatives of a recorded path, not new controlled flights. |

```bash
bash shell/run_navigation_tests.sh run --source '/data/actual closed-loop run' \
  --output outputs/navigation_tests/recorded_replay

bash shell/run_navigation_tests.sh run --trajectory '/data/actual closed-loop run' \
  --suite all --output outputs/navigation_tests/recorded_rerender
```

`--source` and `--trajectory` are mutually exclusive. The launcher selects
`--suite main` for `--source` and rejects any conflicting suite. A single fixed
record must never be relabeled as independent weather/motion/seed replications
or counted as 240 actual flights. Rerendered variants also retain their shared
trajectory provenance; report the number of independent source flights. Preserve
the original records and sensor hashes for analysis and resume.

Imports preserve the source's preview/exclusion status. Failed or interrupted
sources are rejected; sources without all three task stages remain excluded
from complete-mission formal statistics. A CPU preview cannot become a formal
result merely by importing it with `run`. Resume checks both sensor records and
per-method result hashes; missing or damaged result logs are rerun.

## Frozen experiment and interpretation

The primary design is **4 weather conditions × 3 motion levels × 20 seeds = 240
paired sorties**. Weather is clear, fog, night and night_fog. Low/medium/high
motion scales (0.5/1.0/1.5) are internal perturbation categories, not certified
sea-state classes. Each pair shares archived observations among methods;
method replays do not increase the independent sortie count. Gimbal and stress
studies are additional cases, not additional primary independent sorties. The
default `--suite all` plan has **240 main + 480 gimbal + 504 stress = 1,224 cases**
and **480 unique generated sensor recordings**. Tracking cases reuse the 240
primary recordings; fixed-gimbal variants supply the other 240. Stress cases
reuse primary observations with prescribed faults. These counts describe the
generated matrix, not the single-flight `--source` replay mode.

The host/navigation clock is **0.02 s (50 Hz)**, camera 10 Hz and LiDAR 5 Hz.
Freeze the config, code, checkpoint, source identifiers and seed list before
collecting formal results. Keep `frozen.json`, its fingerprint,
`manifest.json`, per-record sensor hashes and all case outputs together. The
preflight's config/weight hashes are additional provenance, not a substitute for
the benchmark's full frozen snapshot. Do not retune estimator noise, health
gates, geometry, training weights or method parameters during collection.

| Method | Benchmark definition |
| --- | --- |
| N0 | Classical white-H detector with the full backend. |
| N1 | Learned ordered corners plus single-frame PnP/covariance; no ESKF, LiDAR, mask/ROI recovery or reference feedforward. Evaluate at successful camera times; no held/shaped pose or invented velocity. |
| N2 | Learned navigation pipeline without LiDAR. |
| N3 | Full learned navigation pipeline. |
| A1 | N3 without ROI and all mask-geometry recovery. |
| A2 | N3 without gimbal-reference feedforward. |
| A3 | N3 with the `recovery_pending` gate bypassed; `reacquisition_confirmations=2` and the other numeric parameters remain unchanged. |

Evaluation thresholds are **provisional navigation criteria independent of the
unchanged online health gates**. `false_valid_error_m: 1.0` identifies metre-scale
erroneous outputs marked valid; `max_outage_s: 0.5` represents five camera periods.
Report sensitivity over `[0.5, 1.0, 2.0]` m and `[0.2, 0.5, 1.0]` s, using the
same frozen data. These are not touchdown-success, aircraft-safety or
certification criteria. An installation/inference check passing does not mean
the navigation study passed. If a custom config explicitly uses `null`, leave
that threshold-dependent decision unscored; never invent a replacement cutoff.

Analysis writes the following artifacts directly in the experiment directory:

| Deliverable | Files |
| --- | --- |
| Table 5-13, N0–N3 | `table_5_13.csv` |
| Table 5-14, ablations, condition views and paired differences | `table_5_14.csv` |
| Figures 5-8, 5-9, 5-10 and their numerical evidence | `figure_5_8.png/.csv`, `figure_5_9.png/.csv`, `figure_5_10.png/.csv` |
| Per-sortie statistics and paired comparisons | `flight_metrics.csv/.json`, `paired_differences.csv/.json`, `statistical_raw.csv` |
| Threshold sensitivity | `sensitivity.csv` |
| Coverage, failures and interpretation | `summary.json`, `runs.json`, `failure_manifest.json`, `analysis.md` |
| Frame/step evidence and events | `samples.csv`, `raw_steps.jsonl`, `events.csv`, `categorical_diagnostics.csv` |

Preserve the numerical tables, figures, summary/provenance and failure diagnostics.
Report errors together with availability, false-valid/outage/recovery metrics
and timing; do not hide failed or missing cases by averaging only successes.
Use paired sortie/source-level statistics rather than treating frames as
independent trials. Label CPU smoke, partial/reduced matrices, analytical
kinematic runs and derivatives of actual recorded paths explicitly in captions
and result tables. Rendering wall time is not navigation inference latency.

**CPU inference alone does not invalidate accuracy.** FP32 inference on CPU with
Blender-generated or actual recorded sensor inputs can support accuracy
evaluation; identify the Torch version and evaluate numerical consistency when
comparing environments. Timing remains hardware-specific and must not be
presented as CUDA/OPTIX throughput. Smoke and the synthetic `--renderer cpu`
sensor generator are excluded from scientific results regardless of inference
device. With `--source`, saved sensor provenance governs accuracy; no renderer
is actually used. The local macOS Torch 2.5.1 environment is supported without
forcing a Torch 2.8 upgrade and must be distinguished from the production pins.
