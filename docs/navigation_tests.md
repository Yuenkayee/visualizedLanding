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

The checker also exercises `cv2.resize`, colour conversion and the PyTorch/NumPy
round trip in a fresh Python process. It requires a single provider of the `cv2`
namespace: `opencv-python-headless==4.10.0.84`, with `numpy==1.26.4`.
Preinstalled `opencv-python`/`opencv-contrib-python` variants can overwrite the
same files even while the required headless distribution's metadata looks correct.
If the native probe fails or multiple providers are found, normal `check` removes
confirmed NumPy residue first as described below. If further repair is needed,
it removes the installed OpenCV wheel variants, reinstalls only the pinned
NumPy/headless wheels without cache, and probes again. Torch is not reinstalled by this repair.
`--check-only` reports the conflict without changing packages. Logs include
`opencv-numpy-before.json` and, after a repair, `opencv-numpy-after.json`, with
package versions, imported paths and errors. A failed `environment.json` also
contains the Python path and full traceback.

If reinstalling the correct wheels still gives the same `resize` error, inspect
the report's `numpy_core` entries. NumPy 1.26.4 forwards `numpy._core` through
Python compatibility shims to `numpy.core`; an old `_core/*.so` left by another
installation can take precedence and expose a different `ndarray` type to
OpenCV. pip's uninstall/reinstall only removes files listed in its RECORD, so
untracked binaries can survive repeated reinstalls. The checker records both
resolved/loaded core paths and array-type identity. With normal `check`, it
first moves confirmed untracked native files under the active NumPy 1.26.4
`_core` directory to `.tools/navigation-numpy-backups/`, preserving hashes and
original paths in `numpy-quarantine.json`. It never moves RECORD-owned files,
symlinks, files from a different NumPy path, or NumPy 2's legitimate core.
A successful fresh-process probe is recorded in
`opencv-numpy-after-quarantine.json` and avoids another wheel download.

For `resize: src is not a numpy array` during dependency checks, sync the updated
checker and rerun `check` without `--check-only`. The inference probe uses a
generated NumPy image; no dataset is read. If repair still fails, use the logged
import paths to check for Conda, user-site or `PYTHONPATH` shadowing, or select a
clean Python 3.12 virtual environment through `PYTHON_BIN`. Do not change the
navigation parameters or checkpoint to work around an environment failure.

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

# Two records (main/tracking shared + fixed), two workers, two simulated seconds.
bash shell/run_navigation_tests.sh smoke --suite gimbal --workers 2 \
  --output outputs/navigation_tests/smoke_concurrency

# Local smoke using synthetic CPU rendering; exclude from scientific results.
PYTHON_BIN=.venv/bin/python bash shell/run_navigation_tests.sh smoke \
  --device cpu --renderer cpu --render-device CPU \
  --output outputs/navigation_tests/local_cpu_smoke

# Formal full suite, two workers, in tmux; thread limits apply to each worker.
bash shell/run_navigation_tests.sh run --suite all --workers 2 \
  --render-threads 4 --torch-threads 2 \
  --output outputs/navigation_tests/study_01 \
  --background --session navigation_study

# Serial collection for isolated timing; use a separate output and idle hardware.
bash shell/run_navigation_tests.sh run --suite all --workers 1 \
  --render-threads 4 --torch-threads 2 \
  --output outputs/navigation_tests/study_serial \
  --background --session navigation_serial

# Resume with exactly the original config, suite, sources and other overrides.
bash shell/run_navigation_tests.sh run --suite all --workers 2 \
  --render-threads 4 --torch-threads 2 --resume \
  --output outputs/navigation_tests/study_01 \
  --background --session navigation_study

# Rebuild analysis from an existing experiment without starting a simulation.
bash shell/run_navigation_tests.sh analyze --output outputs/navigation_tests/study_01
```

The main entry point is `python -m main.navigation_tests {smoke,run,analyze}`.
Its config defaults to `configs/navigation_tests.yaml`. Omitting `--output`
autogenerates a timestamped directory below `outputs/navigation_tests/` (`run_…`
for runs, `smoke_…` for smoke). Analysis and resume require an existing explicit
output. A new run must use a new directory. `--resume` is run-only; the main module
checks frozen fingerprints and sensor integrity before reusing completed work.
Changed code/configuration/weights/input require a new experiment directory.
Worker count and thread limits are part of the frozen configuration: changing
`--workers` (including switching from 2 to 1 for isolated timing),
`--render-threads` or `--torch-threads` requires a **new output directory**.
Storage policy/retention settings are frozen too. Updating to the cache-retirement
implementation requires a new experiment; it does not silently prune old runs.

`--suite main|gimbal|stress|all` is optional; the full config defaults to `all`.
Smoke defaults to one clear/medium primary case, one independent sensor record
and two seconds. Even with `--workers 2`, that single record cannot exercise
multi-record concurrency. Use `smoke --suite gimbal --workers 2` as above: it
creates two records, with main/tracking sharing one and fixed-gimbal using the
other, so both worker slots can run during the two-second simulation.
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
execution. Attach with `tmux attach -t navigation_study`; detach with Ctrl-B D.
The shell worker preserves argument boundaries and selected runtime/GPU variables
even when an existing tmux server has an older environment.

## Concurrency and timing

These overrides accept positive integers in either `--option value` or
`--option=value` form. They are **smoke/run-only**, including resumed runs;
`check`, standalone `--check-only` and `analyze` reject them before preflight.
The launcher forwards them only to `main.navigation_tests`, never to the
dependency checker. Omitted values come from the main module/config.

| Option | Default | Scope |
| --- | --- | --- |
| `--workers N` | `workers: 2` | Maximum simultaneous independent record groups. |
| `--render-threads N` | `render_threads: 4` | Renderer CPU threads per worker. |
| `--torch-threads N` | `torch_threads: 2` | PyTorch CPU threads per worker. |

The coordinator groups cases by `record_id`. All cases sharing a record run
serially on the same worker, including their method replays; the shared sensor
record is generated once. Distinct record groups run concurrently in spawned
processes using Python's standard library, with no extra dependency. Actual
concurrency is bounded by both the worker limit and available record groups.
A fixed-input `--source` replay supplies only one record. The grouping preserves
the paired design and does not increase the number of independent sorties.

Thread limits are per worker, not a machine-wide pool. With two workers, budget
for two renderers and two inference processes sharing CPU, GPU memory and GPU
compute. Each worker retains chronological, causal **0.02 s** navigation steps;
concurrency is across records, not across timestamps within a replay. Navigation
algorithms, method definitions, estimator settings and checkpoint weights stay
unchanged.

Keep the per-worker log files under the experiment output together with the
runtime metadata in `manifest.json`, `frozen.json`, records and results. Worker
logs are `workers/worker_<index>_<pid>.log`; terminal progress includes `worker`
and `pid` so concurrent cases can be distinguished. Use the worker
logs to diagnose record/replay failures and the runtime metadata to report the
worker/thread settings and hardware. The launcher's `console.log` and shell
`worker.sh` describe the overall invocation; retain these preflight/launch logs
as well when archiving an experiment.

Only the coordinator writes the global manifest and starts final analysis.
An output-directory lock rejects a second simultaneous run/resume. Ctrl-C,
SIGTERM and terminal SIGHUP trigger bounded cleanup of workers and their renderer
children. An abrupt worker crash interrupts the experiment instead of reporting
success; completed files remain available for a matching `--resume`.
`record_checks/` holds durable input-validation state, and each replay result is
bound to its exact sensor-record fingerprint. Regenerating a record invalidates
its old results across later interruptions as well. Keep this directory with
the records/results. The manifest preserves `attempts` history on resume; per-run
metadata records the originating runtime and the recording runtime so retained
timings are not attributed to a replacement server.

Concurrent per-step/method timings include contention for shared resources.
Report them as **shared-resource timings**; report total wall time and completed
records per unit time separately. They are not isolated inference latency. For isolated
latency measurements, collect a separate experiment with `--workers 1` on idle
hardware; do not run the concurrent and serial examples simultaneously. Keep
device, render backend, thread settings and weights comparable and disclose all
runtime settings. Rendering wall time remains distinct from navigation latency.

## Sensor storage policy

The default `sensor_retention: compact` preserves the entire experiment matrix,
all numerical result rows and all §5.2.3(9) table/figure inputs. It changes only
the lifetime of generated sensor files. `--sensor-retention all` retains the
previous full-archive behaviour. No new packages are required; the same Ubuntu
dependency checker runs before either mode.

Each worker records one `record_id`, replays **every** dependent main, gimbal and
stress case, verifies their completed durations, checksums and exact input
binding, and only then retires observations. Files under `results/`, truth and
encoder histories, hashes, metadata and diagnostic/statistical outputs are not
deleted or downsampled.

Default compact retention:

- `retain_replicates: 1` keeps the first seed in each weather/motion/gimbal cell
  in full: **24 full records** in the default all suite. This is a prespecified
  selection independent of accuracy. `--retain-replicates N` accepts zero.
- Other successful groups keep up to `anomaly_windows_per_record: 3` diagnostic
  clips with `anomaly_context_s: 1.0` seconds on either side of the event. Events
  are the largest declared-valid 3D error above the evaluation limit and the
  start/end of the longest outage exceeding the evaluation limit, considering
  all dependent methods and gear/CG/shaped-CG references. N1 contributes only
  camera-time errors, not a fictitious continuous-time outage. Windows may
  overlap or be clipped by flight boundaries. They include the original sensor
  arrays/labels. Null evaluation limits disable the corresponding selection.
- **Execution failures** preserve the group's full available sensor archive.
  Statistical anomalies are represented by the bounded clips and all numerical
  evidence, not by unlimited complete-flight image archives. Clip selection is
  for diagnosis; it does not filter any scientific statistics.
- Imported `--source` files belong to the caller and are never removed. With
  `--trajectory`, only newly generated observations in this experiment are
  eligible; source trajectory files remain untouched.

`records/<id>/storage.json` is written atomically before unlinking observations.
It contains the original input identity, every dependent result-manifest hash,
retained sensor hashes, clip selections and logical byte counts. On `--resume`,
intact retired groups are skipped without rendering; an interrupted deletion
is completed idempotently. Missing/corrupt numerical results or retained clips
cause regeneration of the **whole** record and rerunning **all** its dependent
methods, preserving input equality. No old result is mixed with newly generated
sensors. Corrupt metadata/retirement identities fail closed for inspection.
The new attempt receives new recording/timing provenance; retained results keep
their original provenance. Do not manually edit hashes or delete the receipts.

Terminal `sensor_storage` events report each group's retention state and saved
bytes. `storage_summary.json` is generated on exit and by analysis; counts cover
receipts only, and exclude numerical files, logs, external/unpruned records and
filesystem overhead. Use `du -sh` and `df -h` for actual disk usage. Disk demand
includes active workers, the full representative records, clips and all result
files; many execution failures can still retain substantial data. Compact mode
is not a fixed disk quota.

To analyze locally without sensor payloads, keep the experiment's `manifest.json`,
`frozen.json`, `record_checks/`, **all of `results/`**, and the record metadata
(`run.json`, `steps.jsonl`, `sensor_hashes.json`, `record_hashes.json`, `storage.json`).
Keep generated tables/reports and launch/worker logs for audit. Analysis verifies
retired-result checksums and can regenerate every numerical plot/table without
reading images. Full sensor replay of retired groups instead requires the
matching run's `--resume` regeneration when evidence needs repair, or a new
experiment with `--sensor-retention all`; selected clips alone are not full flights.

Ubuntu example for the 22-vCPU / RTX PRO 6000 server, from the repository root:

```bash
LANDING_DOWNLOAD_PROFILE=china bash shell/run_navigation_tests.sh check \
  --device cuda --renderer blender --render-device OPTIX

# Foreground, two records, 2 seconds; exercise retirement even on the first seed.
bash shell/run_navigation_tests.sh smoke --suite all --duration 2 \
  --workers 2 --render-threads 2 --torch-threads 1 \
  --sensor-retention compact --retain-replicates 0 \
  --device cuda --renderer blender --render-device OPTIX \
  --output outputs/navigation_tests/compact_smoke_01

# Full matrix, eight workers, background; default representative retention.
bash shell/run_navigation_tests.sh run --suite all \
  --workers 8 --render-threads 2 --torch-threads 1 \
  --sensor-retention compact --retain-replicates 1 \
  --device cuda --renderer blender --render-device OPTIX \
  --output outputs/navigation_tests/compact_full_01 \
  --background --session navigation_compact

# Resume only after the prior process/session has ended; all settings stay fixed.
bash shell/run_navigation_tests.sh run --suite all \
  --workers 8 --render-threads 2 --torch-threads 1 \
  --sensor-retention compact --retain-replicates 1 \
  --device cuda --renderer blender --render-device OPTIX \
  --output outputs/navigation_tests/compact_full_01 --resume \
  --background --session navigation_compact

bash shell/run_navigation_tests.sh analyze \
  --output outputs/navigation_tests/compact_full_01
```

Use fresh output directories for a new test, changed settings or updated code.
Existing full runs are not migrated or cleaned automatically. The initial
one-seed pilot retains all records with the default `retain_replicates: 1`; use
`--retain-replicates 0` when specifically testing cache retirement. Smoke still
does not provide formal accuracy evidence. Neither mode shuts down the server.

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
