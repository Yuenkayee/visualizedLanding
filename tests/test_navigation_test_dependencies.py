"""Navigation deployment contracts; no downloads, system installs or MATLAB."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from shell import navigation_test_dependencies as deploy


def test_profile_has_exact_platform_pins_and_no_matlab_or_bpy():
    requirements = deploy.dependencies.requirements(deploy.PROFILE)
    names = {r.name.lower() for r in requirements}
    assert {"numpy", "scipy", "opencv-python-headless", "pyyaml", "matplotlib", "torch"} <= names
    assert not {"bpy", "matlabengine"} & names
    for system, version in (("linux", "2.8.0"), ("darwin", "2.5.1")):
        torch = [r for r in requirements if r.name == "torch" and r.marker.evaluate({"sys_platform": system})]
        assert len(torch) == 1 and str(torch[0].specifier) == "==" + version
    assert "requirements-replay" not in deploy.PROFILE.read_text()


def test_base_config_supplies_checkpoint_without_matlab(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("checkpoint: /weights/best.pt\nrequired_toolboxes: [Simulink]\nrender_threads: 8\n")
    config = tmp_path / "navigation.yaml"
    config.write_text(f"base_config: {base}\nrender_threads: 4\n")
    result, hashes = deploy.load_config(config)
    assert result["checkpoint"] == "/weights/best.pt" and result["render_threads"] == 4
    assert hashes == {str(config): deploy.sha256(config), str(base): deploy.sha256(base)}


def test_linux_selection_does_not_reuse_local_python311(monkeypatch):
    monkeypatch.delenv("PYTHON_BIN", raising=False)
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(deploy.platform, "machine", lambda: "x86_64")
    seen = []

    def matches(python, versions):
        seen.append(versions)
        return str(python) == "python3.12"

    monkeypatch.setattr(deploy.bootstrap, "python_matches", matches)
    monkeypatch.setattr(deploy.bootstrap, "absolute_executable", lambda p: "/bin/" + str(p))
    assert deploy.select_python(True) == "/bin/python3.12"
    assert all(versions == ["3.12"] for versions in seen)


def test_missing_python_check_never_installs(monkeypatch):
    monkeypatch.delenv("PYTHON_BIN", raising=False)
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(deploy.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(deploy.bootstrap, "python_matches", lambda *a: False)
    monkeypatch.setattr(deploy.bootstrap, "ensure_apt", lambda *a: pytest.fail("install attempted"))
    monkeypatch.setattr(deploy.bootstrap, "ensure_uv", lambda: pytest.fail("download attempted"))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        deploy.select_python(check=True)


def test_missing_pip_check_never_installs(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: argparse.Namespace(returncode=1))
    monkeypatch.setattr(deploy.dependencies, "run", lambda *a: pytest.fail("install attempted"))
    with pytest.raises(RuntimeError, match="pip is missing"):
        deploy.ensure_pip("/missing/python", True)


def cv_report(*providers, error=None):
    return dict(status="failed" if error else "ready", error=error,
                packages=[dict(name=p, version="4.10.0.84") for p in providers],
                numpy=dict(version="1.26.4", path="/env/numpy/__init__.py"),
                cv2=dict(version="4.10.0", path="/env/cv2/__init__.py"))


def test_real_opencv_probe_in_fresh_process():
    report = deploy.inspect_opencv()
    assert report["status"] == "ready", report
    assert report["resize"] == report["torch_numpy"] == "passed"
    assert report["numpy"]["path"] and report["cv2"]["path"]


def test_opencv_probe_reports_native_error(monkeypatch):
    import cv2

    def fail(*args, **kwargs):
        raise cv2.error("src is not a numpy array, neither a scalar")

    monkeypatch.setattr(cv2, "resize", fail)
    report = deploy.opencv_probe()
    assert report["status"] == "failed"
    assert "not a numpy array" in report["error"]
    assert "Traceback" in report["traceback"]
    assert report["numpy"]["path"] and report["cv2"]["path"]


@pytest.mark.parametrize("conflict", [True, False])
def test_opencv_check_only_preserves_broken_environment(tmp_path, monkeypatch, conflict):
    report = (cv_report("opencv-python", "opencv-python-headless") if conflict else
              cv_report("opencv-python-headless", error="src is not a numpy array"))
    monkeypatch.setattr(deploy, "inspect_opencv", lambda: report)
    monkeypatch.setattr(deploy.dependencies, "run", lambda *a: pytest.fail("install attempted"))
    with pytest.raises(RuntimeError, match="rerun without --check-only"):
        deploy.ensure_opencv(True, tmp_path)
    assert json.loads((tmp_path / "opencv-numpy-before.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("conflict", [True, False])
def test_opencv_repairs_shared_namespace_or_broken_binary_once(tmp_path, monkeypatch, conflict):
    bad = (cv_report("opencv-contrib-python", "opencv-python-headless") if conflict else
           cv_report("opencv-python-headless", error="src is not a numpy array"))
    reports = iter([bad, cv_report("opencv-python-headless")])
    monkeypatch.setattr(deploy, "inspect_opencv", lambda: next(reports))
    commands = []
    monkeypatch.setattr(deploy.dependencies, "run", lambda cmd: commands.append(cmd))
    assert deploy.ensure_opencv(False, tmp_path)["status"] == "ready"
    assert commands[0][:5] == [sys.executable, "-m", "pip", "uninstall", "-y"]
    assert set(commands[0][5:]) == {p["name"] for p in bad["packages"]}
    assert set(commands[1][3:]) == {"install", "--no-cache-dir", "--force-reinstall", "--no-deps",
                                   "numpy==1.26.4", "opencv-python-headless==4.10.0.84"}
    assert json.loads((tmp_path / "opencv-numpy-before.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "opencv-numpy-after.json").read_text())["status"] == "ready"


def test_opencv_shadowing_stops_after_one_repair(monkeypatch):
    def shadowed():
        report = cv_report("opencv-python-headless")
        report["numpy"]["version"] = "2.3.2"
        return report

    monkeypatch.setattr(deploy, "inspect_opencv", shadowed)
    commands = []
    monkeypatch.setattr(deploy.dependencies, "run", lambda cmd: commands.append(cmd))
    with pytest.raises(RuntimeError, match="repair failed in a fresh interpreter.*NumPy 2.3.2"):
        deploy.ensure_opencv(False)
    assert len(commands) == 2


def test_opencv_probe_crash_is_failure(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw:
                        argparse.Namespace(returncode=-11, stdout="", stderr="native crash"))
    report = deploy.inspect_opencv()
    assert report["status"] == "failed" and report["returncode"] == -11
    assert report["stderr"] == "native crash"


def test_numpy_quarantine_preserves_owned_and_unrelated_files(tmp_path, monkeypatch):
    site = tmp_path / "site"
    root = site / "numpy"
    core = root / "_core"
    core.mkdir(parents=True)
    orphan = core / "_multiarray_umath.cpython-312-test.so"
    owned = core / "owned.so"
    shim = core / "_multiarray_umath.py"
    for file in (orphan, owned, shim):
        file.write_bytes(b"test bytes")
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    (core / "symlink.so").symlink_to(outside)
    dist = argparse.Namespace(version="1.26.4", files=["numpy/_core/owned.so"],
                              locate_file=lambda p: site / p)
    monkeypatch.setattr(deploy.metadata, "distribution", lambda name: dist)
    monkeypatch.setattr(deploy, "ROOT", tmp_path / "repo")
    report = dict(numpy=dict(version="1.26.4", path=str(root / "__init__.py")))
    assert deploy.untracked_numpy_extensions(root) == [orphan]
    manifest = deploy.quarantine_numpy_extensions(report, tmp_path / "logs")
    assert len(manifest) == 1 and not orphan.exists()
    assert Path(manifest[0]["backup"]).read_bytes() == b"test bytes"
    assert manifest[0]["sha256"] == deploy.sha256(Path(manifest[0]["backup"]))
    assert owned.exists() and shim.exists() and outside.exists() and (core / "symlink.so").is_symlink()
    assert json.loads((tmp_path / "logs/numpy-quarantine.json").read_text()) == manifest
    # Without reliable ownership metadata, and for NumPy 2's legitimate _core,
    # there is no authorization to classify a binary as an orphan.
    orphan.write_bytes(b"test bytes")
    dist.version = "2.0.0"
    assert deploy.untracked_numpy_extensions(root) == []
    dist.version, dist.files = "1.26.4", None
    assert deploy.untracked_numpy_extensions(root) == []


def test_numpy_quarantine_rechecks_probe_paths(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    fake = tmp_path / "unrelated/numpy/_core"
    fake.mkdir(parents=True)
    file = fake / "_multiarray_umath.so"
    file.touch()
    dist = argparse.Namespace(version="1.26.4", files=["numpy/__init__.py"],
                              locate_file=lambda p: site / p)
    monkeypatch.setattr(deploy.metadata, "distribution", lambda name: dist)
    assert deploy.quarantine_numpy_extensions(dict(numpy=dict(
        version="1.26.4", path=str(fake.parent / "__init__.py")))) == []
    assert file.exists()


def test_real_orphan_numpy_extension_reproduces_resize_failure_and_recovers(tmp_path):
    """Only corrupt a temporary copy; original installed packages stay intact."""
    import numpy as np

    source = Path(np.__file__).parent
    isolated = tmp_path / "site"
    isolated.mkdir()
    shutil.copytree(source, isolated / "numpy")
    dist = deploy.metadata.distribution("numpy")
    metadata_file = next(p for p in dist.files if str(p).endswith(".dist-info/METADATA"))
    metadata_dir = Path(dist.locate_file(metadata_file)).parent
    shutil.copytree(metadata_dir, isolated / metadata_dir.name)
    # Linux wheels keep dependent shared libraries next to the numpy package.
    libraries = source.parent / "numpy.libs"
    if libraries.exists():
        shutil.copytree(libraries, isolated / libraries.name)
    binary = next((source / "core").glob("_multiarray_umath*.so"))
    orphan = isolated / "numpy/_core" / binary.name
    shutil.copy2(binary, orphan)
    env = {**os.environ, "PYTHONPATH": str(isolated)}

    def execute(code, *args):
        result = subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=deploy.ROOT,
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    probe = "import json; from shell.navigation_test_dependencies import opencv_probe; print(json.dumps(opencv_probe()))"
    before = execute(probe)
    assert before["status"] == "failed" and "not a numpy array" in before["error"]
    assert before["numpy"]["version"] == "1.26.4"
    assert before["numpy_core"]["untracked_extensions"] == [str(orphan.resolve())]
    assert before["numpy_core"]["modules"]["numpy._core._multiarray_umath"]["ndarray_matches"] is False
    # Recovery must use a fresh interpreter after moving an already loaded .so.
    repair = '''
import sys, json
from pathlib import Path
from shell import navigation_test_dependencies as deploy
profile = (deploy.ROOT / "requirements-data.txt").read_text()
deploy.ROOT = Path(sys.argv[1])
deploy.ROOT.mkdir()
(deploy.ROOT / "requirements-data.txt").write_text(profile)
deploy.dependencies.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected pip reinstall"))
print(json.dumps(deploy.ensure_opencv(False, deploy.ROOT / "logs")))
'''
    after = execute(repair, tmp_path / "repo")
    assert after["status"] == "ready" and after["resize"] == "passed"
    assert after["numpy_core"]["untracked_extensions"] == []
    assert not orphan.exists()
    assert (tmp_path / "repo/logs/numpy-quarantine.json").exists()
    assert (tmp_path / "repo/logs/opencv-numpy-after-quarantine.json").exists()


def test_real_checkpoint_inference_runs_on_cpu():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = deploy.check_inference(deploy.ROOT / "visual_training/checkpoints/four_weather_v1/best.pt", "cpu")
    finally:
        torch.set_num_threads(previous)
    assert report["inference"] == "passed"
    assert report["output_shapes"]["keypoints"] == [1, 4, 2]
    assert report["sha256"] == "1605b4a258414d931f08c630dc1bd83c3946f1a91bfa595614ee0e54df186ab3"


def test_nonfinite_network_output_is_not_success(tmp_path, monkeypatch):
    import torch
    from navigation.vision import infer_h

    class Model(torch.nn.Module):
        def forward(self, value):
            return dict(mask_logits=torch.tensor([float("nan")]), keypoints=torch.ones(1, 4, 2))

    class Detector:
        def __init__(self, *args):
            self.model, self.image_size = Model(), 8

        def __call__(self, image):
            self.model(torch.from_numpy(image))
            return None

    monkeypatch.setattr(infer_h, "HDetector", Detector)
    with pytest.raises(RuntimeError, match="Non-finite"):
        deploy.check_inference(tmp_path / "fake.pt", "cpu")


@pytest.mark.parametrize("source,analysis", [("/recorded", False), (None, True)])
def test_fixed_replay_and_analysis_skip_renderer(tmp_path, monkeypatch, source, analysis):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    config.write_text(f"checkpoint: {checkpoint}\n")
    calls = []
    monkeypatch.setattr(deploy, "ensure_pip", lambda *a: None)
    monkeypatch.setattr(deploy.dependencies, "ensure_requirements", lambda *a: calls.append(a))
    monkeypatch.setattr(deploy, "ensure_opencv", lambda *a: {"status": "ready"})
    monkeypatch.setattr(deploy.dependencies, "ensure_torch", lambda *a: pytest.fail("GPU check attempted"))
    monkeypatch.setattr(deploy.bootstrap, "ensure_matlab", lambda *a: pytest.fail("MATLAB install attempted"))
    monkeypatch.setattr(deploy, "setup_renderer", lambda *a: pytest.fail("renderer was required"))
    monkeypatch.setattr(deploy, "check_inference", lambda *a: {"inference": "passed"})
    args = argparse.Namespace(check_only=True, analysis_only=analysis, device="cpu", renderer="blender",
                              render_device="OPTIX", render_python=None, config=config, checkpoint=None,
                              report_dir=None, with_tmux=False, source=source, trajectory=None)
    report = deploy.setup(args)
    assert calls == [(sys.executable, deploy.ROOT / "requirements-data.txt", True),
                     (sys.executable, deploy.PROFILE, True)]
    assert report["render_python"] is None and report["matlab_required"] is False
    assert report["scientific_result"] is False


@pytest.mark.parametrize("override", [None, "blender"])
def test_config_defaults_and_cli_precedence(tmp_path, monkeypatch, override, capsys):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    config.write_text(f"device: cpu\nrenderer: cpu\nrender_device: CPU\ncheckpoint: {checkpoint}\n")
    monkeypatch.setattr(deploy, "ensure_pip", lambda *a: None)
    monkeypatch.setattr(deploy.dependencies, "ensure_requirements", lambda *a: None)
    monkeypatch.setattr(deploy, "ensure_opencv", lambda *a: {"status": "ready"})
    monkeypatch.setattr(deploy.dependencies, "ensure_torch", lambda *a: pytest.fail("CPU config ignored"))
    monkeypatch.setattr(deploy, "check_inference", lambda *a: {"inference": "passed"})
    monkeypatch.setattr(deploy, "setup_renderer", lambda *a: pytest.fail("recorded inputs need no renderer"))
    args = argparse.Namespace(check_only=True, analysis_only=False, device=None, renderer=override,
                              render_device=None, render_python=None, config=config, checkpoint=None,
                              report_dir=None, with_tmux=False, source="/recorded", trajectory=None)
    report = deploy.setup(args)
    assert report["device"] == "cpu" and report["render_device"] == "CPU"
    assert report["renderer"] == (override or "cpu")
    messages = capsys.readouterr().out
    assert "CPU FP32 inference can evaluate accuracy" in messages
    assert "NOT scientific" not in messages


def test_blender_probe_uses_isolated_environment_and_selected_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    matlab = tmp_path / "MATLAB"
    monkeypatch.setenv("MATLAB_ROOT", str(matlab))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(matlab / "bin/glnxa64") + ":/cuda/lib64")
    monkeypatch.setattr(deploy.bootstrap, "python_matches", lambda *a: True)
    monkeypatch.setattr(deploy, "ensure_pip", lambda *a: None)
    checks = []
    monkeypatch.setattr(deploy.dependencies, "ensure_libraries", lambda check: checks.append(check))
    profiles = []
    monkeypatch.setattr(deploy.dependencies, "ensure_requirements", lambda *a: profiles.append(a))
    monkeypatch.setattr(deploy, "probe_renderer", lambda *a: {"status": "ready"})
    calls = []
    monkeypatch.setattr(deploy.dependencies, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    python, report = deploy.setup_renderer(True, "/isolated/bin/python", "OPTIX", tmp_path)
    assert checks == [True] and profiles == [(python, deploy.RENDER_PROFILE, True)]
    assert report["status"] == "ready"
    command, kwargs = calls[0]
    assert command[:3] == [python, "-m", "visual_modeling.blender.cycles_device"]
    assert command[command.index("--device")+1] == "OPTIX"
    assert kwargs["env"]["LD_LIBRARY_PATH"] == "/cuda/lib64"
    assert str(matlab) in os.environ["LD_LIBRARY_PATH"]


@pytest.mark.parametrize("version,valid", [("tmux 3.2a", True), ("tmux 3.5a", True), ("tmux 3.1c", False)])
def test_background_requires_argument_safe_tmux(monkeypatch, version, valid):
    monkeypatch.setattr(deploy.shutil, "which", lambda *a: "/usr/bin/tmux")
    monkeypatch.setattr(deploy.dependencies, "run", lambda *a, **kw: argparse.Namespace(stdout=version))
    if valid:
        deploy.ensure_tmux(True)
    else:
        with pytest.raises(RuntimeError, match="tmux >=3.2"):
            deploy.ensure_tmux(True)


def test_deserialization_failure_writes_failed_report(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.touch()
    report = tmp_path / "report"

    def fail(_args):
        import pickle
        raise pickle.UnpicklingError("invalid checkpoint")

    monkeypatch.setattr(deploy, "setup", fail)
    assert deploy.main(["--runtime", "--config", str(config), "--report-dir", str(report), "--check-only"]) == 1
    failure = json.loads((report / "environment.json").read_text())
    assert failure["status"] == "failed" and failure["error"] == "invalid checkpoint"
    assert failure["python"] == sys.executable and "UnpicklingError" in failure["traceback"]


@pytest.fixture
def launcher(tmp_path):
    # Quotes, command substitutions, backticks, whitespace and newlines must
    # survive BOTH the foreground worker and the tmux command boundary.
    root = tmp_path / "repo ' $(touch INJECTED) `touch INJECTED2`\nspace"
    shell = root / "shell"
    shell.mkdir(parents=True)
    for name in ("run_navigation_tests.sh", "download_mirrors.sh"):
        (shell / name).write_text((deploy.ROOT / "shell" / name).read_text())
    (shell / "navigation_test_dependencies.py").write_text('''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path('preflight.json').write_text(json.dumps(args))
print('preflight output', flush=True)
if os.environ.get('FAIL_CHECK'): sys.exit(17)
report = Path(args[args.index('--report-dir')+1])
(report/'python.txt').write_text(sys.executable+'\\n')
(report/'render-python.txt').write_text('/isolated render/python\\n')
for name, default in [('device', 'cuda'), ('renderer', 'blender'), ('render-device', 'OPTIX')]:
    option = '--' + name
    value = args[args.index(option)+1] if option in args else default
    (report/(name+'.txt')).write_text(value+'\\n')
''')
    main = root / "main"
    main.mkdir()
    (main / "__init__.py").touch()
    (main / "navigation_tests.py").write_text('''
import json, os, sys
from pathlib import Path
Path('launched.json').write_text(json.dumps(dict(args=sys.argv[1:], render_python=os.environ.get('RENDER_PYTHON'))))
print('benchmark output', flush=True)
sys.exit(int(os.environ.get('FAIL_RUN', '0')))
''')
    commands = root / "bin"
    commands.mkdir()
    tmux = commands / "tmux"
    tmux.write_text('''#!'''+sys.executable+'''
import json, subprocess, sys
from pathlib import Path
Path('tmux.json').write_text(json.dumps(sys.argv[1:]))
if sys.argv[1] == 'new-session': subprocess.run(sys.argv[-2:])
''')
    tmux.chmod(0o755)

    def invoke(*args, **extra_env):
        env = {**os.environ, "PYTHON_BIN": sys.executable, "LANDING_DOWNLOAD_PROFILE": "default",
               "PATH": str(commands) + os.pathsep + os.environ["PATH"], **extra_env}
        return subprocess.run(["bash", str(shell / "run_navigation_tests.sh"), *args],
                              cwd=tmp_path, env=env, capture_output=True, text=True)

    return root, invoke


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("equals", [False, True])
def test_launcher_preserves_argv_and_logs(launcher, background, equals):
    root, invoke = launcher
    output = "outputs/quoted ' $(touch OUTPUT_INJECTED) `touch OUTPUT_INJECTED2`\nspace"
    args = ["run" if background else "smoke", "--output", output, "--config=config name.yaml",
            "--device=cpu", "--renderer", "cpu", "--suite", "gimbal", "--seeds-per-cell=2"]
    concurrency = {"--workers": "2", "--render-threads": "4", "--torch-threads": "2"}
    for option, value in concurrency.items():
        args += [f"{option}={value}"] if equals else [option, value]
    if background:
        args += ["--background", "--session", "navigation_test"]
    result = invoke(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    launched = json.loads((root / "launched.json").read_text())
    argv = launched["args"]
    assert argv[argv.index("--config") + 1] == "config name.yaml"
    assert argv[argv.index("--output") + 1] == output
    assert argv[argv.index("--suite") + 1] == "gimbal"
    assert argv[argv.index("--seeds-per-cell") + 1] == "2"
    assert ("--duration" in argv) == (not background)
    if not background:
        assert argv[argv.index("--duration") + 1] == "2"
    assert launched["render_python"] == "/isolated render/python"
    preflight = json.loads((root / "preflight.json").read_text())
    assert "--output" not in preflight and "--suite" not in preflight
    for option, value in concurrency.items():
        assert argv.count(option) == 1 and argv[argv.index(option) + 1] == value
        assert option not in preflight
    assert ("--with-tmux" in preflight) == background
    assert not any(root.glob("*INJECTED*"))
    logs = list(root.glob("outputs/logs/navigation_tests/*/console.log"))
    assert len(logs) == 1
    assert "preflight output" in logs[0].read_text() and "benchmark output" in logs[0].read_text()
    assert logs[0].with_name("exit.code").read_text().strip() == "0"
    if background:
        tmux = json.loads((root / "tmux.json").read_text())
        assert tmux[-2] == "bash" and Path(tmux[-1]).name == "worker.sh"


@pytest.mark.parametrize("mode", ["smoke", "run"])
def test_launcher_leaves_concurrency_defaults_to_main_config(launcher, mode):
    root, invoke = launcher
    result = invoke(mode)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads((root / "launched.json").read_text())["args"]
    preflight = json.loads((root / "preflight.json").read_text())
    for option in ("--workers", "--render-threads", "--torch-threads"):
        assert option not in argv and option not in preflight


@pytest.mark.parametrize("option", ["--workers", "--render-threads", "--torch-threads"])
@pytest.mark.parametrize("value", [
    "", "0", "-1", "1.5", "1e2", "nan", "true", "2 3", "2; touch INJECTED",
    "$(touch INJECTED)", "`touch INJECTED`",
])
def test_launcher_rejects_invalid_concurrency_before_bootstrap(launcher, option, value):
    root, invoke = launcher
    result = invoke("smoke", f"{option}={value}")
    assert result.returncode == 2, result.stdout + result.stderr
    assert option in result.stderr
    assert ("positive integer" if value else "Empty value") in result.stderr
    assert not (root / "preflight.json").exists()
    assert not (root / "launched.json").exists()
    assert not any(root.glob("*INJECTED*"))


@pytest.mark.parametrize("option", ["--workers", "--render-threads", "--torch-threads"])
def test_launcher_requires_concurrency_value_before_bootstrap(launcher, option):
    root, invoke = launcher
    result = invoke("run", option)
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"Missing value for {option}" in result.stderr
    assert not (root / "preflight.json").exists()


@pytest.mark.parametrize("option", ["--workers", "--render-threads", "--torch-threads"])
@pytest.mark.parametrize("action", [[], ["check"], ["--check-only"], ["check", "--check-only"], ["analyze"]])
def test_launcher_rejects_concurrency_for_check_and_analyze(launcher, option, action):
    root, invoke = launcher
    args = list(action)
    if action == ["analyze"]:
        existing = root / "existing output"
        existing.mkdir()
        args += ["--output", str(existing)]
    result = invoke(*args, option, "2")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "allowed only for smoke/run" in result.stderr
    assert not (root / "preflight.json").exists()
    assert not (root / "launched.json").exists()


@pytest.mark.parametrize("failure,code", [("FAIL_CHECK", 17), ("FAIL_RUN", 19)])
def test_launcher_propagates_failures_through_tee(launcher, failure, code):
    root, invoke = launcher
    result = invoke("smoke", **{failure: str(code)})
    assert result.returncode == code, result.stdout + result.stderr
    assert (root / "launched.json").exists() == (failure == "FAIL_RUN")
    status = next(root.glob("outputs/logs/navigation_tests/*/exit.code"))
    assert status.read_text().strip() == str(code)


def test_standalone_check_only_does_not_launch(launcher):
    root, invoke = launcher
    result = invoke("--check-only", "--renderer=cpu", "--device=cpu")
    assert result.returncode == 0, result.stderr
    assert "--check-only" in json.loads((root / "preflight.json").read_text())
    assert not (root / "launched.json").exists()


def test_resume_and_analyze_forward_existing_output(launcher):
    root, invoke = launcher
    existing = root / "existing output"
    existing.mkdir()
    result = invoke("run", "--resume", "--output", str(existing),
                    "--workers", "1", "--render-threads", "1", "--torch-threads", "1")
    assert result.returncode == 0, result.stderr
    argv = json.loads((root / "launched.json").read_text())["args"]
    assert "--resume" in argv
    for option in ("--workers", "--render-threads", "--torch-threads"):
        assert argv[argv.index(option) + 1] == "1"
    result = invoke("analyze", "--output", str(existing))
    assert result.returncode == 0, result.stderr
    assert "--analysis-only" in json.loads((root / "preflight.json").read_text())


@pytest.mark.parametrize("option", ["--source", "--trajectory"])
def test_source_forwarding(launcher, option):
    root, invoke = launcher
    result = invoke("run", option, "/actual recorded run")
    assert result.returncode == 0, result.stderr
    args = json.loads((root / "launched.json").read_text())["args"]
    assert args[args.index(option)+1] == "/actual recorded run"
    assert option in json.loads((root / "preflight.json").read_text())
    if option == "--source":
        assert args[args.index("--suite")+1] == "main"


@pytest.mark.parametrize("arguments", [
    ["smoke", "--background"], ["run", "--duration", "2"], ["smoke", "--resume"],
    ["run", "--resume"], ["analyze"], ["run", "--check-only"], ["run", "--device=typo"],
    ["run", "--source", "record", "--suite=all"], ["run", "--source=x", "--trajectory=y"],
    ["run", "--seeds-per-cell", "0"], ["smoke", "--duration=nan"], ["run", "--config"],
])
def test_invalid_options_fail_before_bootstrap(launcher, arguments):
    root, invoke = launcher
    result = invoke(*arguments)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (root / "preflight.json").exists()
