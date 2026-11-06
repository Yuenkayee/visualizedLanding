"""Exercise the coordinator's imports and numerical operations in a fresh Python."""

import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "LANDING_NAVIGATION_PROBE="

# Import the real entry point first: preloading torch/scipy.stats can hide an
# import-order failure that occurs in the fresh process launched by the wrapper.
# Importing main.run does not start MATLAB, render images or run a simulation.
_PROBE = r'''
import importlib.metadata, importlib.util, json, os, sys, traceback
report = dict(status="failed", python=sys.executable, python_version=sys.version,
              stage="main.run import", checks=[],
              LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
              LD_PRELOAD=os.environ.get("LD_PRELOAD", ""),
              PYTHONPATH=os.environ.get("PYTHONPATH", ""))
try:
    import main.run
    report["checks"].append("main.run import")
    import numpy as np
    from scipy.stats import chi2
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation
    from navigation.fusion.measurement_gating import mahalanobis_gate

    report["stage"] = "NumPy/SciPy operations"
    np.testing.assert_allclose(chi2.ppf(0.95, 3), 7.814727903251179, rtol=1e-12)
    np.testing.assert_allclose(interp1d([0., 1., 2.], [0., 2., 4.])(0.5), 1.)
    np.testing.assert_allclose(Rotation.from_matrix(np.eye(3)).as_rotvec(), np.zeros(3))
    if mahalanobis_gate(np.ones(3), np.eye(3)) != (True, 3.):
        raise RuntimeError("Mahalanobis gate rejected an inlier")
    if mahalanobis_gate([10., 0., 0.], np.eye(3)) != (False, 100.):
        raise RuntimeError("Mahalanobis gate accepted an outlier")
    report["checks"].append("NumPy/SciPy operations")

    report["stage"] = "OpenCV/NumPy operations"
    import cv2
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    gray = cv2.cvtColor(cv2.resize(rgb, (2, 2)), cv2.COLOR_RGB2GRAY)
    np.testing.assert_array_equal(gray, np.full((2, 2), 76, dtype=np.uint8))
    report["checks"].append("OpenCV/NumPy operations")

    report["stage"] = "Torch/NumPy operations"
    import torch
    array = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal((torch.from_numpy(array) + 1).numpy(), array + 1)
    report["checks"].append("Torch/NumPy operations")
    report["stage"] = "matplotlib import"
    import matplotlib
    report["checks"].append("matplotlib import")
    report.update(status="ready", stage="complete")
except Exception as exc:
    report.update(error_type=type(exc).__name__, error=str(exc),
                  traceback=traceback.format_exc(limit=12))
finally:
    # Distribution metadata can look healthy while native imports fail. Record
    # both the installed versions and the actual loaded modules, even on failure.
    report["packages"] = {}
    for name, distribution in [("numpy", "numpy"), ("scipy", "scipy"),
                               ("cv2", "opencv-python-headless"), ("torch", "torch"),
                               ("matplotlib", "matplotlib")]:
        info = {}
        try:
            info["installed_version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            info["installed_version"] = None
        module = sys.modules.get(name)
        if module is not None:
            info.update(version=getattr(module, "__version__", None),
                        path=getattr(module, "__file__", None))
        else:
            spec = importlib.util.find_spec(name)
            info["path"] = spec.origin if spec else None
        report["packages"][name] = info
    report["native_modules"] = {
        name: getattr(sys.modules[name], "__file__", None)
        for name in ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath",
                     "scipy.interpolate._dfitpack", "scipy.interpolate._fitpack")
        if name in sys.modules}
    print("LANDING_NAVIGATION_PROBE=" + json.dumps(report), flush=True)
sys.exit(0 if report["status"] == "ready" else 1)
'''


def probe_navigation(python, report_path=None):
    """Read-only check using the same interpreter, cwd and library env as run."""
    print("[environment] Checking navigation imports and numerical operations", flush=True)
    result = subprocess.run([str(python), "-c", _PROBE], cwd=ROOT,
                            capture_output=True, text=True)
    reports = [line[len(PREFIX):] for line in result.stdout.splitlines()
               if line.startswith(PREFIX)]
    report = json.loads(reports[-1]) if reports else dict(
        status="failed", python=str(python),
        error="Navigation process exited before writing its report")
    report["returncode"] = result.returncode
    if result.returncode:
        report["status"] = "failed"
    if result.stderr:
        report["stderr"] = result.stderr[-4000:]
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "ready":
        print(json.dumps(report, indent=2), flush=True)
        location = f"; diagnostic report: {report_path}" if report_path else ""
        repair = shlex.join([str(python), "-m", "pip", "install", "--no-cache-dir",
                             "--force-reinstall", "--no-deps", "--only-binary=:all:",
                             "numpy==1.26.4", "scipy==1.13.1"])
        raise RuntimeError(
            f"Navigation runtime check failed at {report.get('stage', 'startup')}: "
            f"{report.get('error', '')}{location}. "
            f"For NumPy/SciPy installation failures, reinstall with this interpreter "
            f"and rerun the shell check: {repair}"
        )
    print("[environment] Navigation imports and numerical operations: OK", flush=True)
    return report
