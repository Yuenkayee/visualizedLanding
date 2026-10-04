"""MATLAB Engine is optional; startup occurs only when MATLAB backend is selected."""

import json
from pathlib import Path


class MatlabSession:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.engine = None

    def __enter__(self):
        try:
            import matlab.engine
        except ImportError as exc:
            raise RuntimeError(
                "install requirements-matlab.txt with MATLAB R2024b present"
            ) from exc
        try:
            self.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
            self.engine.addpath(str(self.root / "simulation/matlab"), nargout=0)
            self.engine.setup_paths(str(self.root), nargout=0)
        except Exception:
            if self.engine is not None:
                self.engine.quit()
                self.engine = None
            raise
        return self

    def call(self, name, *args):
        if self.engine is None:
            raise RuntimeError("MATLAB session is not open")
        values = [
            json.dumps(a, allow_nan=False) if isinstance(a, (dict, list)) else a
            for a in args
        ]
        raw = getattr(self.engine, name)(*values, nargout=1)
        return json.loads(raw) if isinstance(raw, str) and raw else raw

    def __exit__(self, *exc):
        if self.engine is not None:
            try:
                self.engine.finalize_external_model(nargout=1)
            finally:
                self.engine.quit()
                self.engine = None
