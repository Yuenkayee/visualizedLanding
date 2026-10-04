import json
from pathlib import Path
import numpy as np


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w")
        return self

    def write(self, row):
        if self.stream is None:
            raise RuntimeError("logger not open")
        self.stream.write(json.dumps(serializable(row), allow_nan=False) + "\n")
        self.stream.flush()

    def __exit__(self, *exc):
        if self.stream:
            self.stream.close()
            self.stream = None


def read_log(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
