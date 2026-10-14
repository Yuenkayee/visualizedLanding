"""JSON-safe bridge state. Truth is evaluation/rendering-only, never navigation input."""

from dataclasses import dataclass
import numpy as np
from navigation.common.contracts import finite_array
from navigation.common.frames import transform
from navigation.common.timestamps import validate_timestamp


@dataclass
class SensorTruth:
    timestamp: float
    T_world_deck: np.ndarray
    T_deck_camera: np.ndarray
    velocity_deck: np.ndarray
    T_body_camera: np.ndarray | None = None
    gear_position_deck: np.ndarray | None = None
    gear_velocity_deck: np.ndarray | None = None

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        for name in ("T_world_deck", "T_deck_camera"):
            T = finite_array(getattr(self, name), (4, 4), name)
            transform(T[:3, :3], T[:3, 3])
            if not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("invalid homogeneous transform")
            setattr(self, name, T)
        self.velocity_deck = finite_array(self.velocity_deck, (3,), "velocity")
        if self.T_body_camera is not None:
            from navigation.common.contracts import GimbalSample
            self.T_body_camera = GimbalSample(self.timestamp, self.T_body_camera).T_body_camera
        for name in ("gear_position_deck", "gear_velocity_deck"):
            if getattr(self, name) is not None:
                setattr(self, name, finite_array(getattr(self, name), (3,), name))

    @classmethod
    def from_dict(cls, row):
        return cls(
            T_body_camera=row.get("T_body_camera"),
            gear_position_deck=row.get("gear_position_deck"),
            gear_velocity_deck=row.get("gear_velocity_deck"), **{
                k: row[k]
                for k in ("timestamp", "T_world_deck", "T_deck_camera", "velocity_deck")
            }
        )

    def to_dict(self):
        return dict(
            timestamp=self.timestamp,
            T_world_deck=self.T_world_deck.tolist(),
            T_deck_camera=self.T_deck_camera.tolist(),
            velocity_deck=self.velocity_deck.tolist(),
            T_body_camera=self.T_body_camera.tolist() if self.T_body_camera is not None else None,
            gear_position_deck=self.gear_position_deck.tolist() if self.gear_position_deck is not None else None,
            gear_velocity_deck=self.gear_velocity_deck.tolist() if self.gear_velocity_deck is not None else None,
        )
