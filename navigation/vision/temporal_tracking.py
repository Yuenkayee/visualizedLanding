import cv2
import numpy as np


class TemporalTracker:
    def __init__(self):
        self.gray = None
        self.points = None

    def update(self, image, detected=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        points = detected
        if points is None and self.gray is not None and self.points is not None:
            p = self.points.astype("float32").reshape(-1, 1, 2)
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, p, None)
            if nxt is not None:
                back, ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None)
                if (
                    back is not None
                    and status.all()
                    and ok.all()
                    and np.max(np.linalg.norm(back - p, axis=2)) < 1.5
                ):
                    points = nxt.reshape(-1, 2)
        self.gray = gray
        self.points = None if points is None else np.asarray(points).copy()
        return self.points

    def reset(self):
        self.gray = None
        self.points = None
