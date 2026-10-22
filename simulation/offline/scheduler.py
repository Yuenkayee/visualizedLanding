"""Integer tick scheduler avoids modulo and accumulated floating point drift."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Event:
    timestamp: float
    kind: str
    index: int


class Scheduler:
    def __init__(self, rates, duration):
        if (
            duration < 0
            or not math.isfinite(duration)
            or any(r <= 0 or not math.isfinite(r) for r in rates.values())
        ):
            raise ValueError("invalid rates/duration")
        self.rates = dict(rates)
        self.duration = duration

    def __iter__(self):
        import heapq

        heap = [(0.0, order, name, 0) for order, name in enumerate(self.rates)]
        heapq.heapify(heap)
        while heap:
            t, order, name, index = heapq.heappop(heap)
            if t > self.duration + 1e-10:
                continue
            yield Event(t, name, index)
            following = (index + 1) / self.rates[name]
            if following <= self.duration + 1e-10:
                heapq.heappush(heap, (following, order, name, index + 1))

    def groups(self):
        """All events at one state time form one read/update/feedback transaction."""
        pending = []
        for event in self:
            if pending and not math.isclose(
                event.timestamp, pending[0].timestamp, rel_tol=0, abs_tol=1e-10
            ):
                yield pending
                pending = []
            pending.append(event)
        if pending:
            yield pending
