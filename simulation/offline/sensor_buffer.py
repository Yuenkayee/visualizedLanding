import heapq
import itertools
from navigation.common.timestamps import validate_timestamp


class SensorBuffer:
    def __init__(self, max_size=10000):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.heap = []
        self.counter = itertools.count()
        self.max_size = max_size
        self.last_delivered = -1.0

    def push(self, kind, frame):
        timestamp = validate_timestamp(frame.timestamp)
        if timestamp < self.last_delivered:
            raise ValueError("late sensor packet requires rollback/replay")
        if len(self.heap) >= self.max_size:
            raise BufferError("sensor buffer is full")
        heapq.heappush(self.heap, (timestamp, next(self.counter), kind, frame))

    def pop_until(self, timestamp):
        while self.heap and self.heap[0][0] <= timestamp:
            t, _, kind, frame = heapq.heappop(self.heap)
            self.last_delivered = t
            yield kind, frame

    def __len__(self):
        return len(self.heap)
