"""Bound short runs at the sampler and explicitly own DataLoader worker lifetime."""

from itertools import islice

from torch.utils.data import DataLoader, Sampler


class LimitedBatchSampler(Sampler):
    """Do not submit work past a short run's limit, including prefetched work.

    Wrapping the batch sampler retains shuffling on every training epoch and
    handles a partial final batch exactly as the underlying sampler does.
    """

    def __init__(self, batch_sampler, limit):
        if limit < 1:
            raise ValueError("batch limit must be positive")
        self.batch_sampler = batch_sampler
        self.limit = limit

    def __iter__(self):
        return islice(self.batch_sampler, self.limit)

    def __len__(self):
        return min(len(self.batch_sampler), self.limit)


class TrainingDataLoader(DataLoader):
    """Retain the active iterator so its shutdown errors cannot hide in __del__.

    PyTorch 2.5/2.8 expose no public DataLoader.close(). Keep the private
    _shutdown_workers dependency here, including the non-persistent case where
    DataLoader itself does not keep a reference to the iterator.
    """

    def __iter__(self):
        iterator = super().__iter__()
        self._training_iterator = iterator
        return iterator

    def close(self):
        iterator = getattr(self, "_training_iterator", None)
        if iterator is None or not self.num_workers:
            self._training_iterator = None
            return
        workers = list(iterator._workers)
        try:
            iterator._shutdown_workers()
        finally:
            # _shutdown_workers may terminate a stuck worker without joining it.
            # Reap all children even if one shutdown raises; check exit codes
            # below instead of silently accepting an abort/forced termination.
            for worker in workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
            self._training_iterator = None
            self._iterator = None
        failed = [
            (worker.pid, worker.exitcode) for worker in workers if worker.exitcode != 0
        ]
        if failed:
            raise RuntimeError(
                f"DataLoader worker shutdown failed (pid, exitcode): {failed}"
            )
