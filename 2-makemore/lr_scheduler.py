class LearningRateScheduler:
    def __init__(self, schedule: list[tuple[int, float]]) -> None:
        self._epoch = 0
        self._lrs: list[float] = []
        for epochs, lr in schedule:
            self._lrs.extend([lr] * epochs)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, float]:
        if self._epoch > len(self._lrs) - 1:
            raise StopIteration
        try:
            return self._epoch, self._lrs[self._epoch]
        finally:
            self._epoch += 1
