class Epsilon:

    def __init__(self, start, end, decay):
        self._start = start
        self._end = end
        self._decay = decay
        self._current = start

    def reset(self):
        self._current = self._start

    def __next__(self):
        self._current = max(self._current * self._decay, self._end)
        return self._current

    @property
    def value(self):
        return self._current

    @property
    def start(self):
        return self._start

    @property
    def decay(self):
        return self._decay

    @property
    def end(self):
        return self._end

    def __str__(self):
        return f'Epsilon(start={self.start}, end={self.end}, decay={self.decay}, current={self.value})'

    def __repr__(self):
        return self.__str__()
