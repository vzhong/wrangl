from collections import defaultdict


class RunningAverage:
    """
    Computes exponential moving averages averages.
    """

    def __init__(self, mix_rate: float = 0.95):
        self.mix_rate = mix_rate
        self.avgs = defaultdict(lambda: None)

    def record(self, name: str, value: float, ignore_nan=True):
        """
        Args:
            name: name of value.
            value: value to record.
            ignore_nan: ignore nan values and do not record them (they will mess up the averages).
        """
        if ignore_nan and (value != value or value is None):
            return self.avgs[name]
        if self.avgs.get(name) is None:
            self.avgs[name] = value
        else:
            self.avgs[name] = self.mix_rate * self.avgs[name] + (1-self.mix_rate) * value
        return self.avgs[name]
