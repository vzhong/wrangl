from collections import defaultdict


class RunningAverage:
    """
    Computes running averages.
    """

    def __init__(self, mix_rate=0.95):
        self.mix_rate = mix_rate
        self.avgs = defaultdict(lambda: None)

    def record(self, name, value):
        if name not in self.avgs:
            self.avgs[name] = value
        else:
            self.avgs[name] = self.mix_rate * self.avgs[name] + (1-self.mix_rate) * value
        return self.avgs[name]
