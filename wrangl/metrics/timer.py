from collections import defaultdict


class Timer:

    def __init__(self, mix_rate=0.95):
        self.mix_rate = mix_rate
        self.elapsed = defaultdict(lambda: None)

    def record(self, name, elapsed):
        if name not in self.elapsed:
            self.elapsed[name] = elapsed
        else:
            self.elapsed[name] = self.mix_rate * self.elapsed[name] + (1-self.mix_rate) * elapsed
        return self.elapsed[name]
