import ray
import unittest
from wrangl.dataloader import Dataloader, Processor


class MyDataloader(Dataloader):

    def __init__(self, strings, pool: ray.util.ActorPool, cache_size: int = 1024):
        super().__init__(pool, cache_size=cache_size)
        self.current = 0
        self.strings = strings

    def reset(self):
        self.current = 0

    def load_next(self):
        if self.current < len(self.strings):
            ret = self.strings[self.current]
            self.current += 1
            return ret
        else:
            return None


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return raw * 10


if __name__ == '__main__':
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    loader = MyDataloader(strings, pool, cache_size=5)

    out = []
    for batch in loader.batch(2):
        out.extend(batch)

    expect = [s * 10 for s in strings]
    unittest.TestCase().assertListEqual(expect, out)
