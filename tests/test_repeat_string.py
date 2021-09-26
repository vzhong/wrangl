import ray
import unittest
from wrangl.data import Dataloader, Processor


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


class TestRepeatString(unittest.TestCase):


    def setUp(self):
        ray.init()
        self.strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def tearDown(self):
        ray.shutdown()

    def test_no_pool_loader(self):
        serial_loader = MyDataloader(self.strings, None, cache_size=5)
        serial_out = []
        for batch in serial_loader.batch(2):
            serial_out.extend(batch)
        self.assertListEqual(self.strings, serial_out)

    def test_repeat(self):
        pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
        loader = MyDataloader(self.strings, pool, cache_size=5)
        out = []
        for batch in loader.batch(2, ordered=True):
            out.extend(batch)
        expect = [s * 10 for s in self.strings]
        self.assertListEqual(expect, out)

    def test_repeat_unordered(self):
        pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
        loader = MyDataloader(self.strings, pool, cache_size=5)
        out = []
        for batch in loader.batch(2, ordered=False):
            out.extend(batch)
        expect = [s * 10 for s in self.strings]
        self.assertSetEqual(set(expect), set(out))
