import ray
from wrangl.data import Dataloader, Processor


strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']


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


def run_no_process():
    loader = MyDataloader(strings, None, cache_size=5)
    out = []
    for batch in loader.batch(2):
        out.extend(batch)
    return out


def run_ordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = MyDataloader(strings, pool, cache_size=5)
    out = []
    for batch in loader.batch(2, ordered=True):
        out.extend(batch)
    return out


def run_unordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = MyDataloader(strings, pool, cache_size=5)
    out = []
    for batch in loader.batch(2, ordered=False):
        out.extend(batch)
    return out


if __name__ == '__main__':
    ray.init()

    out = run_no_process()
    assert strings == out
    print(out)

    out = run_ordered()
    expect = [s * 10 for s in strings]
    assert expect == out
    print(out)

    expect = [s * 10 for s in strings]
    assert set(expect) == set(out)
    print(out)
