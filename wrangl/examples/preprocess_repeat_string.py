import ray
from wrangl.data import IterableDataset, Processor


class MyDataset(IterableDataset):

    strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def __init__(self, pool: ray.util.ActorPool, cache_size: int = 1024, shuffle: bool = False, timeout: int = 10):
        super().__init__(self.strings, pool, cache_size, shuffle, timeout)


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return raw * 10


def run_no_process():
    out = []
    for s in MyDataset.strings:
        out.append(s * 10)
    return out


def run_ordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    d = MyDataset(pool, cache_size=5, shuffle=False)
    return list(d)


def run_unordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    d = MyDataset(pool, cache_size=5, shuffle=True)
    return list(d)


if __name__ == '__main__':
    expect = [s * 10 for s in MyDataset.strings]

    out = run_no_process()
    assert expect == out
    print(out)

    out = run_ordered()
    assert expect == out
    print(out)

    out = run_unordered()
    assert set(expect) == set(out)
    print(out)
