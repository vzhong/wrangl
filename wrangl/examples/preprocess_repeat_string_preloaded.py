import ray
from wrangl.data import PreloadedDataloader, Processor


strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return raw * 10


def run_no_process():
    loader = PreloadedDataloader(strings, None, cache_size=5)
    out = []
    for batch in loader.batch(2):
        out.extend(batch)
    return out


def run_ordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = PreloadedDataloader(strings, pool, cache_size=5)
    out = []
    for batch in loader.batch(2, ordered=True):
        out.extend(batch)
    return out


def run_unordered():
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = PreloadedDataloader(strings, pool, cache_size=5)
    out = []
    for batch in loader.batch(2, ordered=False):
        out.extend(batch)
    return out


if __name__ == '__main__':
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
