# Wrangl

Ray-based parallel data preprocessing for NLP and ML.
See [docs here](https://www.victorzhong.com/wrangl).

```bash
pip install -e .  # add [dev] if you want to run tests and build docs.

# for latest
pip install git+https://github.com/vzhong/wrangl
```

See [examples](https://github.com/vzhong/wrangl/tree/main/example) and [test cases](https://github.com/vzhong/wrangl/tests) for usage.


## How to process in parallel


Here is a trivial example where we simple repeat the input many times.
```
import ray
from wrangl.data import Dataloader, Processor


# define how to load your data
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


# define how a worker processes each example
@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return raw * 10


if __name__ == '__main__':
    ray.init()
    strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = MyDataloader(self.strings, pool, cache_size=5)
    out = []
    for batch in loader.batch(2, ordered=True):  # you can use ordered=False here for faster speed if you do not care about retrieving examples in order.
        out.extend(batch)
    expect = [s * 10 for s in self.strings]
    assert expect == out
```


## Additional utilities

Annotate data in commandline:

```
wannotate -h
```


## Run tests

```
python -m unittest discover tests
```
