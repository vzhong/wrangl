#!/usr/bin/env python
import ray
import bz2
from pathlib import Path
from typing import Union, List


class Processor:

    def process(self):
        raise NotImplementedError()


class Cache:

    def __init__(self, pool, cache_size=1024):
        self.pool = pool
        self.cache_size = cache_size
        self.in_pool = 0
        if pool is None:
            # emulate a cache with no processors
            self.items = []

    def add(self, o):
        if self.pool is None:
            self.items.append(o)
        else:
            self.pool.submit(lambda a, v: a.process.remote(v), o)
        self.in_pool += 1

    def is_full(self):
        return self.in_pool >= self.cache_size

    def get_batch(self, batch_size=10, ordered=True, timeout=None):
        batch = []
        while len(batch) < batch_size and self.in_pool > 0:
            if self.pool is None:
                batch.append(self.items.pop(0))
            else:
                get_next = self.pool.get_next if ordered else self.pool.get_next_unordered
                batch.append(get_next(timeout=timeout))
            self.in_pool -= 1
        return batch


class Dataloader:

    def __init__(self, pool: ray.util.ActorPool, cache_size: int = 1024):
        self.cache = Cache(pool, cache_size=cache_size)

    def load_next(self):
        raise NotImplementedError()

    def reset(self):
        return

    def batch(self, batch_size: int = 1, ordered: bool = True, timeout=None):
        self.reset()
        while True:
            while not self.cache.is_full():
                o = self.load_next()
                if o is None:
                    break
                self.cache.add(o)
            batch = self.cache.get_batch(batch_size, ordered=ordered, timeout=timeout)
            if not batch:
                return
            else:
                yield batch

    def close(self):
        pass


class Fileloader(Dataloader):

    def __init__(self, fnames: List[Union[str, Path]], pool: ray.util.ActorPool, cache_size: int = 1024):
        assert fnames
        super().__init__(pool, cache_size=cache_size)
        self.fnames = fnames
        self.current_file_idx = self.file = None
        self.reset()

    def load_next(self):
        try:
            line = next(self.file).rstrip('\n')
            return line
        except StopIteration:
            self.current_file_idx += 1
            if self.current_file_idx >= len(self.fnames):
                return None
            else:
                self.file = self.open_file(self.fnames[self.current_file_idx])
                return self.load_next()

    def open_file(self, fname):
        return bz2.open(fname, 'rt') if fname.endswith('.bz2') else open(fname, 'rt')

    def reset(self):
        self.current_file_idx = 0
        self.file = self.open_file(self.fnames[self.current_file_idx])

    def close(self):
        super().close()
        self.file.close()
