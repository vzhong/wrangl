#!/usr/bin/env python
import ray


class Processor:

    def process(self):
        raise NotImplementedError()


class Cache:

    def __init__(self, pool, cache_size=1024):
        self.pool = pool
        self.cache_size = cache_size
        self.in_pool = 0

    def add(self, o):
        self.pool.submit(lambda a, v: a.process.remote(v), o)
        self.in_pool += 1

    def is_full(self):
        return self.in_pool >= self.cache_size

    def get_batch(self, batch_size=10, random=False):
        batch = []
        while len(batch) < batch_size and self.pool.has_next():
            get_next = self.pool.get_next_unordered if random else self.pool.get_next
            batch.append(get_next())
            self.in_pool -= 1
        return batch


class Dataloader:

    def __init__(self, pool: ray.util.ActorPool, cache_size: int = 1024):
        self.cache = Cache(pool, cache_size=cache_size)
        self.reset()

    def load_next(self):
        raise NotImplementedError()

    def reset(self):
        return

    def batch(self, batch_size: int = 1, random: bool = False):
        self.reset()
        while True:
            while not self.cache.is_full():
                o = self.load_next()
                if o is None:
                    break
                self.cache.add(o)
            batch = self.cache.get_batch(batch_size, random=random)
            if not batch:
                raise StopIteration
            else:
                yield batch

    @classmethod
    def init(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def shutdown(cls):
        ray.shutdown()
