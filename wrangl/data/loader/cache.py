import ray


class FullCacheException(Exception):
    """
    Exception that the cache is full
    """

    def __init__(self, cache_size: int):
        return super().__init__('Cache size exceeds maximum of {}!'.format(cache_size))


class Cache:
    """
    A cache of examples to be processed in parallel by workers.
    """

    def __init__(self, pool: ray.util.actor_pool.ActorPool, cache_size: int = 1024):
        """
        Args:
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each examples.
            cache_size: how many examples to keep in the cache.
        """
        self.pool = pool
        self.cache_size = cache_size
        self.in_pool = 0
        if pool is None:
            # emulate a cache with no processors
            self.items = []

    def add(self, o):
        """
        Adds an example `o` to the cache.

        Args:
            o: example to add.
        """
        if self.in_pool + 1 > self.cache_size:
            raise FullCacheException(self.cache_size)
        if self.pool is None:
            self.items.append(o)
        else:
            self.pool.submit(lambda a, v: a.process.remote(v), o)
        self.in_pool += 1

    def is_full(self):
        """
        Returns whether number of examples in the cache exceeds `self.cache_size`.
        """
        return self.in_pool >= self.cache_size

    def get_batch(self, batch_size: int = 10, ordered: bool = True, timeout: int = None):
        """
        Fetches a batch of processed examples from the cache.

        Args:
            batch_size: number of examples to retrieve.
            ordered: whether to retrieve examples in insertion order.
            timeout: number of seconds to wait for retrieval. `None` means wait indefinitely.
        """
        batch = []
        while len(batch) < batch_size and self.in_pool > 0:
            if self.pool is None:
                batch.append(self.items.pop(0))
            else:
                get_next = self.pool.get_next if ordered else self.pool.get_next_unordered
                batch.append(get_next(timeout=timeout))
            self.in_pool -= 1
        return batch
