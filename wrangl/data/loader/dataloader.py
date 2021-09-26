import ray
from .cache import Cache


class Dataloader:
    """
    A generic dataloader that processes examples in parallel.
    """

    def __init__(self, pool: ray.util.ActorPool, cache_size: int = 1024):
        """
        Args:
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each examples.
            cache_size: how many examples to keep in the cache.
        """
        self.cache = Cache(pool, cache_size=cache_size)

    def load_next(self):
        """
        Loads unprocessed examples.

        Returns:
            The next example. Otherwise returns `None` if there are no more examples to load.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the loader to load examples from scratch.
        """
        return

    def batch(self, batch_size: int = 1, ordered: bool = True, timeout=None):
        """
        A generator of batches of processed examples.

        Args:
            batch_size: how many examples to yield.
            ordered: whether to yield examples in the order they were loaded.
            timeout: how long to wait for examples to load. `None` means wait indefinitely.

        Returns:
            A generator of list of examples.
        """
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
        """
        Closes the loader and cleans up open resources.
        """
        pass
