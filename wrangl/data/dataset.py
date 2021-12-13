import ray
import bz2
import sqlite3
from typing import Union, List
from collections.abc import Iterable
from pathlib import Path
from torch.utils.data import IterableDataset as Base


class FullCacheException(Exception):
    pass


class Dataset(Base):
    """
    A generic torch.utils.data.IterableDataset that processes examples in parallel.
    """

    def __init__(self, pool: ray.util.actor_pool.ActorPool, cache_size: int = 1024, shuffle: bool = False, timeout: int = 10):
        """
        Args:
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each examples.
            cache_size: how many examples to keep in the cache.
            shuffle: pseudo random order (local shuffling within cache).
            timeout: processor timeout.
        """
        self.pool = pool
        self.cache_size = cache_size
        self.shuffle = shuffle
        self.timeout = timeout
        self.in_pool = 0

    def add(self, o):
        """
        Adds an example `o` to be processed.

        Args:
            o: example to add.
        """
        if self.cache_is_full():
            raise FullCacheException('Cache is full at {}'.format(self.cache_size))
        self.pool.submit(lambda a, v: a.process.remote(v), o)
        self.in_pool += 1

    def cache_is_full(self):
        """
        Returns whether number of examples in the cache exceeds `self.cache_size`.
        """
        return self.in_pool >= self.cache_size

    def cache_is_empty(self):
        """
        Returns whether cache is empty.
        """
        return self.in_pool == 0

    def pop(self, ordered: bool = True, timeout: int = None):
        """
        Fetches a batch of processed examples from the cache.

        Args:
            ordered: whether to retrieve examples in insertion order.
            timeout: number of seconds to wait for retrieval. `None` means wait indefinitely.
        """
        get_next = self.pool.get_next if ordered else self.pool.get_next_unordered
        o = get_next(timeout=timeout)
        self.in_pool -= 1
        return o

    def iterate_unprocessed(self):
        """
        Loads unprocessed examples.

        Returns:
            Iterator of the next example.
        """
        raise NotImplementedError()

    def __iter__(self):
        for o in self.iterate_unprocessed():
            if self.cache_is_full():
                yield self.pop(ordered=not self.shuffle, timeout=self.timeout)
            self.add(o)

        while not self.cache_is_empty():
            yield self.pop(ordered=not self.shuffle, timeout=self.timeout)

    def close(self):
        """
        Clean up any resources.
        """
        pass


class IterableDataset(Dataset):

    def __init__(self, iterable: Iterable, pool: ray.util.ActorPool, cache_size: int = 1024, shuffle: bool = False, timeout: int = 10):
        """
        Processes list of examples.

        Args:
            iterable: list of examples to process.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each example. If not specified, then no processing will be done.
            cache_size: how many examples to keep in the cache.
            shuffle: pseudo random order (local shuffling within cache).
            timeout: processor timeout.
        """
        super().__init__(pool, cache_size=cache_size, shuffle=shuffle, timeout=timeout)
        self.iterable = iterable

    def iterate_unprocessed(self):
        """
        Loads unprocessed examples.

        Returns:
            Iterator of the next example.
        """
        for ex in self.iterable:
            yield ex


class SQLDataset(Dataset):

    def __init__(self, fdb: Union[str, Path], query: str, pool: ray.util.ActorPool, cache_size: int = 1024, shuffle: bool = False, timeout: int = 10):
        """
        Loads rows from SQL database.

        Args:
            fdb: path to SQL database.
            query: SQL query to retrieve data.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each example. If not specified, then no processing will be done.
            cache_size: how many examples to keep in the cache.
            shuffle: pseudo random order (local shuffling within cache).
            timeout: processor timeout.
        """
        super().__init__(pool, cache_size=cache_size, shuffle=shuffle, timeout=timeout)
        self.fdb = fdb
        self.query = query
        self.db = sqlite3.connect(self.fdb, isolation_level=None)
        self.cursor = self.db.cursor()
        self.rows = None

    def iterate_unprocessed(self):
        """
        Loads unprocessed examples.

        Returns:
            Iterator of the next example.
        """
        rows = self.cursor.execute(self.query)
        for row in rows:
            yield row

    def close(self):
        super().close()
        self.cursor.close()
        self.db.close()


class FileDataset(Dataset):

    def __init__(self, fnames: List[Union[str, Path]], pool: ray.util.ActorPool, cache_size: int = 1024, shuffle: bool = False, timeout: int = 10):
        """
        Loads lines from input files.

        Args:
            fnames: list of text or `bz2` files to loader from.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each example. If not specified, then no processing will be done.
            cache_size: how many examples to keep in the cache.
            shuffle: pseudo random order (local shuffling within cache).
            timeout: processor timeout.
        """
        super().__init__(pool, cache_size=cache_size, shuffle=shuffle, timeout=timeout)
        assert fnames
        self.fnames = fnames

    def iterate_unprocessed(self):
        """
        Loads unprocessed examples.

        Returns:
            Iterator of the next example.
        """
        for fname in self.fnames:
            with self.open_file(fname) as f:
                for line in f:
                    yield line

    def open_file(self, fname):
        return bz2.open(fname, 'rt') if fname.endswith('.bz2') else open(fname, 'rt')
