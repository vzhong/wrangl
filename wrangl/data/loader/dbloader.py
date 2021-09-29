import ray
import sqlite3
from pathlib import Path
from typing import List, Union
from .dataloader import Dataloader


class SQLDBloader(Dataloader):

    def __init__(self, fdb: Union[str, Path], query: str, pool: ray.util.ActorPool = None, cache_size: int = 1024):
        """
        Loads rows from SQL database.

        Args:
            fdb: path to SQL database.
            query: SQL query to retrieve data.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each example. If not specified, then no processing will be done.
            cache_size: how many examples to keep in the cache.
        """
        super().__init__(pool, cache_size=cache_size)
        self.fdb = fdb
        self.query = query
        self.db = sqlite3.connect(self.fdb, isolation_level=None)
        self.cursor = self.db.cursor()
        self.rows = None
        self.reset()

    def load_next(self):
        try:
            line = next(self.rows)
            return line
        except StopIteration:
            return None

    def reset(self):
        self.rows = self.cursor.execute(self.query)

    def close(self):
        super().close()
        self.cursor.close()
        self.db.close()
