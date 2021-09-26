import ray
import bz2
from pathlib import Path
from typing import List, Union
from .dataloader import Dataloader


class Fileloader(Dataloader):

    def __init__(self, fnames: List[Union[str, Path]], pool: ray.util.ActorPool, cache_size: int = 1024):
        """
        Loads lines from input files.

        Args:
            fnames: list of text or `bz2` files to loader from.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each examples.
            cache_size: how many examples to keep in the cache.
        """
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
