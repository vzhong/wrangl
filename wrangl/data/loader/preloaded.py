import ray
import random
from .dataloader import Dataloader


class PreloadedDataloader(Dataloader):
    """
    A preloaded dataset from a list input.
    """

    def __init__(self, preloaded_data: list, pool: ray.util.ActorPool = None, cache_size: int = 1024, shuffle_seed: int = None):
        """
        Preloads another dataloader.

        Args:
            preloaded_data: data to process.
            pool: pool of `wrangl.data.loader.processor.Processor`s to process each example. If not specified, then no processing will be done.
            cache_size: how many examples to keep in the cache.
            shuffle_seed: if specified, then the seed will be used to set a RNG for shuffling on reset.
        """
        super().__init__(pool, cache_size=cache_size)
        self.shuffle_seed = shuffle_seed
        self.preloaded_data = preloaded_data
        self.current_idx = 0
        if self.shuffle_seed is not None:
            self.rng = random.Random(self.shuffle_seed)

    def __len__(self):
        return len(self.preloaded_data)

    def reset(self):
        super().reset()
        self.current_idx = 0
        if self.shuffle_seed is not None:
            self.rng.shuffle(self.preloaded_data)

    def load_next(self):
        if self.current_idx < len(self.preloaded_data):
            ret = self.preloaded_data[self.current_idx]
            self.current_idx += 1
            return ret
        else:
            return None

    def populate_from_loader(cls, loader: Dataloader):
        """
        Preloads from another dataloader.

        Args:
            loader: loader to preload, note that this argument loader will not be preprocessed.
        """
        data = []
        while True:
            o = loader.load_next()
            if o is None:
                break
            self.preloaded_data.append(o)
