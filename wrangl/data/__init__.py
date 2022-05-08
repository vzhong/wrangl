"""
Wrangl consists of two data types:

- `wrangl.data.dataset` are Torch Dataset wrappers that iterate over and process raw data
- `wrangl.data.processor` are Ray processors that transform each piece of the raw data

See [here for examples](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/preprocess) on how to combine these to preprocess raw data in parallel.
"""
from .dataset import IterableDataset, SQLDataset, FileDataset
from .processor import Processor
