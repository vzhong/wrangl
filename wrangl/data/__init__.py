"""
Wrangl consists of two data types:

- `wrangl.data.dataset` are Torch Dataset wrappers that iterate over and process raw data
- `wrangl.data.processor` are Ray processors that transform each piece of the raw data

.. include:: ../examples/preprocess/README.md
"""
from .dataset import IterableDataset, SQLDataset, FileDataset
from .processor import Processor
