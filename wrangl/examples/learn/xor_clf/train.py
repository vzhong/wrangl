#!/usr/bin/env python
import ray
import hydra
import torch
import random
import logging
import warnings
from wrangl.learn import SupervisedModel
from wrangl.data import IterableDataset, Processor


warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


logger = logging.getLogger(__name__)


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return dict(
            feat=torch.tensor([raw['x'], raw['y']]),
            label=1 if raw['x'] > 0 and raw['y'] > 0 else 0,
        )


def generate_dataset(n, seed=0):
    rng = random.Random(seed)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    iterable = [dict(x=rng.uniform(-1, 1), y=rng.uniform(-1, 1)) for _ in range(n)]
    return IterableDataset(iterable, pool)


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    train = generate_dataset(10000)
    val = generate_dataset(1000)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
