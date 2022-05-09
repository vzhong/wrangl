#!/usr/bin/env python
import ray
import json
import hydra
import logging
from wrangl.learn import SupervisedModel
from wrangl.data import IterableDataset, Processor


logger = logging.getLogger(__name__)


@ray.remote
class MyProcessor(Processor):

    def process(self, row):
        return dict(sent=row['sentence'], label_text=row['gold_label'], label_idx=['positive', 'neutral', 'negative'].index(row['gold_label']))


def load_data(fname):
    data = []
    with open(fname) as f:
        for line in f:
            row = json.loads(line)
            if row['gold_label'] not in {None, 'mixed'}:
                data.append(row)
    return data


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(cfg.num_workers)])
    train = IterableDataset(load_data(cfg.ftrain), pool=pool, shuffle=True)
    val = IterableDataset(load_data(cfg.feval), pool=pool)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
