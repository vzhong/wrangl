#!/usr/bin/env python
import ray
import json
import hydra
import logging
from wrangl.learn import SupervisedModel
from wrangl.data import FileDataset, Processor


logger = logging.getLogger(__name__)


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        row = json.loads(raw)
        return dict(sent=row['sentence'], label_text=row['gold_label'], label_idx=['positive', 'neutral', 'negative', 'mixed'].index(row['gold_label']))


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(cfg.num_workers)])
    train = FileDataset([cfg.ftrain], pool=pool, shuffle=True)
    val = FileDataset([cfg.feval], pool=pool)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
