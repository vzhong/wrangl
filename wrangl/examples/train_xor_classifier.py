import ray
import torch
import random
import pprint
from torch import nn
from torch.nn import functional as F
from wrangl.learn import SupervisedModel
from wrangl.data import Dataloader, Processor
from wrangl.metrics import Accuracy


# here is a silly dataloader that generates the XOR classification task
class MyDataloader(Dataloader):

    def __init__(self, pool: ray.util.ActorPool, cache_size: int = 1024, size: int = 2048, seed: int = 0):
        super().__init__(pool, cache_size=cache_size)
        rng = random.Random(seed)
        self.current = 0
        self.xs = [dict(a=rng.choice([0, 1]), b=rng.choice([0, 1])) for _ in range(size)]

    def reset(self):
        self.current = 0

    def load_next(self):
        if self.current < len(self.xs):
            ret = self.xs[self.current]
            self.current += 1
            return ret
        else:
            return None


# as a silly example, the processor will attach a label to each input
@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        x = torch.tensor([raw['a'], raw['b']])
        y = torch.tensor([1 if (raw['a'] and not raw['b']) or (raw['b'] and not raw['a']) else 0])
        return dict(x=x, y=y)


class MyModel(SupervisedModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 2),
        )

    def __call__(self, batch):
        x = torch.stack([ex['x'] for ex in batch], dim=0).float()
        out = self.net(x)
        return dict(scores=out)

    def compute_loss(self, batch, out):
        y = torch.cat([ex['y'] for ex in batch], dim=0)
        return F.cross_entropy(out['scores'], y)

    def extract_pred(self, batch, out):
        return [dict(y=i) for i in out['scores'].max(1)[1].tolist()]

    def get_metrics(self):
        return Accuracy('y')


def main(args):
    torch.manual_seed(1)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(args.num_procs)])
    eval_loader = MyDataloader(pool, cache_size=args.cache_size, seed=1)
    if not args.test:
        train_loader = MyDataloader(pool, cache_size=args.cache_size, seed=0)
        model = MyModel(args)
        model.run_train(train_loader, eval_loader)
    else:
        model = MyModel.load_checkpoint(args.test, override_hparams=args)
        eval_preds, eval_loss = model.run_preds(eval_loader, compute_loss=True, verbose=True)
        eval_metrics = model.get_metrics()(eval_preds)
        eval_metrics['loss'] = eval_loss
        pprint.pprint(eval_metrics)
        return eval_metrics

if __name__ == '__main__':
    parser = MyModel.get_parser(num_train_steps=1500, eval_period=200, order='ordered')
    parser.add_argument('--num_procs', default=3, type=int, help='number of processors.')
    parser.add_argument('--cache_size', default=10, type=int, help='preprocessing cache size.')
    parser.add_argument('--test', help='test checkpoint.')
    args = parser.parse_args()
    main(args)
