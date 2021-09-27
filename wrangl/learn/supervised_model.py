import os
import json
import tqdm
import torch
import pprint
import logging
import pathlib
from torch import nn
from typing import Tuple, List, Dict, Union
from ..data import Dataloader
from ..metrics import Metric
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter


class SupervisedModel(nn.Module):

    @classmethod
    def get_parser(cls, **defaults) -> ArgumentParser:
        """
        returns the default parser
        """
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--learning_rate', default=defaults.get('learning_rate', 1e-3), type=float, help='initial learning rate.')
        parser.add_argument('--batch_size', default=defaults.get('batch_size', 24), type=int, help='batch size.')
        parser.add_argument('--silent', action='store_true', help='do not print progress bars and logs.')
        parser.add_argument('--order', choices=('ordered', 'unordered'), default=defaults.get('order', 'unordered'), help='whether to train in order.')
        parser.add_argument('--timeout', default=defaults.get('timeout', 60), type=int, help='time out in seconds for data loading.')
        parser.add_argument('--num_train_steps', default=defaults.get('num_train_steps', 5000), type=int, help='number of training steps.')
        parser.add_argument('--eval_period', default=defaults.get('eval_period', 100), type=int, help='number of steps between evaluation.')
        parser.add_argument('--dout', default=defaults.get('dout', 'checkpoints/{}'.format(cls.__name__.lower())), help='where to save experiment.')
        parser.add_argument('--resume', help='where to resume experiment from. If `"auto"` then resume from the default checkpoint.')
        return parser

    def __init__(self, hparams: Namespace = None):
        """
        Args:
            hparams: hyperparameters for the model. Defaults to `self.get_parser().parse_args()`.
        """
        super().__init__()
        if hparams is None:
            hparams = self.get_parser().parse_args()
        self.hparams = hparams
        self.train_steps = 0
        self.logger = self.get_logger()

    def get_logger(self) -> logging.Logger:
        """
        Returns the default logger
        """
        logger = logging.getLogger(self.__class__.__name__.lower())
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        flog = pathlib.Path(self.hparams.dout).joinpath('train.log')
        fh = logging.FileHandler(flog)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def get_metrics(self) -> Metric:
        raise NotImplementedError()

    def get_optimizer_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(lr=self.hparams.learning_rate, params=self.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (self.hparams.num_train_steps - step) / self.hparams.num_train_steps)
        return optimizer, scheduler

    def extract_pred(self, batch, out) -> List[Dict]:
        """
        Extracts predictions from model output.
        """
        raise NotImplementedError()

    def compute_loss(self, batch, out) -> torch.Tensor:
        """
        Computes loss fro model output.
        """
        raise NotImplementedError()

    def get_fresume(self) -> pathlib.Path:
        """
        Returns the resume checkpoint file path.
        """
        if self.hparams.resume == 'auto':
            fresume = pathlib.Path(self.hparams.dout).joinpath('ckpt.tar')
        elif self.hparams.resume:
            fresume = pathlib.Path(self.hparams.resume)
            assert fresume.exists(), 'Requested resume checkpoint does not exist at {}'.format(fresume.name)
        else:
            fresume = None
        return fresume

    def run_train(self, train_data: Dataloader, eval_data: Dataloader):
        """
        Trains model on supervised dataset with early stopping.

        Args:
            train_data: training dataset.
            eval_data: evaluation dataset.
        """
        self.ensure_dout()

        generator = train_data.batch(self.hparams.batch_size, ordered=self.hparams.order == 'ordered', timeout=self.hparams.timeout, auto_reset=True)

        metrics = self.get_metrics()
        optimizer, scheduler = self.get_optimizer_scheduler()

        train_loss = 0
        num_steps = 0
        train_preds = []
        best_eval_metric = None

        fresume = self.get_fresume()
        if fresume and fresume.exists():
            self.load_checkpoint(fresume, override_hparams=self.hparams, model=self, optimizer=optimizer, scheduler=scheduler)

        self.logger.info('Starting train')
        if not self.hparams.silent:
            bar = tqdm.tqdm(desc='training', total=self.hparams.num_train_steps)
            bar.update(self.train_steps)
        self.train()

        while self.train_steps < self.hparams.num_train_steps:
            optimizer.zero_grad()

            batch = next(generator)
            batch_out = self(batch)
            batch_loss = self.compute_loss(batch, batch_out)
            for ex, pred in zip(batch, self.extract_pred(batch, batch_out)):
                train_preds.append((ex, pred))
            train_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
            bar.update(1)
            self.train_steps += 1
            num_steps += 1

            if self.train_steps % self.hparams.eval_period == 0 or self.train_steps == self.hparams.num_train_steps:
                train_loss /= num_steps
                eval_preds, eval_loss = self.run_preds(eval_data, compute_loss=True, verbose=not self.hparams.silent)
                train_metrics = metrics(train_preds)
                train_metrics['loss'] = train_loss
                key_metric_name, key_train_metric = metrics.extract_key_metric(train_metrics)
                eval_metrics = metrics(eval_preds)
                eval_metrics['loss'] = eval_loss
                _, key_eval_metric = metrics.extract_key_metric(eval_metrics)
                combined_metrics = dict(train=train_metrics, eval=eval_metrics, best=key_eval_metric, train_steps=self.train_steps)

                if metrics.should_early_stop(key_eval_metric, best_eval_metric):
                    self.save_checkpoint(combined_metrics, optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), model_state=self.state_dict())
                    best_eval_metric = key_eval_metric

                self.logger.info('\n' + pprint.pformat(combined_metrics))
                bar.set_description('L train {:.3g} eval {:.3g} | {} train {:.3g} eval {:.3g} best {:.3g}'.format(train_loss, eval_loss, key_metric_name, key_train_metric, key_eval_metric, best_eval_metric))
                self.train()
                train_loss = num_steps = 0
                train_preds.clear()
        bar.close()

    def run_preds(self, eval_data: Dataloader, compute_loss: bool = False, verbose: bool = False) -> List[Dict]:
        """
        Generates predictions for dataset.

        Args:
            eval_data: dataset to generate predictions for.
            compute_loss: whether to additionally return the loss (requires `eval_data` return labels).
            verbose: whether to show progress bar.

        Returns:
            model prediction.
        """
        generator = eval_data.batch(self.hparams.batch_size, ordered=True, timeout=self.hparams.timeout, auto_reset=False)
        if verbose:
            bar = tqdm.tqdm(desc='evaluating', leave=False)

        preds = []
        loss = 0
        num_steps = 0
        self.eval()
        with torch.no_grad():
            for batch in generator:
                batch_out = self(batch)
                for ex, pred in zip(batch, self.extract_pred(batch, batch_out)):
                    preds.append((ex, pred))
                if compute_loss:
                    loss += self.compute_loss(batch, batch_out).item()
                num_steps += 1
                if verbose:
                    bar.update(1)
        loss /= num_steps
        if verbose:
            bar.close()
        if compute_loss:
            return preds, loss
        else:
            return preds

    def ensure_dout(self):
        """
        Makes sure that `self.hparams.dout` exists.
        """
        if not os.path.isdir(self.hparams.dout):
            if not self.hparams.silent:
                self.logger.info('making directory: {}'.format(self.hparams.dout))
            os.makedirs(self.hparams.dout)

    @classmethod
    def load_checkpoint(cls, fckpt: Union[str, pathlib.Path], override_hparams: Namespace = None, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> nn.Module:
        """
        Loads model from a checkpoint.

        Args:
            fckpt: checkpoint to load.
            override_hparams: parameter values to overwrite checkpoint parameters with.
            model: model whose parameters to load into.
            optimizer: optimizer whose parameters to load into.
            scheduler: scheduler whose parameters to load into.
        """
        d = torch.load(fckpt)
        for k, v in vars(override_hparams).items():
            setattr(d['hparams'], k, v)
        if model is None:
            model = cls(d['hparams'])
        model.hparams = d['hparams']
        model.load_state_dict(d['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(d['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(d['scheduler_state'])
        model.train_steps = d['train_steps']
        model.logger.info('Resuming from step {} with {}'.format(model.train_steps, fckpt))
        return model

    def save_checkpoint(self, metrics: dict, model_state: dict = None, optimizer_state: dict = None, scheduler_state: dict = None):
        """
        Saves training state to checkpoint files.
        """
        dout = pathlib.Path(self.hparams.dout)
        self.logger.info('Saving to {}'.format(dout.name))
        with dout.joinpath('hparams.json').open('wt') as f:
            json.dump(vars(self.hparams), f, indent=2)
        with dout.joinpath('metrics.best.json').open('wt') as f:
            json.dump(metrics, f, indent=2)
        flog = dout.joinpath('metrics.log.json')
        if flog.exists():
            with flog.open('rt') as f:
                log = json.load(f)
        else:
            log = []
        log.append(metrics)
        with flog.open('wt') as f:
            json.dump(log, f, indent=2)
        d = dict(
            hparams=self.hparams,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            train_steps=self.train_steps,
        )
        torch.save(d, dout.joinpath('ckpt.tar'))
