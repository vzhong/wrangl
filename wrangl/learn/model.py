import os
import json
import torch
import logging
import pathlib
from typing import Tuple, Union
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


class Model(torch.nn.Module):

    @classmethod
    def get_parser(cls, **defaults) -> ArgumentParser:
        """
        returns the default parser
        """
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--learning_rate', default=defaults.get('learning_rate', 1e-3), type=float, help='initial learning rate.')
        parser.add_argument('--batch_size', default=defaults.get('batch_size', 24), type=int, help='batch size.')
        parser.add_argument('--silent', action='store_true', help='do not print progress bars and logs.')
        parser.add_argument('--num_train_steps', default=defaults.get('num_train_steps', 5000), type=int, help='number of training steps.')
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
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self.get_logger()
        return self._logger

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

    def get_optimizer_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(lr=self.hparams.learning_rate, params=self.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (self.hparams.num_train_steps - step) / self.hparams.num_train_steps)
        return optimizer, scheduler

    def ensure_dout(self):
        """
        Makes sure that `self.hparams.dout` exists.
        """
        if not os.path.isdir(self.hparams.dout):
            os.makedirs(self.hparams.dout)
            if not self.hparams.silent:
                self.logger.info('made directory: {}'.format(self.hparams.dout))

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

    @classmethod
    def load_checkpoint(cls, fckpt: Union[str, pathlib.Path], override_hparams: Namespace = None, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> torch.nn.Module:
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

    def save_checkpoint(self, metrics: dict = None, model_state: dict = None, optimizer_state: dict = None, scheduler_state: dict = None):
        """
        Saves training state to checkpoint files.
        """
        dout = pathlib.Path(self.hparams.dout)
        self.logger.info('Saving to {}'.format(dout.name))
        with dout.joinpath('hparams.json').open('wt') as f:
            json.dump(vars(self.hparams), f, indent=2)
        if metrics is not None:
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
