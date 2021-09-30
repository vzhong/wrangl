import tqdm
import time
import torch
import pprint
from argparse import ArgumentParser
from typing import List, Dict
from ...data import Dataloader
from ...metrics import Metric, Timer
from ..model import Model as BaseModel


class SupervisedModel(BaseModel):

    @classmethod
    def get_parser(cls, **defaults) -> ArgumentParser:
        """
        returns the default parser
        """
        parser = super().get_parser(**defaults)
        parser.add_argument('--order', choices=('ordered', 'unordered'), default=defaults.get('order', 'unordered'), help='whether to train in order.')
        parser.add_argument('--timeout', default=defaults.get('timeout', 5), type=int, help='time out in seconds for data loading.')
        parser.add_argument('--eval_period', default=defaults.get('eval_period', 100), type=int, help='number of steps between evaluation.')
        return parser

    def get_metrics(self) -> Metric:
        raise NotImplementedError()

    def extract_pred(self, batch, out) -> List[Dict]:
        """
        Extracts predictions from model output.
        """
        raise NotImplementedError()

    def extract_gold(self, batch) -> List[Dict]:
        """
        Extracts label from input batch.
        """
        raise NotImplementedError()

    def compute_loss(self, batch, out) -> torch.Tensor:
        """
        Computes loss fro model output.
        """
        raise NotImplementedError()

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
        timer = Timer()
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

            time_start = time.time()
            batch = next(generator)
            time_batch = time.time()
            batch_out = self(batch)
            time_forward = time.time()
            timer.record('batch', time_batch-time_start)
            timer.record('forward', time_forward-time_batch)
            batch_loss = self.compute_loss(batch, batch_out)
            for gold, pred in zip(self.extract_gold(batch), self.extract_pred(batch, batch_out)):
                train_preds.append((gold, pred))
            train_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if not self.hparams.silent:
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
                combined_metrics = dict(train=train_metrics, eval=eval_metrics, best=key_eval_metric, train_steps=self.train_steps, elapsed=timer.elapsed)

                if metrics.should_early_stop(key_eval_metric, best_eval_metric):
                    self.save_checkpoint(combined_metrics, optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict() if scheduler else None, model_state=self.state_dict())
                    best_eval_metric = key_eval_metric

                self.logger.info('\n' + pprint.pformat(combined_metrics))
                if not self.hparams.silent:
                    bar.set_description('L tr {:.3g} ev {:.3g} | {} tr {:.3g} ev {:.3g} bst {:.3g} | bat {:.3g} fwd {:.3g}'.format(train_loss, eval_loss, key_metric_name, key_train_metric, key_eval_metric, best_eval_metric, timer.elapsed['batch'], timer.elapsed['forward']))
                self.train()
                train_loss = num_steps = 0
                train_preds.clear()
        if not self.hparams.silent:
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
                for gold, pred in zip(self.extract_gold(batch), self.extract_pred(batch, batch_out)):
                    preds.append((gold, pred))
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
