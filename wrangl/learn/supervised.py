"""
Boilerplate that combines Hydra conf and PytorchLightning for supervised training.
"""
import os
import json
import torch
import random
import logging
import pytorch_lightning as pl
from typing import List
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from .callbacks import WandbTableCallback, S3Callback, GitCallback
from .metrics import Accuracy
from .model import BaseModel
from pytorch_lightning import callbacks as C
from pytorch_lightning.loggers import WandbLogger, CSVLogger


class SupervisedModel(BaseModel):
    """
    Supervised learning Pytorch lightning module.
    You should overload this model as you see fit.
    """

    def get_callbacks(self):
        """
        Returns a list of `pytorch_lightning.callbacks.Callback`s to use for training.
        """
        callbacks = []
        if self.hparams.git.enable:
            callbacks.append(GitCallback())
        if self.hparams.wandb.enable:
            callbacks.append(WandbTableCallback())
        if self.hparams.s3.enable:
            callbacks.append(S3Callback(self.hparams.s3))
        return callbacks

    def featurize(self, batch: list):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        return batch

    def compute_metrics(self, pred: list, gold: list) -> dict:
        """
        Computes metrics between predictions and ground truths.
        """
        m = Accuracy()
        return m(pred, gold)

    def compute_loss(self, out, feat, batch) -> torch.Tensor:
        """
        Computes loss from inputs and model output.
        You must implement this function.

        Args:
            out: model output from `wrangl.learn.supervised.Model.forward`.
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.

        """
        raise NotImplementedError()

    def extract_context(self, out, feat, batch) -> List:
        """
        Extract example context from inputs and model output.
        You must implement this function.

        Args:
            out: model output from `wrangl.learn.supervised.Model.forward`.
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.

        """

        raise NotImplementedError()

    def extract_pred(self, out, feat, batch) -> List:
        """
        Extract model predictions from inputs and model output.
        You must implement this function.

        Args:
            out: model output from `wrangl.learn.supervised.Model.forward`.
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.
        """

        raise NotImplementedError()

    def extract_gold(self, feat, batch) -> List:
        """
        Extract ground truths from inputs and model output.
        You must implement this function.

        Args:
            out: model output from `wrangl.learn.supervised.Model.forward`.
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.

        """

        raise NotImplementedError()

    def forward(self, feat, batch):
        """
        Computes model output from inputs.
        You must implement this function.

        Args:
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.

        """
        raise NotImplementedError()

    def infer(self, feat, batch):
        """
        Computes model output from inputs for inference.
        Defaults to `wrangl.learn.supervised.Model.forward`.

        Args:
            feat: featurize output from `wrangl.learn.supervised.Model.featurize`.
            batch: Dataset iterator output.

        """
        return self.forward(feat, batch)

    def infer_batch_size(self, batch):
        """
        Infers batch size from batch of raw examples.
        """
        if isinstance(batch, list):
            return len(batch)
        if isinstance(batch, dict):
            first_key = list(batch.keys())[0]
            return len(batch[first_key])
        return None

    def training_step(self, batch, batch_id):
        """
        One training step in PytorchLightning.
        You should not have to modify this.
        For more information, see `pytorch_lightning.training_step`.
        """
        feat = self.featurize(batch)
        out = self.forward(feat, batch)
        loss = self.compute_loss(out, feat, batch)
        self.log('loss', loss, batch_size=self.infer_batch_size(batch))
        perplexity = torch.exp(loss)
        self.log('ppl', perplexity, batch_size=self.infer_batch_size(batch))
        return loss

    def test_step(self, batch, batch_id, split='test'):
        """
        One test step in PytorchLightning.
        You should not have to modify this.
        For more information, see `pytorch_lightning.validation_step`.
        """
        return self.validation_step(batch, batch_id, split='test')

    def predict_step(self, batch, batch_id):

        """
        One prediction step in PytorchLightning.
        You should not have to modify this.
        For more information, see `pytorch_lightning.prediction_step`.
        """
        feat = self.featurize(batch)
        out = self.infer(feat, batch)
        pred = self.extract_pred(out, feat, batch)
        return pred

    def validation_step(self, batch, batch_id, split='val'):
        """
        One validation step in PytorchLightning.
        You should not have to modify this.
        For more information, see `pytorch_lightning.validation_step`.

        A core difference here is that Wrangl will
        1. compute metrics automatically
        2. extract example contexts, predictions, and ground truths
        3. save predictions to disk (and upload to S3, Wandb etc.) for debugging
        """
        feat = self.featurize(batch)
        context = self.extract_context(feat, batch)
        gold = self.extract_gold(feat, batch)

        out = self.infer(feat, batch)
        pred = self.extract_pred(out, feat, batch)

        metrics = self.compute_metrics(pred, gold)
        for k, v in metrics.items():
            self.log('{}_{}'.format(split, k), v, batch_size=self.infer_batch_size(batch))

        # for logging to wandb
        sample_ids = list(range(len(context)))
        random.shuffle(sample_ids)
        sample_ids = sample_ids[:self.hparams.val_sample_size]

        random.shuffle(self.pred_samples)
        for i in sample_ids:
            self.pred_samples.insert(0, (context[i], pred[i], gold[i]))
        self.pred_samples = self.pred_samples[:self.hparams.val_sample_size]

        with open('pred_samples.json', 'wt') as f:
            data = [dict(context=repr(context), gen=repr(gen), gold=repr(gold)) for context, gen, gold in self.pred_samples]
            json.dump(data, f, indent=2)

    @classmethod
    def run_train_test(cls, cfg: OmegaConf, train_dataset: torch.utils.data.DataLoader, eval_dataset: torch.utils.data.DataLoader, model_kwargs=None):
        """
        Creates a model according to config and trains/evaluates it.

        Args:
            cfg: config to use.
            train_dataset: dataset for training.
            eval_dataset: dataset for validation.
            model_kwargs: key ward arguments for model constructor.

        This will additionally set up according to your config file
            - manual seeds
            - early stopping
            - checkpoint saving
            - gradient clipping
            - callbacks (including an always on CSV logger)
        """
        model_kwargs = model_kwargs or {}
        pl.utilities.seed.seed_everything(seed=cfg.seed, workers=True)
        dout = os.getcwd()

        logger = logging.getLogger(name='{}:train_test'.format(cls.__name__))
        logger.info('Logging to {}'.format(dout))

        checkpoint = C.ModelCheckpoint(
            dirpath=dout,
            monitor=cfg.early_stopping.monitor,
            mode=cfg.early_stopping.mode,
            filename='{epoch:02d}-{' + cfg.early_stopping.monitor + ':.4f}',
            every_n_epochs=1,
            save_top_k=3,
            save_last=True,
            verbose=True,
        )

        train_logger = [CSVLogger(save_dir=dout, name='logs', flush_logs_every_n_steps=cfg.flush_logs_every_n_steps)]
        if cfg.wandb.enable:
            train_logger.append(WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity, save_dir=cfg.wandb.dir))

        fconfig = os.path.join(dout, 'config.yaml')
        OmegaConf.save(config=cfg, f=fconfig)

        model = cls(cfg, **model_kwargs)

        logger.info('Loading data')
        if cfg.collate_fn == 'auto':
            collate_fn = None
        elif cfg.collate_fn == 'ignore':
            def collate_fn(batch):
                return batch
        else:
            collate_fn = cls.collate_fn
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)

        trainer = pl.Trainer(
            precision=cfg.precision,
            strategy=cfg.strategy,
            gpus=cfg.gpus,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            auto_select_gpus=cfg.gpus > 0,
            benchmark=True,
            default_root_dir=dout,
            gradient_clip_val=cfg.grad_clip_norm,
            gradient_clip_algorithm='norm',
            log_every_n_steps=cfg.log_every_n_steps,
            val_check_interval=1 if cfg.debug else cfg.val_check_interval,
            limit_val_batches=cfg.limit_val_batches,
            max_steps=cfg.max_steps,
            logger=train_logger,
            callbacks=[checkpoint] + model.get_callbacks(),
        )
        if not cfg.test_only:
            ckpt_path = None
            if cfg.autoresume:
                ckpt_path = os.path.join(dout, 'last.ckpt')
                if not os.path.isfile(ckpt_path):
                    ckpt_path = None
            trainer.fit(model, train_loader, eval_loader, ckpt_path=ckpt_path)

        model = cls.load_from_checkpoint(os.path.join(dout, 'last.ckpt'))
        result = trainer.test(model, eval_loader, verbose=True)
        with open(os.path.join(dout, 'test_results.json'), 'wt') as f:
            json.dump(result, f, indent=2)
        logger.info('Finished!')

    @classmethod
    def run_inference(cls, cfg: OmegaConf, fcheckpoint: str, eval_dataset: torch.utils.data.DataLoader, model_kwargs=None, test=False):
        """
        Loads a model according to config and runs inference.

        Args:
            cfg: config to use.
            fcheckpoint: model checkpoint to load.
            eval_dataset: dataset for validation.
            model_kwargs: key ward arguments for model constructor.
            test: whether to evaluate metrics

        Returns:
            if `test==True` then returns a dictionary of scores, otherwise returns a list of model predictions.
        """
        model_kwargs = model_kwargs or {}
        model = cls.load_from_checkpoint(fcheckpoint, **model_kwargs)

        if cfg.collate_fn == 'auto':
            collate_fn = None
        elif cfg.collate_fn == 'ignore':
            def collate_fn(batch):
                return batch
        else:
            collate_fn = cls.collate_fn
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)

        trainer = pl.Trainer(
            precision=cfg.precision,
            strategy=cfg.strategy,
            gpus=cfg.gpus,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            auto_select_gpus=cfg.gpus > 0,
            benchmark=True,
        )
        if test:
            result = trainer.test(model, eval_loader, verbose=True)
        else:
            result = trainer.predict(model, eval_loader)
        return result
