import os
import json
import torch
import random
import logging
import importlib.util
import pytorch_lightning as pl
from typing import List
from torch.optim import Adam
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd
from .callbacks import WandbTableCallback
from .metrics import Accuracy
from pytorch_lightning import callbacks as C
from pytorch_lightning.loggers import WandbLogger


class SupervisedModel(pl.LightningModule):

    @classmethod
    def load_model_class(cls, model_name):
        fname = os.path.join(get_original_cwd(), 'model', '{}.py'.format(model_name))
        spec = importlib.util.spec_from_file_location(model_name, fname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Model = module.Model
        return Model

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.pred_samples = []

    def get_callbacks(self):
        return [WandbTableCallback()] if self.hparams.wandb.enable else []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def featurize(self, batch):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        return batch

    def compute_metrics(self, pred, gold) -> dict:
        m = Accuracy()
        return m(pred, gold)

    def compute_loss(self, out, feat, batch) -> torch.Tensor:
        raise NotImplementedError()

    def extract_context(self, out, feat, batch) -> List:
        raise NotImplementedError()

    def extract_pred(self, out, feat, batch) -> List:
        raise NotImplementedError()

    def extract_gold(self, feat, batch) -> List:
        raise NotImplementedError()

    def forward(self, feat, batch):
        raise NotImplementedError()

    def infer(self, feat, batch):
        return self.forward(feat, batch)

    def training_step(self, batch, batch_id):
        feat = self.featurize(batch)
        out = self.forward(feat, batch)
        loss = self.compute_loss(out, feat, batch)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_id, split='val'):
        feat = self.featurize(batch)
        context = self.extract_context(feat, batch)
        gold = self.extract_gold(feat, batch)

        out = self.infer(feat, batch)
        pred = self.extract_pred(out, feat, batch)

        metrics = self.compute_metrics(pred, gold)
        for k, v in metrics.items():
            self.log('{}_{}'.format(split, k), v)

        # for logging to wandb
        sample_ids = list(range(len(context)))
        random.shuffle(sample_ids)
        sample_ids = sample_ids[:self.hparams.val_sample_size]
        self.pred_samples = [(context[i], pred[i], gold[i]) for i in sample_ids]

    @classmethod
    def run_train_test(cls, cfg, train_dataset, eval_dataset):
        logger = logging.getLogger(name='{}:train_test'.format(cls.__name__))
        pl.utilities.seed.seed_everything(seed=cfg.seed, workers=True)
        dout = os.getcwd()

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

        train_logger = None
        if cfg.wandb.enable:
            train_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)

        fconfig = os.path.join(os.getcwd(), 'config.yaml')
        OmegaConf.save(config=cfg, f=fconfig)

        model = cls(cfg)

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
            strategy='dp',
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
            weights_save_path=os.path.join(dout, 'weights'),
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
        logger.info('Finished!')
