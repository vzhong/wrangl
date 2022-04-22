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
from pytorch_lightning.loggers import WandbLogger, CSVLogger


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

    def infer_batch_size(self, batch):
        if isinstance(batch, list):
            return len(batch)
        if isinstance(batch, dict):
            first_key = list(batch.keys())[0]
            return len(batch[first_key])
        return None

    def training_step(self, batch, batch_id):
        feat = self.featurize(batch)
        out = self.forward(feat, batch)
        loss = self.compute_loss(out, feat, batch)
        self.log('loss', loss, batch_size=self.infer_batch_size(batch))
        perplexity = torch.exp(loss)
        self.log('ppl', perplexity, batch_size=self.infer_batch_size(batch))
        return loss

    def test_step(self, batch, batch_id, split='test'):
        return self.validation_ste(batch, batch_id, split='test')

    def predict_step(self, batch, batch_id):
        feat = self.featurize(batch)
        out = self.infer(feat, batch)
        pred = self.extract_pred(out, feat, batch)
        return pred

    def validation_step(self, batch, batch_id, split='val'):
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

    @classmethod
    def run_train_test(cls, cfg, train_dataset, eval_dataset, model_kwargs=None):
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

        model = cls.load_from_checkpoint(os.path.join(dout, 'last.ckpt'))
        result = trainer.test(model, eval_loader, verbose=True)
        with open(os.path.join(dout, 'test_results.json'), 'wt') as f:
            json.dump(result, f, indent=2)
        logger.info('Finished!')

    @classmethod
    def run_pred(cls, cfg, fcheckpoint, eval_dataset, model_kwargs=None):
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
        result = trainer.predict(model, eval_loader)
        return result

    @classmethod
    def run_test(cls, cfg, fcheckpoint, eval_dataset, model_kwargs=None):
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
        result = trainer.test(model, eval_loader, verbose=True)
        return result
