"""
Callbacks that can be used during PytorchLightning training.
"""

import os
import glob
import json
import wandb
import torch
import tempfile
import pytorch_lightning as pl
from ..cloud import S3Client
from hydra.utils import get_original_cwd


class WandbTableCallback(pl.Callback):
    """
    Uploads sample predictions to Wandb.
    """

    def on_validation_end(self, trainer, model):
        wandb_logger = [exp for exp in trainer.logger.experiment if isinstance(exp, wandb.sdk.wandb_run.Run)]
        assert wandb_logger
        run = wandb.Api().run(path=wandb_logger[0].path)
        for artifact in run.logged_artifacts():
            artifact.delete(delete_aliases=True)
        table = wandb.Table(columns=['context', 'pred', 'gold'])
        for context, gen, gold in model.pred_samples:
            table.add_data(repr(context), repr(gen), repr(gold))
        wandb.log(dict(gen=table))


class S3Callback(pl.Callback):
    """
    Uploads experiment config, git diffs, logs, best checkpoints, and plots to S3.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.client = S3Client(fcredentials=cfg.fcredentials)

    def on_save_checkpoint(self, trainer, model, checkpoint):
        if self.cfg.checkpoint:
            with tempfile.NamedTemporaryFile('w') as f:
                torch.save(checkpoint, f.name)
                f.flush()
                self.client.upload_file(self.cfg.project_id, self.cfg.experiment_id, 'last.ckpt', f.name, content_type='application/pytorch')

    def on_validation_end(self, trainer, model):
        self.client.upload_experiment(os.getcwd())
        if self.cfg.plot:
            self.client.plot_experiment(
                project_id=self.cfg.project_id,
                experiment_id=self.cfg.experiment_id,
                **self.cfg.plot
            )

        for fname in ['pred_samples.json']:
            if os.path.isfile(fname):
                self.client.upload_file(self.cfg.project_id, self.cfg.experiment_id, fname, fname, content_type='text/plain')

    def on_fit_start(self, trainer, model):
        for fname in glob.glob('git.*'):
            self.client.upload_file(self.cfg.project_id, self.cfg.experiment_id, fname, fname, content_type='text/plain')


class GitCallback(pl.Callback):
    """
    Dumps git diffs to work directory.
    """

    def on_init_end(self, trainer=None):
        import git
        repo = git.Repo(get_original_cwd(), search_parent_directories=True)
        with open('git.patch.diff', 'wt') as f:
            f.write(repo.git.diff(repo.head.commit.tree))
        with open('git.head.json', 'wt') as f:
            c = repo.head.commit
            json.dump(dict(hexsha=c.hexsha, message=c.message, date=c.committed_date, committer=c.committer.name), f, indent=2)
