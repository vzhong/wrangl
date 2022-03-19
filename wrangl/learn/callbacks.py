import os
import pytorch_lightning as pl


class WandbTableCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer, model):
        import wandb
        wandb_logger = [exp for exp in trainer.logger.experiment if isinstance(exp, wandb.sdk.wandb_run.Run)]
        assert wandb_logger
        run = wandb.Api().run(path=wandb_logger[0].path)
        for artifact in run.logged_artifacts():
            artifact.delete(delete_aliases=True)
        table = wandb.Table(columns=['context', 'pred', 'gold'])
        for context, gen, gold in model.pred_samples:
            table.add_data(repr(context), repr(gen), repr(gold))
        wandb.log(dict(gen=table))


class AppWriteCallback(pl.Callback):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_validation_end(self, trainer, model):
        from ..cloud.client import AppwriteClient
        cfg = model.hparams
        client = AppwriteClient()
        client.upload_experiment(os.getcwd())
        client.plot_experiment(
            project_id=cfg.project,
            experiment_id=cfg.name,
            **cfg.appwrite.plot
        )
