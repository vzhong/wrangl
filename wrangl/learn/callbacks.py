import os
import json
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


class S3Callback(pl.Callback):

    def on_validation_end(self, trainer, model):
        from ..cloud import S3Client
        cfg = model.hparams
        client = S3Client(url=cfg.s3.url, key=cfg.s3.key, secret=cfg.s3.secret, bucket=cfg.s3.bucket)
        client.upload_experiment(os.getcwd())
        if cfg.s3.plot:
            client.plot_experiment(
                project_id=cfg.project,
                experiment_id=cfg.name,
                **cfg.s3.plot
            )
        if cfg.s3.checkpoint:
            client.upload_file(cfg.project, cfg.name, 'last.ckpt', 'last.ckpt', content_type='application/pytorch')
        if cfg.s3.pred_samples:
            data = [dict(context=repr(context), gen=repr(gen), gold=repr(gold)) for context, gen, gold in model.pred_samples]
            client.upload_content(cfg.project, cfg.name, 'pred_samples.json', json.dumps(data))
