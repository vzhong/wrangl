import wandb
import pytorch_lightning as pl


class WandbTableCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer, model):
        run = wandb.Api().run(path=trainer.logger.experiment.path)
        for artifact in run.logged_artifacts():
            artifact.delete(delete_aliases=True)
        table = wandb.Table(columns=['context', 'pred', 'gold'])
        for context, gen, gold in model.pred_samples:
            table.add_data(repr(context), repr(gen), repr(gold))
        wandb.log(dict(gen=table))
