import os
import importlib
import pytorch_lightning as pl
from torch import optim
from hydra.utils import get_original_cwd


class BaseModel(pl.LightningModule):

    @classmethod
    def load_model_class(cls, model_name, model_dir='model', root_dir=None):
        """
        Loads a model from file. Note that model class must be `Model`.

        Args:
            model_name: name of model to load.
            model_dir: directory of model files.
            root_dir: source directory. If not specified, then it is inferred from HydraConfig's launch directory.

        Suppose we have our model files in a directory `mymodels`, and in `mymodels/cool_model.py` we have:

        ```
        from wrangl.learn import SupervisedModel
        class Model(SupervisedModel):
            ...
        ```

        To load this model we would call in `train.py`:
        ```python
        MyModel = SupervisedModel.load_model_class('cool_model', 'mymodels')
        ```
        """
        if root_dir is None:
            root_dir = get_original_cwd()
        fname = os.path.join(root_dir, model_dir, '{}.py'.format(model_name))
        spec = importlib.util.spec_from_file_location(model_name, fname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Model = module.Model
        return Model

    def __init__(self, cfg):
        """
        Instiate a model from config file.

        Args:
            cfg: `omegaconf.OmegaConf` config to use.

        By default, this config is created by [Hydra](https://hydra.cc) from your config files.
        If you'd like to use a `dict`, instead:

        ```python
        conf = OmegaConf.create({"var1" : "v", "var2" : [1, {"a": "1", "b": "2", 3: "c"}]})
        model = Model(conf)
        ```

        For more info on OmegaConf, see [their docs](https://omegaconf.readthedocs.io/).
        """
        super().__init__()
        self.save_hyperparameters(cfg)
        self.pred_samples = []

    def configure_optimizers(self):
        """
        Returns a the `torch.optim.Optimizer` to use for this model.
        """
        opt = self.hparams.optim
        Optimizer = getattr(optim, opt.name)
        return Optimizer(self.parameters(), **opt.params)
