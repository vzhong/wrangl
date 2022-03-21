This example trains a XOR classifier using a combination of
- wrangl
- hydra
- pytorch lightning

In order to run this example, you must install the `learn` module dependencies via

To train locally (settings in `conf/default.yaml`):

```bash
python train.py
```

To train using Slurm via the local launcher (settings in `conf/hydra/launcher/local.yaml`):

```bash
python train.py --multirun
```

To train using Slurm via the Slurm launcher (settings in `conf/hydra/launcher/slurm.yaml`):

```bash
python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=<name of your partition>
```
