"""
Wrangl implements two learning boilerplates:

- `wrangl.learn.supervised` for Supervised Learning, mostly using PytorchLightning.
- `wrangl.learn.rl` for Reinforcement Learning, mostly using Moolib and IMPALA.

These are combined with [Hydra](https://hydra.cc) for config management and job parallelization (e.g. sweeping via Slurm).

See [here for examples](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/learn) on how to quickly set up experiments.
The basic workflow is:

1. Copy an existing example
```bash
wrangl project xor-clf
```

2. Adjust configuration in `conf/default.yaml`. More config options are in found in [the git repo](https://github.com/vzhong/wrangl/tree/main/wrangl/conf).

3. Adjust model implementation in `model/<model name>.py`

4. Run experiment via `python train.py`.
"""
from .supervised import SupervisedModel
