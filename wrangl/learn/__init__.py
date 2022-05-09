"""
Wrangl implements two learning boilerplates:

- `wrangl.learn.supervised` for Supervised Learning, mostly using PytorchLightning.
- `wrangl.learn.rl` for Reinforcement Learning, mostly using Moolib and IMPALA.

.. include:: ../examples/learn/README.md
"""
from .supervised import SupervisedModel
__all__ = ['callbacks', 'metrics', 'supervised']
