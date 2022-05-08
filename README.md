# Wrangl

[![Tests](https://github.com/vzhong/wrangl/actions/workflows/tests.yml/badge.svg)](https://github.com/vzhong/wrangl/actions/workflows/tests.yml)
[![Docs](https://github.com/vzhong/wrangl/actions/workflows/docs.yml/badge.svg)](https://www.victorzhong.com/wrangl)

Parallel data preprocessing and fast boilerplates for NLP and ML.
See [docs here](https://www.victorzhong.com/wrangl).

## Installation

```bash
pip install -e .  # add [dev] if you want to run tests and build docs.

# for latest
pip install git+https://github.com/vzhong/wrangl

# pypi release
pip install wrangl
```

If [moolib](https://github.com/facebookresearch/moolib) install fails because you do not have CUDA you can try installing it yourself with `env USE_CUDA=0 pip install moolib`.

## Usage

See `wrangl.examples` for usage.
In particular `wrangl.examples.learn.xor_clf` shows an example of using Wrangl to quickly set up a supervised classification task.
For parallel data preprocessing `wrangl.examples.preprocess.using_stanza` shows an example of using Stanford NLP Stanza to parse text in parallel across CPU cores.
Here are some common use cases.


### Preprocess data in parallel:

```python
import io, ray, tqdm, stanza, contextlib
from wrangl.data import IterableDataset, Processor

@ray.remote
class MyProcessor(Processor):

    def __init__(self):
        self.nlp = stanza.Pipeline('en')

    def process(self, raw):
        return self.nlp(raw).text

if __name__ == '__main__':
    # we will use Python's Zen poem as an example
    zen = io.StringIO()
    with contextlib.redirect_stdout(zen):
        import this
    text = [zen.getvalue()] * 20
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])

    # parallel
    loader = IterableDataset(text, pool, cache_size=10, shuffle=False)
    processed = list(tqdm.tqdm(loader, desc='parallel ordered', total=len(text)))
```


### Set up quick supervised learning experiment

1. Copy an example experiment with the bash command `wrangl project`. Alternatively `wrangl project xor-clf` to copy a particular example.
2. Modify `conf/default.yaml`, `train.py`, and `model/mymodel.py` to your liking. Default configs are in the wrangl library in `wrangl.conf`.
3. Run the experiment with `python train.py`
4. To introduce a new model, just add its file to `model/mymodel2.py` and run `python train.py model=mymodel2`.
5. Slurm parallelization comes free: `python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=<name of your partition>`.


If you find this work helpful, please consider citing

```
@misc{zhong2021wrangl,
  author = {Zhong, Victor},
  title = {Wrangl: Parallel data preprocessing for NLP and ML},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vzhong/wrangl}}
}
```

## Run tests

```bash
wrangl autotest
```

## Generate docs

```bash
wrangl autodoc
```

## Dependencies
Wrangl heavily depends on the following libraries:
- supervised learning
  - [Ray](https://ray.io)
  - [Pytorch Lightning](https://www.pytorchlightning.ai)
- reinforcement learning
  - [moolib](https://github.com/facebookresearch/moolib)
- experiment config management
  - [Hydra](https://hydra.cc)
