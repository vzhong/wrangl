# Wrangl

[![Tests](https://github.com/vzhong/wrangl/actions/workflows/test.yml/badge.svg)](https://github.com/vzhong/wrangl/actions/workflows/test.yml)
 task

Parallel data preprocessing and fast experiments for NLP and ML.
See [docs here](https://www.victorzhong.com/wrangl).

## Why?
I built this library to prototype ideas quickly.
In essence it combines [Hydra](https://hydra.cc), [Pytorch Lightning](https://www.pytorchlightning.ai), [moolib](https://github.com/facebookresearch/moolib), and [Ray](https://ray.io) for some fast data processing and (supervised/reinforcement) learning.
The following are supported with command line or config tweaks (e.g. no additional boilerplate code):

- checkpointing
- early stopping
- auto git diffs
- logging to S3 (along with auto-generated seaborn plot), wandb
- Slurm launcher


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

See [the documentation](https://victorzhong.com/wrangl) for how to use Wrangl.
Examples of projects using Wrangl are found in `wrangl.examples`.
In particular `wrangl.examples.learn.xor_clf` shows an example of using Wrangl to quickly set up a supervised classification task.
`wrangl.examples.learn.atari_rl` shows an example of reinforcement learning using IMPALA VTrace.
For parallel data preprocessing `wrangl.examples.preprocess.using_stanza` shows an example of using Stanford NLP Stanza to parse text in parallel across CPU cores.

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
