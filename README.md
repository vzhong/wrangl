# Wrangl

[![Tests](https://github.com/vzhong/wrangl/actions/workflows/tests.yml/badge.svg)](https://github.com/vzhong/wrangl/actions/workflows/tests.yml)
[![Docs](https://github.com/vzhong/wrangl/actions/workflows/docs.yml/badge.svg)](https://www.victorzhong.com/wrangl)

Parallel data preprocessing for NLP and ML.
See [docs here](https://www.victorzhong.com/wrangl).
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

The supervised learning dataset parallelization component of this library uses [Ray](https://ray.io).
The reinforcement learning environment parallelization component of this library uses [Torchbeast](https://github.com/facebookresearch/torchbeast).


## Installation

```bash
pip install -e .  # add [dev] if you want to run tests and build docs.

# for latest
pip install git+https://github.com/vzhong/wrangl

# pypi release
pip install wrangl
```

## Usage

See [examples](https://github.com/vzhong/wrangl/tree/main/wrangl/examples) for usage.
Here are some common use cases:

* process data in parallel
  * [repeat string with generator dataset](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/preprocess_repeat_string.py)
  * [loading from SQL database](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/preprocess_sql_db.py)
  * [loading from bzipped jsonl files](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/preprocess_jsonl_files.py)
  * [parse text using Stanza](https://github.com/vzhong/wrangl/tree/main/wrangl/examples/preprocess_using_stanza.py)
* train models

## Commandline utilities

Current supports:

- annotating text files
- plotting learning curves
- autodocumenting this package

```
wrangl -h
```


## Run tests

```
python -m unittest discover tests
```

## Generate docs

```
wrangl autodoc
```
