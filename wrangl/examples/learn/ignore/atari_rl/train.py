# Copyright (c) Facebook, Inc. and its affiliates.
import hydra
from wrangl.learn import MoolibVtrace
from atari import environment
from atari import models


@hydra.main(config_path="conf", config_name="default")
def main(cfg):
    MoolibVtrace.launch(cfg, environment.create_env, models.create_model)


if __name__ == "__main__":
    main()
