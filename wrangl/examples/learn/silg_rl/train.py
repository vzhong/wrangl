# Copyright (c) Facebook, Inc. and its affiliates.
import gym
import hydra
import importlib
from silg import envs
from wrangl.learn import MoolibVtrace


def create_env(cfg):
    return gym.make(cfg.env.name, time_penalty=cfg.env.time_penalty)


def create_model(cfg):
    env = create_env(cfg)
    Model = importlib.import_module('model.{}'.format(cfg.model.name)).Model
    return Model(cfg, env).to(cfg.device)


@hydra.main(config_path="conf", config_name="default")
def main(cfg):
    MoolibVtrace.launch(cfg, create_env, create_model)


if __name__ == "__main__":
    main()
