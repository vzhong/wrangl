# Copyright (c) Facebook, Inc. and its affiliates.
import os
import gym
import hydra
import importlib
from silg.envs.rtfm import RTFMS1
from silg.envs.msgr import MessengerWrapper
from silg.envs.alfred import TWAlfred
from silg.envs.touchdown.gym_wrapper import TDWrapper, TD_ROOT, PARENT
from wrangl.learn import MoolibVtrace


def create_env(cfg):
    from silg import envs
    # if cfg.env.name == 'rtfm':
    #     return RTFMS1(time_penalty=cfg.env.time_penalty)
    # elif cfg.env.name == 'msgr':
    #     return MessengerWrapper(stage=1, split='train-all', time_penalty=cfg.env.time_penalty)
    # elif cfg.env.name == 'alf':
    #     return TFAlfred(time_penalty=cfg.env.time_penalty)
    # elif cfg.env.name == 'td':
    #     return TFAlfred(
    #         features_path=os.path.join(TD_ROOT, 'maj_ds_a10.npz'),
    #         data_json=str(PARENT.joinpath('data/train.json')),
    #         feat_type='segs',
    #         path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
    #         time_penalty=cfg.env.time_penalty,
    #     )
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
