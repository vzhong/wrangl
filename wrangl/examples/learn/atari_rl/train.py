import gym
import hydra
import pprint
from wrangl.learn.rl import MoolibVtrace


def create_env(flags):
    env = gym.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
        flags.env.name,
        obs_type="grayscale",  # "ram", "rgb", or "grayscale".
        frameskip=1,  # Action repeats. Done in wrapper b/c of noops.
        repeat_action_probability=flags.env.repeat_action_probability,  # Sticky actions.
        full_action_space=True,  # Use all actions.
        render_mode=None,  # None, "human", or "rgb_array".
    )

    # Using wrapper from seed_rl in order to do random no-ops _before_ frameskipping.
    # gym.wrappers.AtariPreprocessing doesn't play well with the -v5 versions of the game.
    import wrappers
    env = wrappers.AtariPreprocessing(
        env,
        frame_skip=flags.env.num_action_repeats,
        terminal_on_life_loss=False,
        screen_size=84,
        max_random_noops=flags.env.noop_max,  # Max no-ops to apply at the beginning.
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


@hydra.main(config_path="conf", config_name="default")
def main(cfg):
    Model = MoolibVtrace.load_model_class(cfg.model, model_dir='model')
    if not cfg.test_only:
        Model.run_train_test(cfg, create_train_env=create_env, create_eval_env=create_env)
    else:
        checkpoint_path = 'last.ckpt'
        eval_envs = Model.create_env_pool(cfg, create_env, override_actor_batches=1)
        test_results = Model.run_test(cfg, eval_envs, checkpoint_path, eval_steps=cfg.eval_steps)
        pprint.pprint(test_results)


if __name__ == "__main__":
    main()
