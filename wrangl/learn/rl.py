# Copyright (c) Facebook, Inc. and its affiliates.
"""
Adapted from https://github.com/facebookresearch/moolib
"""
import copy
import dataclasses
import logging
import os
import pprint
import signal
import time
import omegaconf
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import wandb

import socket
import coolname

import moolib
from moolib.examples.common import nest
from moolib.examples.common import vtrace
from moolib.examples.common import record
from moolib.examples import common

from pytorch_lightning.loggers import CSVLogger
from .callbacks import GitCallback, S3Callback
from .model import BaseModel


__all__ = ['MoolibVtrace']


def uid():
    return "%s:%i:%s" % (socket.gethostname(), os.getpid(), coolname.generate_slug(2))


omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)


@dataclasses.dataclass
class LearnerState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    model_version: int = 0
    num_previous_leaders: int = 0
    train_time: float = 0
    last_checkpoint: float = 0
    last_checkpoint_history: float = 0
    global_stats: Optional[dict] = None

    def save(self):
        r = dataclasses.asdict(self)
        r["model"] = self.model.state_dict()
        r["optimizer"] = self.optimizer.state_dict()
        r["scheduler"] = self.scheduler.state_dict()
        return r

    def load(self, state):
        for k, v in state.items():
            if k not in ("model", "optimizer", "scheduler", "global_stats"):
                setattr(self, k, v)
        self.model_version = state["model_version"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

        for k, v in state["global_stats"].items():
            if k in self.global_stats:
                self.global_stats[k] = type(self.global_stats[k])(**v)


class MoolibVtrace(BaseModel):
    """
    This interface is based on the reference VTrace implementation from moolib.

    The functions you need to implement are:

    - `wrangl.learn.rl.MoolibVtrace.__init__`
    - `wrangl.learn.rl.MoolibVtrace.forward`

    In addition, if you want to use the default state tracking, you must

    - implement a `self.state_tracker` module to compute the next state
    - override `wrangl.learn.rl.MoolibVtrace.initial_state` to provide initial model states
    """

    @classmethod
    def compute_baseline_loss(cls, actor_baseline, learner_baseline, target, clip_delta_value=None):
        # Adjustments courtesy of Eric Hambro
        baseline_loss = (target - learner_baseline) ** 2

        if clip_delta_value:
            # Common PPO trick - clip a change in baseline fn
            # (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
            delta_baseline = learner_baseline - actor_baseline
            clipped_baseline = actor_baseline + torch.clamp(
                delta_baseline, -clip_delta_value, clip_delta_value
            )
            clipped_baseline_loss = (target - clipped_baseline) ** 2
            baseline_loss = torch.max(baseline_loss, clipped_baseline_loss)
        return 0.5 * torch.mean(baseline_loss)

    def compute_policy_gradient_loss(actor_log_prob, learner_log_prob, advantages, normalize_advantages=False, clip_delta_policy=None):
        # Adjustments courtesy of Eric Hambro
        advantages = advantages.detach()
        if normalize_advantages:
            # Common PPO trick (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
            adv = advantages
            advantages = (adv - adv.mean()) / max(1e-3, adv.std())
        if clip_delta_policy:
            # APPO policy loss - clip a change in policy fn
            ratio = torch.exp(learner_log_prob - actor_log_prob)
            policy_loss = ratio * advantages
            clip_high = 1 + clip_delta_policy
            clip_low = 1.0 / clip_high
            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
            clipped_policy_loss = clipped_ratio * advantages
            policy_loss = torch.min(policy_loss, clipped_policy_loss)
        else:
            # IMPALA policy loss
            policy_loss = learner_log_prob * advantages
        return -torch.mean(policy_loss)

    @classmethod
    def compute_entropy_loss(cls, logits):
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
        return -torch.mean(entropy_per_timestep)

    @classmethod
    def create_scheduler(cls, optimizer, FLAGS):
        factor = FLAGS.unroll_length * FLAGS.virtual_batch_size / FLAGS.max_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: max(1 - epoch * factor, 0)
        )
        return scheduler

    @classmethod
    def compute_gradients(cls, FLAGS, data, learner_state, stats):
        model = learner_state.model

        env_outputs = data["env_outputs"]
        actor_outputs = data["actor_outputs"]
        initial_core_state = data["initial_core_state"]

        model.train()

        learner_outputs, _ = model(env_outputs, initial_core_state)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from env_outputs[t] -> action[t] to action[t] -> env_outputs[t].
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)
        env_outputs = nest.map(lambda t: t[1:], env_outputs)
        actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)

        rewards = env_outputs["reward"] * FLAGS.reward_scale
        if FLAGS.reward_clip:
            rewards = torch.clip(rewards, -FLAGS.reward_clip, FLAGS.reward_clip)

        # TODO: reward normalization ?

        discounts = (~env_outputs["done"]).float() * FLAGS.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=actor_outputs["action"],
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        entropy_loss = FLAGS.entropy_cost * cls.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )
        pg_loss = cls.compute_policy_gradient_loss(
            vtrace_returns.behavior_action_log_probs,
            vtrace_returns.target_action_log_probs,
            vtrace_returns.pg_advantages,
            FLAGS.normalize_advantages,
            FLAGS.appo_clip_policy,
        )
        baseline_loss = FLAGS.baseline_cost * cls.compute_baseline_loss(
            actor_outputs['baseline'],
            learner_outputs['baseline'],
            vtrace_returns.vs,
            FLAGS.appo_clip_baseline,
        )
        total_loss = entropy_loss + pg_loss + baseline_loss
        total_loss.backward()
        stats["env_train_steps"] += FLAGS.unroll_length * FLAGS.batch_size

    @classmethod
    def step_optimizer(cls, FLAGS, learner_state, stats):
        unclipped_grad_norm = nn.utils.clip_grad_norm_(learner_state.model.parameters(), FLAGS.grad_clip_norm)
        learner_state.optimizer.step()
        learner_state.scheduler.step()
        learner_state.model_version += 1
        stats["unclipped_grad_norm"] += unclipped_grad_norm.item()
        stats["optimizer_steps"] += 1
        stats["model_version"] += 1

    @classmethod
    def log(cls, FLAGS, logger, stats, step, is_global=False):
        stats_values = {}
        prefix = "global/" if is_global else "local/"
        for k, v in stats.items():
            stats_values[prefix + k] = v.result()
            v.reset()

        logging.info(stats_values)
        if is_global:
            logger.log_metrics(stats_values, step=step)
            logger.save()

        if FLAGS.wandb.enable:
            wandb.log(stats_values, step=step)

    @classmethod
    def save_checkpoint(cls, FLAGS, checkpoint_path, learner_state):
        tmp_path = "%s.tmp.%s" % (checkpoint_path, moolib.create_uid())

        logging.info("saving global stats %s", learner_state.global_stats)

        checkpoint = {
            "learner_state": learner_state.save(),
            "flags": omegaconf.OmegaConf.to_container(FLAGS),
        }

        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)

        logging.info("Checkpoint saved to %s", checkpoint_path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path, learner_state):
        checkpoint = torch.load(checkpoint_path)
        learner_state.load(checkpoint["learner_state"])

    @classmethod
    def calculate_sps(cls, stats, delta, prev_steps):
        env_train_steps = stats["env_train_steps"].result()
        logging.info("calculate_sps %g steps in %g", env_train_steps - prev_steps, delta)
        stats["SPS"] += (env_train_steps - prev_steps) / delta
        return env_train_steps

    @classmethod
    def create_env_pool(cls, FLAGS, create_env, override_actor_batches: int = None) -> moolib.EnvPool:
        """
        Returns a batched environment.

        Args:
            FLAGS: experiment config.
            create_env: function `f(FLAGS)` that returns a single Gym environment in the batch.
            override_actor_batches: optional argument to override the number of actor batches.
        """
        return moolib.EnvPool(
            lambda: create_env(FLAGS),
            num_processes=FLAGS.num_actor_cpus,
            batch_size=FLAGS.actor_batch_size,
            num_batches=FLAGS.num_actor_batches if override_actor_batches is None else override_actor_batches,
        )

    @classmethod
    def run_train_test(cls, FLAGS, create_train_env, create_eval_env, model_kwargs=None):
        """
        Trains a policy.

        Args:
            FLAGS: experiment config.
            create_train_env: function `f(FLAGS)` that returns a single Gym environment for training.
            create_eval_env: function `f(FLAGS)` that returns a single Gym environment for validation.
            model_kwargs: model keyword args.
        """
        dsave = os.getcwd()
        localdir = os.path.join(dsave, 'peers', FLAGS.local_name)
        fconfig = os.path.join(dsave, 'config.yaml')
        omegaconf.OmegaConf.save(config=FLAGS, f=fconfig)

        envs = cls.create_env_pool(FLAGS, create_train_env)
        eval_envs = cls.create_env_pool(FLAGS, create_eval_env, override_actor_batches=1)

        logging.info("Flags:\n%s\n", pprint.pformat(omegaconf.OmegaConf.to_container(FLAGS, resolve=True)))
        logging.info("Save directory: %s", dsave)

        train_id = "%s/%s/%s" % (FLAGS.wandb.entity, FLAGS.wandb.project, FLAGS.wandb.name)

        logging.info("train_id: %s", train_id)

        if not os.path.isdir(localdir):
            os.makedirs(localdir)

        print("EnvPool started")

        device = 'cpu'
        if FLAGS.gpus > 0:
            device = 'cuda:0'
            # hack for moolib
        moolib_args = Namespace(**omegaconf.OmegaConf.to_container(FLAGS, resolve=True))
        moolib_args.device = device

        model_kwargs = model_kwargs or {}
        model = cls(FLAGS, **model_kwargs).to(device)
        optimizer = model.configure_optimizers()
        scheduler = cls.create_scheduler(optimizer, FLAGS)
        learner_state = LearnerState(model, optimizer, scheduler)

        model_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Number of model parameters: %i", model_numel)

        if FLAGS.wandb.enable:
            wandb.init(
                project=str(FLAGS.wandb.project),
                config=omegaconf.OmegaConf.to_container(FLAGS),
                entity=FLAGS.wandb.entity,
                name=FLAGS.wandb.name,
            )

        env_states = [
            common.EnvBatchState(moolib_args, model) for _ in range(FLAGS.num_actor_batches)
        ]
        eval_env_state = common.EnvBatchState(moolib_args, model)

        rpc = moolib.Rpc()
        rpc.set_name(FLAGS.local_name)
        rpc.connect(FLAGS.connect)

        rpc_group = moolib.Group(rpc, name=train_id)

        accumulator = moolib.Accumulator(
            group=rpc_group,
            name="model",
            parameters=model.parameters(),
            buffers=model.buffers(),
        )
        accumulator.set_virtual_batch_size(FLAGS.virtual_batch_size)

        learn_batcher = moolib.Batcher(FLAGS.batch_size, device, dim=1)

        stats = {
            "mean_episode_return": common.StatMean(),
            "mean_episode_step": common.StatMean(),
            "SPS": common.StatMean(),
            "eval_mean_episode_return": common.StatMean(),
            "eval_running_reward": common.StatMean(),
            "env_act_steps": common.StatSum(),
            "env_train_steps": common.StatSum(),
            "optimizer_steps": common.StatSum(),
            "running_reward": common.StatMean(),
            "running_step": common.StatMean(),
            "steps_done": common.StatSum(),
            "episodes_done": common.StatSum(),
            "unclipped_grad_norm": common.StatMean(),
            "model_version": common.StatSum(),
            "virtual_batch_size": common.StatMean(),
            "num_gradients": common.StatMean(),
        }
        learner_state.global_stats = copy.deepcopy(stats)

        checkpoint_path = FLAGS.ckpt_path

        if FLAGS.autoresume and os.path.exists(checkpoint_path):
            logging.info("Loading checkpoint: %s" % checkpoint_path)
            cls.load_checkpoint(checkpoint_path, learner_state)
            accumulator.set_model_version(learner_state.model_version)
            logging.info("loaded stats %s", learner_state.global_stats)

        global_stats_accumulator = common.GlobalStatsAccumulator(
            rpc_group, learner_state.global_stats
        )

        terminate = False
        previous_signal_handler = {}

        def signal_handler(signum, frame):
            nonlocal terminate
            logging.info(
                "Got signal %s, quitting!",
                signal.strsignal(signum) if hasattr(signal, "strsignal") else signum,
            )
            terminate = True
            previous_handler = previous_signal_handler[signum]
            if previous_handler is not None:
                previous_signal_handler[signum] = None
                signal.signal(signum, previous_handler)

        previous_signal_handler[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, signal_handler
        )
        previous_signal_handler[signal.SIGINT] = signal.signal(
            signal.SIGINT, signal_handler
        )

        if torch.backends.cudnn.is_available():
            logging.info("Optimising CuDNN kernels")
            torch.backends.cudnn.benchmark = True

        # Run.
        now = time.time()
        warm_up_time = FLAGS.warmup
        prev_env_train_steps = 0
        prev_global_env_train_steps = 0
        next_env_index = 0
        last_log = now
        last_reduce_stats = now
        is_leader = False
        is_connected = False

        logger = CSVLogger(save_dir=dsave, name='logs', flush_logs_every_n_steps=FLAGS.flush_logs_every_n_steps)
        logger.log_hyperparams(FLAGS)

        if FLAGS.git.enable:
            git = GitCallback()
            git.on_init_end()

        if FLAGS.s3.enable:
            s3 = S3Callback(FLAGS.s3)
            s3.on_fit_start(trainer=None, model=None)

        while not terminate:
            prev_now = now
            now = time.time()

            steps = learner_state.global_stats["env_train_steps"].result()
            if steps >= FLAGS.max_steps:
                logging.info("Stopping training after %i steps", steps)
                break

            rpc_group.update()
            accumulator.update()
            if accumulator.wants_state():
                assert accumulator.is_leader()
                accumulator.set_state(learner_state.save())
            if accumulator.has_new_state():
                assert not accumulator.is_leader()
                learner_state.load(accumulator.state())

            was_connected = is_connected
            is_connected = accumulator.connected()
            if not is_connected:
                if was_connected:
                    logging.warning("Training interrupted!")
                # If we're not connected, sleep for a bit so we don't busy-wait
                logging.info("Your training will commence shortly.")
                time.sleep(10)
                continue

            was_leader = is_leader
            is_leader = accumulator.is_leader()
            if not was_connected:
                logging.info(
                    "Training started. Leader is %s, %d members, model version is %d"
                    % (
                        "me!" if is_leader else accumulator.get_leader(),
                        len(rpc_group.members()),
                        learner_state.model_version,
                    )
                )
                prev_global_env_train_steps = learner_state.global_stats[
                    "env_train_steps"
                ].result()

                if warm_up_time > 0:
                    logging.info("Warming up for %g seconds", warm_up_time)

            if warm_up_time > 0:
                warm_up_time -= now - prev_now

            learner_state.train_time += now - prev_now
            if now - last_reduce_stats >= 2:
                last_reduce_stats = now
                global_stats_accumulator.reduce(stats)
            if now - last_log >= FLAGS.log_every_n_steps:
                delta = now - last_log
                last_log = now

                global_stats_accumulator.reduce(stats)
                global_stats_accumulator.reset()

                prev_env_train_steps = cls.calculate_sps(stats, delta, prev_env_train_steps)
                prev_global_env_train_steps = cls.calculate_sps(
                    learner_state.global_stats, delta, prev_global_env_train_steps
                )

                steps = learner_state.global_stats["env_train_steps"].result()

                cls.log(FLAGS, logger, stats, step=steps, is_global=False)
                cls.log(FLAGS, logger, learner_state.global_stats, step=steps, is_global=True)
                if FLAGS.s3.enable:
                    s3.on_validation_end(trainer=None, model=None)

                if warm_up_time > 0:
                    logging.info(
                        "Warming up up for an additional %g seconds", round(warm_up_time)
                    )

            if is_leader:
                if not was_leader:
                    leader_filename = os.path.join(
                        dsave, "leader-%03d" % learner_state.num_previous_leaders
                    )
                    record.symlink_path(localdir, leader_filename)
                    logging.info(
                        "Created symlink %s -> %s", leader_filename, localdir
                    )
                    learner_state.num_previous_leaders += 1
                if not was_leader and not os.path.exists(checkpoint_path):
                    logging.info("Training a new model from scratch.")
                if learner_state.train_time - learner_state.last_checkpoint >= FLAGS.val_check_interval:
                    learner_state.last_checkpoint = learner_state.train_time
                    cls.save_checkpoint(FLAGS, checkpoint_path, learner_state)
                    test_results = cls.run_test(FLAGS, eval_envs, checkpoint_path, env_state=eval_env_state, model_kwargs=model_kwargs, eval_steps=FLAGS.eval_steps)
                    for k in ['mean_episode_return', 'running_reward']:
                        if test_results[k] is not None:
                            stats['eval_{}'.format(k)] += test_results[k]

                    if FLAGS.s3.enable:
                        s3.on_save_checkpoint(trainer=None, model=None, checkpoint=learner_state.save())

            if accumulator.has_gradients():
                gradient_stats = accumulator.get_gradient_stats()
                stats["virtual_batch_size"] += gradient_stats["batch_size"]
                stats["num_gradients"] += gradient_stats["num_gradients"]
                cls.step_optimizer(FLAGS, learner_state, stats)
                accumulator.zero_gradients()
            elif not learn_batcher.empty() and accumulator.wants_gradients():
                cls.compute_gradients(FLAGS, learn_batcher.get(), learner_state, stats)
                accumulator.reduce_gradients(FLAGS.batch_size)
            else:
                if accumulator.wants_gradients():
                    accumulator.skip_gradients()

                # Generate data.
                cur_index = next_env_index
                next_env_index = (next_env_index + 1) % FLAGS.num_actor_batches

                env_state = env_states[cur_index]
                if env_state.future is None:
                    env_state.future = envs.step(cur_index, env_state.prev_action)
                cpu_env_outputs = env_state.future.result()

                env_outputs = nest.map(
                    lambda t: t.to(device, copy=True), cpu_env_outputs
                )

                env_outputs["prev_action"] = env_state.prev_action
                prev_core_state = env_state.core_state
                model.eval()
                with torch.no_grad():
                    actor_outputs, env_state.core_state = model(
                        nest.map(lambda t: t.unsqueeze(0), env_outputs),
                        env_state.core_state,
                    )
                actor_outputs = nest.map(lambda t: t.squeeze(0), actor_outputs)
                action = actor_outputs["action"]
                env_state.update(cpu_env_outputs, action, stats)
                # envs.step invalidates cpu_env_outputs
                del cpu_env_outputs
                env_state.future = envs.step(cur_index, action)

                stats["env_act_steps"] += action.numel()

                last_data = {
                    "env_outputs": env_outputs,
                    "actor_outputs": actor_outputs,
                }
                if warm_up_time <= 0:
                    env_state.time_batcher.stack(last_data)

                if not env_state.time_batcher.empty():
                    data = env_state.time_batcher.get()
                    data["initial_core_state"] = env_state.initial_core_state
                    learn_batcher.cat(data)

                    # We need the last entry of the previous time batch
                    # to be put into the first entry of this time batch,
                    # with the initial_core_state to match
                    env_state.initial_core_state = prev_core_state
                    env_state.time_batcher.stack(last_data)
        if is_connected and is_leader:
            cls.save_checkpoint(FLAGS, checkpoint_path, learner_state)
        logging.info("Graceful exit. Bye bye!")

    @classmethod
    def run_test(cls, FLAGS: omegaconf.OmegaConf, envs: moolib.EnvPool, fcheckpoint: str, env_state=None, eval_steps=100, model_kwargs=None) -> dict:
        """
        Evaluates checkpoint of policy.

        Args:
            FLAGS: experiment config.
            envs: batched environments.
            fcheckpoint: saved checkpoint.
            env_state: batched environment state, if not given then will be created internally.
            eval_steps: how many steps to evaluate.
            model_kwargs: model keyword args.

        Returns:
            a dictionary of evaluation results.
        """
        model_kwargs = model_kwargs or {}

        device = 'cpu'
        if FLAGS.gpus > 0:
            device = 'cuda:0'
            # hack for moolib

        model = cls(FLAGS, **model_kwargs).to(device)

        if env_state is None:
            device = 'cpu'
            if FLAGS.gpus > 0:
                device = 'cuda:0'
                # hack for moolib
            moolib_args = Namespace(**omegaconf.OmegaConf.to_container(FLAGS, resolve=True))
            moolib_args.device = device
            env_state = common.EnvBatchState(moolib_args, model)

        env_act_steps = 0

        stats = {
            "mean_episode_return": common.StatMean(),
            "mean_episode_step": common.StatMean(),
            "env_act_steps": common.StatSum(),
            "env_train_steps": common.StatSum(),
            "running_reward": common.StatMean(),
            "running_step": common.StatMean(),
            "steps_done": common.StatSum(),
            "episodes_done": common.StatSum(),
        }

        while env_act_steps < eval_steps:
            # Generate data.
            if env_state.future is None:
                env_state.future = envs.step(0, env_state.prev_action)
            cpu_env_outputs = env_state.future.result()

            env_outputs = nest.map(
                lambda t: t.to(device, copy=True), cpu_env_outputs
            )

            env_outputs["prev_action"] = env_state.prev_action
            prev_core_state = env_state.core_state
            model.eval()
            with torch.no_grad():
                actor_outputs, env_state.core_state = model(
                    nest.map(lambda t: t.unsqueeze(0), env_outputs),
                    env_state.core_state,
                )
            actor_outputs = nest.map(lambda t: t.squeeze(0), actor_outputs)
            action = actor_outputs["action"]
            env_state.update(cpu_env_outputs, action, stats)

            # envs.step invalidates cpu_env_outputs
            del cpu_env_outputs
            env_state.future = envs.step(0, action)

            stats["env_act_steps"] += action.numel()
            env_act_steps += action.numel()

            last_data = {
                "env_outputs": env_outputs,
                "actor_outputs": actor_outputs,
            }

            if not env_state.time_batcher.empty():
                data = env_state.time_batcher.get()
                data["initial_core_state"] = env_state.initial_core_state

                # We need the last entry of the previous time batch
                # to be put into the first entry of this time batch,
                # with the initial_core_state to match
                env_state.initial_core_state = prev_core_state
                env_state.time_batcher.stack(last_data)

        stats_values = {}
        for k, v in stats.items():
            stats_values[k] = v.result()
        return stats_values

    def initial_state(self, batch_size=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the initial states for the policy learner.
        """
        if not self.hparams.stateful:
            return tuple()
        return tuple(
            torch.zeros(self.state_tracker.num_layers, batch_size, self.state_tracker.hidden_size)
            for _ in range(2)
        )

    def make_state_tracker(self, din, dout):
        return nn.LSTM(din, dout, num_layers=1) if self.hparams.stateful else None

    def update_state(self, state_tracker_input, state_tracker_state, done):
        if not self.hparams.stateful:
            return state_tracker_input, state_tracker_state
        T, B, *_ = done.size()
        state_tracker_input = state_tracker_input.view(T, B, -1)
        state_tracker_output_list = []
        notdone = (~done).float()
        for input, nd in zip(state_tracker_input.unbind(), notdone.unbind()):
            # Reset state_tracker state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # states:
            nd = nd.view(1, -1, 1)
            state_tracker_state = nest.map(nd.mul, state_tracker_state)
            output, state_tracker_state = self.state_tracker(input.unsqueeze(0), state_tracker_state)
            state_tracker_output_list.append(output)
        state_tracker_output = torch.cat(state_tracker_output_list)
        return state_tracker_output, state_tracker_state

    def forward(self, inputs, state_tracker_state=None) -> Tuple[dict, tuple]:
        """
        Computing one step of the policy.

        Args:
            inputs: policy inputs, each is of shape [T, B, _*] where T is the unroll length and B is the batch size.
            state_tracker_state: previous learner states, each is of shape [T, B, _*].

        This method must return a tuple `(output, state_tracker_state)` of the following format:

        ```python
        output = dict(
            policy_logits=policy_logits  # shape [T, B, num_actions],
            baseline=baseline  # shape [T, B],
            action=action  # shape [T, B],
        )
        return output, state_tracker_state
        ```
        """
        raise NotImplementedError()
