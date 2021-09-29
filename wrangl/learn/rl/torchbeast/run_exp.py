import argparse
import pathlib
import logging
import os
import sys
import tqdm
import importlib
import copy
import json
from pathlib import Path

import threading
import time
import timeit
import traceback
import pprint
import typing
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp

import gym
import random

from .core import environment
from .core import vtrace


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
        model: torch.nn.Module, buffers: Buffers, initial_agent_state_buffers, flags, create_env):
    try:
        logging.info('Actor %i started.', i)

        gym_env = create_env(flags)
        seed = i ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env = environment.Environment(gym_env)
        env_output = env.initial()

        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                action = agent_output['action']
                env_output = env.step(action)

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

            full_queue.put(index)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              initial_agent_state_buffers,
              flags,
              device,
              lock=threading.Lock()) -> typing.Dict[str, torch.Tensor]:
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    for m in indices:
        free_queue.put(m)
    batch = {
        k: t.to(device=device, non_blocking=True)
        for k, t in batch.items()
    }
    initial_agent_state = tuple(
        t.to(device=device, non_blocking=True) for t in initial_agent_state
    )
    return batch, initial_agent_state


def learn(actor_model,
          model,
          batch,
          initial_agent_state,
          optimizer,
          scheduler,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    # logging.info('Learner started on device {}.'.format(initial_agent_state[0].device))
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = learner_outputs['baseline'][-1]

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in actor_batch and learner_outputs at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        clipped_rewards = model.clip_rewards(rewards, clip_method=flags.reward_clipping)

        discounts = (~batch['done']).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = model.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * model.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * model.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_lens = batch['episode_step'][batch['done']]
        won = batch['reward'][batch['done']] > 0.8
        stats = {
            'mean_win_rate': torch.mean(won.float()).item(),
            'mean_episode_len': torch.mean(episode_lens.float()).item(),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }

        optimizer.zero_grad()
        model.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        scheduler.step()

        # Interestingly, this doesn't require moving off cuda first?
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(observation_shapes, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    for k, (shape, dtype) in observation_shapes.items():
        specs[k] = dict(size=(T + 1, *shape), dtype=dtype)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags, create_env, create_eval_env, get_env_shapes, create_model, write_result, write_eval_result, device=None):  # pylint: disable=too-many-branches, too-many-statements
    assert os.environ['OMP_NUM_THREADS'] == '1', 'Must run with OMP_NUM_THREADS=1'
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    T = flags.unroll_length
    B = flags.batch_size
    dout = pathlib.Path(flags.dout)

    env = create_env(flags)
    observation_shapes, num_actions = get_env_shapes(env)
    model = create_model(flags, observation_shapes, num_actions)
    model.ensure_dout()
    buffers = create_buffers(observation_shapes, num_actions, flags)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, buffers, initial_agent_state_buffers, flags, create_env))
        actor.start()
        actor_processes.append(actor)

    learner_model = create_model(flags, observation_shapes, num_actions).to(device=device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.num_train_steps) / flags.num_train_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stats = {}

    fresume = learner_model.get_fresume()
    if fresume and fresume.exists():
        learner_model.load_checkpoint(fresume, override_hparams=learner_model.hparams, model=learner_model, optimizer=optimizer, scheduler=scheduler)

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal stats
        while model.train_steps < flags.num_train_steps:
            batch, agent_state = get_batch(free_queue, full_queue, buffers, initial_agent_state_buffers, flags, device)

            stats = learn(model, learner_model, batch, agent_state, optimizer, scheduler, flags)

            with lock:
                model.train_steps += T * B
                stats['train_steps'] = model.train_steps
                write_result(stats)

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)

    def terminate(status=1):
        for _ in range(flags.num_actors):
            free_queue.put(None)
        if status == 0:
            for actor in actor_processes:
                actor.join(timeout=1)
        else:
            model.logger.error('TERMINATING. You may exit with Ctrl+C')
            for p in actor_processes:
                p.terminate()
                p.kill()
            sys.exit(status)

    def checkpoint():
        model.save_checkpoint(metrics=None, optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), model_state=model.state_dict())

    eval_env = create_eval_env(flags)

    test_flags = copy.deepcopy(flags)
    test_flags.resume = 'auto'

    def val_checkpoint():
        model.logger.info('Testing checkpoint')
        test_result = test(test_flags, flags.test_eps, eval_env, get_env_shapes, create_model, write_eval_result)
        test_result['train_steps'] = model.train_steps
        write_eval_result(test_result)
        s = 'Evaluating after {} steps:\n{}'.format(model.train_steps, pprint.pformat(test_result))
        model.logger.critical(s)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while model.train_steps < flags.num_train_steps:
            for t in threads:
                if not t.is_alive():
                    terminate()
            for p in actor_processes:
                if not p.is_alive():
                    terminate()

            start_step = model.train_steps
            start_time = timer()
            time.sleep(flags.print_period_seconds)

            if timer() - last_checkpoint_time > flags.save_period_seconds:
                checkpoint()
                val_checkpoint()
                last_checkpoint_time = timer()

            fps = (model.train_steps - start_step) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats['mean_episode_return']
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            s = 'After {} steps: loss {} @ {:.2f} fps. {} Stats:\n{}'.format(model.train_steps, total_loss, fps, mean_return, pprint.pformat(stats))
            model.logger.critical(s)
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info('Learning finished after %d steps.', model.train_steps)
    finally:
        terminate(status=0)

    checkpoint()


def test(flags, num_eps, eval_env, get_env_shapes, create_model, write_eval_result):
    observation_shapes, num_actions = get_env_shapes(eval_env)
    model = create_model(flags, observation_shapes, num_actions)
    fresume = model.get_fresume()
    model.load_checkpoint(fresume, override_hparams=model.hparams, model=model)
    model.eval()

    env = environment.Environment(eval_env)
    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(observation, agent_state)
    start_time = time.time()

    while len(won) < num_eps:
        done = False
        steps = 0
        while not done:
            agent_outputs, agent_state = model(observation, agent_state)
            observation = env.step(agent_outputs['action'])
            policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
            log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
            e = -torch.sum(policy * log_policy, dim=-1)
            entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)
                agent_state = model.initial_state(batch_size=1)

    fps = steps / (time.time() - start_time)
    return{
        'mean_episode_returns': sum(returns)/len(returns),
        'mean_win_rate': sum(won)/len(returns),
        'mean_episode_len': sum(ep_len)/len(ep_len),
        'mean_fps': fps,
    }
