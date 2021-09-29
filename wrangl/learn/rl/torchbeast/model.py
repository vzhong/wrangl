import json
import torch
import typing
import pathlib
from torch.nn import functional as F
from ...model import Model as BaseModel
from .run_exp import train, test
from argparse import ArgumentParser, Namespace


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


class TorchbeastModel(BaseModel):
    """
    Torchbeast model.
    """

    @classmethod
    def get_parser(cls, **defaults) -> ArgumentParser:
        parser = super().get_parser(**defaults)

        # Training settings.
        parser.add_argument('--num_actors', default=defaults.get('num_actors', 1), type=int, metavar='N', help='Number of actors.')
        parser.add_argument('--unroll_length', default=defaults.get('unroll_length', 36), type=int, metavar='T', help='The unroll length (time dimension; default: 64).')
        parser.add_argument('--queue_timeout', default=defaults.get('queue_timeout', 1), type=int, metavar='S', help='Error timeout for queue.')
        parser.add_argument('--num_buffers', default=defaults.get('num_buffers', 2), type=int, metavar='N', help='Number of shared-memory buffers.')
        parser.add_argument('--num_threads', default=defaults.get('num_threads', 4), type=int, metavar='N', help='Number learner threads.')
        parser.add_argument('--save_period_seconds', default=defaults.get('save_seconds', 5*60), type=int, metavar='N', help='How often to save.')
        parser.add_argument('--print_period_seconds', default=defaults.get('print_seconds', 5), type=int, metavar='N', help='How often to print.')
        parser.add_argument('--test_eps', default=defaults.get('test_eps', 10), type=int, metavar='N', help='How many episodes to test.')

        # Loss settings.
        parser.add_argument('--entropy_cost', default=defaults.get('entropy_cost', 0.05), type=float, help='Entropy cost/multiplier.')
        parser.add_argument('--baseline_cost', default=defaults.get('baseline_cost', 0.5), type=float, help='Baseline cost/multiplier.')
        parser.add_argument('--discounting', default=defaults.get('discounting', 0.99), type=float, help='Discounting factor.')
        parser.add_argument('--reward_clipping', default=defaults.get('reward_clipping', 'abs_one'), choices=['abs_one', 'soft_asymmetric', 'none'], help='Reward clipping.')
    
        # Optimizer settings.
        parser.add_argument('--alpha', default=defaults.get('alpha', 0.99), type=float, help='RMSProp smoothing constant.')
        parser.add_argument('--momentum', default=defaults.get('momentum', 0), type=float, help='RMSProp momentum.')
        parser.add_argument('--epsilon', default=defaults.get('epsilon', 0.01), type=float, help='RMSProp epsilon.')
        return parser

    @classmethod
    def run_train(cls, flags: Namespace, create_env: typing.Callable, create_eval_env: typing.Callable, get_env_shapes: typing.Callable):
        """
        Runs Monobeast training.

        Args:
            flags: training arguments.
            create_env: function with signature `f(flags)` that creates training gym environment.
            create_eval_env: function with signature `f(flags)` that creates evaluation gym environment.
            get_env_shapes: function with signature `f(env)` that returns a tuple of the observation shapes and the number of actions.

        Note:
            the observation shapes should be a dictionary that maps from observation name to a tuple (`shape`, `dtype`).
        """
        dout = pathlib.Path(flags.dout)
        flog = dout.joinpath('train.log.json')

        def write_result(res, mode='train'):
            res['type'] = mode
            with flog.open('at') as f:
                f.write(json.dumps(res) + '\n')

        def write_eval_result(res):
            write_result(res, mode='eval')

        return train(
            flags,
            create_env=create_env,
            create_eval_env=create_eval_env,
            get_env_shapes=get_env_shapes,
            create_model=cls,
            write_result=write_result,
            write_eval_result=write_eval_result,
        )

    def __init__(self, hparams, observation_shapes: dict, num_actions: int):
        super().__init__(hparams)
        self.observation_shapes = observation_shapes
        self.num_actions = num_actions

    def initial_state(self, batch_size=1):
        """
        Returns tuple of initial model state
        """
        return tuple()

    def forward(self, obs: typing.Dict[str, torch.Tensor], prev_state: typing.Tuple) -> typing.Tuple[typing.Dict[str, torch.Tensor], tuple]:
        """
        Computes action distribution given input and previous model state.

        Args:
            obs: dictionary of input observations.
            prev_state: memory from last time step.

        Note:
            all input shapes are (T, B, *shapes), where T is the `unroll_length` and B is the `batch_size`.
            the return arguments should also be in this format.

        Returns:
            a dictionary of tensors containing `polic_logits`, `baseline` value estimate, and `action`.
            the next model state.
            
        """
        raise NotImplementedError()

    @staticmethod
    def compute_baseline_loss(advantages):
        # Take the mean over batch, sum over time.
        return 0.5 * torch.sum(torch.mean(advantages ** 2, dim=1))

    @staticmethod
    def compute_entropy_loss(logits):
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
        return -torch.sum(torch.mean(entropy_per_timestep, dim=1))

    @staticmethod
    def compute_policy_gradient_loss(logits, actions, advantages):
        cross_entropy = F.nll_loss(
            F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
            target=torch.flatten(actions, 0, 1),
            reduction='none')
        cross_entropy = cross_entropy.view_as(advantages)
        advantages.requires_grad = False
        policy_gradient_loss_per_timestep = cross_entropy * advantages
        return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))

    @staticmethod
    def clip_rewards(rewards, clip_method='none'):
        if clip_method == 'abs_one':
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif clip_method == 'soft_asymmetric':
            squeezed = torch.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed,
                                          squeezed) * 5.0
        elif clip_method == 'none':
            clipped_rewards = rewards
        else:
            raise NotImplementedError()
        return clipped_rewards
