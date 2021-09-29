import gym
import torch
from torch import nn
from wrangl.learn import TorchbeastModel


class MyRLModel(TorchbeastModel):

    def __init__(self, hparams, observation_shapes, num_actions):
        super().__init__(hparams, observation_shapes=observation_shapes, num_actions=num_actions)
        shape, dtype = observation_shapes['obs']
        self.rep = nn.Sequential(
            nn.Linear(sum(shape), 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh()
        )
        self.policy = nn.Linear(200, num_actions)
        self.baseline = nn.Linear(200, 1)


    def forward(self, obs, core):
        inp = obs['obs']  # T, B, dim
        rep = self.rep(inp)
        policy = self.policy(rep)
        out = dict(
            policy_logits=policy,
            baseline=self.baseline(rep).squeeze(-1),
            action=policy.max(-1)[1],
        )
        return out, core


def create_env(flags):
    return gym.make('CartPole-v0')


def get_env_shapes(env):
    return dict(obs=(env.observation_space.shape, torch.float)), env.action_space.n


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = MyRLModel.get_parser(batch_size=10, num_actors=10, num_buffers=20, num_train_steps=int(1e6), print_period_second=2)
    args = parser.parse_args()
    MyRLModel.run_train(args, create_env=create_env, create_eval_env=create_env, get_env_shapes=get_env_shapes)
