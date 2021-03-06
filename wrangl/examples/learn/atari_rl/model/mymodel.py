# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F

from wrangl.learn.rl import MoolibVtrace


class Model(MoolibVtrace):

    def __init__(self, FLAGS, num_actions=18, input_channels=4):
        super().__init__(FLAGS)
        self.num_actions = num_actions

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        # FC output size + one-hot of last action + last reward.
        core_input_size = core_output_size = self.fc.out_features + num_actions + 1
        if FLAGS.stateful:
            core_output_size = 256

        self.state_tracker = self.make_state_tracker(core_input_size, core_output_size)

        self.policy = nn.Linear(core_output_size, num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def forward(self, inputs, core_state=None):
        reward = inputs["reward"]
        x = inputs["state"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["prev_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(reward, -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        core_output, core_state = self.update_state(core_input, core_state, done=inputs['done'])

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        action = torch.multinomial(F.softmax(policy_logits.reshape(T*B, -1), dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
        )
        return output, core_state
