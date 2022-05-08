import torch
from torch import nn
from model.multi import Model as Base


class Model(Base):

    def __init__(self, flags, env):
        flags.demb = 5
        super().__init__(flags, env)
        self.core_trans = nn.Linear(self.drep, 64)

    def encode_cell(self, inputs):
        return inputs['features'].float()  # T, B, H, W, channels

    def score_actions(self, inputs, conv, core):
        TB, channels, W, H = conv.size()
        pool = conv.max(3)[0].transpose(1, 2)  # TB, W, channels
        slices = []
        for x_i, feat_i in zip(inputs['x'].view(TB, -1), pool):
            slices_i = torch.index_select(feat_i, 0, x_i)  # select width channel
            slices.append(slices_i)
        commands = torch.stack(slices, dim=0)  # TB, num_actions, channels

        core_trans = self.core_trans(core)  # TB, num_actions, drep
        rep_exp = core_trans.unsqueeze(1).expand_as(commands)
        policy_logits = rep_exp.mul(commands).sum(2)
        return policy_logits
