from torch import nn
from model.multi import Model as Base


class Model(Base):

    def __init__(self, flags, env):
        super().__init__(flags, env)
        self.cmd_rnn = nn.LSTM(self.demb, self.drnn, bidirectional=True, batch_first=True)
        self.cmd_trans = nn.Linear(2*self.drnn, self.drep)
        self.cmd_scorer = nn.Linear(self.drep, 1)

    def score_actions(self, inputs, conv, core):
        command = inputs['command'].long()
        command_len = inputs['command_len'].long()
        T, B, num_cmd, max_cmd_len = command.size()

        command_rnn = self.run_rnn(self.cmd_rnn, command.view(-1, max_cmd_len), command_len.view(-1))
        command_trans = self.cmd_trans(command_rnn)

        # rep is (T*B, drep)
        rep_exp = core.unsqueeze(1).expand(core.size(0), num_cmd, core.size(1))

        attn, _ = self.run_attn(command_trans, command_len.reshape(-1), rep_exp.reshape(-1, core.size(1)))
        policy_logits = self.cmd_scorer(attn).squeeze(-1).view(T*B, -1)
        return policy_logits
