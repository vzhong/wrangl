import os
os.environ['OMP_NUM_THREADS'] = '1'
import gym
import torch
import pprint
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from rtfm.tasks import GroupsSimpleStationary, GroupsSimpleStationaryDev
from rtfm import featurizer as X
from wrangl.learn import TorchbeastModel


class RTFMEasy(gym.Env):

    TEXT_FIELDS = ['wiki', 'task', 'inv']
    RTFM_ENV = GroupsSimpleStationary

    def __init__(self, featurizer=X.Concat([X.Text(), X.ValidMoves(), X.Position(), X.RelativePosition()]), room_shape=(6, 6), max_name=8, max_inv=8, max_wiki=80, max_task=40, max_text=80, partially_observable=False, max_placement=1, shuffle_wiki=False, time_penalty=-0.02):
        self.rtfm_env = self.RTFM_ENV(
            room_shape=room_shape, featurizer=featurizer, partially_observable=partially_observable, max_placement=max_placement, max_name=max_name, max_inv=max_inv, max_wiki=max_wiki, max_task=max_task, time_penalty=time_penalty, shuffle_wiki=shuffle_wiki
        )
        self.vocab = self.rtfm_env.vocab
        self.steps_taken = 0
        self.max_steps = 80
        self.height, self.width = room_shape
        self.max_text = max_text
        self.max_grid = self.height * self.width * max_name
        self.num_actions = len(self.rtfm_env.action_space)
        self.observation_shapes = {
            'name': ((self.height, self.width, max_placement, max_name), torch.long),
            'name_len': ((self.height, self.width, max_placement), torch.long),
            'wiki': ((max_wiki, ), torch.long),
            'wiki_len': ((1, ), torch.long),
            'inv': ((max_inv, ), torch.long),
            'inv_len': ((1, ), torch.long),
            'task': ((max_task, ), torch.long),
            'task_len': ((1, ), torch.long),
            'rel_pos': ((self.height, self.width, 2), torch.float),
            'valid': ((len(self.rtfm_env.action_space), ), torch.float),  # a 1-0 vector that is a mask for valid actions, should be the same length as `self.action_space`
        }

    def reset(self):
        obs = self.rtfm_env.reset()
        self.steps_taken = 0
        return self.reformat(obs)

    def step(self, action):
        obs, reward, done, info = self.rtfm_env.step(action)
        self.steps_taken += 1
        if self.steps_taken > self.max_steps and reward < 1:
            done = True
            reward = -1
        return self.reformat(obs), reward, done, info

    def reformat(self, obs):
        del obs['position']
        return obs


class RTFMEasyDev(RTFMEasy):
    RTFM_ENV = GroupsSimpleStationaryDev


class DoubleFILM(nn.Module):
    # from https://arxiv.org/pdf/1806.01946.pdf

    def __init__(self, drnn, demb, dchannel, conv):
        super().__init__()
        self.drnn = drnn
        self.conv = conv
        self.gamma_beta_trans = nn.Linear(2*drnn+2*demb, 2*dchannel)

        self.gamma_beta_conv = nn.Conv2d(conv.in_channels, 2*dchannel, kernel_size=(3, 3), padding=1)
        self.trans_text = nn.Linear(2*drnn+2*demb, dchannel)
        self.image_summ = nn.Linear(dchannel, drnn)

    def forward(self, prev, wiki, nonwiki, pos, wiki_inv_attn):
        T, B, *_ = wiki.size()
        prev = torch.cat([prev, pos], dim=1)
        text = torch.cat([wiki, nonwiki, wiki_inv_attn], dim=1)
        gamma_beta_trans = self.gamma_beta_trans(text)
        gamma, beta = torch.chunk(gamma_beta_trans, 2, dim=1)

        conv = self.conv(prev)
        gamma = gamma.unsqueeze(2).unsqueeze(2).expand_as(conv)
        beta = beta.unsqueeze(2).unsqueeze(2).expand_as(conv)
        image_modulated_with_text = ((1+gamma) * conv + beta).relu()

        gamma_conv, beta_conv = torch.chunk(self.gamma_beta_conv(prev), 2, dim=1)
        text_trans = self.trans_text(text).unsqueeze(2).unsqueeze(2).expand_as(gamma_conv)
        text_modulated_with_image = ((1+gamma_conv) * text_trans + beta_conv).relu()

        mix = image_modulated_with_text + text_modulated_with_image
        image_summ = self.image_summ(mix.max(3)[0].max(2)[0])
        return mix, image_summ


class MyRTFMModel(TorchbeastModel):

    @classmethod
    def get_parser(cls):
        parser = super().get_parser(num_actors=30, batch_size=20, unroll_length=80, num_train_steps=int(1e8), print_period_seconds=60, save_period_seconds=60*5)
        parser.add_argument('--demb', type=int, default=30)
        parser.add_argument('--drnn', type=int, default=200)
        parser.add_argument('--drep', type=int, default=400)
        parser.add_argument('--stateful', action='store_true')
        parser.add_argument('--test', action='store_true')
        return parser

    def __init__(self, hparams, observation_shapes, num_actions, vocab_size, padding_idx, text_fields):
        super().__init__(hparams, observation_shapes=observation_shapes, num_actions=num_actions)
        self.text_fields = text_fields
        self.emb = nn.Embedding(vocab_size, hparams.demb, padding_idx=padding_idx)
        self.conv = nn.Sequential(
            nn.Conv2d(3*hparams.demb, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )

        self.film1 = DoubleFILM(2*hparams.drnn, hparams.drnn, 16, nn.Conv2d(hparams.demb+2, 16, kernel_size=(3, 3), padding=1))
        self.film2 = DoubleFILM(2*hparams.drnn, hparams.drnn, 32, nn.Conv2d(16+2, 32, kernel_size=(3, 3), padding=1))
        self.film3 = DoubleFILM(2*hparams.drnn, hparams.drnn, 64, nn.Conv2d(32+2, 64, kernel_size=(3, 3), padding=1))
        self.film4 = DoubleFILM(2*hparams.drnn, hparams.drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
        self.film5 = DoubleFILM(2*hparams.drnn, hparams.drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
        self.c0_trans = nn.Linear(hparams.demb+2, 2*hparams.drnn)

        self.text_rnn = nn.LSTM(hparams.demb, hparams.drnn, bidirectional=True, batch_first=True)
        self.nonwiki_scorer = nn.Linear(2*hparams.drnn, 1)
        self.wiki_attn_scorer = nn.Linear(2*hparams.drnn, 1)

        for t in text_fields:
            setattr(self, t + '_text_scorer', nn.Linear(2*hparams.drnn, 1))

        self.fc = nn.Sequential(nn.Linear(64, hparams.drep), nn.Tanh())
        self.policy = nn.Linear(hparams.drep, num_actions)
        self.baseline = nn.Linear(hparams.drep, 1)

    def make_state_rnn(self):
        if self.hparams.stateful:
            return nn.LSTM(self.drep, self.drep, num_layers=1)
        else:
            return None

    def encode_text(self, inputs, rnn, key, key_len, scorer=None):
        T, B, max_len = inputs[key].size()
        xlens = inputs[key_len].view(-1)
        x = inputs[key].view(-1, max_len).long()
        h = self.run_rnn(rnn, x, xlens)
        if scorer is None:
            return h
        else:
            selfattn, _ = self.run_selfattn(h, xlens, scorer)
            return h, selfattn

    def encode_cell(self, inputs):
        placement = self.emb(inputs['name'].long()).sum(-2)  # sum over placement
        return placement.sum(4)  # sum over words

    def run_film(self, tb, wiki, wiki_lens, nonwiki, pos, wiki_attn):
        c0 = tb.transpose(1, 3)  # (T*B, demb, W, H)
        s0 = self.c0_trans(torch.cat([c0, pos], dim=1).max(3)[0].max(2)[0])
        a0, _ = self.run_attn(wiki, wiki_lens, cond=s0)
        c1, s1 = self.film1(c0, a0, nonwiki, pos, wiki_attn)
        a1, _ = self.run_attn(wiki, wiki_lens, cond=s1)
        c2, s2 = self.film2(c1, a1, nonwiki, pos, wiki_attn)
        a2, _ = self.run_attn(wiki, wiki_lens, cond=s2)
        c3, s3 = self.film3(c2, a2, nonwiki, pos, wiki_attn)
        a3, _ = self.run_attn(wiki, wiki_lens, cond=s3)
        c4, s4 = self.film4(c3, a3, nonwiki, pos, wiki_attn)
        a4, _ = self.run_attn(wiki, wiki_lens, cond=s4)
        c5, s5 = self.film5(c4+c3, a4, nonwiki, pos, wiki_attn)
        last = c5
        conv_out = last.max(3)[0].max(2)[0]  # pool over spatial dimensions
        return conv_out, last

    def fuse(self, inputs, cell):
        T, B, H, W, demb = cell.size()
        tb = torch.flatten(cell, 0, 1)  # (T*B, H, W, 3*demb)
        pos = inputs['rel_pos'].float().view(T*B, H, W, -1).transpose(1, 3)

        wiki_field = self.text_fields[0]
        nonwiki_fields = self.text_fields[1:]

        wiki = self.encode_text(inputs, self.text_rnn, wiki_field, wiki_field+'_len')
        wiki_lens = inputs[wiki_field+'_len'].view(-1).long()

        if nonwiki_fields:
            nonwiki_reps = []
            wiki_attns = []
            for f in nonwiki_fields:
                rnn, weighted_ave = self.encode_text(inputs, self.text_rnn, f, f+'_len', getattr(self, '{}_text_scorer'.format(f)))
                wiki_attn, _ = self.run_attn(wiki, wiki_lens, cond=weighted_ave)
                nonwiki_reps.append(weighted_ave)
                wiki_attns.append(wiki_attn)

            nonwiki_options = torch.stack(nonwiki_reps, dim=0)
            nonwiki_scores = F.softmax(self.nonwiki_scorer(nonwiki_options), dim=0)
            nonwikis = nonwiki_scores.expand_as(nonwiki_options).mul(nonwiki_options).sum(0)

            wiki_attn_options = torch.stack(wiki_attns, dim=0)
            wiki_attn_scores = F.softmax(self.wiki_attn_scorer(wiki_attn_options), dim=0)
            wiki_attn = wiki_attn_scores.expand_as(wiki_attn_options).mul(wiki_attn_options).sum(0)
        else:
            nonwikis, _ = self.run_selfattn(wiki, wiki_lens, self.nonwiki_scorer)
            wiki_attn, _ = self.run_attn(wiki, wiki_lens, cond=nonwikis)

        conv_out, clast = self.run_film(tb, wiki, wiki_lens, nonwikis, pos, wiki_attn)
        flat = conv_out.view(T * B, -1)  # (T*B, -1)
        return self.fc(flat), clast  # (T*B, drep)

    def score_actions(self, inputs, conv, core):
        return self.policy(core)

    def forward(self, inputs, core_state):
        T, B, *_ = inputs['valid'].size()

        # encode everything
        cell = self.encode_cell(inputs)
        rep, conv = self.fuse(inputs, cell)

        if self.hparams.stateful:
            core, core_state = self.compute_state(rep, core_state, inputs['done'])
            rep = rep + core

        policy_logits = self.score_actions(inputs, conv, rep)
        baseline = self.baseline(rep)

        # mask out invalid actions
        action_mask = inputs['valid'].view(T*B, -1)
        policy_logits -= (1-action_mask) * 1e20

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

    def run_rnn(self, rnn, x, lens):
        # embed
        emb = self.emb(x.long())
        # rnn
        packed = rnn_utils.pack_padded_sequence(emb, lengths=lens.cpu().long(), batch_first=True, enforce_sorted=False)
        packed_h, _ = rnn(packed)
        h, _ = rnn_utils.pad_packed_sequence(packed_h, batch_first=True, padding_value=0.)
        return h

    def run_selfattn(self, h, lens, scorer):
        mask = self.get_mask(lens, max_len=h.size(1)).unsqueeze(2)
        raw_scores = scorer(h)
        # raise Exception(f"mask: {mask.shape} raw_scores: {raw_scores.shape} h: {h.shape}")
        scores = F.softmax(raw_scores - (1-mask)*1e20, dim=1)
        context = scores.expand_as(h).mul(h).sum(dim=1)
        return context, scores

    def run_rnn_selfattn(self, rnn, x, lens, scorer):
        rnn = self.run_rnn(rnn, x, lens)
        context, scores = self.run_selfattn(rnn, lens, scorer)
        # attn = [(w, s) for w, s in zip(self.vocab.index2word(seq[0][0].tolist()), scores[0].tolist()) if w != 'pad']
        # print(attn)
        return context

    @classmethod
    def get_mask(cls, lens, max_len=None):
        m = max_len if max_len is not None else lens.max().item()
        mask = torch.tensor([[1]*li + [0]*(m-li) for li in lens.tolist()], device=lens.device, dtype=torch.float)
        return mask

    @classmethod
    def run_attn(cls, seq, lens, cond):
        raw_scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        mask = cls.get_mask(lens, max_len=seq.size(1))
        raw_scores -= (1-mask) * 1e20
        scores = F.softmax(raw_scores, dim=1)

        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, raw_scores


def get_env_shapes(env):
    additional = dict(
        vocab_size=len(env.vocab),
        padding_idx=env.vocab.word2index('pad'),
        text_fields=env.TEXT_FIELDS,
    )
    return env.observation_shapes, env.num_actions, additional


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = MyRTFMModel.get_parser()
    args = parser.parse_args()
    if not args.test:
        MyRTFMModel.run_train(args, create_env=lambda flags: RTFMEasy(), create_eval_env=lambda flags: RTFMEasyDev(), get_env_shapes=get_env_shapes)
    print('Evaluating...')
    res = MyRTFMModel.run_test(args, num_eps=100, create_env=lambda flags: RTFMEasyDev(), get_env_shapes=get_env_shapes, verbose=True)
    pprint.pprint(res)
