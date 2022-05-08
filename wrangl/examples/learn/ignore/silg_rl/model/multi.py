# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from numpy import sqrt as sqrt
from .modules import DoubleFILM


class Model(nn.Module):

    @property
    def device(self):
        return self.emb.weight.device

    def __init__(self, flags, env):
        super().__init__()
        self.fields = env.TEXT_FIELDS

        self.flags = flags
        self.observation_shape = env.observation_space
        self.num_actions = env.max_actions

        self.stateful = flags.model.stateful
        self.use_local_conv = flags.model.use_local_conv
        self.dual_wiki_encode = True

        self.drep = flags.model.drep
        self.drnn = flags.model.drnn

        if flags.model.field_attn:
            self.field_attn = True
            self.demb = 2*self.drnn
            self.attn_scale = sqrt(self.demb)
            self.key_trans = nn.Linear(self.demb, self.demb)
            self.val_trans = nn.Linear(self.demb, self.demb)
        else:
            self.field_attn = False
            self.demb = flags.model.demb

        self.dconv_out = 1
        room_height, room_width = env.height, env.width
        self.room_height_conv_out = room_height // 2
        self.room_width_conv_out = room_width // 2

        self.vocab = vocab = env.vocab
        self.emb = nn.Embedding(len(vocab), self.demb, padding_idx=env.tokenizer.pad_token_id)
        if hasattr(env, 'grid_vocab'):
            self.name_emb = nn.Embedding(env.grid_vocab, self.demb)
        else:
            self.name_emb = self.emb

        self.policy = nn.Linear(self.drep, self.num_actions)
        self.baseline = nn.Linear(self.drep, 1)

        drnn = self.drnn

        self.make_film_layers()
        if self.use_local_conv:
            self.make_film_layers(suffix='_local')

        if self.stateful:
            self.core = nn.LSTM(self.drep, self.drep, num_layers=1)

        self.text_rnn = nn.LSTM(self.demb, drnn, bidirectional=True, batch_first=True)

        self.nonwiki_scorer = nn.Linear(drnn*2, 1)
        self.wiki_attn_scorer = nn.Linear(drnn*2, 1)

        for f in self.fields:
            setattr(self, '{}_text_scorer'.format(f), nn.Linear(drnn*2, 1))

        self.fc = nn.Sequential(
            nn.Linear(64, self.drep),
            nn.Tanh(),
        )

    def make_film_layers(self, suffix=''):
        layers = {}

        layers['c0_trans'] = nn.Linear(self.demb+2, 2*self.drnn)

        layers['film1'] = DoubleFILM(2*self.drnn, self.drnn, 16, nn.Conv2d(self.demb+2, 16, kernel_size=(3, 3), padding=1))
        layers['film2'] = DoubleFILM(2*self.drnn, self.drnn, 32, nn.Conv2d(16+2, 32, kernel_size=(3, 3), padding=1))
        layers['film3'] = DoubleFILM(2*self.drnn, self.drnn, 64, nn.Conv2d(32+2, 64, kernel_size=(3, 3), padding=1))

        assert self.flags.model.num_film in {3, 5}
        if self.flags.model.num_film == 5:
            layers['film4'] = DoubleFILM(2*self.drnn, self.drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
            layers['film5'] = DoubleFILM(2*self.drnn, self.drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))

        for k, v in layers.items():
            setattr(self, k + suffix, v)

    def initial_state(self, batch_size=1):
        if not self.stateful:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size).to(self.device)
            for _ in range(2)
        )

    def encode_text(self, inputs, rnn, key, key_len, scorer=None):
        T, B, max_len = inputs[key].size()
        # raise Exception(f"key: {key} inputs size: {inputs[key + '_emb'].size()}")
        xlens = inputs[key_len].view(-1)

        x = inputs[key].view(-1, max_len).long()
        h = self.run_rnn(rnn, x, xlens)

        if scorer is None:
            return h
        else:
            selfattn, _ = self.run_selfattn(h, xlens, scorer)
            return h, selfattn

    def encode_cell(self, inputs):
        placement = self.name_emb(inputs['name'].long()).sum(-2)
        return placement.sum(4)

    def _batch_field_attention(self, query, key, value):
        # batch attention from EMMA
        # query: (bs, k, h, w, c), key, value: (bs, num_sent, emb_dim)
        # return: (bs, k, h, w, c)
        bs = query.shape[0]
        kq = torch.bmm(
            query.view(bs, -1, self.demb),
            key.permute(0, 2, 1)
        )
        kq = kq / self.attn_scale
        weights = F.softmax(kq, dim=-1)
        weights = weights * (kq != 0)
        weights = weights.view(*query.shape[:-1], -1)

        return torch.mean(
            weights.unsqueeze(-1) * value.view(bs, 1, 1, 1, *value.shape[1:]), dim=-2)

    def field_attention(self, fields, cell):
        # fields: (num_nonwiki, T*B, demb), cell: (T*B, H, W, demb)
        # return: (T*B, H, W, demb)
        bs, H, W, demb = cell.shape
        assert demb == self.demb
        fields = fields.permute(1, 0, 2)  # (bs, num_nonwiki, demb)
        key = self.key_trans(fields)  # (bs, num_nonwiki, demb)
        value = self.val_trans(fields)  # (bs, num_nonwiki, demb)
        query = cell.unsqueeze(1)  # (bs, 1, h, w, demb)

        cell = self._batch_field_attention(query, key, value)
        cell = cell.view(bs, H, W, demb)
        return cell

    def run_film(self, tb, wiki, wiki_lens, nonwiki, pos, wiki_attn, suffix=''):
        c0_trans = getattr(self, 'c0_trans' + suffix)
        film1 = getattr(self, 'film1' + suffix)
        film2 = getattr(self, 'film2' + suffix)
        film3 = getattr(self, 'film3' + suffix)

        c0 = tb.transpose(1, 3)  # (T*B, demb, W, H)
        s0 = c0_trans(torch.cat([c0, pos], dim=1).max(3)[0].max(2)[0])
        a0, _ = self.run_attn(wiki, wiki_lens, cond=s0)
        c1, s1 = film1(c0, a0, nonwiki, pos, wiki_attn)
        a1, _ = self.run_attn(wiki, wiki_lens, cond=s1)
        c2, s2 = film2(c1, a1, nonwiki, pos, wiki_attn)
        a2, _ = self.run_attn(wiki, wiki_lens, cond=s2)
        c3, s3 = film3(c2, a2, nonwiki, pos, wiki_attn)
        last = c3

        if self.flags.model.num_film == 5:
            film4 = getattr(self, 'film4' + suffix)
            film5 = getattr(self, 'film5' + suffix)
            a3, _ = self.run_attn(wiki, wiki_lens, cond=s3)
            c4, s4 = film4(c3, a3, nonwiki, pos, wiki_attn)
            a4, _ = self.run_attn(wiki, wiki_lens, cond=s4)
            c5, s5 = film5(c4+c3, a4, nonwiki, pos, wiki_attn)
            last = c5

        conv_out = last.max(3)[0].max(2)[0]  # pool over spatial dimensions
        return conv_out, last

    def fuse(self, inputs, cell):
        T, B, H, W, demb = cell.size()
        tb = torch.flatten(cell, 0, 1)  # (T*B, H, W, 3*demb)
        pos = inputs['rel_pos'].float().view(T*B, H, W, -1).transpose(1, 3)

        if self.use_local_conv:
            local_cell = self.extract_neighbourhood(inputs, cell)
            _, _, local_H, local_W, _ = local_cell.size()
            local_tb = torch.flatten(local_cell, 0, 1)  # (T*B, H, W, 3*demb)

        wiki_field = self.fields[0]
        nonwiki_fields = self.fields[1:]

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

            if self.field_attn:
                # override the grid (tb)
                tb = self.field_attention(nonwiki_options, tb)
                if self.use_local_conv:
                    local_tb = self.field_attention(nonwiki_options, local_tb)

        else:
            nonwikis, _ = self.run_selfattn(wiki, wiki_lens, self.nonwiki_scorer)
            wiki_attn, _ = self.run_attn(wiki, wiki_lens, cond=nonwikis)

        conv_out, clast = self.run_film(tb, wiki, wiki_lens, nonwikis, pos, wiki_attn)

        if self.use_local_conv:
            local_pos = self.extract_neighbourhood(inputs, inputs['rel_pos'])
            local_pos = local_pos.float().view(T*B, local_H, local_W, -1).transpose(1, 3)
            local_conv_out, local_clast = self.run_film(local_tb, wiki, wiki_lens, nonwikis, local_pos, wiki_attn, suffix='_local')
            conv_out = conv_out + local_conv_out

        flat = conv_out.view(T * B, -1)  # (T*B, -1)
        return self.fc(flat), clast  # (T*B, drep)

    def score_actions(self, inputs, conv, core):
        return self.policy(core)

    def extract_neighbourhood(self, inputs, grid_enc, size=3):
        assert size % 2 == 1, 'Crop size must be odd, got {} instead'.format(size)
        T, B, H, W, demb = grid_enc.size()
        grid_enc = grid_enc.view(T*B, H, W, demb)

        # pad grid
        pad_size = size // 2
        Hpad = torch.zeros(T*B, pad_size, W, demb).to(self.device)
        padded = torch.cat([Hpad, grid_enc, Hpad], dim=1)  # (T*B, H+2pad, W, demb)
        Wpad = torch.zeros(T*B, H+2*pad_size, pad_size, demb).to(self.device)
        padded = torch.cat([Wpad, padded, Wpad], dim=2)  # (T*B, H+2pad, W+2pad, demb)

        # pull out crops
        Hpos = inputs['pos'][:, :, 0].view(-1)  # (T*B)
        Wpos = inputs['pos'][:, :, 1].view(-1)  # (T*B)
        crops = []
        for hi, wi, im in zip(Hpos.tolist(), Wpos.tolist(), padded):
            crop = im[hi:hi + 2*pad_size + 1, wi:wi + 2*pad_size + 1]
            crops.append(crop)
        crops = torch.stack(crops, dim=0)
        TB, HH, WW, demb = crops.size()
        return crops.view(T, B, HH, WW, demb)

    def forward(self, inputs, core_state):
        T, B, *_ = inputs['valid'].size()

        # encode everything
        cell = self.encode_cell(inputs)
        rep, conv = self.fuse(inputs, cell)

        if self.stateful:
            core, core_state = self.compute_core(inputs, rep, core_state)
            rep = rep + core

        policy_logits = self.score_actions(inputs, conv, rep)
        baseline = self.baseline(rep)

        # mask out invalid actions
        action_mask = inputs['valid'].float().view(T*B, -1)
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

    def compute_core(self, inputs, rep, core_state):
        T, B, *_ = inputs['valid'].size()
        assert self.stateful
        core_input = rep.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for inp, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # states:
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(inp.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        return core_output, core_state

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
        mask = torch.tensor([[1]*l + [0]*(m-l) for l in lens.tolist()], device=lens.device, dtype=torch.float)
        return mask

    @classmethod
    def run_attn(cls, seq, lens, cond):
        raw_scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        mask = cls.get_mask(lens, max_len=seq.size(1))
        raw_scores -= (1-mask) * 1e20
        scores = F.softmax(raw_scores, dim=1)

        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, raw_scores
