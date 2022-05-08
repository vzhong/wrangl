import torch
from torch import nn


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
