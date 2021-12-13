from wrangl.learn import SupervisedModel
from torch import nn
from torch.nn import functional as F


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(2, cfg.dhid),
            nn.ReLU(),
            nn.Linear(cfg.dhid, 2),
        )

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['label'])

    def extract_context(self, feat, batch):
        return feat['feat'].tolist()

    def extract_pred(self, out, feat, batch):
        return out.max(1)[1].tolist()

    def extract_gold(self, feat, batch):
        return feat['label'].tolist()

    def forward(self, feat, batch):
        return self.mlp(feat['feat'])
