from wrangl.learn import SupervisedModel, metrics as M
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.labels = ['positive', 'neutral', 'negative', 'mixed']
        self.mlp = nn.Linear(self.lm.hidden_size, 4)  # there are 4 classes

    def featurize(self, batch: list):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        return batch

    def compute_metrics(self, pred: list, gold: list) -> dict:
        return self.acc(pred, gold)

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['label'])

    def extract_context(self, feat, batch):
        return batch['sent']

    def extract_pred(self, out, feat, batch):
        return [self.labels[x for x in out.max(1)[1].tolist()]

    def extract_gold(self, feat, batch):
        return batch['label_text']

    def forward(self, feat, batch):
        sent = self.tokenizer.batch_encode_plus(feat['sent'], add_special_tokens=True, padding='max_length', truncation=True, max_length=80, return_tensors='pt')
        out = self.lm(**sent).last_hidden_state
        return self.mlp(out[:, 0])  # use the [CLS] token for classification
