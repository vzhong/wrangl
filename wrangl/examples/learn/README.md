# Learning examples

This directory contains learning experiment examples using Wrangl.
They are:

- Supervised learning experiments
  - learning a XOR : `wrangl.examples.learn.xor_clf`
- Reinforcement learning experiments
  - TBD

Moreover, `wrangl.examples.learn.dynasent_clf` illustrates an example of how to use `wrangl project` to quickly create a project from scratch.


## Supervised learning from scratch

We'll build a question answering model on the [DynaSent dataset](https://github.com/cgpotts/dynasent).
First, let's bootstrap our model using the `xor_clf` example and download the dataset.

```bash
wrangl project --source xor_clf --name dynasent_clf
cd dynasent_clf
wget https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip
unzip dynasent-v1.1.zip
rm -rf __MACOSX dynasent-v1.1.zip
mv dynasent-v1.1 data
```

Now we'll modify `train.py` to use these data files.

```python
# train.py
# ...
@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        row = json.loads(raw)
        return dict(sent=row['sentence'], label_text=row['gold_label'], label_idx=['positive', 'neutral', 'negative', 'mixed'].index(row['gold_label']))

# ...
@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(cfg.num_workers)])
    train = FileDataset([cfg.ftrain], pool=pool, shuffle=True)
    val = FileDataset([cfg.feval], pool=pool)
    Model.run_train_test(cfg, train, val)
```

Next, we'll update the config file to use these data files, and specify what pretrained LM we'll use.
```yaml
# conf/default.yaml
ftrain: '${oc.env:PWD}/data/dynasent-v1.1-round01-yelp-train.jsonl'
feval: '${oc.env:PWD}/data/dynasent-v1.1-round01-yelp-dev.jsonl'
num_workers: 4
lm: bert-base-uncased
```

Next, we'll overload the existing model as follows:
- tell it out how extract out the context, predictions, and ground truth labels for this dataset.
- instantiate a BERT encoder and use it during the forward pass.

```python
# model/mymodel.py
...
from transformers import AutoModel, AutoTokenizer

...
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.labels = ['positive', 'neutral', 'negative', 'mixed']
        self.mlp = nn.Linear(self.lm.hidden_size, 4)  # there are 4 classes

...
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
```
