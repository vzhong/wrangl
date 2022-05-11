### Supervised learning from scratch

You can find examples of how to use Wrangl for supervised learning in `wrangl.examples.learn.xor_clf`.

Here, we'll build from scratch a sentiment classifier on the [DynaSent dataset](https://github.com/cgpotts/dynasent).
The finished project for this example is in `wrangl.examples.learn.dynasent_clf`.
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

    def process(self, row):
        return dict(sent=row['sentence'], label_text=row['gold_label'], label_idx=['positive', 'neutral', 'negative'].index(row['gold_label']))


def load_data(fname):
    with open(fname) as f:
        for line in f:
            row = json.loads(line)
            if row['gold_label'] not in {None, 'mixed'}:
                yield row


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(cfg.num_workers)])
    train = IterableDataset(load_data(cfg.ftrain), pool=pool, shuffle=True)
    val = IterableDataset(load_data(cfg.feval), pool=pool)
    Model.run_train_test(cfg, train, val)
```

Next, we'll update the config file to use these data files, and specify what pretrained LM we'll use.
We'll also drop the learning rate a bit since we'll be finetuning a pretrained LM.

```yaml
# conf/default.yaml
...
learning_rate: 0.00001
...
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


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.labels = ['positive', 'neutral', 'negative']
        self.mlp = nn.Linear(self.lm.config.hidden_size, len(self.labels))
        self.acc = M.Accuracy()

...

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['label_idx'])

    def extract_context(self, feat, batch):
        return batch['sent']

    def extract_pred(self, out, feat, batch):
        return [self.labels[x] for x in out.max(1)[1].tolist()]

    def extract_gold(self, feat, batch):
        return batch['label_text']

    def forward(self, feat, batch):
        sent = self.tokenizer.batch_encode_plus(feat['sent'], add_special_tokens=True, padding='max_length', truncation=True, max_length=80, return_tensors='pt').to(self.device)
        out = self.lm(**sent).last_hidden_state
        return self.mlp(out[:, 0])  # use the [CLS] token for classification
```

Let's train this and save its output to S3 (you can delete the S3 flags if you don't have that set up).

```bash
python train.py gpus=1 git.enable=true s3.enable=true s3.fcredentials=/data/home/vzhong/wrangl_sync/s3.json
```

Here is the validation accuracy across steps, logged onto S3:

![validation curve](https://github.com/vzhong/wrangl/raw/main/wrangl/examples/learn/dynasent_clf/static/step_vs_val_acc.jpg)


### Reinforcement learning from scratch

Example of RL can be found in `wrangl.examples.learn.atari_rl`.
