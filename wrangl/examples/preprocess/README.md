### Preprocess data in parallel:

You can find the following examples of how to use Wrangl for parallel data processing in `wrangl.examples.preprocess`:

- text strings: `wrangl.examples.preprocess.repeat_string`
- multiple BZip2 compressed JSONL files: `wrangl.examples.preprocess.jsonl_files`
- query output from SQLite database: `wrangl.examples.preprocess.sql_db`
- parsing using StanfordNLP Stanza: `wrangl.examples.preprocess.using_stanza`

For example, here is how to preprocess some lines of text (the Python Zen poem) in parallel using Stanford Stanza.

```python
import io, ray, tqdm, stanza, contextlib
from wrangl.data import IterableDataset, Processor

@ray.remote
class MyProcessor(Processor):

    def __init__(self):
        self.nlp = stanza.Pipeline('en')

    def process(self, raw):
        return self.nlp(raw).text

if __name__ == '__main__':
    # we will use Python's Zen poem as an example
    zen = io.StringIO()
    with contextlib.redirect_stdout(zen):
        import this
    text = [zen.getvalue()] * 20
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])

    # parallel
    loader = IterableDataset(text, pool, cache_size=10, shuffle=False)
    processed = list(tqdm.tqdm(loader, desc='parallel ordered', total=len(text)))
```
