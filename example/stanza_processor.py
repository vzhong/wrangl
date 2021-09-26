import ray
import time
from wrangl.data import Dataloader, Processor
import contextlib
import tqdm
import io
import stanza
import unittest


class MyDataloader(Dataloader):

    def __init__(self, text, pool: ray.util.ActorPool, cache_size: int = 1024):
        super().__init__(pool, cache_size=cache_size)
        self.current = 0
        self.strings = text

    def reset(self):
        self.current = 0

    def load_next(self):
        if self.current < len(self.strings):
            ret = self.strings[self.current]
            self.current += 1
            return ret
        else:
            return None


@ray.remote
class MyProcessor(Processor):

    def __init__(self):
        self.nlp = stanza.Pipeline('en')

    def process(self, raw):
        return self.nlp(raw).text


if __name__ == '__main__':
    ray.init()
    tc = unittest.TestCase()

    zen = io.StringIO()
    with contextlib.redirect_stdout(zen):
        import this
    text = [zen.getvalue()] * 20

    serial_loader = MyDataloader(text, None, cache_size=10)
    out = []
    for batch in tqdm.tqdm(serial_loader.batch(5), desc='identity'):
        out.extend(batch)
    tc.assertListEqual(text, out)

    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    loader = MyDataloader(text, pool, cache_size=10)

    # parallel
    start = time.time()
    parallel_ordered_out = []
    for batch in tqdm.tqdm(loader.batch(5), desc='parallel ordered'):
        parallel_ordered_out.extend(batch)
    parallel_ordered = time.time() - start

    start = time.time()
    parallel_unordered_out = []
    for batch in tqdm.tqdm(loader.batch(5, ordered=False), desc='parallel unordered'):
        parallel_unordered_out.extend(batch)
    parallel_unordered = time.time() - start

    # serial
    nlp = stanza.Pipeline('en')
    start = time.time()
    serial_out = []
    for line in tqdm.tqdm(text, desc='serial'):
        serial_out.append(nlp(line).text)
    serial = time.time() - start

    tc.assertListEqual(
        sorted(parallel_ordered_out),
        sorted(parallel_unordered_out),
        'Parallel outputs are not equal!\nOrdered:\n{}\nUnordered\n{}'.format('\n'.join(sorted(parallel_ordered_out)), '\n'.join(sorted(parallel_unordered_out))),
    )

    tc.assertListEqual(
        parallel_ordered_out,
        serial_out,
        'Parallel output is not equal to serial!\nOrdered:\n{}\nSerial\n{}'.format('\n'.join(parallel_ordered_out), '\n'.join(serial_out)),
    )

    print('parallel ordered: {}s, parallel unordered: {}s, serial: {}s'.format(parallel_ordered, parallel_unordered, serial))
