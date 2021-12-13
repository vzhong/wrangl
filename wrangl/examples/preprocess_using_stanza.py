import ray
import time
from wrangl.data import IterableDataset, Processor
import contextlib
import tqdm
import io
import stanza
import unittest


@ray.remote
class MyProcessor(Processor):

    def __init__(self):
        self.nlp = stanza.Pipeline('en')

    def process(self, raw):
        return self.nlp(raw).text


if __name__ == '__main__':
    tc = unittest.TestCase()

    zen = io.StringIO()
    with contextlib.redirect_stdout(zen):
        import this
    text = [zen.getvalue()] * 20

    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])

    # parallel
    loader = IterableDataset(text, pool, cache_size=10, shuffle=False)
    start = time.time()
    parallel_ordered_out = list(tqdm.tqdm(loader, desc='parallel ordered', total=len(text)))
    parallel_ordered = time.time() - start

    loader = IterableDataset(text, pool, cache_size=10, shuffle=True)
    start = time.time()
    parallel_unordered_out = list(tqdm.tqdm(loader, desc='parallel unordered', total=len(text)))
    parallel_unordered = time.time() - start

    # serial
    nlp = stanza.Pipeline('en')
    start = time.time()
    serial_out = [nlp(line).text for line in tqdm.tqdm(text, desc='serial', total=len(text))]
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
