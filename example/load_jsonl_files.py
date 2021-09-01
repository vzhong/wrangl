import os
import ray
import bz2
import tempfile
import unittest
import ujson as json
from wrangl.dataloader import Fileloader, Processor


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return json.loads(raw)


def create_file(num_lines, start=0):
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl.bz2')
    f.close()
    data = [dict(value=start + i) for i in range(num_lines)]
    with bz2.open(f.name, 'wt') as fzip:
        for row in data:
            fzip.write(json.dumps(row) + '\n')
    return f.name, data


if __name__ == '__main__':
    ray.init()
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    fnames, data = [], []
    for i in range(3):
        fname_i, data_i = create_file(4, start=i*100)
        fnames.append(fname_i)
        data.extend(data_i)

    try:
        loader = Fileloader(fnames, pool, cache_size=5)
        output = []
        for batch in loader.batch(2):
            output.extend(batch)
    finally:
        for f in fnames:
            os.remove(f)
    unittest.TestCase().assertListEqual(data, output)
