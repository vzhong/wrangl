import os
import ray
import bz2
import tempfile
import ujson as json
from wrangl.data import FileDataset, Processor


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


def load_files():
    fnames, data = [], []
    for i in range(3):
        fname_i, data_i = create_file(4, start=i*100)
        fnames.append(fname_i)
        data.extend(data_i)

    try:
        pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
        loader = FileDataset(fnames, pool, cache_size=5)
        output = list(loader)
    finally:
        for f in fnames:
            os.remove(f)
    return data, output


if __name__ == '__main__':
    data, output = load_files()
    assert data == output
    print(output)
