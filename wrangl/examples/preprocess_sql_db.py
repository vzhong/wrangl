import os
import ray
import sqlite3
import tempfile
from wrangl.data import SQLDataset, Processor


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return dict(id=raw[0], value=raw[1])


def create_db(num_lines):
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    f.close()
    data = [dict(value='a' * i, id=i) for i in range(num_lines)]
    db = sqlite3.connect(f.name, isolation_level=None)
    cursor = db.cursor()
    cursor.execute('CREATE TABLE data(id INTEGER PRIMARY KEY, value TEXT)')

    for row in data:
        cursor.execute('INSERT INTO data VALUES (?, ?)', (row['id'], row['value']))
    db.commit()
    cursor.close()
    db.close()
    return f.name, data


def load_db():
    fname, data = create_db(5)
    query = 'SELECT id, value FROM data ORDER BY id'

    try:
        pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
        loader = SQLDataset(fname, query, pool, cache_size=5)
        output = list(loader)
    finally:
        os.remove(fname)
    return data, output


if __name__ == '__main__':
    data, output = load_db()
    assert data == output
    print(output)
