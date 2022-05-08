import unittest
from wrangl.examples.preprocess import sql_db as T


class TestLoadSQLDB(unittest.TestCase):

    def test_dbloader(self):
        data, output = T.load_db()
        self.assertListEqual(data, output)


if __name__ == '__main__':
    unittest.main()
