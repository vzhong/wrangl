import unittest
from wrangl.examples.preprocess import jsonl_files as T


class TestLoadJsonlFiles(unittest.TestCase):

    def test_fileloader(self):
        data, output = T.load_files()
        self.assertListEqual(data, output)


if __name__ == '__main__':
    unittest.main()
