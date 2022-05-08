import unittest
from wrangl.examples.preprocess import repeat_string as T


class TestRepeatString(unittest.TestCase):
    expect = [s * 10 for s in T.MyDataset.strings]

    def test_no_pool_loader(self):
        out = T.run_no_process()
        self.assertListEqual(self.expect, out)

    def test_ordered(self):
        out = T.run_ordered()
        self.assertListEqual(self.expect, out)

    def test_repeat_unordered(self):
        out = T.run_unordered()
        self.assertSetEqual(set(self.expect), set(out))
