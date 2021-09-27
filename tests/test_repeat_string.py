import unittest
from wrangl.examples import preprocess_repeat_string as T


class TestRepeatString(unittest.TestCase):

    def test_no_pool_loader(self):
        out = T.run_no_process()
        self.assertListEqual(T.strings, out)

    def test_ordered(self):
        out = T.run_ordered()
        expect = [s * 10 for s in T.strings]
        self.assertListEqual(expect, out)

    def test_repeat_unordered(self):
        out = T.run_unordered()
        expect = [s * 10 for s in T.strings]
        self.assertSetEqual(set(expect), set(out))
