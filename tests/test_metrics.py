import unittest
from wrangl import metrics as M


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        m = M.Accuracy()
        self.assertTrue(m.single_forward('foo', 'foo'))
        self.assertFalse(m.single_forward('foo', 'bar'))
        self.assertDictEqual(dict(accuracy=0.5), m.forward([('foo', 'foo'), ('foo', 'bar')]))

    def test_precision(self):
        m = M.Precision()
        self.assertEqual(0.25, m.single_forward({1, 2}, {2, 4, 5, 6}))
        self.assertEqual(1, m.single_forward({1, 2, 4, 5}, {1, 4}))
        self.assertDictEqual(dict(precision=1.25/2), m.forward([({1, 2}, {2, 4, 5, 6}), ({1, 2, 4, 5}, {1, 4})]))

    def test_recall(self):
        m = M.Recall()
        self.assertEqual(0.5, m.single_forward({1, 2}, {2, 4, 5, 6}))
        self.assertEqual(0.5, m.single_forward({1, 2, 4, 5}, {1, 4}))
        self.assertDictEqual(dict(recall=0.5), m.forward([({1, 2}, {2, 4, 5, 6}), ({1, 2, 4, 5}, {1, 4})]))

    def test_f1(self):
        m = M.F1Score()
        a = 2*0.5*0.25/0.75
        b = 2*0.5/1.5
        self.assertEqual(a, m.single_forward({1, 2}, {2, 4, 5, 6}))
        self.assertEqual(b, m.single_forward({1, 2, 4, 5}, {1, 4}))
        self.assertDictEqual(dict(f1score=(a+b)/2), m.forward([({1, 2}, {2, 4, 5, 6}), ({1, 2, 4, 5}, {1, 4})]))
