import unittest
from wrangl.learn import metrics as M


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        m = M.Accuracy()
        self.assertDictEqual(dict(acc=True), m.compute_one('foo', 'foo'))
        self.assertDictEqual(dict(acc=False), m.compute_one('bar', 'foo'))
        self.assertDictEqual(dict(acc=0.5), m(['foo', 'foo'], ['foo', 'bar']))

    def test_f1(self):
        m = M.SetF1()
        ax = {2, 4, 5, 6}, {1, 2}
        bx = {1, 4}, {1, 2, 4, 5}
        ay = dict(precision=1/4, recall=1/2, f1=2*0.25*0.5/(0.25+0.5))
        by = dict(precision=2/2, recall=2/4, f1=2*1*0.5/(1+0.5))
        self.assertDictEqual(ay, m.compute_one(*ax))
        self.assertDictEqual(by, m.compute_one(*bx))
        self.assertDictEqual({k: (ay[k]+by[k])/2 for k in ay.keys()}, m([ax[0], bx[0]], [ax[1], bx[1]]))
