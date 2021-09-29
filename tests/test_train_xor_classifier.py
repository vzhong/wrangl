import os
import json
import shutil
import pathlib
import unittest
import tempfile
from wrangl.examples import train_xor_classifier


class TestTrainXorClassifier(unittest.TestCase):

    def setUp(self):
        self.mydir = tempfile.TemporaryDirectory()
        self.dout = pathlib.Path(self.mydir.name)

        self.parser = parser = train_xor_classifier.MyModel.get_parser(num_train_steps=1500, eval_period=200, order='ordered', dout=self.mydir.name)
        parser.add_argument('--num_procs', default=3, type=int, help='number of processors.')
        parser.add_argument('--cache_size', default=10, type=int, help='preprocessing cache size.')
        parser.add_argument('--test', help='test checkpoint.')

    def tearDown(self):
        if os.path.isdir(self.dout):
            shutil.rmtree(self.dout)

    def load_metrics_log(self):
        with open(self.dout.joinpath('metrics.log.json')) as f:
            log = json.load(f)
        return log

    def test_train_test(self):
        args = self.parser.parse_args([])

        # train from scratch
        train_xor_classifier.main(args)

        self.assertTrue(self.dout.joinpath('ckpt.tar').exists())
        self.assertTrue(self.dout.joinpath('hparams.json').exists())
        self.assertTrue(self.dout.joinpath('metrics.best.json').exists())
        self.assertTrue(self.dout.joinpath('metrics.log.json').exists())
        self.assertTrue(self.dout.joinpath('train.log').exists())

        log = self.load_metrics_log()
        self.assertDictEqual(dict(
            train=dict(loss=0.4270490446686745, accuracy=1.0),
            eval=dict(loss=0.40561878346815344, accuracy=1.0),
            best=1.0,
            train_steps=1500,
        ), log[-1])

        # resume training
        args.num_train_steps = 2000
        args.resume = 'auto'
        train_xor_classifier.main(args)

        expect = dict(
            train=dict(loss=0.259609357342124, accuracy=1.0),
            eval=dict(loss=0.23055222477854753, accuracy=1.0),
            best=1.0,
            train_steps=2000,
        )
        log = self.load_metrics_log()
        self.assertDictEqual(expect, log[-1])

        # try training again
        train_xor_classifier.main(args)
        log = self.load_metrics_log()
        self.assertDictEqual(expect, log[-1])

        # do inference
        args.test = str(self.dout.joinpath('ckpt.tar'))
        eval_metrics = train_xor_classifier.main(args)
        for k, v in expect['eval'].items():
            self.assertAlmostEqual(v, eval_metrics[k])
