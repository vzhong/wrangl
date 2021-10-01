from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from . import annotator, plotter, docs
import unittest


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    subs = parser.add_subparsers(title='command', dest='cmd')

    annotator.add_parser_arguments(subs.add_parser('annotate'))
    plotter.add_parser_arguments(subs.add_parser('plot'))
    docs.add_parser_arguments(subs.add_parser('autodoc'))

    autotest = subs.add_parser('autotest')
    autotest.add_argument('--dtest', default='tests', help='test directory')

    args = parser.parse_args()

    if args.cmd == 'annotate':
        annotator.main(args)
    elif args.cmd == 'plot':
        plotter.main(args)
    elif args.cmd == 'autodoc':
        docs.main(args)
    elif args.cmd == 'autotest':
        tests = unittest.TestLoader().discover(args.dtest)
        unittest.runner.TextTestRunner().run(tests)
