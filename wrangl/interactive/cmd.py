from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from . import plotter, docs, projects, sync
import unittest


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    subs = parser.add_subparsers(title='command', dest='cmd')

    plotter.add_parser_arguments(subs.add_parser('plot'))
    docs.add_parser_arguments(subs.add_parser('autodoc'))
    projects.add_parser_arguments(subs.add_parser('project'))
    sync.add_parser_arguments(subs.add_parser('sync'))

    autotest = subs.add_parser('autotest')
    autotest.add_argument('--dtest', default='tests', help='test directory')

    args = parser.parse_args()

    if args.cmd == 'plot':
        plotter.main(args)
    elif args.cmd == 'project':
        projects.main(args)
    elif args.cmd == 'autodoc':
        docs.main(args)
    elif args.cmd == 'autotest':
        tests = unittest.TestLoader().discover(args.dtest)
        unittest.runner.TextTestRunner().run(tests)
    elif args.cmd == 'sync':
        sync.main(args)


if __name__ == '__main__':
    main()
