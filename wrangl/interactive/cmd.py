from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from . import annotator
from . import plotter


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    subs = parser.add_subparsers(title='command', dest='cmd')
    annotator.add_parser_arguments(subs.add_parser('annotate'))
    plotter.add_parser_arguments(subs.add_parser('plot'))
    args = parser.parse_args()
    print(args)

    if args.cmd == 'annotate':
        annotator.main(args)
    elif args.cmd == 'plot':
        plotter.main(args)
