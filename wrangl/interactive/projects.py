"""
Autodocuments this library.
"""
import os
import shutil
from pathlib import Path


EXAMPLEDIR = Path(__file__).parent.parent.joinpath('examples', 'learn', 'xor_clf')


def add_parser_arguments(parser):
    parser.add_argument('--name', default='myproj', help='project name')


def main(args):
    dst = os.path.join(os.getcwd(), args.name)
    print('making directory {}'.format(dst))
    shutil.copytree(src=EXAMPLEDIR, dst=dst)
