"""
Autodocuments this library.
"""
import os
import shutil
from pathlib import Path


EXAMPLEROOT = Path(__file__).parent.parent.joinpath('examples', 'learn')


def add_parser_arguments(parser):
    parser.add_argument('--source', default='xor_clf', help='source example name')
    parser.add_argument('--name', default='myproj', help='project name')


def main(args):
    dst = Path(os.getcwd()).joinpath(args.name)
    print('making directory {}'.format(dst))
    shutil.copytree(src=EXAMPLEROOT.joinpath(args.source), dst=dst)
    if dst.joinpath('__init__.py').is_file():
        os.remove(dst.joinpath('__init__.py'))
    if dst.joinpath('saves').is_dir():
        shutil.rmtree(dst.joinpath('saves'))
