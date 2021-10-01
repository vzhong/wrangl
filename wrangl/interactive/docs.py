"""
Autodocuments this library.
"""
import os
import shutil
from pdoc import pdoc, render
from pathlib import Path
from datetime import date


DOCDIR = Path(__file__).parent.parent.parent.joinpath('docs')


def add_parser_arguments(parser):
    parser.add_argument('--host', default=os.environ.get('WRANGL_DOCS_HOST'), help='where the docs will be hosted')
    parser.add_argument('--format', default='google', help='doc format')
    parser.add_argument('--logo', help='logo image')
    parser.add_argument('--logo_link', help='logo link')


def main(args):
    render.configure(
        docformat=args.format,
        edit_url_map=dict(wrangl=args.host) if args.host else None,
        footer_text='Copyright @{} by Victor Zhong'.format(date.today().year),
        logo=args.logo,
        logo_link=args.logo_link,
        math=True,
        search=True,
        show_source=True,
        template_directory=DOCDIR.joinpath('templates'),
    )
    pdoc('wrangl', output_directory=DOCDIR.joinpath('build'), format='html')
    os.remove(DOCDIR.joinpath('build', 'index.html'))
    # use the package html as the index
    shutil.copy(DOCDIR.joinpath('build', 'wrangl.html'), DOCDIR.joinpath('build', 'index.html'))
