"""
Autodocuments this library.
"""
import os
from pdoc import pdoc, render
from pathlib import Path
from datetime import date


DOCDIR = Path(__file__).parent.parent.parent.joinpath('docs')


def generate_docs():
    render.configure(
        docformat='google',
        edit_url_map=os.environ.get('WRANGL_DOCS_HOST', None),
        footer_text='Copyright @{} by Victor Zhong'.format(date.today().year),
        logo=None,
        logo_link=None,
        math=True,
        search=True,
        show_source=True,
        template_directory=DOCDIR.joinpath('templates'),
    )
    pdoc('wrangl', output_directory=DOCDIR, format='html')
