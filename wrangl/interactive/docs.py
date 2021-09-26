"""
Autodocuments this library.
"""
import os
import shutil
from pdoc import pdoc, render
from pathlib import Path
from datetime import date


DOCDIR = Path(__file__).parent.parent.parent.joinpath('docs')


def generate_docs():
    host = os.environ.get('WRANGL_DOCS_HOST', None)
    render.configure(
        docformat='google',
        edit_url_map=dict(wrangl=host) if host else None,
        footer_text='Copyright @{} by Victor Zhong'.format(date.today().year),
        logo=None,
        logo_link=None,
        math=True,
        search=True,
        show_source=True,
        template_directory=DOCDIR.joinpath('templates'),
    )
    pdoc('wrangl', output_directory=DOCDIR.joinpath('build'), format='html')
    os.remove(DOCDIR.joinpath('build', 'index.html'))
    # use the package html as the index
    shutil.copy(DOCDIR.joinpath('build', 'wrangl.html'), DOCDIR.joinpath('build', 'index.html'))
