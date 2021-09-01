import os
import py_cui
import argparse
import ujson as json
from pathlib import Path
from collections import Counter

from .dataloader import Fileloader


class Annotator:

    def __init__(self, get_data, write_annotation, stats=None, height=10, width=10, top_k=10):
        control = py_cui.PyCUI(height, width)
        control.set_title('Annotator')
        self.control = control

        self.data_generator = get_data()
        self.write_annotation = write_annotation
        self.stats = stats or Counter()
        self.top_k = top_k
        self.current_identifier = self.current_example = None

        self.input_cell = control.add_scroll_menu('Input', 0, 0, row_span=height//10*7, column_span=width//10*7)
        self.annotation_cell = control.add_text_box('Annotation', height//10*7, 0, row_span=height//10*3, column_span=width//10*7)
        self.stats_cell = control.add_scroll_menu('Stats', 0, width//10*7, row_span=10, column_span=width//10*3)

        self.annotation_cell.add_key_command(py_cui.keys.KEY_ENTER, self.submit_annotation)

        self.get_next_example()
        self.update_stats()
        control.set_selected_widget(self.annotation_cell.get_id())
        control.move_focus(self.annotation_cell)

    def get_next_example(self):
        self.input_cell.clear()
        try:
            self.current_identifier, self.current_example = next(self.data_generator)
            self.input_cell.add_item(self.current_example)
        except StopIteration:
            self.control.show_message_popup('Annotation finished!', 'You are done!')

    def submit_annotation(self):
        content = self.annotation_cell.get()
        self.write_annotation(self.current_identifier, content)
        self.get_next_example()
        self.annotation_cell.clear()
        self.stats[content] += 1
        self.update_stats()

    def update_stats(self):
        self.stats_cell.clear()
        total = sum(self.stats.values())
        self.stats_cell.add_item('Statistics (top {})'.format(self.top_k))
        for m, c in self.stats.most_common(self.top_k):
            self.stats_cell.add_item('{} ({} -> {}%)'.format(m, c, round(c/total * 100, 2)))

    def start(self):
        self.control.start()


def annotate():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', nargs='+', help='files to annotate')
    parser.add_argument('--dout', default='annotation', help='output directory')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    dout = Path(args.dout)
    fannotated = dout.joinpath('annotated.jsonl')

    if args.overwrite:
        if fannotated.exists():
            os.remove(f)

    loader = Fileloader(args.files, pool=None)
    iterator = enumerate(loader.batch(1))

    stats = Counter()
    if fannotated.exists():
        with fannotated.open('rt') as f:
            stats = Counter([json.loads(l)['annotation'] for l in f])

    def get_data():
        seen = sum(stats.values())
        for i, batch in iterator:
            if i < seen:
                continue
            yield (i, batch[0])

    fann = fannotated.open('at')

    def write_annotation(identifier, result):
        fann.write(json.dumps(dict(id=identifier, annotation=result)) + '\n')
        fann.flush()

    annotator = Annotator(get_data, write_annotation, stats=stats)
    annotator.start()
    fann.close()
