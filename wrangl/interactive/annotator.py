"""
Annotator class for annotating individual examples.
You should run this with `wannotate -h`.
"""
import os
import py_cui
import argparse
import ujson as json
from pathlib import Path
from typing import Callable, List
from collections import defaultdict

from ..data import Fileloader


class Annotator:
    """
    Annotation interface for individual examples.
    """

    def __init__(self, get_data: Callable, write_annotation: Callable, grouped: defaultdict[List[str]] = None, height: int = 10, width : int = 10, top_k: int = 10, max_char: int = 40, name: str = 'Annotator'):
        """
        Args:
            get_data: generator function with the signature `identifier, example = f()` to retrieve new examples.
            write_annotation: function with the signature `f(identifier, example, annotation)` to write annotation.
            grouped: mapping of annotation to examples.
            height: height of annotation interface.
            width: width of annotation interface.
            top_k: number of most frequent label classes to visualize.
            max_char: max number of characters per line to show before wrapping.
        """
        control = py_cui.PyCUI(height, width)
        control.set_title(name)
        self.control = control

        self.data_generator = get_data()
        self.write_annotation = write_annotation
        self.grouped = grouped or defaultdict(list)
        self.top_k = top_k
        self.max_char = max_char
        self.current_identifier = self.current_example = None
        self.selected_examples = []

        self.main_cell = control.add_scroll_menu('Input', 0, 0, row_span=height//10*7, column_span=width//10*7)
        self.annotation_cell = control.add_text_box('Annotation', height//10*7, 0, row_span=height//10*3, column_span=width//10*7)
        self.stats_cell = control.add_scroll_menu('Stats (top {})'.format(top_k), 0, width//10*7, row_span=10, column_span=width//10*3)

        # focus colours
        for w in [self.annotation_cell, self.stats_cell]:
            w.set_focus_border_color(py_cui.colors.RED_ON_BLACK)
            w.set_selected_color(py_cui.colors.RED_ON_BLACK)
        self.main_cell.set_focus_border_color(py_cui.colors.RED_ON_BLACK)

        # submit annotation
        self.annotation_cell.add_key_command(py_cui.keys.KEY_ENTER, self.submit_annotation)

        # view detailed statistics
        self.stats_cell.add_key_command(py_cui.keys.KEY_ENTER, self.update_detailed_stats)
        self.stats_cell.add_key_command(py_cui.keys.KEY_Q_LOWER, self.update_example)

        # view detailed example
        self.main_cell.add_key_command(py_cui.keys.KEY_ENTER, self.update_detailed_example)
        self.main_cell.add_key_command(py_cui.keys.KEY_Q_LOWER, self.update_example)

        # vim bindings
        for w in [self.main_cell, control, self.stats_cell]:
            w.add_key_command(py_cui.keys.KEY_H_LOWER, lambda: control._handle_key_presses(py_cui.keys.KEY_LEFT_ARROW))
            w.add_key_command(py_cui.keys.KEY_L_LOWER, lambda: control._handle_key_presses(py_cui.keys.KEY_RIGHT_ARROW))
            w.add_key_command(py_cui.keys.KEY_K_LOWER, lambda: control._handle_key_presses(py_cui.keys.KEY_UP_ARROW))
            w.add_key_command(py_cui.keys.KEY_J_LOWER, lambda: control._handle_key_presses(py_cui.keys.KEY_DOWN_ARROW))

        # default display
        self.get_next_example()
        self.update_example()
        self.update_stats()

        # default focus
        control.set_selected_widget(self.annotation_cell.get_id())
        control.move_focus(self.annotation_cell)

    def get_next_example(self):
        try:
            self.current_identifier, self.current_example = next(self.data_generator)
        except StopIteration:
            self.control.show_message_popup('Annotation finished!', 'You are done!')
            self.current_identifier = self.current_example = None

    def submit_annotation(self):
        content = self.annotation_cell.get()

        if self.current_example is not None:
            self.grouped[content].append(self.current_example)

            self.write_annotation(self.current_identifier, self.current_example, content)
        self.get_next_example()
        self.update_example()
        self.update_stats()
        self.annotation_cell.clear()

    def update_example(self):
        self.main_cell.set_title('Current example')
        self.main_cell.clear()
        self.render_example(self.current_example, self.main_cell)

    def update_stats(self):
        self.stats_cell.clear()
        total = sum(len(v) for v in self.grouped.values())
        ordered = [(k, len(v)) for k, v in self.grouped.items()]
        ordered.sort(key=lambda tup: tup[1], reverse=True)
        items = []
        for m, c in ordered[:self.top_k]:
            items.append('{} ({} -> {}%)'.format(m, c, round(c/total * 100, 2)))
        self.stats_cell.add_item_list(items)

    def update_detailed_stats(self):
        selected = self.stats_cell.get()
        self.main_cell.set_title('Detailed stats for: {}'.format(selected))
        self.main_cell.clear()
        ann = selected.split('(')[0].strip()
        self.selected_examples = self.grouped[ann][:self.top_k]
        for i, ex in enumerate(self.selected_examples):
            self.render_example_summary(ex, identifier=i, cell=self.main_cell)

    def update_detailed_example(self):
        if not self.main_cell.get_title().startswith('Detailed stats'):
            return
        selected = self.main_cell.get()
        index = int(selected.split(':')[0])
        ex = self.selected_examples[index]
        self.main_cell.set_title('Detailed example')
        self.main_cell.clear()
        self.render_example(ex, self.main_cell)

    def render_example_summary(self, ex, cell, identifier=None):
        if len(ex) > self.max_char:
            ex = ex[:self.max_char] + '...'
        if identifier is not None:
            ex = '{}: {}'.format(identifier, ex)
        cell.add_item(ex)

    def render_example(self, ex, cell, line_width=50):
        if ex is None:
            return
        items = [ex[i:i+line_width] for i in range(0, len(ex), line_width)]
        cell.add_item_list(items)

    def start(self):
        self.control.start()


def annotate():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', nargs='+', help='files to annotate')
    parser.add_argument('--dout', default='annotation', help='output directory')
    parser.add_argument('--name', default='Annotator', help='output directory')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    dout = Path(args.dout)
    fannotated = dout.joinpath('annotated.jsonl')

    if args.overwrite:
        if fannotated.exists():
            os.remove(fannotated)

    loader = Fileloader(args.files, pool=None)

    grouped = defaultdict(list)
    if fannotated.exists():
        with fannotated.open('rt') as f:
            for l in f:
                ann = json.loads(l)
                grouped[ann['annotation']].append(ann['example'])

    iterator = enumerate(loader.batch(1))
    def get_data():
        seen = sum(len(v) for v in grouped.values())
        for i, batch in iterator:
            if i < seen:
                continue
            yield (i, batch[0])

    fann = fannotated.open('at')
    def write_annotation(identifier, example, result):
        fann.write(json.dumps(dict(id=identifier, example=example, annotation=result)) + '\n')
        fann.flush()

    annotator = Annotator(get_data, write_annotation, grouped=grouped, name=args.name)
    annotator.start()
    fann.close()
