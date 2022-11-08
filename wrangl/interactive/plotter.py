import os
import csv
import typing
import pathlib
import logging
import plotille
import pandas as pd
import ujson as json


def add_parser_arguments(parser):
    parser.add_argument('dlogs', nargs='+', help='log directory')
    parser.add_argument('--type', choices=('rl', 'supervised', 'auto'), default='auto', help='type of experiment')
    parser.add_argument('--curves', choices=('train', 'eval', 'both'), default='eval', help='type of curve to plot')
    parser.add_argument('--window', help='smoothing window', type=int, default=1)
    parser.add_argument('--width', help='plot width', type=int, default=60)
    parser.add_argument('--height', help='plot height', type=int, default=10)
    parser.add_argument('-n', '--path_parts', help='how many last folders to use as name', type=int, default=2)
    parser.add_argument('-x', help='x axis', default='auto')
    parser.add_argument('-y', help='y axis', default='auto')


def detect_log_type(dlogs):
    dlog = dlogs[0]
    if dlog.joinpath('metrics.log.json').exists():
        return 'supervised'
    elif dlog.joinpath('metrics.log.jsonl').exists():
        return 'rl'
    else:
        return NotImplementedError('Unknown log type in {}'.format(dlog))


def load_supervised(dlogs: typing.List[pathlib.Path], n: int = 2):
    ret = []
    for dlog in dlogs:
        for root, dirs, files in os.walk(dlog):
            if 'metrics.csv' in files:
                fmetric = os.path.join(root, 'metrics.csv')
                log = []
                with open(fmetric, 'rt') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    key = '/'.join(root.split('/')[-n:])
                    for r in reader:
                        d = dict(zip(header, r))
                        d['train_steps'] = d['step']
                        del d['step']
                        for k, v in d.items():
                            x = pd.to_numeric(v, errors='ignore')
                            if not pd.isna(x):
                                d[k] = x
                        train = {k: v for k, v in d.items() if not k.startswith('val')}
                        train['type'] = 'train'
                        val = d.copy()
                        val['type'] = 'eval'
                        log.append(train)
                        log.append(val)
                ret.append((key, log))
    return ret


def load_rl(dlogs: typing.List[pathlib.Path], n: int = 2):
    ret = []
    for dlog in dlogs:
        name = '/'.join(dlog.parts[-n:])
        log = []
        flog = dlog.joinpath('metrics.log.jsonl')
        if not flog.exists():
            logging.error('Could not find {}'.format(flog))
            continue
        with flog.open('rt') as f:
            for i, line in enumerate(f):
                try:
                    log.append(json.loads(line))
                except Exception as e:
                    print('Could not load line {}: {}'.format(i+1, line))
                    raise e
        ret.append((name, log))
    return ret


def main(args):
    args.dlogs = [pathlib.Path(d) for d in args.dlogs]
    if args.type == 'auto':
        args.type = detect_log_type(args.dlogs)
    if args.x == 'auto':
        args.x = 'train_steps'
    if args.type == 'supervised':
        logs = load_supervised(args.dlogs, n=args.path_parts)
        if args.y == 'auto':
            args.y = 'loss'
    elif args.type == 'rl':
        logs = load_rl(args.dlogs, n=args.path_parts)
        if args.y == 'auto':
            args.y = 'mean_episode_return'
    else:
        raise NotImplementedError()

    fig = plotille.Figure()
    kwargs = {}
    for exp, log in logs:
        if log:
            if args.curves in {'train', 'both'}:
                df = pd.DataFrame([e for e in log if e['type'] == 'train' and e[args.x] not in {'', None} and e.get(args.y) not in {'', None}])
                fig.plot(
                    X=df[args.x],
                    Y=df[args.y].rolling(args.window, min_periods=1).mean(),
                    label='train {}'.format(exp),
                    **kwargs
                )
            if args.curves in {'eval', 'both'}:
                df = pd.DataFrame([e for e in log if e['type'] == 'eval' and e[args.x] not in {'', None} and e.get(args.y) not in {'', None}])
                fig.plot(
                    X=df[args.x],
                    Y=df[args.y].rolling(args.window, min_periods=1).mean(),
                    label='eval {}'.format(exp),
                    **kwargs
                )
    fig.x_label = args.x
    fig.y_label = args.y
    fig.height = args.height
    fig.width = args.width

    print(fig.show(legend=True))
