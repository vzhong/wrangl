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
        name = '/'.join(dlog.parts[-n:])
        flog = dlog.joinpath('metrics.log.json')
        if not flog.exists():
            logging.error('Could not find {}'.format(flog))
            continue
        with flog.open('rt') as f:
            raw_log = json.load(f)
        log = []
        for entry in raw_log:
            train = entry['train']
            train['type'] = 'train'
            val = entry['eval']
            val['type'] = 'eval'
            train['train_steps'] = val['train_steps'] = entry['train_steps']
            log.append(train)
            log.append(val)
        ret.append((name, log))
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
            for line in f:
                log.append(json.loads(line))
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
                df = pd.DataFrame([e for e in log if e['type'] == 'train' and e[args.x] is not None and e.get(args.y) is not None])
                fig.plot(
                    X=df[args.x],
                    Y=df[args.y].rolling(args.window, min_periods=1).mean(),
                    label='train {}'.format(exp),
                    **kwargs
                )
            if args.curves in {'eval', 'both'}:
                df = pd.DataFrame([e for e in log if e['type'] == 'eval' and e[args.x] is not None and e.get(args.y) is not None])
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
