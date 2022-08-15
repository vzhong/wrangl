"""
Autodocuments this library.
"""
import os
import csv
import json
import tqdm
from ..cloud import s3


def add_parser_arguments(parser):
    parser.add_argument('--fconfig', default=os.environ.get('WRANGL_S3_FCREDENTIALS'), help='S3 config file')
    parser.add_argument('action', choices=('push', 'pull', 'projpush'), help='command')
    parser.add_argument('src', help='source path')
    parser.add_argument('--tgt', help='target path, default to current directory')
    parser.add_argument('--proj', default='sync', help='project')
    parser.add_argument('--exp', default='default', help='project')


def main(args):
    client = s3.S3Client(fcredentials=args.fconfig)
    if args.tgt is None:
        args.tgt = os.path.join(os.getcwd(), os.path.basename(args.src))
    print('{} from {} to {}'.format(args.action, args.src, args.tgt))
    if args.action == 'push':
        client.upload_file(
            project_id=args.proj,
            experiment_id=args.exp,
            fname=args.tgt,
            from_fname=args.src,
            content_type='application/octet-stream',
        )
    elif args.action == 'pull':
        client.download_file(
            project_id=args.proj,
            experiment_id=args.exp,
            fname=args.tgt,
            from_fname=args.src,
            content_type='application/octet-stream',
        )
    elif args.action == 'projpush':
        match = []
        for root, _, files in os.walk(args.src):
            if root.endswith('.hydra') or '/wandb/' in root:
                continue
            for fname in files:
                if fname == 'config.yaml':
                    match.append((root))
        bar = tqdm.tqdm(match, desc='uploading projects')
        for root in bar:
            bar.set_description(root)
            exp_id = os.path.basename(root)
            for tgt in ['config.yaml']:
                src = os.path.join(root, tgt)
                client.upload_file(
                    project_id=args.proj,
                    experiment_id=exp_id,
                    fname=tgt,
                    from_fname=src,
                    content_type='application/octet-stream',
                )
            # concat logs
            log = []
            for logdir in os.listdir(os.path.join(root, 'logs')):
                flog = os.path.join(root, 'logs', logdir, 'metrics.csv')
                if os.path.isfile(flog):
                    with open(flog, 'rt') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        for row in reader:
                            log.append(dict(zip(header, row)))
            client.upload_content(args.proj, exp_id, 'metrics.json', content=json.dumps(log))
        bar.close()
