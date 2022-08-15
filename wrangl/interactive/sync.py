"""
Autodocuments this library.
"""
import os
import tqdm
from ..cloud import s3


def add_parser_arguments(parser):
    parser.add_argument('--fconfig', default=os.environ.get('WRANGL_S3_FCREDENTIALS'), help='S3 config file')
    parser.add_argument('action', choices=('push', 'pull', 'projpush', 'projpull'), help='command')
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
            client.upload_experiment(root, overwrite_project=args.proj)
        bar.close()
    elif args.action == 'projpull':
        match = client.list_experiments(args.src)
        bar = tqdm.tqdm(match, desc='downloading projects')
        for dexp in bar:
            bar.set_description(dexp)
            exp_id = os.path.basename(dexp)
            proj_id = os.path.dirname(dexp)
            client.download_experiment(proj_id, exp_id, dout=args.tgt)
            client.get_experiment(proj_id, exp_id)
        bar.close()
