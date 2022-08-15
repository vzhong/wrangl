"""
Autodocuments this library.
"""
import os
import tqdm
from ..cloud import s3


def add_parser_arguments(parser):
    parser.add_argument('--fconfig', default=os.environ.get('WRANGL_S3_FCREDENTIALS'), help='S3 config file')
    parser.add_argument('action', choices=('push', 'pull'), help='command')
    parser.add_argument('--local', help='local path, defaults to local directory', default=os.getcwd())
    parser.add_argument('--remote', help='remote path')
    parser.add_argument('--ignore_extensions', nargs='*', default=('.ckpt', 'pred_samples.json', '.log', 'hparams.yaml'), help='ignore these extensions')
    parser.add_argument('--ignore_directories', nargs='*', default=('wandb', '.hydra', 'submitit', 'lightning_logs'), help='ignore these subdirectories')


def should_ignore(src, ignore_extensions, ignore_directories):
    for ext in ignore_extensions:
        if src.endswith(ext):
            return True
    for folder in src.split('/'):
        if folder in ignore_directories:
            return True
    return False


def main(args):
    args.ignore_extensions = set(args.ignore_extensions)
    args.ignore_directories = set(args.ignore_directories)
    client = s3.S3Client(fcredentials=args.fconfig)
    print('{} local {} remote {}'.format(args.action, args.local, args.remote))
    if args.action == 'push':
        files = []
        if os.path.isfile(args.local):
            files.append(args.local)
        else:
            for root, _, fs in os.walk(args.local):
                for f in fs:
                    files.append(os.path.join(root, f))
        files = [f for f in files if not should_ignore(f, args.ignore_extensions, args.ignore_directories)]
        bar = tqdm.tqdm(files, desc='uploading')
        for src in bar:
            sub = src.replace(args.local, '').strip('/')
            tgt = os.path.join(args.remote, sub) if sub else args.remote
            client.client.fput_object(client.bucket, tgt, src)
            bar.set_description(sub)
        bar.close()
    elif args.action == 'pull':
        files = client.client.list_objects(client.bucket, recursive=True, prefix=args.remote)
        files = [f.object_name for f in files if not should_ignore(f.object_name, args.ignore_extensions, args.ignore_directories)]
        bar = tqdm.tqdm(files, desc='downloading')
        for tgt in bar:
            sub = tgt.replace(args.remote, '').strip('/')
            src = os.path.join(args.local, sub) if sub else args.local
            client.client.fget_object(client.bucket, tgt, src)
            bar.set_description(sub)
        bar.close()
