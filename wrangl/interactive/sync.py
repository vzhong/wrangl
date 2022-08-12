"""
Autodocuments this library.
"""
import os
from ..cloud import s3


def add_parser_arguments(parser):
    parser.add_argument('--fconfig', default=os.environ.get('WRANGL_S3_FCREDENTIALS'), help='S3 config file')
    parser.add_argument('action', choices=('push', 'pull'), help='command')
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
