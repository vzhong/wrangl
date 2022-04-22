import minio
import pathlib
import tempfile
import os
import io
import json
import pandas as pd
from omegaconf import OmegaConf


class S3Client:
    """
    Instantiates a S3 Client.
    You must have a S3 bucket as well as API access.
    The following environment variables are required.

    - WRANGL_S3_URL: URL to API.
    - WRANGL_S3_KEY: API key.
    - WRANGL_S3_SECRET: API secret.
    - WRANGL_S3_BUCKET: bucket name.
    """

    def __init__(self, url=None, key=None, secret=None, bucket=None):
        url = url or os.environ['WRANGL_S3_URL']
        key = key or os.environ['WRANGL_S3_KEY']
        secret = secret or os.environ['WRANGL_S3_SECRET']
        self.bucket = bucket or os.environ['WRANGL_S3_BUCKET']
        self.client = minio.Minio(endpoint=url, access_key=key, secret_key=secret)

    def upload_content(self, project_id, experiment_id, fname, content, content_type='application/json'):
        data = io.BytesIO(content.encode('utf-8'))
        res = self.client.put_object(self.bucket, self.get_path(project_id, experiment_id, fname), data, length=len(content), content_type=content_type)
        return res

    def upload_file(self, project_id, experiment_id, fname, from_fname, content_type='application/json'):
        return self.client.fput_object(self.bucket, self.get_path(project_id, experiment_id, fname), from_fname, content_type=content_type)

    def upload_experiment(self, dexp):
        dexp = pathlib.Path(dexp)
        config = OmegaConf.to_container(OmegaConf.load(dexp.joinpath('config.yaml')), resolve=True)
        project_id = config['project']

        # make sure collection exists for project
        df = []
        for flog in dexp.joinpath('logs').glob('*/*.csv'):
            df.append(pd.read_csv(flog))
        if df:
            logs = pd.concat(df).to_dict()
        else:
            logs = {}

        # make sure document does not exist
        experiment_id = config['name']

        config_response = self.upload_content(project_id=project_id, experiment_id=experiment_id, fname='config.json', content=json.dumps(config, indent=2))
        log_response = self.upload_content(project_id=project_id, experiment_id=experiment_id, fname='logs.json', content=json.dumps(logs, indent=2))
        return dict(config=config_response, log=log_response)

    @classmethod
    def get_path(cls, project_id, experiment_id, fname):
        return '{}/{}/{}'.format(project_id, experiment_id, fname)

    def download_content(self, project_id, experiment_id, fname):
        obj = json.loads(self.client.get_object(self.bucket, self.get_path(project_id, experiment_id, fname)).data.decode())
        return obj

    def get_experiment(self, project_id, experiment_id):
        config = self.download_content(project_id, experiment_id, 'config.json')
        logs = self.download_content(project_id, experiment_id, 'logs.json')
        df = pd.DataFrame(logs)
        return config, df

    def plot_experiment(self, project_id, experiment_id, x='step', y='loss', **plot_kwargs):
        import seaborn as sns
        from matplotlib import pyplot as plt
        config, df = self.get_experiment(project_id=project_id, experiment_id=experiment_id)

        if not len(df):
            return

        sns.set(font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=x, y=y, data=df, ax=ax, **plot_kwargs)

        fname = '{}_vs_{}.pdf'.format(x, y)
        with tempfile.TemporaryDirectory() as tempdir_path:
            tempdir = pathlib.Path(tempdir_path)
            ffig = tempdir.joinpath(fname)
            fig.savefig(ffig, bbox_inches='tight')
            return self.upload_file(project_id, experiment_id, fname, ffig, content_type='application/pdf')


if __name__ == '__main__':
    mydir = pathlib.Path(os.path.abspath(__file__))
    dexp = mydir.parent.parent.joinpath('examples', 'learn', 'xor_clf', 'saves', 'wrangle-example-xor-clf', 'mymodel-default')
    client = S3Client()
    client.upload_experiment(dexp)
    config, df = client.get_experiment('wrangle-example-xor-clf', 'mymodel-default')
