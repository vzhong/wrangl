from appwrite.client import Client as Base
from appwrite.services import storage
from appwrite.exception import AppwriteException
import pathlib
import tempfile
import os
import json
import pandas as pd
from omegaconf import OmegaConf


class AppwriteClient:
    """
    Instantiates an Appwrite Client.
    You must have an Appwrite server as well as API access.
    The following environment variables are required.

    - WRANGL_APPWRITE_URL: URL to the Appwrite API.
    - WRANGL_APPWRITE_SECRET: API secret.
    - WRANGL_APPWRITE_PROJECT: Appwrite project id - you must create this project ID in Appwrite first.
    """

    def __init__(self, url=os.environ['WRANGL_APPWRITE_URL'], secret=os.environ['WRANGL_APPWRITE_SECRET'], project=os.environ['WRANGL_APPWRITE_PROJECT']):
        self.appwrite_client = Base()
        self.appwrite_client.set_endpoint(url)
        self.appwrite_client.set_project(project)
        self.appwrite_client.set_key(secret)

    def upload_content(self, bucket_id, file_id, content, overwrite=False):
        with tempfile.TemporaryDirectory() as tempdir_path:
            tempdir = pathlib.Path(tempdir_path)
            fname = tempdir.joinpath(file_id)
            with fname.open('wt') as f:
                f.write(content)
            return self.upload_file(bucket_id=bucket_id, fname=fname, overwrite=overwrite)

    def upload_file(self, bucket_id, fname, overwrite=False):
        file_id = os.path.basename(fname)
        s = storage.Storage(self.appwrite_client)
        if overwrite:
            try:
                s.delete_file(bucket_id=bucket_id, file_id=file_id)
            except AppwriteException:
                pass
        response = s.create_file(bucket_id=bucket_id, file_id=file_id, file=fname)
        return response

    def upload_experiment(self, dexp):
        s = storage.Storage(self.appwrite_client)
        dexp = pathlib.Path(dexp)
        config = OmegaConf.to_container(OmegaConf.load(dexp.joinpath('config.yaml')), resolve=True)
        project_id = config['project']

        # make sure collection exists for project
        try:
            s.create_bucket(
                bucket_id=project_id,
                name=project_id,
                permission='file'
            )
        except AppwriteException:
            # already exists
            pass

        df = []
        for flog in dexp.joinpath('logs').glob('*/*.csv'):
            df.append(pd.read_csv(flog))
        if df:
            logs = pd.concat(df).to_dict()
        else:
            logs = {}

        # make sure document does not exist
        experiment_id = config['name']
        fconfig = '{}.config.json'.format(experiment_id)
        flog = '{}.log.json'.format(experiment_id)

        config_response = self.upload_content(bucket_id=project_id, file_id=fconfig, content=json.dumps(config, indent=2), overwrite=True)
        log_response = self.upload_content(bucket_id=project_id, file_id=flog, content=json.dumps(logs, indent=2), overwrite=True)
        return dict(config=config_response, log=log_response)

    def get_experiment(self, project_id, experiment_id):
        fconfig = '{}.config.json'.format(experiment_id)
        flog = '{}.log.json'.format(experiment_id)
        s = storage.Storage(self.appwrite_client)
        raw_config = s.get_file_download(bucket_id=project_id, file_id=fconfig)
        raw_logs = s.get_file_download(bucket_id=project_id, file_id=flog)
        config = raw_config
        logs = json.loads(raw_logs)
        df = pd.DataFrame(logs)
        return config, df

    def plot_experiment(self, project_id, experiment_id, x='step', y='loss', **plot_kwargs):
        import seaborn as sns
        from matplotlib import pyplot as plt
        config, df = self.get_experiment(project_id=project_id, experiment_id=experiment_id)

        sns.set(font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=x, y=y, data=df, ax=ax, **plot_kwargs)

        fname = '{}_vs_{}.pdf'.format(x, y)
        with tempfile.TemporaryDirectory() as tempdir_path:
            tempdir = pathlib.Path(tempdir_path)
            ffig = tempdir.joinpath(fname)
            fig.savefig(ffig, bbox_inches='tight')
            return self.upload_file(bucket_id=project_id, fname=ffig, overwrite=True)


if __name__ == '__main__':
    cloud = AppwriteClient()
    print(cloud.upload_experiment('/Users/victor/project/wrangl/wrangl/examples/learn/xor_clf/saves/wrangle-example-xor-clf/mymodel-default/'))
    print(cloud.plot_experiment('wrangle-example-xor-clf', 'mymodel-default'))
