import minio
import pathlib
import tempfile
import os
import io
import json
import pandas as pd
from omegaconf import OmegaConf
from typing import Tuple, Union, List


class S3Client:
    """
    Instantiates a Minio Client.
    You must have a Minio-compatible S3 bucket as well as API access.
    The following environment variables are used as default, unless a credentials file is specified with the same content in a JSON file.

    - `WRANGL_S3_URL`: URL to API endpoint.
    - `WRANGL_S3_KEY`: API key.
    - `WRANGL_S3_SECRET`: API secret.
    - `WRANGL_S3_BUCKET`: bucket name.
    """

    def __init__(self, url=None, key=None, secret=None, bucket=None, fcredentials=None):
        """
        Args:
            url: S3 endpoint.
            key: S3 access key.
            secret: S3 secret key.
            bucket: S3 bucket name.
            fcredentials: alternatively specify these infomation in the form of `WRANGL_S3_{}` dictionary in a JSON credentials file.
        """
        if fcredentials:
            with open(fcredentials) as f:
                credentials = json.load(f)
        else:
            credentials = os.environ
        url = url or credentials['WRANGL_S3_URL']
        key = key or credentials['WRANGL_S3_KEY']
        secret = secret or credentials['WRANGL_S3_SECRET']
        self.bucket = bucket or credentials['WRANGL_S3_BUCKET']
        self.client = minio.Minio(endpoint=url, access_key=key, secret_key=secret)

    def upload_content(self, project_id: str, experiment_id: str, fname: str, content: str, content_type='application/json'):
        """
        Uploads raw content to a project's experiment into a file `fname`.

        The file will be accessible at `<WRANGL_S3_URL>/<WRANGL_S3_BUCKET>/<project_id>/<experiment_id>/<fname>`.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
            fname: name of file in S3 directory.
            content: content to upload.
            content_type: S3 content type.
        """
        data = io.BytesIO(content.encode('utf-8'))
        res = self.client.put_object(self.bucket, self.get_path(project_id, experiment_id, fname), data, length=len(content), content_type=content_type)
        return res

    def upload_file(self, project_id: str, experiment_id: str, fname: str, from_fname: str, content_type='application/json'):
        """
        Uploads a file to a project's experiment.

        The file will be accessible at `<WRANGL_S3_URL>/<WRANGL_S3_BUCKET>/<project_id>/<experiment_id>/<fname>`.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
            fname: name of file in S3 directory.
            from_fname: local path to file to upload.
            content_type: S3 content type.
        """
        return self.client.fput_object(self.bucket, self.get_path(project_id, experiment_id, fname), from_fname, content_type=content_type)

    def download_file(self, project_id: str, experiment_id: str, fname: str, from_fname: str, content_type='application/json'):
        """
        Downloads a file from a project's experiment.

        The file will be downloaded from `<WRANGL_S3_URL>/<WRANGL_S3_BUCKET>/<project_id>/<experiment_id>/<fname>`.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
            fname: local path to file to download.
            from_fname: name of file in S3 directory.
            content_type: S3 content type.
        """
        data = self.client.get_object(self.bucket, self.get_path(project_id, experiment_id, from_fname)).data
        with open(fname, 'wb') as f:
            f.write(data)

    def upload_experiment(self, dexp):
        """
        Uploads an experiment, including its config and logs.

        Args:
            dexp: local path to experiment directory.
        """
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
        """
        Returns the relative path of file in S3.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
            fname: name of file to look up.
        """
        return '{}/{}/{}'.format(project_id, experiment_id, fname)

    def download_content(self, project_id, experiment_id, fname):
        """
        Returns the JSON object corresponding to the uploaded file at `<WRANGL_S3_URL>/<WRANGL_S3_BUCKET>/<project_id>/<experiment_id>/<fname>`.
        """
        obj = json.loads(self.client.get_object(self.bucket, self.get_path(project_id, experiment_id, fname)).data.decode())
        return obj

    def get_experiment(self, project_id: str, experiment_id: str) -> Tuple[OmegaConf, pd.DataFrame]:
        """
        Returns the experiment's config and logs.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
        """
        config = self.download_content(project_id, experiment_id, 'config.json')
        logs = self.download_content(project_id, experiment_id, 'logs.json')
        df = pd.DataFrame(logs)
        return config, df

    def plot_experiment(self, project_id: str, experiment_id: str, x='step', y: Union[str, List[str]] = 'loss', **plot_kwargs):
        """
        Generates a line plot and uploads the result to `<WRANGL_S3_URL>/<WRANGL_S3_BUCKET>/<project_id>/<experiment_id>/<x>_vs_<y>.pdf`.
        You must have `seaborn` installed for this to work.

        Args:
            project_id: name of project.
            experiment_id: name of experiment.
            x: x axis to plot.
            y: y axis (or a list of axes) to plot.
            plot_kwargs: keyword arguments to `seaborn.lineplot`.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt
        config, df = self.get_experiment(project_id=project_id, experiment_id=experiment_id)

        if not len(df):
            return

        sns.set(font_scale=1.5)

        if isinstance(y, str):
            y = [y]

        for yi in y:
            yi = yi.strip()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=x, y=yi, data=df, ax=ax, **plot_kwargs)

            fname = '{}_vs_{}.pdf'.format(x.replace('/', '--'), yi.replace('/', '--'))
            with tempfile.TemporaryDirectory() as tempdir_path:
                tempdir = pathlib.Path(tempdir_path)
                ffig = tempdir.joinpath(fname)
                fig.savefig(ffig, bbox_inches='tight')
                self.upload_file(project_id, experiment_id, fname, ffig, content_type='application/pdf')


if __name__ == '__main__':
    mydir = pathlib.Path(os.path.abspath(__file__))
    dexp = mydir.parent.parent.joinpath('examples', 'learn', 'xor_clf', 'saves', 'wrangle-example-xor-clf', 'mymodel-default')
    client = S3Client()
    client.upload_experiment(dexp)
    config, df = client.get_experiment('wrangle-example-xor-clf', 'mymodel-default')
