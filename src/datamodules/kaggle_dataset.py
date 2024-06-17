from typing import Dict, List, Union

import os
import hydra
import lightning as L
import torchvision
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import zipfile


class KaggleDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        os.environ["KAGGLE_CONFIG_DIR"] = cfg.paths.kaggle_config_dir
        from kaggle import api
        api.authenticate()
        self.api = api
        self.datasets = datasets

    def download_dataset(self):
        dataset_id = self.datasets.dataset_id
        data_dir = self.paths.data_dir

        type, id = dataset_id.split('/')

        if type in ["competitions", "c"]:
            self.api.competition_download_files(id, data_dir, quiet=False)
            zip_fname = data__dir + '/' + id + '.zip'
            extract_archive(zip_fname, data_dir)
            try:
                os.remove(zip_fname)
            except OSError as e:
                print('Could not delete zip file, got' + str(e))
        else:
            self.api.dataset_download_files(dataset_id, target_dir, quiet=False, unzip=True)


def extract_archive(zip_fname, effective_path):
    """Taken from kaggle library"""

    try:
        with zipfile.ZipFile(outfile) as z:
            z.extractall(effective_path)
    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)

    try:
        os.remove(outfile)
    except OSError as e:
        print('Could not delete zip file, got %s' % e)
