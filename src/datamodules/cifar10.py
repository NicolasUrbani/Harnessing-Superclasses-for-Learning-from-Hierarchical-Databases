import hydra
import lightning as L
import torchvision
from omegaconf import DictConfig
from typing import Union, List, Dict
from torch.utils.data import DataLoader

from .datamodule import DataModule


class Cifar10DataModule(DataModule):
    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.datasets.data_dir, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.CIFAR10(
            self.datasets.data_dir,
            train=True,
            transform=self.transforms.train,
            download=False,
        )

        self.valid_dataset = torchvision.datasets.CIFAR10(
            self.datasets.data_dir,
            train=False,
            transform=self.transforms.valid,
            download=False,
        )
