from typing import Dict, List, Union

import hydra
import lightning as L
import torchvision
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.loaders = loaders
        self.transforms = transforms

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, **self.loaders.get("train"))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_dataset, **self.loaders.get("valid"))
