from typing import Dict, List, Union

import hydra
import lightning as L
import torchvision
from datasets import load_dataset, ClassLabel
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.random import sample_without_replacement



class HFImageClassificationDataset(Dataset):
    """Load a dataset from HuggingFace

    Reorder the labels according to `labels`.
    """

    def __init__(self, path, transform=None, split=None, labels=None, reduction=None, seed=42):
        super().__init__()

        self.dataset = load_dataset(path, split=split)

        self.labels = labels

        if labels is not None:
            mapping = self.get_mapping()
            self.dataset = self.dataset.align_labels_with_mapping(mapping, "label")

        self.transform = transform
        if reduction is not None and split=="train":
            len_train = len(self.dataset)
            n_samples = len_train / reduction
            print(len_train)
            print(n_samples)

            train_idx = sample_without_replacement(len_train, n_samples, random_state=seed)
            self.dataset = self.dataset.select(train_idx)

    def get_mapping(self):
        return {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]["image"]), self.dataset[idx]["label"]
