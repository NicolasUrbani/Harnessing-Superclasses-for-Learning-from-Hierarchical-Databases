import torchvision
import torch
from datasets import load_dataset, ClassLabel, DatasetDict

from .datamodule import DataModule
from .dataset import HFImageClassificationDataset


class TinyImagenetDataset(HFImageClassificationDataset):
    remap = {
        "n02666347": "n02666196",
        "n03373237": "n02190166",
        "n04465666": "n04465501",
        "n04598010": "n04597913",
        "n07056680": "n04067472",
        "n07646821": "n01855672",
        "n07647870": "n03250847",
        "n07657664": "n07579787",
        "n07975909": "n02206856",
        "n08496334": "n02730930",
        "n08620881": "n03976657",
        "n08742578": "n02085620",
        "n12520864": "n02906734",
        "n13001041": "n07734744",
        "n13652335": "n03804744",
        "n13652994": "n02999410",
        "n13719102": "n01945685",
        "n14991210": "n07747607",
    }

    def __init__(self, path, transform=None, split=None, labels=None):
        # Don't apply labels first
        super().__init__(path, transform=transform, split=split, labels=None)
        self.labels = labels

        # Remapping labels
        if split is None:
            old_names = self.dataset["train"].features["label"].names
            new_names = [self.remap[e] if e in self.remap else e for e in old_names]
            self.dataset = DatasetDict({
                k: dataset.cast_column("label", ClassLabel(names=new_names))
                for k, dataset in self.dataset.items()
            })
        else:
            old_names = self.dataset.features["label"].names
            new_names = [self.remap[e] if e in self.remap else e for e in old_names]
            self.dataset = self.dataset.cast_column("label", ClassLabel(names=new_names))

        # Apply labels now
        if self.labels is not None:
            mapping = self.get_mapping()
            self.dataset = self.dataset.align_labels_with_mapping(mapping, "label")


class TinyImagenetDataModule(DataModule):
    def __init__(self, datasets, loaders, transforms) -> None:
        super().__init__(datasets, loaders, transforms)
        self.labels = [e.pos_offset for e in self.datasets.hierarchy.leaves]

    def prepare_data(self):
        # Download dataset
        self.dataset = TinyImagenetDataset(self.datasets.path, labels=self.labels)

    def setup(self, stage=None):
        self.train_dataset = TinyImagenetDataset(
            self.datasets.path, transform=self.transforms.train, split="train", labels=self.labels
        )
        self.valid_dataset = TinyImagenetDataset(
            self.datasets.path, transform=self.transforms.valid, split="valid", labels=self.labels
        )
