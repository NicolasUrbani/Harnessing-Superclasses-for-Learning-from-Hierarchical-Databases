import os

from sklearn.model_selection import train_test_split
import hydra
import lightning as L
import torchvision
from omegaconf import DictConfig
from typing import Union, List, Dict
from torch.utils.data import DataLoader, Subset
from sklearn.utils.random import sample_without_replacement


from src.tree import load_tree_from_file

from .datamodule import DataModule

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class Imagenet64SubsetDataModule(DataModule):

    def __init__(self, datasets, loaders, transforms, paths):
        super().__init__(datasets, loaders, transforms)
        self.paths = paths

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        tree_target = os.path.join(
        self.paths.hierarchy_dir, self.datasets.hierarchy_filename)
        tree = load_tree_from_file(tree_target)
        all_leaves = [leaf.name for leaf in tree.leaves]
        self.train_dataset = ImageFolderNotAlphabetic(
                    self.datasets.path+"train", classes=all_leaves, transform=self.transforms.train
                )

        self.valid_dataset = ImageFolderNotAlphabetic(
                    self.datasets.path+"val", classes=all_leaves, transform=self.transforms.valid
                )


        len_train = len(self.train_dataset)
        n_samples = len_train / self.datasets.reduction_factor
        print(len_train)
        print(n_samples)

        self.seed = self.datasets.seed if self.datasets.seed is not None else 42

        train_idx = sample_without_replacement(len_train, n_samples, random_state=self.seed)
        self.train_dataset = Subset(self.train_dataset, train_idx)

class ImageFolderNotAlphabetic(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root,
        classes,
        transform=None,
        target_transform=None,
        loader=torchvision.datasets.folder.default_loader,
        is_valid_file=None,
    ):
        self.classes = classes
        super(ImageFolderNotAlphabetic, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(self, directory):
        print("Override!")
        classes = self.classes
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
     # def find_classes(self, directory):
    #     classes = [get_class_code_from_label(c) for c in self.true_classes]
    #     if not classes:
    #         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    #     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    #     return classes, class_to_idx
