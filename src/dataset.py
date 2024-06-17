from torchvision import datasets


class ImageFolderOrdered(datasets.ImageFolder):
    """Regular image folder dataset with prescribed order of classes"""

    def __init__(
        self,
        root: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def find_classes(self):
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if set(classes) != set(self.classes):
            raise Exception("Mismatch between classes and folders")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        return self.classes, class_to_idx
