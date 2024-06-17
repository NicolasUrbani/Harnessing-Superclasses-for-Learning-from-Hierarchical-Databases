"""Resnet50 models for OOD"""

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def model(**kwargs):
    m = resnet50(weights=ResNet50_Weights.DEFAULT, **kwargs)
    return m


if __name__ == "__main__":
    model = model()
    import pdb

    pdb.set_trace()
