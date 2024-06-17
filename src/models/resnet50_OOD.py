"""Resnet50 models for OOD"""

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def model(**kwargs):
    out_features = kwargs.pop("out_features")

    m = resnet50(**kwargs)
    m.fc = nn.Linear(m.fc.in_features, out_features)
    m.out_features = out_features
    m.last_layer_name = "fc"
    return m

def pretrained_model(**kwargs):
    out_features = kwargs.pop("out_features")

    m = resnet50(weights=ResNet50_Weights.DEFAULT, **kwargs)
    m.fc = nn.Linear(m.fc.in_features, out_features)
    m.out_features = out_features
    m.last_layer_name = "fc"
    return m


if __name__ == "__main__":
    model = model()
    import pdb

    pdb.set_trace()
