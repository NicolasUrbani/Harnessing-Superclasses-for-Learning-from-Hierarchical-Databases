"""Resnet18 models for OOD"""

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def pretrained_model(**kwargs):
    kwargs["weights"] = ResNet18_Weights.DEFAULT
    return model(**kwargs)


def model(**kwargs):
    if "out_features" in kwargs:
        out_features = kwargs.pop("out_features")
        m = resnet18(**kwargs)
        m.fc = nn.Linear(m.fc.in_features, out_features)
        m.out_features = out_features
        m.last_layer_name = "fc"
    else:
        m = resnet18(**kwargs)

    return m


if __name__ == "__main__":
    model = model()
    import pdb

    pdb.set_trace()
