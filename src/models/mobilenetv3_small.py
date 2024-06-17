"""Resnet50 models for OOD"""

from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def model(**kwargs):
    out_features = kwargs.pop("out_features")

    m = mobilenet_v3_small(**kwargs)
    m.fc = nn.Linear(m.fc.in_features, out_features)
    m.out_features = out_features
    m.last_layer_name = "fc"
    return m

def pretrained_model(**kwargs):
    out_features = kwargs.pop("out_features")

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT, **kwargs)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, out_features)
    m.out_features = out_features
    m.last_layer_name = "classifier.3"

    return m


if __name__ == "__main__":
    model = model()
    import pdb

    pdb.set_trace()
