"""DenseNet121 models with custom last layer for OOD"""

from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights


def model(**kwargs):
    out_features = kwargs.pop("out_features")

    m = densenet121(weights=DenseNet121_Weights.DEFAULT, **kwargs)
    m.classifier = nn.Linear(m.classifier.in_features, out_features)
    return m


if __name__ == "__main__":
    model = model()
    import pdb

    pdb.set_trace()
